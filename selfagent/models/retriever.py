"""Finding the passages an answer will be built from.

BM25 over an inverted index, which is a lexical match rather than a semantic one. That is the right
first stage and not a placeholder: it needs no trained weights, so it works before the encoder does,
and a finance question carries the rare terms BM25 is strongest on — a ticker, "demat", "1099-B",
"repo corridor". A neural reranker over these candidates is the second stage and comes later; it
reorders what this finds, so this stage sets the ceiling on everything downstream.

Scoring walks posting lists rather than a passage-by-passage loop. With 117k passages a Python loop
per query was the difference between minutes and hours over 62k training questions.
"""

from collections import defaultdict

from ..backend import xp

# Standard BM25 constants. k1 bounds how much repeating a term can help, b how much a long passage is
# penalised for it. Left at the published defaults because we have no relevance judgements to tune on
# — S13 is what would earn a different value.
K1 = 1.5
B = 0.75


def embed(model, tokenizer, texts, length):
    """One unit vector per text from the frozen encoder, so a dot product is a cosine.

    Mean over the real tokens, not [CLS]: masked language modelling never gives [CLS] a job, so it comes
    out near constant and every pair of texts scores above 0.97. The decoder cross-attends to all the
    token states anyway, so this is the representation it actually works from.
    """
    ids = xp.zeros((len(texts), length), dtype=xp.int64)
    keep = xp.zeros((len(texts), length), dtype=xp.uint8)
    for row, text in enumerate(texts):
        encoded = tokenizer.encode(text)[:length]
        ids[row, : len(encoded)] = encoded
        keep[row, : len(encoded)] = 1
    # float64 before any reduction: the encoder runs in a narrower dtype and the norms overflow.
    states = xp.asarray(model.text(ids, keep).data, dtype=xp.float64)
    pooled = (states * keep[..., None]).sum(axis=1) / keep.sum(axis=1, keepdims=True)
    return pooled / xp.linalg.norm(pooled, axis=-1, keepdims=True)


class BM25:
    """An index over passages, searchable by query text.

    `owners` labels each passage with the thread it came from, so a caller can forbid the thread it
    is currently building a training example for. Without that the index returns the answer the model
    is being trained to write.
    """

    def __init__(self, passages, tokenize, owners=None):
        if owners is not None and len(owners) != len(passages):
            raise ValueError(f"{len(owners)} owners for {len(passages)} passages; they must match")
        self.owners = list(owners) if owners is not None else None
        self.tokenize = tokenize

        postings = defaultdict(lambda: defaultdict(int))
        lengths = xp.zeros(len(passages))
        for index, passage in enumerate(passages):
            terms = tokenize(passage)
            lengths[index] = len(terms)
            for term in terms:
                postings[term][index] += 1

        self.lengths = lengths
        # A pool of empty passages would divide by zero here, and the caller cannot see why.
        self.average_length = float(lengths.mean()) if len(passages) and lengths.sum() else 1.0
        self.count = len(passages)
        self.postings = {
            term: (
                xp.fromiter(documents.keys(), dtype=int, count=len(documents)),
                xp.fromiter(documents.values(), dtype=float, count=len(documents)),
            )
            for term, documents in postings.items()
        }
        self.inverse_document_frequency = {
            term: float(xp.log(1.0 + (self.count - len(documents[0]) + 0.5) / (len(documents[0]) + 0.5)))
            for term, documents in self.postings.items()
        }

    def scores(self, query):
        """A BM25 score per passage. Zero means no query term appears in it at all."""
        total = xp.zeros(self.count)
        for term in set(self.tokenize(query)):
            if term not in self.postings:
                continue
            documents, frequencies = self.postings[term]
            normalised = 1.0 - B + B * self.lengths[documents] / self.average_length
            contribution = frequencies * (K1 + 1.0) / (frequencies + K1 * normalised)
            total[documents] += self.inverse_document_frequency[term] * contribution
        return total

    def search(self, query, top_k, exclude_owner=None):
        """Indices of the best passages, best first, skipping anything scoring zero.

        Returning fewer than top_k is deliberate. Padding the list out with unmatched passages would
        hand the generator text that has nothing to do with the question, and the honest response to
        a question we hold no evidence for is to abstain.
        """
        total = self.scores(query)
        if exclude_owner is not None and self.owners is not None:
            for index, owner in enumerate(self.owners):
                if owner == exclude_owner:
                    total[index] = 0.0
        # argpartition beats a full sort: we want the top handful out of >100k passages.
        limit = min(top_k, self.count)
        candidates = xp.argpartition(-total, limit - 1)[:limit] if limit < self.count else xp.arange(self.count)
        ranked = candidates[xp.argsort(-total[candidates])]
        return [int(index) for index in ranked if total[index] > 0.0]
