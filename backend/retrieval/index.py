"""Hybrid retrieval over the chunks: lexical and vector, fused by rank, filtered by date.

Two halves, because they fail differently. BM25 finds the rare term -- a ticker, "repo corridor",
"1099-B" -- and is blind to a question that shares no words with the passage it wants. The encoder finds
the paraphrase and is weakest on exactly those rare terms, because a term seen twice in pretraining has
no useful vector.

**Measured, the vector half does not pay, so `SERVED` is the lexical arm.** On 400 queries over a
44,811-chunk index: lexical recall@5 15.5%, hybrid 12.0%, vector 1.5%, a random picker 0.0%. Fusing a
working retriever with a near-random one costs 3.5 points, which is what fusion does when one input
carries no signal. The vector arm stays because it is the evidence for that sentence and the thing a
trained retriever would replace -- `scripts/eval_retrieval.py` re-runs the comparison -- not because
anything calls it.

Why it fails is worth keeping: an encoder trained only to fill in masked words is never asked to push
unlike passages apart, and it does not. Unrelated chunks sit at cosine 0.925 and the mean embedding has
norm 0.942 out of 1, so nearly every direction is the same direction. Centring fixes the geometry
outright (0.925 to 0.142) and moves recall by half a point, which says the collapse was a symptom and the
missing contrastive objective is the cause.

**Fused by rank, not by score.** A BM25 score is unbounded and a cosine sits in [-1, 1], so a weighted
sum of the two needs a scale constant nothing here has measured. Reciprocal rank fusion needs none: it
reads only the position each half put a chunk in, so the two halves cannot be mis-weighted by accident.

**The date filter runs before ranking, not after.** Filtering a top-5 list down to what existed on the
day would return two results and call it five. The mask is applied to the scores, so the top-5 is the
best five that existed.
"""

from selfagent.backend import xp
from selfagent.models.retriever import BM25

from .chunk import Chunk

# The published RRF constant. It sets how fast rank 1 stops mattering more than rank 2; left at 60
# because we have no relevance judgements to tune it on, which is the same reason BM25's k1 and b are
# at their defaults.
RRF_K = 60

LEXICAL, VECTOR, HYBRID = "lexical", "vector", "hybrid"
ARMS = (LEXICAL, VECTOR, HYBRID)

# What `search` uses when a caller does not choose, set by the measurement above rather than by taste.
SERVED = LEXICAL


def save(path, chunks, vectors=None):
    """One .npz holding the chunks and, when they were embedded, their vectors.

    Texts go in as a newline-joined blob, which is lossless because chunking collapsed whitespace. Joined
    rather than stored as an object array so the file loads without `allow_pickle`: an index is data, and
    data that can execute on load is not worth the convenience.
    """
    texts = [chunk.text for chunk in chunks]
    if any("\n" in text for text in texts):
        raise ValueError("a chunk contains a newline; the joined blob would not round-trip")
    xp.savez_compressed(
        path,
        text="\n".join(texts),
        document=xp.array([chunk.document for chunk in chunks]),
        day=xp.array([chunk.day for chunk in chunks]),
        **({} if vectors is None else {"vectors": vectors}),
    )


def load(path):
    """(chunks, vectors or None) from a saved index. The ids are not stored: only the embedder needed
    them, and keeping 40k × 128 of them would double the file to save re-tokenising a text nobody
    re-embeds."""
    held = xp.load(path)
    texts = str(held["text"]).split("\n")
    if not len(texts) == len(held["document"]) == len(held["day"]):
        raise ValueError(f"{path} is inconsistent: {len(texts)} texts, {len(held['document'])} documents")
    chunks = [Chunk(text, [], str(document), str(day))
              for text, document, day in zip(texts, held["document"], held["day"])]
    return chunks, held["vectors"] if "vectors" in held else None


def centred(vectors):
    """The corpus mean direction, the one every embedding in a collapsed space mostly points along."""
    return xp.asarray(vectors, dtype=xp.float64).mean(axis=0)


def without(vectors, centre):
    """`vectors` with the common direction removed and renormalised, so a dot product is still a cosine.

    Measured on this index: unrelated chunks sit at cosine 0.925 raw and 0.142 centred, and the mean
    embedding itself has norm 0.942 out of 1. An encoder trained only to fill in masked words is never
    asked to push unlike passages apart, so it does not, and subtracting the direction they share is the
    whole of the fix. It costs one vector and no training, which is why it is on by default.
    """
    moved = xp.asarray(vectors, dtype=xp.float64) - centre
    moved /= xp.linalg.norm(moved, axis=-1, keepdims=True)
    return moved.astype(xp.asarray(vectors).dtype)


def fuse(rankings, k=RRF_K):
    """Reciprocal rank fusion: {index: score} from several best-first lists of indices."""
    fused = {}
    for ranking in rankings:
        for position, index in enumerate(ranking):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + position + 1)
    return fused


class Hybrid:
    """An index over `Chunk`s, searchable by one arm at a time.

    `vectors` and `embed_query` arrive together: a matrix with no way to embed a query is unusable, and
    making one optional without the other would fail at the first vector search instead of here.
    """

    def __init__(self, chunks, tokenize, vectors=None, embed_query=None, k=RRF_K, centre=True):
        if (vectors is None) != (embed_query is None):
            raise ValueError("vectors and embed_query must be given together or not at all")
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks; they must match")
        self.chunks = list(chunks)
        self.lexical = BM25([chunk.text for chunk in self.chunks], tokenize)
        self.centre = centred(vectors) if centre and vectors is not None else None
        self.vectors = vectors if self.centre is None else without(vectors, self.centre)
        # Checked once here so the searches can trust it: a single NaN row would silently win every
        # comparison it is in, and a half-written index is the likeliest way to get one.
        if self.vectors is not None and not xp.isfinite(self.vectors).all():
            raise ValueError("the index holds a non-finite vector; rebuild it with scripts/build_index.py")
        self.embed_query = embed_query
        self.k = k
        # Compared as strings: an ISO date orders correctly that way, and parsing 200k of them into
        # objects would cost more than every search this index will serve.
        self.days = xp.array([chunk.day for chunk in self.chunks])

    def visible(self, as_of=None):
        """A 1/0 mask over the chunks: what a question asked on `as_of` is allowed to see."""
        if as_of is None:
            return xp.ones(len(self.chunks), dtype=bool)
        return self.days <= as_of

    def search(self, query, top_k, as_of=None, arm=SERVED):
        """Chunk indices, best first, for one arm. Shorter than `top_k` when little enough matched."""
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        if arm != LEXICAL and self.vectors is None:
            raise ValueError(f"the {arm} arm needs vectors, and this index was built without them")
        allowed = self.visible(as_of)
        if not allowed.any():
            return []
        if arm == HYBRID:
            # Each half is asked for top_k, then fusion reorders the union. Asking each for fewer would
            # let a chunk both halves rank sixth beat nothing, which is the case fusion exists for.
            ranked = fuse([self._lexical(query, top_k, allowed), self._vector(query, top_k, allowed)],
                          self.k)
            best = sorted(ranked, key=lambda index: -ranked[index])
            return best[:top_k]
        return (self._lexical if arm == LEXICAL else self._vector)(query, top_k, allowed)

    def cite(self, index):
        """The chunk at `index` with its document key and date, for the answer's evidence line."""
        return self.chunks[index]

    def _lexical(self, query, top_k, allowed):
        scores = self.lexical.scores(query) * allowed
        return self._best(scores, top_k, above=0.0)

    def _vector(self, query, top_k, allowed):
        wanted = self.embed_query(query)
        if self.centre is not None:
            wanted = without(wanted[None, :], self.centre)[0]
        # Cast to the matrix dtype: mixing float32 and float64 upcasts the whole index per query, which
        # is a 95 MB temporary for 46k chunks, and makes numpy report BLAS's own status flags as errors.
        wanted = xp.asarray(wanted, dtype=self.vectors.dtype)
        # Rows are unit vectors, so the dot product is the cosine. Below -1 is below any real score,
        # which is how a masked chunk is kept out without deciding what its similarity "should" be.
        # errstate because Apple's BLAS raises its own status flags out of a gemv on finite operands, and
        # numpy reports them as this line's fault. The inputs were checked finite when the index was built.
        with xp.errstate(all="ignore"):
            scores = xp.where(allowed, self.vectors @ wanted, -2.0)
        # No floor, unlike the lexical arm: a BM25 zero means no query term appears anywhere in the
        # passage, where a cosine of zero means nothing in particular -- and after centring half of them
        # are negative. So this arm always returns top_k and never abstains, which the lexical one does.
        return self._best(scores, top_k, above=-2.0)

    @staticmethod
    def _best(scores, top_k, above):
        """The top_k indices scoring above a floor, best first, and fewer when fewer qualify.

        Padding the list out with unmatched chunks would hand the generator text that has nothing to do
        with the question, and the honest answer to a question we hold no evidence for is to abstain.
        """
        limit = min(top_k, len(scores))
        candidates = (xp.argpartition(-scores, limit - 1)[:limit] if limit < len(scores)
                      else xp.arange(len(scores)))
        ranked = candidates[xp.argsort(-scores[candidates])]
        return [int(index) for index in ranked if scores[index] > above]
