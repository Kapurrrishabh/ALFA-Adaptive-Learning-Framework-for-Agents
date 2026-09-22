#!/usr/bin/env python3
"""Does the vector half earn its place next to BM25? C2's gate, measured before either is served.

**The relevance judgements are the hard part, and there are none.** Nobody has labelled which passage
answers which question, and inventing labels with the model under test would measure the model against
itself. So this uses the standard weak-supervision proxy and names it as one: *find the rest of the
document*. One sentence sampled from a document is the query, and a hit is any returned chunk from that
same document, with every chunk containing the query sentence excluded from the target set.

Excluded because otherwise the task is a copy: BM25 would be scoring a passage against words lifted
straight out of it, which is not retrieval, it is string search, and the vector half would lose to an
artefact. With the source chunk removed the query and the target share only what the document is about.

What this does **not** measure is answer-bearing relevance. A Fed press release ends with the same
contact line every time, so a query from one release can find another release's boilerplate and count as
a hit. That inflates every arm equally, which is why the comparison is arm against arm and the random
baseline is printed beside them rather than the absolute number being quoted on its own.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.retrieval import ARMS, LEXICAL, Hybrid, load  # noqa: E402
from backend.retrieval.chunk import sentences  # noqa: E402
from selfagent import pretrained  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.models.retriever import embed  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece, pretokenize  # noqa: E402

# A query has to be a sentence with something in it. Under this it is a heading or a date line, and
# every arm is being asked to find a document from the word "Share".
MIN_QUERY_WORDS = 8


def queries(chunks, rng, wanted, distinctiveness):
    """(query sentence, the chunk indices that count as a hit) for documents with something to find.

    The query sentence is dropped from the hit set wherever it appears, which is what stops the task
    being a string search. A document whose other chunks all contain it is skipped entirely.

    The **most distinctive** sentence of the document is the query, not a random one, scored by the
    index's own inverse document frequency. A random sentence out of a Fed press release is usually its
    standing boilerplate -- "for media inquiries, call 202-452-2955" -- and asking any retriever to find
    a particular release from a line that appears in four thousand of them measures nothing. It is also
    the more faithful task: a reader asks about the specific thing in a document, not its letterhead.
    """
    by_document = {}
    for index, chunk in enumerate(chunks):
        by_document.setdefault(chunk.document, []).append(index)
    usable = [key for key, held in by_document.items() if len(held) > 1]
    if not usable:
        raise ValueError("no document has two chunks; there is nothing to retrieve")

    built = []
    for key in (usable[index] for index in rng.permutation(len(usable))):
        held = by_document[key]
        pool = [sentence for index in held for sentence in sentences(chunks[index].text)
                if len(sentence.split()) >= MIN_QUERY_WORDS]
        if not pool:
            continue
        query = max(pool, key=distinctiveness)
        targets = [index for index in held if query not in chunks[index].text]
        if targets:
            built.append((query, set(targets)))
        if len(built) == wanted:
            break
    return built


def recall(index, asked, top_k, arm):
    """(recall@k, mean reciprocal rank, mean results returned) for one arm."""
    hits, reciprocal, returned = [], [], []
    for query, targets in asked:
        found = index.search(query, top_k, arm=arm)
        returned.append(len(found))
        ranked = [position for position, result in enumerate(found, 1) if result in targets]
        hits.append(bool(ranked))
        reciprocal.append(1.0 / ranked[0] if ranked else 0.0)
    return float(np.mean(hits)), float(np.mean(reciprocal)), float(np.mean(returned))


def chance(chunks, asked, top_k, rng):
    """What recall@k a uniformly random picker gets, which is the floor every arm has to clear."""
    hits = [bool(set(rng.permutation(len(chunks))[:top_k]) & targets) for _, targets in asked]
    return float(np.mean(hits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--index", default="index.npz")
    parser.add_argument("--checkpoint", default="advisory_combined.npz",
                        help="the encoder queries are embedded with; must be the one the index used")
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-centring", action="store_true",
                        help="also measure the raw vectors, which is the run that decided the default")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    chunks, vectors = load(artifacts / args.index)
    if vectors is None:
        raise ValueError(f"{args.index} holds no vectors; rebuild it with a --checkpoint")
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    config, weights = pretrained.load(artifacts / args.checkpoint)
    model = GroundedGenerator(config)
    model.load_pretrained_encoder(weights)
    model.eval()

    def embed_query(text):
        return embed(model, tokenizer, [text], config.max_text_length)[0]

    rng = np.random.default_rng(args.seed)
    built = {centre: Hybrid(chunks, pretokenize, vectors, embed_query, centre=centre)
             for centre in ((True, False) if args.compare_centring else (True,))}
    # The queries are chosen with the index's own idf and are the same set for every arm: a comparison
    # where each arm is asked different questions is not a comparison.
    rare = next(iter(built.values())).lexical.inverse_document_frequency
    asked = queries(chunks, rng, args.queries,
                    lambda text: np.mean([rare.get(term, 0.0) for term in pretokenize(text)] or [0.0]))
    documents = len({chunk.document for chunk in chunks})
    print(f"{len(chunks)} chunks over {documents} documents, {len(asked)} queries, "
          f"recall@{args.top_k}")

    scored = {}
    for centre, index in built.items():
        seen = index.vectors[:200].astype(np.float64)
        # Same BLAS flag quirk the index comments: finite operands, flags raised inside the gemv.
        with np.errstate(all="ignore"):
            unrelated = (seen @ seen.T)[~np.eye(len(seen), dtype=bool)]
        print(f"\n  vectors {'centred' if centre else 'raw    '}, unrelated chunks at cosine "
              f"{unrelated.mean():+.3f}")
        print(f"  {'arm':9}{'recall':>9}{'MRR':>8}{'results returned':>19}")
        for arm in ARMS:
            found, reciprocal, returned = recall(index, asked, args.top_k, arm)
            scored[centre, arm] = found
            print(f"  {arm:9}{found:>9.1%}{reciprocal:>8.3f}{returned:>15.1f}/{args.top_k}")
    print(f"  {'random':9}{chance(chunks, asked, args.top_k, rng):>9.1%}")

    # The gate, stated by the run rather than left for a reader to work out from the table.
    best = max(scored, key=lambda key: scored[key])
    lexical = scored[True, LEXICAL]
    print(f"\ngate: best arm is {best[1]} at {scored[best]:.1%} against lexical alone at {lexical:.1%} -- "
          + ("the vector half pays and is served" if scored[best] > lexical
             else "the vector half does not pay and is dropped"))


if __name__ == "__main__":
    sys.exit(main())
