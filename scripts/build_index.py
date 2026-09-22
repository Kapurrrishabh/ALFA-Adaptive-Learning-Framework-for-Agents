#!/usr/bin/env python3
"""Chunk the dated corpus, embed it with the frozen encoder, and write one searchable index.

Sized by measurement, not by ambition. Chunked at the encoder's 128-token window the two dated sources
come out wildly different: 4,613 Fed press releases make ~17k chunks and fit whole, while 13,728 SEC
filings make ~5.1M, which is 19 hours of NumPy encoder passes and an index larger than the corpus it
indexes. So the run takes a **chunk budget** and shares it out: a source that fits entirely is taken
entirely, and the rest is divided between the sources that do not.

A source that does not fit is thinned by **whole documents, spread across its list**, not by keeping the
first chunks of every document. Two reasons. The first pages of a 10-K are its cover and contents, so a
per-document prefix would index the least informative part of every filing. And thinning within a
document leaves 1 window in 370, which is a fragment of something nobody asked about, where a whole
filing is answerable. Spread rather than truncated for the same reason dataforge strides its corpus
budget: the point of a decade of filings is that they span a decade.

The index is therefore a **bounded sample of the corpus and says so** -- `--budget` is printed with the
share of each source that survived it, so no later number can be read as covering everything collected.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.retrieval import chunks, deduplicate, documents, save  # noqa: E402
from backend.retrieval.corpus import DATED_SOURCES  # noqa: E402
from selfagent import pretrained  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.models.retriever import embed  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402

# Documents sampled per source to estimate its chunk count. 40 is enough to separate 3.8 chunks per
# document from 370; a tighter estimate would not change how the budget is shared.
SAMPLED = 40


def estimate(docs, tokenizer, size, rng, sampled=SAMPLED):
    """Mean chunks per document, from a random sample rather than the first few.

    Random because the manifest is in fetch order, which for filings is calendar order, and a 2015 10-K
    is not the length of a 2024 one.
    """
    picked = [docs[index] for index in rng.permutation(len(docs))[:sampled]]
    counts = [len(chunks(document, tokenizer, size)) for document in picked]
    return max(1.0, float(np.mean(counts)))


def share(by_source, budget, tokenizer, size, rng):
    """{source: documents to index}, fitting the budget, smallest source first.

    Smallest first is what makes the leftovers work: a source needing less than its even share hands the
    remainder to the larger ones, instead of the budget being split evenly and half of it going unused.
    """
    sizes = {name: estimate(docs, tokenizer, size, rng) for name, docs in by_source.items()}
    order = sorted(by_source, key=lambda name: len(by_source[name]) * sizes[name])

    taken, left = {}, budget
    for position, name in enumerate(order):
        docs, per = by_source[name], sizes[name]
        allowance = left / (len(order) - position)
        affordable = max(1, int(allowance / per))
        if affordable >= len(docs):
            taken[name], left = docs, left - len(docs) * per
            continue
        step = len(docs) / affordable
        taken[name] = [docs[int(index * step)] for index in range(affordable)]
        left -= affordable * per
    return taken, sizes


def build(docs, tokenizer, size, log):
    """Every chunk of every document, in document order."""
    built = []
    for count, document in enumerate(docs, 1):
        built += chunks(document, tokenizer, size)
        if count % 500 == 0:
            log(f"    {count}/{len(docs)} documents, {len(built)} chunks")
    return built


def vectors(model, tokenizer, texts, size, batch, log):
    """One unit vector per chunk, in batches, because the whole set at once will not fit in memory."""
    built, started = [], time.time()
    for start in range(0, len(texts), batch):
        built.append(embed(model, tokenizer, texts[start : start + batch], size))
        done = start + len(built[-1])
        if (start // batch) % 20 == 0 or done == len(texts):
            rate = done / max(1e-9, time.time() - started)
            log(f"    {done}/{len(texts)} embedded, {rate:.0f}/s, "
                f"{(len(texts) - done) / max(1e-9, rate) / 60:.1f} min left")
    return np.concatenate(built).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--checkpoint", default="advisory_combined.npz",
                        help="the encoder the chunks are embedded with; empty for a lexical-only index")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--text", default="data/text")
    parser.add_argument("--sources", nargs="+", default=list(DATED_SOURCES))
    # 40k chunks is ~3 minutes of encoder passes and a ~40 MB index. The corpus would be 5.1M, which is
    # 19 hours; nothing about this number is optimal, it is the largest one that keeps a rebuild cheap.
    parser.add_argument("--budget", type=int, default=40000, help="chunks to index, across all sources")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="index.npz")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    config, weights = pretrained.load(artifacts / (args.checkpoint or "advisory_combined.npz"))
    rng = np.random.default_rng(args.seed)

    every = documents(args.manifest, args.text, args.sources)
    by_source = {name: [document for document in every if document.source == name]
                 for name in args.sources}
    by_source = {name: docs for name, docs in by_source.items() if docs}
    print(f"{len(every)} dated documents over {len(by_source)} sources, "
          f"budget {args.budget} chunks at {config.max_text_length} tokens")

    taken, sizes = share(by_source, args.budget, tokenizer, config.max_text_length, rng)
    for name in sorted(taken):
        whole = len(by_source[name])
        print(f"  {name:11s} {len(taken[name]):>6}/{whole:<6} documents "
              f"({len(taken[name]) / whole:.0%}), ~{sizes[name]:.1f} chunks each")

    built = []
    for name in sorted(taken):
        print(f"  chunking {name}")
        built += build(taken[name], tokenizer, config.max_text_length, print)
    whole = len(built)
    built = deduplicate(built)
    print(f"{len(built)} chunks after dropping {whole - len(built)} exact duplicates "
          f"({(whole - len(built)) / whole:.1%}), {min(c.day for c in built)} to "
          f"{max(c.day for c in built)}")

    embedded = None
    if args.checkpoint:
        model = GroundedGenerator(config)
        # The encoder only: a retriever scores questions against passages and never writes an answer.
        model.load_pretrained_encoder(weights)
        model.eval()
        print(f"  embedding with {args.checkpoint}")
        embedded = vectors(model, tokenizer, [chunk.text for chunk in built],
                           config.max_text_length, args.batch_size, print)

    out = artifacts / args.out
    save(out, built, embedded)
    print(f"\n{len(built)} chunks -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    with open(out.with_suffix(".sources.json"), "w", encoding="utf-8") as handle:
        json.dump({"budget": args.budget, "checkpoint": args.checkpoint,
                   "chunks": len(built), "window": config.max_text_length,
                   "documents": {name: [len(taken[name]), len(by_source[name])] for name in taken}},
                  handle, indent=2)


if __name__ == "__main__":
    sys.exit(main())
