#!/usr/bin/env python3
"""Add newly collected documents to an existing index, without rebuilding it.

The corpus is not a fixed thing. `dataforge` can fetch today's press releases, filings and articles at any
time, and a retrieval path is only as current as the last index build. Rebuilding is the honest way to get
there and also the expensive one -- it re-chunks every document and, on an embedded index, re-runs the
encoder over all of them. This appends instead: it indexes the documents the index does not already hold
and leaves every existing chunk exactly where it was.

Run it after collecting, in this order:

    python3 -m dataforge.collect --source fed_press --source news_rss --tier standard
    python3 -m dataforge.collect --process
    python3 scripts/refresh_index.py --index reference_index.npz

**The shape comes from the index, not from this script's flags.** The window a chunk was cut to, whether
undated documents are allowed in, which sources are in scope and which encoder embedded it are all read
back out of the `.sources.json` the build wrote. A passage cut to a different window than its neighbours
overflows the generator's slot at serving time, and an embedding from a different checkpoint is a vector in
a different space that still compares as a number -- both are wrong in ways no error would announce, so
they are taken from the record rather than re-specified. The sidecar missing is an error, not a default.

What it cannot do is make the index grow forever: `--budget` caps the chunks one run may add, shared out
between sources by the same rule the build used, so a refresh has a known cost.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.retrieval import deduplicate, documents, load, save  # noqa: E402
from build_index import BY_SOURCE, build, share, vectors  # noqa: E402
from selfagent import pretrained  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402

# Smaller than a build's, because a refresh runs often and an index that doubles on every run is not a
# refresh. Provisional: what would settle it is how much a month of collection actually adds.
BUDGET = 8000


def recorded(path):
    """What the build wrote beside the index, or an error naming the file that has to exist."""
    sidecar = path.with_suffix(".sources.json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"no build record at {sidecar}, so the window and encoder this index was built with are "
            f"unknown; rebuild it with scripts/build_index.py rather than appending blind"
        )
    with open(sidecar, encoding="utf-8") as handle:
        return json.load(handle)


def fresh(every, held):
    """{source: documents} for the documents no chunk in the index came from.

    Keyed on the document key, which is the sha256 prefix `data/manifest.jsonl` files each URL under, so a
    file re-fetched unchanged is not re-indexed and a file whose content changed is a different document.
    """
    by_source = {}
    for document in every:
        if document.key not in held:
            by_source.setdefault(document.source, []).append(document)
    return by_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--index", default="reference_index.npz")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--text", default="data/text")
    parser.add_argument("--budget", type=int, default=BUDGET, help="chunks this run may add")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be added and write nothing")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    path = artifacts / args.index
    built = recorded(path)
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    rng = np.random.default_rng(args.seed)

    existing, embedded = load(path)
    if (embedded is None) != (not built["checkpoint"]):
        raise ValueError(
            f"{path} holds {'no ' if embedded is None else ''}vectors but its record names "
            f"checkpoint {built['checkpoint']!r}; one of the two is stale and appending would mix spaces"
        )
    size, sources = built["window"], sorted(built["documents"])
    how = "lexical only" if embedded is None else f"embedded with {built['checkpoint']}"
    print(f"{len(existing)} chunks in {args.index}, cut at {size} tokens "
          f"over {len(sources)} sources, {how}")

    every = documents(args.manifest, args.text, sources, built["undated"])
    by_source = fresh(every, {chunk.document for chunk in existing})
    if not by_source:
        print(f"nothing new in {args.manifest}; collect first, then run this again")
        return 0
    print(f"{sum(len(docs) for docs in by_source.values())} new documents over "
          f"{len(by_source)} sources, budget {args.budget} chunks")

    # An index written before the flag existed was built the one way there was, which is a fact about those
    # files rather than a default chosen here.
    taken, sizes = share(by_source, args.budget, tokenizer, size, rng,
                         built.get("weighting", BY_SOURCE))
    for name in sorted(taken):
        whole = len(by_source[name])
        print(f"  {name:14s} {len(taken[name]):>6}/{whole:<6} new documents "
              f"({len(taken[name]) / whole:.0%}), ~{sizes[name]:.1f} chunks each")

    chunked = []
    for name in sorted(taken):
        chunked += build(taken[name], tokenizer, size, print)

    # Deduplicated against the whole index, not only against each other, because the standing paragraph a
    # press release repeats is in last month's chunks too. Existing chunks come first and were already
    # deduplicated at build time, so none of them can be the copy that gets dropped -- which is what keeps
    # the existing vectors aligned to the rows they were computed for. Checked rather than assumed.
    kept = deduplicate(existing + chunked)
    if any(one is not two for one, two in zip(kept, existing)) or len(kept) < len(existing):
        raise ValueError(f"{path} holds duplicate chunks of its own, so appending would shift its vectors")
    added = kept[len(existing) :]
    print(f"{len(added)} chunks to add, {len(chunked) - len(added)} of {len(chunked)} already held "
          f"({(len(chunked) - len(added)) / max(1, len(chunked)):.1%} duplicates)")
    if args.dry_run or not added:
        print("nothing written" if args.dry_run else "no new chunk survived deduplication")
        return 0

    if embedded is not None:
        config, weights = pretrained.load(artifacts / built["checkpoint"])
        model = GroundedGenerator(config)
        model.load_pretrained_encoder(weights)
        model.eval()
        print(f"  embedding {len(added)} new chunks with {built['checkpoint']}")
        grown = vectors(model, tokenizer, [chunk.text for chunk in added], config.max_text_length,
                        args.batch_size, print)
        embedded = np.concatenate([embedded, grown]).astype(np.float32)

    save(path, kept, embedded)
    dated = [chunk.day for chunk in kept if chunk.day]
    print(f"\n{len(existing)} -> {len(kept)} chunks in {path} "
          f"({path.stat().st_size / 1e6:.1f} MB), latest dated {max(dated) if dated else 'none'}")

    built["chunks"] = len(kept)
    for name in taken:
        was = built["documents"].get(name, [0, 0])
        built["documents"][name] = [was[0] + len(taken[name]), len([d for d in every
                                                                    if d.source == name])]
    with open(path.with_suffix(".sources.json"), "w", encoding="utf-8") as handle:
        json.dump(built, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
