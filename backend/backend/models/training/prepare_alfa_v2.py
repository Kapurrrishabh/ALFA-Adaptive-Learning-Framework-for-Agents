#!/usr/bin/env python3
"""Build alfa_v2, the generator's next training set, out of three encoded sets.

    python -m backend.models.training.prepare_alfa_v2 --wide <dir> --wide-stride <n> --wide-excluded <file>... \
        --facts <dir> --frozen <dir> --out <dir>

run_alfa_v2.sh builds every input and then trains on the result.

The served advisory_casual rows stay whole, so the 101 held stocks do not regress. The wide rows, from
`prepare_advisory --casual` over database/data/prices_wide, add shares outside those 101: there the
generator named another share in half its answers, having only ever copied 101 tickers. The facts rows,
from `prepare_stock_qa generator`, are repeated until they are FACTS_SHARE of the scored target tokens.
"""
from backend.paths import ARTIFACTS, DATA

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from backend.models.core.config import ModelConfig
from backend.models.core.tokenizer.vocab import PAD_ID
from backend.models.core.tokenizer.wordpiece import WordPiece
from backend.models.training.prepare_stock_qa import GENERATOR_DATASET
from backend.models.training.train_generator import load_split

DATASET = "alfa_v2"
SERVED = "advisory_casual"
# Provisional, the share the plan set: what settles it is stockqa loss falling while unseen and casual hold.
FACTS_SHARE = 0.08


def scored_tokens(target):
    """The target positions the loss is a mean over: every one after the first that is not padding."""
    return int((target[:, 1:] != PAD_ID).sum())


def oversampling(facts_tokens, other_tokens, share=FACTS_SHARE):
    """How many copies of each facts row make the facts about `share` of the scored target tokens. Shared out by
    rows, facts targets, half as long again as the others, took 11.7% of the loss for 8% of the rows."""
    return max(1, round(share * other_tokens / ((1 - share) * facts_tokens)))


def check_unseen(train, test_stocks, tokenizer):
    """Stop when a train set names a frozen test stock, which would leak it into training. The served rows are
    exempt: some test stocks are among the 101 they hold, and those rows are kept so they do not regress, so on
    stockqa only the other test stocks measure a share the generator never saw."""
    for name, (source, _, _) in train.items():
        named = rows_naming(source, test_stocks, tokenizer)
        print(f"{name} train rows naming a frozen test stock: {named or 'none'}")
        if named and name != SERVED:
            raise SystemExit(f"{name} train rows name the frozen set's test stocks {named}; rebuild them without")


def rows_naming(source, symbols, tokenizer):
    """{symbol: how many rows of `source` hold it as a whole word}, read on the ids the rows carry."""
    flat = source.reshape(len(source), -1)
    continues = np.zeros(tokenizer.vocab_size, dtype=bool)
    continues[list(tokenizer.continuation_ids)] = True
    after = np.pad(flat, ((0, 0), (0, 1)))
    found = {}
    for symbol in symbols:
        ids = tokenizer.encode(symbol)
        starts = flat.shape[1] - len(ids) + 1
        hit = ~continues[after[:, len(ids):]]
        for at, token in enumerate(ids):
            hit &= flat[:, at:at + starts] == token
        if hit.any():
            found[symbol] = int(hit.any(axis=1).sum())
    return found


def save(out, split, arrays):
    for part, array in zip(("source", "keep", "target"), arrays):
        np.save(out / f"{DATASET}_{split}_{part}.npy", array)
    print(f"  {split}: {len(arrays[0]):,} rows")
    return len(arrays[0])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", required=True, type=Path, help="prepare_advisory --casual's output over prices_wide")
    parser.add_argument("--wide-stride", required=True, type=int, help="the --stride the wide rows were built with")
    parser.add_argument("--wide-excluded", required=True, nargs="*", help="the prices_wide files the wide rows left out")
    parser.add_argument("--facts", required=True, type=Path, help="prepare_stock_qa generator's output")
    parser.add_argument("--frozen", required=True, type=Path, help="the frozen stock Q&A set the facts came from")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    tokenizer = WordPiece.load(ARTIFACTS / "tokenizer.json")
    for built in (args.wide, args.facts):
        if (built / "tokenizer.json").read_bytes() != (ARTIFACTS / "tokenizer.json").read_bytes():
            raise SystemExit(f"{built} was encoded with another tokenizer than {ARTIFACTS}'s; its ids mean other words")
    served = lambda split: load_split(ARTIFACTS, SERVED, split)
    wide = lambda split: load_split(args.wide, SERVED, split)
    facts = lambda split: load_split(args.facts, GENERATOR_DATASET, split)

    test_stocks = json.loads((args.frozen / "split.json").read_text())["test_stocks"]
    train = {SERVED: served("train"), "wide": wide("train"), GENERATOR_DATASET: facts("train")}
    check_unseen(train, test_stocks, tokenizer)

    tokens = {name: scored_tokens(target) for name, (_, _, target) in train.items()}
    copies = oversampling(tokens[GENERATOR_DATASET], tokens[SERVED] + tokens["wide"])
    mixed = [np.concatenate([served_part, wide_part, np.repeat(facts_part, copies, axis=0)])
             for served_part, wide_part, facts_part in zip(*train.values())]
    shares = {"rows": copies * len(train[GENERATOR_DATASET][0]) / len(mixed[0]),
              "scored_tokens": copies * tokens[GENERATOR_DATASET] / (copies * tokens[GENERATOR_DATASET] + tokens[SERVED] + tokens["wide"])}
    print(f"{GENERATOR_DATASET} train rows repeated {copies}x: {shares['rows']:.1%} of {len(mixed[0]):,} rows, "
          f"{shares['scored_tokens']:.1%} of the scored target tokens")

    args.out.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(ModelConfig.seed).permutation(len(mixed[0]))
    counts = {"train": save(args.out, "train", [part[order] for part in mixed])}
    # train_generator scores a split's first rows only, so the facts rows go first; the served rows after them are
    # in the served run's order, so what it scored of them it scored of the same rows
    counts["validation"] = save(args.out, "validation", [np.concatenate(p) for p in zip(facts("validation"), served("validation"))])
    for split in ("unseen", "casual"):
        counts[split] = save(args.out, split, served(split))
    # train_generator scores a split's first rows in file order, and the wide rows come a ticker at a time; one
    # order for all three keeps row i the same snapshot asked three ways.
    wide_order = np.random.default_rng(ModelConfig.seed).permutation(len(wide("validation")[0]))
    for split, name in (("validation", "wide"), ("unseen", "wide_unseen"), ("casual", "wide_casual")):
        counts[name] = save(args.out, name, [part[wide_order] for part in wide(split)])
    counts["stockqa"] = save(args.out, "stockqa", facts("stockqa"))

    # train_generator reads both from --artifacts; copies, so nothing written there reaches the served files
    for name in ("tokenizer.json", "pretrained.npz"):
        shutil.copy2(ARTIFACTS / name, args.out / name)
    (args.out / "MANIFEST.json").write_text(json.dumps({
        "train": {SERVED: len(train[SERVED][0]), "wide": len(train["wide"][0]),
                  GENERATOR_DATASET: len(train[GENERATOR_DATASET][0]), f"{GENERATOR_DATASET}_copies": copies},
        "scored_tokens": tokens, f"{GENERATOR_DATASET}_share": shares,
        "validation_starts_with": {GENERATOR_DATASET: len(facts("validation")[0])},
        "wide": {"stride": args.wide_stride, "excluded": sorted(args.wide_excluded)},
        "facts": json.loads((args.facts / f"{GENERATOR_DATASET}_MANIFEST.json").read_text()),
        "frozen": {"path": str(args.frozen), "manifest_sha1": hashlib.sha1((args.frozen / "MANIFEST.json").read_bytes()).hexdigest()},
        # what promote.py records the checkpoint trained on; the facts set by its path under DATA, finance.STOCK_QA's form
        "trained_on": [name if name != GENERATOR_DATASET else args.frozen.resolve().relative_to(DATA.resolve()).as_posix()
                       for name in train],
        "rows": counts, "seed": ModelConfig.seed}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
