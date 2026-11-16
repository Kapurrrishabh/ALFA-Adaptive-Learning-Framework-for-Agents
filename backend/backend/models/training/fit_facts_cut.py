#!/usr/bin/env python3
"""Fit the answer cut for facts turns on the facts validation rows, and quote it on the stockqa rows.

    python -m backend.models.training.fit_facts_cut --checkpoint alfa_v2_data/alfa_v2.step36000.npz

The advisory cut is fitted on answers whose confidence sits near 1. A facts answer's averaged 0.45, so under that cut
almost none was spoken. This is the same `Abstainer.fit` at the same wanted precision, on facts answers a facts oracle
judged, and it is written under the generator's digest where `ask.build` reads it.
"""
from backend.paths import ARTIFACTS

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from backend.models.agent.run.ask import FACTS_CUT  # noqa: E402
from backend.models.core.tokenizer.wordpiece import pretokenize  # noqa: E402
from backend.models.data import stock_qa  # noqa: E402
from backend.models.guardrails import guardrails  # noqa: E402
from backend.models.learning.abstain import Abstainer, precision_at  # noqa: E402
from backend.models.serving import registry  # noqa: E402
from backend.models.training.check_answers import answer_rows, load_model  # noqa: E402
from backend.models.training.prepare_alfa_v2 import DATASET  # noqa: E402
from backend.models.training.prepare_stock_qa import (GENERATOR_DATASET, GENERATOR_SPLITS, PASSAGES,  # noqa: E402
                                                      read_jsonl)
from backend.models.training.train_generator import load_split  # noqa: E402


def judge(answer, passage, company):
    """(is right, why) by the facts oracle: every figure is the passage's, each under its own metric, and its share named.

    Not the advisory oracle's exact match: a facts answer is free prose, and 0 of 461 stockqa answers matched theirs.
    """
    unsupported = guardrails.unsupported_figures(answer, passage)
    if unsupported:
        return False, f"states figures the passage does not: {unsupported}"
    mislabelled = guardrails.mislabelled_figures(answer, passage, stock_qa.METRICS)
    if mislabelled:
        return False, f"gives a figure as another metric's: {mislabelled}"
    said = f" {' '.join(pretokenize(answer))} "
    if not any(f" {' '.join(pretokenize(name))} " in said for name in stock_qa.own_names(passage) + [company]):
        return False, "never names the share its passage is about"
    return True, "every figure is the passage's, under its own metric, and the share is named"


def answered(model, tokenizer, seed, data, split, passages):
    """[(answer, passage, company, confidence)] for the facts rows of an alfa_v2 split, sampled as the server samples."""
    arrays = load_split(data, DATASET, split)
    scored = answer_rows(model, tokenizer, arrays, np.arange(len(passages)), np.random.default_rng(seed))
    for (_, evidence, _, _, _), row in zip(scored, passages):
        if not evidence.startswith(tokenizer.decode(tokenizer.encode(row["passage"]))):
            raise ValueError(f"{DATASET}'s {split} row for {row['id']} reads another passage than {PASSAGES.name} holds; "
                             f"the two were built from different frozen sets")
    return [(answer, row["passage"], row["company"], sure) for (*_, answer, sure), row in zip(scored, passages)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=str(ARTIFACTS))
    parser.add_argument("--checkpoint", required=True, help="the generator, relative to --artifacts")
    parser.add_argument("--data", default="alfa_v2_data", help="the alfa_v2 build, relative to --artifacts")
    parser.add_argument("--wanted", type=float, default=0.6, help="the stated chance of being right, as ask.py's")
    parser.add_argument("--out", default="", help="the folder the cut is written to; --artifacts by default")
    parser.add_argument("--rows", default="", help="a JSON file to write every judged row to")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    data = artifacts / args.data
    tokenizer, model, config = load_model(artifacts, "", args.checkpoint)
    facts = {"validation": json.loads((data / "MANIFEST.json").read_text())["validation_starts_with"][GENERATOR_DATASET],
             "stockqa": len(load_split(data, DATASET, "stockqa", ("keep",))[0])}
    passages = list(read_jsonl(PASSAGES))
    judged = {}
    for split, count in facts.items():
        rows = [row for row in passages if GENERATOR_SPLITS[row["split"]] == split]
        if len(rows) != count:
            raise ValueError(f"{PASSAGES.name} holds {len(rows)} {split} passages and {DATASET} {count} facts rows")
        judged[split] = [dict(zip(("right", "why"), judge(answer, passage, company)), confidence=sure, answer=answer,
                              passage=passage) for answer, passage, company, sure in
                         answered(model, tokenizer, config.seed, data, split, rows)]

    def columns(split):
        return (np.array([row["confidence"] for row in judged[split]]),
                np.array([row["right"] for row in judged[split]], dtype=bool))
    confidence, right = columns("validation")
    abstainer = Abstainer.fit(confidence, right, args.wanted)
    path = Path(args.out or artifacts) / FACTS_CUT.format(registry.digest(artifacts / args.checkpoint))
    abstainer.save(path)

    print(f"{args.checkpoint}: {len(right)} facts validation rows, {right.mean():.1%} right by the facts oracle")
    for split in judged:
        wrong = collections.Counter(row["why"].split(":")[0] for row in judged[split] if not row["right"])
        print(f"  {split} wrong: {dict(wrong)}")
    print(f"  cut {abstainer.cut:.5f} for {abstainer.wanted:.0%}: expects {abstainer.expected:.1%} right on "
          f"{abstainer.coverage:.1%} of these rows")
    held, held_right = columns("stockqa")
    precision, coverage = precision_at(held, held_right, abstainer.cut)
    print(f"  on the {len(held)} stockqa rows, held out: {precision:.1%} right on {coverage:.1%} "
          f"({held_right.mean():.1%} answering all), before the served guards")
    print(f"wrote {path}")
    if args.rows:
        Path(args.rows).write_text(json.dumps(judged))


if __name__ == "__main__":
    sys.exit(main())
