#!/usr/bin/env python3
"""Build the advisory training set: a question, one evidence passage, and an answer that quotes it.

This replaces the Stack Exchange task for the generator. There, only 31% of an answer's content words
appeared anywhere in the input, so ignoring the evidence was the loss-minimising policy and the model
learned fluent invention. Here every figure in every answer appears in the evidence in the same
characters, so the answer is reachable by copying and checkable by search.

One row in seven asks for something the evidence does not hold, and its answer is a refusal. That
share is provisional: what settles it is the abstention rate on out-of-evidence questions, because
training on answerable rows alone moved their perplexity from 60 to 310.

With --slots the answers name each fact rather than quoting its figure, and serving substitutes the
value. The quoting form does work — copying appears abruptly, and the model reproduces a ticker and a
four figure price exactly — but it still slips a digit, answering +3.3% where the evidence said +3.7%.
Naming the fact takes that last failure off the model, which is the only kind that reaches a reader
looking plausible and being wrong.

Three splits come out, not two. Train and validation differ by date; the third, unseen, holds the
phrasings of each question that training never saw, and the gap between the last two is the only
honest measure of whether the model learned the task or the templates.

With --casual a fourth joins them, asking a trained phrasing the way somebody types it — no full stop,
words contracted, a letter in the wrong place. Every frame here is clean prose and nobody writes that
way, so unseen and casual pull the two kinds of unfamiliarity apart: the wording, and the typing.

Snapshots read no bar after the one they are taken at, and rows are split by date, so neither the
facts nor the split can carry the future backwards.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfagent.agent import guardrails  # noqa: E402
from selfagent.config import ModelConfig  # noqa: E402
from selfagent.data import advisory, prices  # noqa: E402
from selfagent.data.advisory import QUESTION_TOKENS  # noqa: E402
from selfagent.data.encode import build_sources  # noqa: E402
from selfagent.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402

# How often a training question arrives in chat register. Provisional: what settles it is whether the
# casual split closes on validation without validation itself getting worse.
CASUAL_SHARE = 0.5

# A split-adjusted price below a dollar formats to "0.07" and its average range to "0.00", which is a
# figure the answer would be copying for nothing.
MIN_PRICE = 1.0

# Confidences are drawn rather than taken from the price head, on purpose: the generator's job is to
# phrase whatever the head reports, so it should see the whole range and stay independent of one
# checkpoint. Concentrated near uniform because that is what a head with 22% R-squared produces.
OUTLOOK_CONCENTRATION = 4.0


def snapshot_rows(ticker, bars, end, probabilities, split, rng, slots, messy):
    """(split, question, evidence, answer) for one snapshot, one row per intent.

    A training snapshot draws a trained phrasing per intent; a validation one emits the same intent
    twice, once at a trained phrasing and once at a held-out one, so the two are compared on
    identical facts and differ only in wording.

    With `messy` a third held-out row joins them, asking the *trained* phrasing in chat register. That
    separates the two ways a real question is unfamiliar: unseen tells us about the wording, casual
    about the typing, and they are different problems with different fixes.
    """
    facts = advisory.snapshot(bars, end)
    shown = advisory.as_text(facts)
    shown.update(advisory.risk_outlook(probabilities))
    evidence = advisory.render_evidence(ticker, shown)
    # The evidence always carries the figures; only the answer changes, to name them instead.
    spoken = advisory.slot_names(shown) if slots else shown

    rows = []
    for intent in advisory.INTENTS:
        trained = advisory.trained_phrasings(intent)
        phrasing = rng.choice(trained)
        typed = rng if messy and split == "train" and rng.random() < CASUAL_SHARE else None
        question, answer = advisory.row(intent, ticker, facts, spoken, phrasing, typed)
        rows.append((split, question, evidence, answer))
        if split == "train":
            continue
        unseen = [p for p in range(advisory.phrasings(intent)) if p not in trained]
        question, answer = advisory.row(intent, ticker, facts, spoken, rng.choice(unseen))
        rows.append(("unseen", question, evidence, answer))
        if messy:
            question, answer = advisory.row(intent, ticker, facts, spoken, phrasing, rng)
            rows.append(("casual", question, evidence, answer))
    return rows


def collect(price_dir, stride, cutoff, rng, limit, slots, messy):
    """Every row of every snapshot of every ticker, each tagged with the split it belongs to."""
    collected = []
    snapshots = 0
    for path in sorted(Path(price_dir).glob("*.csv")):
        if limit and snapshots >= limit:
            break
        try:
            dates, bars = prices.load_bars(path)
        except ValueError:
            continue
        ticker = path.stem.upper()
        for end in range(advisory.BARS_NEEDED - 1, len(bars), stride):
            if bars[end, 3] < MIN_PRICE:
                continue
            probabilities = rng.dirichlet([OUTLOOK_CONCENTRATION] * 3)
            split = "train" if dates[end] < cutoff else "validation"
            collected.extend(
                snapshot_rows(ticker, bars, end, probabilities, split, rng, slots, messy)
            )
            snapshots += 1
            if limit and snapshots >= limit:
                break
    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/prices")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--cutoff", default="2021-01-01", help="validation starts on this date")
    parser.add_argument("--stride", type=int, default=21,
                        help="bars between snapshots; a day apart they are near-duplicates")
    parser.add_argument("--limit", type=int, default=0, help="cap the snapshots, for a smoke run")
    parser.add_argument("--slots", action="store_true",
                        help="answers name each fact instead of quoting its figure, and serving fills them")
    parser.add_argument("--casual", action="store_true",
                        help="half the training questions arrive in chat register, and a casual split measures it")
    args = parser.parse_args()
    dataset = "advisory" + ("_slots" if args.slots else "") + ("_casual" if args.casual else "")

    artifacts = Path(args.artifacts)
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    length = ModelConfig.max_text_length
    answer_budget = ModelConfig.max_answer_length

    rng = np.random.default_rng(ModelConfig.seed)
    rows = collect(args.prices, args.stride, args.cutoff, rng, args.limit, args.slots, args.casual)
    refusals = sum(answer == advisory.UNSUPPORTED for _, _, _, answer in rows)
    asked = len({question for _, question, _, _ in rows})
    print(f"{dataset}: {len(rows):,} rows, {refusals / len(rows):.1%} refusals, "
          f"{asked:,} distinct questions")

    sources = np.zeros((len(rows), 1, length), dtype=np.uint16)
    keeps = np.zeros((len(rows), 1, length), dtype=np.uint8)
    targets = np.full((len(rows), answer_budget), PAD_ID, dtype=np.uint16)
    splits = np.empty(len(rows), dtype=object)
    truncated = 0

    for row, (split, question, evidence, answer) in enumerate(rows):
        unsupported = guardrails.unsupported_figures(answer, evidence)
        if unsupported:
            raise ValueError(
                f"row {row} answers {question!r} with figures the evidence lacks: {unsupported}. "
                f"A row like this teaches the model to invent and would not show up in the loss."
            )
        evidence_ids = tokenizer.encode(evidence)
        truncated += len(evidence_ids) > length - QUESTION_TOKENS - 3
        sources[row], keeps[row] = build_sources(
            tokenizer.encode(question)[:QUESTION_TOKENS], [evidence_ids], 1, length
        )
        ids = [CLS_ID] + tokenizer.encode(answer)[: answer_budget - 2] + [SEP_ID]
        targets[row, : len(ids)] = ids
        splits[row] = split

    if truncated:
        raise ValueError(
            f"{truncated} of {len(rows)} evidence passages exceed "
            f"{length - QUESTION_TOKENS - 3} tokens; a cut passage drops a figure the answer quotes"
        )

    for name in ("train", "validation", "unseen") + (("casual",) if args.casual else ()):
        picked = np.flatnonzero(splits == name)
        np.save(artifacts / f"{dataset}_{name}_source.npy", sources[picked])
        np.save(artifacts / f"{dataset}_{name}_keep.npy", keeps[picked])
        np.save(artifacts / f"{dataset}_{name}_target.npy", targets[picked])
        print(f"  {name}: {len(picked):,} rows -> {artifacts}/")


if __name__ == "__main__":
    sys.exit(main())
