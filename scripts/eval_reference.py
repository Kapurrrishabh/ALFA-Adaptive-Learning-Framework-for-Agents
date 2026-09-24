#!/usr/bin/env python3
"""What the reference path actually does on questions the advisory router cannot place.

Four rates, and the point of the run is that the first three are free to be bad while the fourth is not:

  coverage    a passage cleared the relevance floor, so something was said at all
  paraphrase  the model's own words were served, because every figure in them was in the passages
  quote       the passages' words were served instead, because the model stated a figure they did not
  invented    a figure in what the **user was served** that the evidence does not carry

`invented` is the claim. It is zero by construction when the quote fires -- a quote of the evidence cannot
state a figure the evidence lacks -- so a non-zero reading here means the construction is wrong, not that
the model is bad. That is the failure this script exists to catch, and it is checked against the served
text rather than against the model's, which is measured separately as `wrote_invented`.

The questions are real ones people asked on Stack Exchange, not questions written for this. Two reasons
they are not the advisory set: that set is what the router places, so it would never reach this path, and
a question invented alongside the system that answers it measures nothing.

**The overlap is reported, not assumed away.** Stack Exchange is one of the nine sources in the index, so
a question whose own thread was indexed could be answered by its own accepted answer. That is correct
behaviour when serving and flattering when measuring, so the share of questions whose asked text appears
in a retrieved passage is printed beside the rates.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask import reference_from  # noqa: E402
from backend.agent import PARAPHRASED  # noqa: E402
from selfagent.agent import guardrails  # noqa: E402
from selfagent.data.qa_pairs import load_threads, question_text  # noqa: E402


def sampled(qa_root, count, rng):
    """`count` questions drawn from the collected threads, as their askers typed them."""
    questions, _ = load_threads(Path(qa_root))
    asked = [question_text(question) for question in questions]
    asked = [text for text in asked if text.strip()]
    return [asked[index] for index in rng.permutation(len(asked))[:count]]


def measure(reference, questions, log):
    """One row per question: its top passage's score, what was served, how, and what it invented.

    The score is taken here rather than carried out of `look_up`, because the run needs the scores of the
    questions the floor **rejected** and a serving path has no reason to report those.
    """
    rows = []
    for count, question in enumerate(questions, 1):
        found = reference.index.search(question, reference.passages)
        top = reference.index.relevance(question, found[0]) if found else 0.0
        rows.append((question, top, reference.look_up(question)))
        if count % 25 == 0:
            log(f"  {count}/{len(questions)}")
    return rows


def report(rows, floor_at):
    """The four rates, the score distribution the floor has to be read against, and the overlap."""
    answered = [(question, looked) for question, _, looked in rows if looked is not None]
    print(f"\n{len(rows)} questions, floor {floor_at}")
    print(f"  coverage     {len(answered)}/{len(rows)} ({len(answered) / len(rows):.1%})")

    tops = np.array([top for _, top, _ in rows])
    below = tops[tops < floor_at]
    print(f"  top score    median {np.median(tops):.2f}, 10th pct {np.percentile(tops, 10):.2f}, "
          f"max {tops.max():.2f}; {len(below)} under the floor"
          + (f", the highest of them {below.max():.2f}" if len(below) else ""))
    if not answered:
        return 0

    paraphrased = [looked for _, looked in answered if looked.how == PARAPHRASED]
    print(f"  paraphrase   {len(paraphrased)}/{len(answered)} "
          f"({len(paraphrased) / len(answered):.1%} of what was answered)")
    print(f"  quote        {len(answered) - len(paraphrased)}/{len(answered)} "
          f"({1 - len(paraphrased) / len(answered):.1%})")

    served = [guardrails.unsupported_figures(looked.served, looked.evidence) for _, looked in answered]
    wrote = [looked.unsupported for _, looked in answered]
    print(f"  invented     {sum(bool(found) for found in served)}/{len(answered)} answers, "
          f"{sum(len(found) for found in served)} figures  <- must be 0")
    print(f"  wrote_invented {sum(bool(found) for found in wrote)}/{len(answered)} answers, "
          f"{sum(len(found) for found in wrote)} figures  <- what the guard caught")

    sure = np.array([looked.confidence for _, looked in answered])
    print(f"  confidence   min {sure.min():.4f}, median {np.median(sure):.4f}, max {sure.max():.4f}")

    overlap = sum(question.lower()[:60] in looked.evidence.lower() for question, looked in answered)
    print(f"  overlap      {overlap}/{len(answered)} ({overlap / len(answered):.1%}) retrieved a passage "
          f"holding the question's own words")
    return sum(bool(found) for found in served)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--qa", default="data/qa")
    parser.add_argument("--index", default="reference_index.npz")
    parser.add_argument("--checkpoint", default="generator.npz")
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=5, help="answers to print in full, for reading")
    args = parser.parse_args()

    # Paraphrasing on, whatever serving does. This run is what decided serving quotes instead, and a
    # measurement that stopped generating could no longer tell anyone whether that is still the right call.
    reference = reference_from(Path(args.artifacts), args.index, args.checkpoint, paraphrase=True)
    rng = np.random.default_rng(args.seed)
    questions = sampled(args.qa, args.questions, rng)
    print(f"{len(reference.index.chunks)} chunks, {len(questions)} questions, {args.checkpoint}")

    rows = measure(reference, questions, print)
    invented = report(rows, reference.floor)

    for question, _, looked in [row for row in rows if row[2] is not None][: args.show]:
        print(f"\n> {question[:120]}")
        print(f"  {looked.how:11s} {looked.served[:300]}")
        if looked.how != PARAPHRASED:
            print(f"  wrote       {looked.answer[:200]!r}")
    # A non-zero count here is the construction failing, so it fails the run rather than being printed.
    return 1 if invented else 0


if __name__ == "__main__":
    sys.exit(main())
