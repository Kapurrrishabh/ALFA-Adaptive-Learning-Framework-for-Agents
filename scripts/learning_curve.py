#!/usr/bin/env python3
"""Does more feedback make the agent better, with the model's weights never touched?

This is the thesis measured as a curve. The adaptation is B4's abstention cut: the agent is told "answer
only where you are at least 60% likely to be right" and solves for the confidence cut that delivers it
from whatever feedback it has. So the question is whether n rows of feedback place that cut better than
n/2 did, on rows the cut never saw.

Three lines to read the curve against, because a curve that rises on its own proves nothing:

  fixed cut 0.9980           the hand-picked threshold. It sees no feedback and so cannot improve; the
                             gap between it and the curve is what the feedback bought.
  answers everything         no abstention at all. The floor.
  fitted on the scored rows  the ceiling. Feedback cannot beat already knowing the answers.

And the number to read is the **miss**, not the precision: how far the delivered precision lands from the
bar that was asked for. A cut fitted on ten rows promises 60% and delivers whatever those ten rows
happened to say. The claim being tested is that the promise comes true as the log grows — which is
improvement from outcomes alone, since no gradient is computed anywhere in this script.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_confidence import judged  # noqa: E402
from fit_abstention import HAND_PICKED  # noqa: E402
from selfagent.learn import AGENT, ORACLE, FeedbackLog  # noqa: E402
from selfagent.learn.abstain import Abstainer, precision_at  # noqa: E402


def draw(rows, holdout, seed):
    """(scored, pool) row indices — disjoint, so nothing the cut is fitted on is also scored.

    Shuffled rather than taken in order: every row in this log came from one batch on one checkpoint, so
    there is no history for a time split to respect and pretending otherwise would dress a random half up
    as something stronger. The scored rows are drawn first and kept fixed across every feedback size in a
    seed, so a change along the curve is the cut moving rather than the scoring set shrinking under it.
    """
    if holdout >= rows:
        raise ValueError(f"cannot hold out {holdout} of {rows} rows; nothing would be left to fit on")
    shuffled = np.random.default_rng(seed).permutation(rows)
    return shuffled[:holdout], shuffled[holdout:]


def delivered(fit_confidence, fit_right, confidence, right, wanted):
    """(precision, coverage) on the scored rows, for the cut a stated bar implies on the fitting rows."""
    cut = Abstainer.fit(fit_confidence, fit_right, wanted).cut
    return precision_at(confidence, right, cut)


def curve(confidence, right, sizes, wanted, holdout, seeds):
    """What each feedback size delivers, beside the three lines that say whether the size mattered."""
    measured = {size: [] for size in sizes}
    controls = {name: [] for name in ("fixed cut", "answers everything", "fitted on the scored rows")}

    for seed in range(seeds):
        scored, pool = draw(len(right), holdout, seed)
        scored_confidence, scored_right = confidence[scored], right[scored]
        controls["fixed cut"].append(precision_at(scored_confidence, scored_right, HAND_PICKED))
        controls["answers everything"].append((float(scored_right.mean()), 1.0))
        controls["fitted on the scored rows"].append(
            delivered(scored_confidence, scored_right, scored_confidence, scored_right, wanted))
        for size in sizes:
            if size <= len(pool):
                taken = pool[:size]
                measured[size].append(
                    delivered(confidence[taken], right[taken], scored_confidence, scored_right, wanted))
    return measured, controls


def average(pairs, wanted):
    """(precision, coverage, miss in points, seeds kept) over the seeds where the cut answered at all.

    A seed where it answered nothing has no precision to average — counting it as zero would report a
    cut that was silent as a cut that was wrong. The miss is the mean of the per-seed absolute misses,
    not the miss of the mean: a cut landing at 40% and 80% by turns has not delivered 60%.
    """
    kept = [(precision, coverage) for precision, coverage in pairs if coverage]
    if not kept:
        return float("nan"), 0.0, float("nan"), 0
    precisions = np.array([precision for precision, _ in kept])
    return (precisions.mean(), np.mean([coverage for _, coverage in kept]),
            100 * np.abs(precisions - wanted).mean(), len(kept))


def report(feedback, labeller, sizes, wanted, holdout, seeds):
    confidence, right = judged(feedback, labeller)
    measured, controls = curve(confidence, right, sizes, wanted, holdout, seeds)

    print(f"\n{labeller}: {len(right)} rows, asked for {wanted:.0%}, {holdout} scored, averaged over "
          f"{seeds} seeds")
    print(f"  {'feedback rows':>14}{'delivered':>11}{'coverage':>10}{'miss vs bar':>13}{'seeds':>7}")
    for size in sizes:
        if not measured[size]:
            continue
        precision, coverage, miss, seeds_kept = average(measured[size], wanted)
        print(f"  {size:>14}{precision:>11.1%}{coverage:>10.1%}{miss:>12.1f} pts{seeds_kept:>7}")
    misses = {}
    for name, pairs in controls.items():
        precision, coverage, miss, _ = average(pairs, wanted)
        misses[name] = miss
        print(f"  {name:<28}{precision:>7.1%} on {coverage:>6.1%}, miss {miss:.1f} pts")

    # Splitting the miss in two is what makes the curve readable. A bar the model cannot reach at any
    # coverage leaves a miss no quantity of feedback can close, and that part must not be read as a
    # failure to learn.
    drawn = {size: average(measured[size], wanted)[2] for size in sizes if measured[size]}
    at, lowest = min(drawn.items(), key=lambda row: row[1])
    ceiling = misses["fitted on the scored rows"]
    print(f"  best at {at} rows: miss {lowest:.1f} pts, of which {ceiling:.1f} is a bar the model cannot "
          f"reach at this coverage floor and {lowest - ceiling:.1f} is not yet learned")
    moved = drawn[min(drawn)] - drawn[max(drawn)]
    print(f"  the miss moved {moved:+.1f} pts between {min(drawn)} and {max(drawn)} rows of feedback"
          + ("" if abs(moved) > 1 else " — no trend worth claiming at this log size"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="artifacts/feedback.sqlite")
    parser.add_argument("--wanted", type=float, default=0.6, help="the stated chance of being right")
    parser.add_argument("--holdout", type=int, default=80, help="rows scored on, never fitted on")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 60, 80, 100, 118])
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()

    with FeedbackLog(args.log) as feedback:
        for labeller in (ORACLE, AGENT):
            report(feedback, labeller, args.sizes, args.wanted, args.holdout, args.seeds)


if __name__ == "__main__":
    main()
