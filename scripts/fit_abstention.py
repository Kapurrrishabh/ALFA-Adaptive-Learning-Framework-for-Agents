#!/usr/bin/env python3
"""Fit the abstention cut on half the feedback log and report what it does on the other half.

The number this replaces — 53.8% correct on the 40% it answers, at a cut of 0.9980 — was chosen on the
same rows it was quoted on. check_answers.py says so in its own docstring: that is an upper bound, not a
baseline. So it is quoted here twice, once as the in-sample bound it is and once fitted out of sample,
because the gap between those two is the whole reason B4 exists.

Both are on one precision-coverage curve, which no threshold can move. The comparison that matters is
therefore at matched coverage, and the win being claimed is a cut chosen by a stated bar rather than by
hand — a number that still means something on the next checkpoint.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_confidence import judged, split_rows  # noqa: E402
from selfagent.learn import AGENT, ORACLE, FeedbackLog  # noqa: E402
from selfagent.learn.abstain import COVERAGE_FLOOR, Abstainer, precision_at  # noqa: E402

HAND_PICKED = 0.9980

# The coverage the number B4 is told to beat was quoted at.
GATE_COVERAGE = 0.40


def report(feedback, labeller, wanted_levels, seeds):
    """For each stated bar, the cut it implies and what that cut does on rows it was not fitted on."""
    confidence, right = judged(feedback, labeller)
    curve = [(*precision_at(confidence, right, cut), cut) for cut in np.unique(confidence)]
    best = max(row for row in curve if row[1] >= COVERAGE_FLOOR)
    hand = precision_at(confidence, right, HAND_PICKED)

    print(f"\n{labeller}: {len(right)} rows, {right.mean():.1%} right if it answers everything")
    print(f"  best cut fitted and quoted on the same rows: {best[0]:.1%} correct on {best[1]:.1%} at "
          f"{best[2]:.4f} — an upper bound, and the number B4 is told to beat")
    print(f"  hand-picked {HAND_PICKED}: {hand[0]:.1%} correct on {hand[1]:.1%}")
    print(f"  {'asked for':>10}{'cut':>9}{'held-out correct':>19}{'coverage':>11}{'vs answer all':>15}")

    for wanted in wanted_levels:
        precisions, coverages, cuts, bases = [], [], [], []
        for seed in range(seeds):
            fit_confidence, fit_right, out_confidence, out_right = split_rows(feedback, labeller, seed)
            abstainer = Abstainer.fit(fit_confidence, fit_right, wanted)
            precision, coverage = precision_at(out_confidence, out_right, abstainer.cut)
            if coverage:
                precisions.append(precision)
                coverages.append(coverage)
                cuts.append(abstainer.cut)
                bases.append(out_right.mean())
        lift = 100 * (np.mean(precisions) - np.mean(bases))
        print(f"  {wanted:>10.0%}{np.mean(cuts):>9.4f}{np.mean(precisions):>18.1%}"
              f"{np.mean(coverages):>11.1%}{lift:>+12.1f} pts")

    # The gate is stated as a coverage, so it gets answered as one. Asking for a precision lands on
    # whatever coverage the curve allows, which is not comparable to a number quoted at 40%.
    matched = []
    for seed in range(seeds):
        fit_confidence, _, out_confidence, out_right = split_rows(feedback, labeller, seed)
        cut = np.quantile(fit_confidence, 1 - GATE_COVERAGE)
        matched.append(precision_at(out_confidence, out_right, cut))
    print(f"  cut placed for {GATE_COVERAGE:.0%} coverage on the fitting half: "
          f"{np.mean([p for p, _ in matched]):.1%} correct on {np.mean([c for _, c in matched]):.1%} "
          f"held out")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="artifacts/feedback.sqlite")
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--wanted", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    args = parser.parse_args()

    with FeedbackLog(args.log) as feedback:
        for labeller in (ORACLE, AGENT):
            report(feedback, labeller, args.wanted, args.seeds)


if __name__ == "__main__":
    main()
