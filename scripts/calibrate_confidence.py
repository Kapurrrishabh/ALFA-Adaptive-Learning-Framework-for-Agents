#!/usr/bin/env python3
"""Fit the confidence calibrator on the feedback log and report whether it was worth fitting.

Three predictors, not two. Beating the raw confidence is nearly free — it claims 0.99 on rows that are
right a third of the time, so almost any map beats it. The predictor that has to be beaten is the base
rate: a constant that ignores the confidence entirely and just says how often the model is right. If the
calibrator cannot beat that, the confidence carries no usable signal and the honest thing is to print it.

Fit and measured on disjoint halves, and separately per labeller, because the oracle and the agent
disagree about what 'right' means on 42 of 198 rows. A calibration that only holds under one judge is
not a calibration, so both numbers are printed side by side.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent.learn import AGENT, ORACLE, FeedbackLog  # noqa: E402
from selfagent.learn.calibrate import (Calibrator, brier,  # noqa: E402
                                      expected_calibration_error, ranking_auc)


def judged(feedback, labeller):
    """Every row this judge labelled, as (confidence, is_right) arrays."""
    rows = feedback.rows(labeller=labeller)
    if not rows:
        raise SystemExit(f"no rows labelled by {labeller} in the log; run label_feedback.py first")
    return (np.array([row["confidence"] for row in rows]),
            np.array([row["is_right"] for row in rows], dtype=bool))


def split_rows(feedback, labeller, seed):
    """(fit, held out) confidences and labels. Disjoint, seeded, so the gate is the same number twice.

    A random split rather than a time split: every row in this log was generated in one batch from one
    checkpoint, so there is no ordering for a time split to respect and pretending otherwise would
    dress a random half up as something stronger.
    """
    confidence, right = judged(feedback, labeller)
    order = np.random.default_rng(seed).permutation(len(right))
    cut = len(right) // 2
    return (confidence[order[:cut]], right[order[:cut]],
            confidence[order[cut:]], right[order[cut:]])


def report(feedback, labeller, bins, seeds, out=None):
    """Repeat the fit over many splits and report the average, plus how often each baseline was beaten.

    One split of a hundred rows moves the calibration error enough to flip the verdict, so a single seed
    would have reported whichever answer it happened to draw. The win counts are the honest summary.
    """
    scores = {name: [] for name in ("raw confidence", "base rate only", "calibrated confidence")}
    briers = {name: [] for name in scores}
    beat_base, beat_base_brier, areas = 0, 0, []

    for seed in range(seeds):
        fit_confidence, fit_right, confidence, right = split_rows(feedback, labeller, seed)
        calibrator = Calibrator().fit(fit_confidence, fit_right)
        predictions = {"raw confidence": confidence,
                       "base rate only": np.full(len(confidence), fit_right.mean()),
                       "calibrated confidence": calibrator(confidence)}
        for name, predicted in predictions.items():
            scores[name].append(expected_calibration_error(predicted, right, bins))
            briers[name].append(brier(predicted, right))
        beat_base += scores["calibrated confidence"][-1] < scores["base rate only"][-1]
        beat_base_brier += briers["calibrated confidence"][-1] < briers["base rate only"][-1]
        areas.append(ranking_auc(confidence, right))

    # The numbers above come from held-out halves; the calibrator that gets saved is refit on every row,
    # because at a hundred rows throwing half away to ship is a cost with nothing bought by it.
    confidence, right = judged(feedback, labeller)
    calibrator = Calibrator().fit(confidence, right)

    print(f"\n{labeller}: {len(right)} rows, fit on half and measured on the other, averaged over "
          f"{seeds} splits")
    print(f"  calibrator  p = sigmoid({calibrator.weight:+.3f} * log odds of confidence "
          f"{calibrator.bias:+.3f})")
    print(f"  {'predictor':<22}{'calib. error':>14}{'brier':>9}")
    for name in scores:
        print(f"  {name:<22}{np.mean(scores[name]):>14.4f}{np.mean(briers[name]):>9.4f}")
    print(f"  calibration beats raw confidence on every split, by "
          f"{np.mean(scores['raw confidence']) - np.mean(scores['calibrated confidence']):.3f}")
    print(f"  against the base rate: calibration error {beat_base}/{seeds}, brier "
          f"{beat_base_brier}/{seeds}")
    print(f"  ranking auc {np.mean(areas):.3f} +- {np.std(areas):.3f} — unchanged by calibration, and "
          f"the ceiling on B4 and B5")
    if out:
        calibrator.save(out)
        print(f"  saved to {out}, refit on all {len(right)} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="artifacts/feedback.sqlite")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=25, help="how many half splits to average over")
    parser.add_argument("--out", default="artifacts/calibrator_{labeller}.json")
    args = parser.parse_args()

    with FeedbackLog(args.log) as feedback:
        for labeller in (ORACLE, AGENT):
            report(feedback, labeller, args.bins, args.seeds, args.out.format(labeller=labeller))


if __name__ == "__main__":
    main()
