"""Turning the model's own confidence into a number that means what it says.

The model's mean token probability averages 0.990 on answers that are right 31% of the time. As a
probability it is close to worthless. As an *ordering* it may still separate right from wrong, and
those are two different questions that one number is usually asked to answer at once.

So this module reports them apart. Calibration fixes what a score means; it cannot change what the
score orders, because the map is monotone on purpose. Whatever ranking power the raw confidence has is
therefore all that B4's threshold and B5's reranker will ever get out of it, and the AUC here is the
ceiling on both. A calibrator that flattered the ordering by reordering would hide that ceiling.
"""

import json
from pathlib import Path

from ..backend import xp


def _sigmoid(x):
    return 1.0 / (1.0 + xp.exp(-xp.clip(x, -60.0, 60.0)))


def _log_odds(probability, floor=1e-6):
    """Confidences reach exactly 1.0, whose log odds is infinite. The floor is what keeps them finite."""
    clipped = xp.clip(xp.asarray(probability, dtype=xp.float64), floor, 1.0 - floor)
    return xp.log(clipped / (1.0 - clipped))


class Calibrator:
    """Platt scaling: one slope and one intercept on the log odds of the model's own confidence.

    Two parameters because there are a couple of hundred labelled rows. Isotonic regression would fit
    the training slice better and would spend the freedom memorising which of a hundred rows were right.
    """

    def __init__(self, weight=1.0, bias=0.0):
        self.weight = float(weight)
        self.bias = float(bias)

    def fit(self, confidence, is_right, steps=100, ridge=1e-8):
        """Newton steps on the log loss, against Platt's smoothed targets rather than hard 0 and 1.

        At this sample size the smoothing is the difference between a fit and a memorised slice: with
        hard targets the slope runs off to separate the few rows at the edges of the score range.
        """
        right = xp.asarray(is_right, dtype=bool)
        if right.all() or not right.any():
            raise ValueError(
                f"cannot fit a calibrator on {len(right)} rows that are all "
                f"{'right' if right.all() else 'wrong'}; it has no difference to learn"
            )
        design = xp.stack([_log_odds(confidence), xp.ones(len(right))], axis=1)
        target = xp.where(right, (right.sum() + 1) / (right.sum() + 2), 1 / ((~right).sum() + 2))

        beta = xp.zeros(2)
        for _ in range(steps):
            predicted = _sigmoid(design @ beta)
            curvature = xp.maximum(predicted * (1 - predicted), 1e-12)
            hessian = design.T @ (design * curvature[:, None]) + ridge * xp.eye(2)
            step = xp.linalg.solve(hessian, design.T @ (target - predicted))
            beta += step
            if xp.abs(step).max() < 1e-10:
                break
        self.weight, self.bias = float(beta[0]), float(beta[1])
        return self

    def __call__(self, confidence):
        """The probability the answer is right, given the confidence the model reported writing it."""
        return _sigmoid(self.weight * _log_odds(confidence) + self.bias)

    def save(self, path):
        Path(path).write_text(json.dumps({"weight": self.weight, "bias": self.bias}) + "\n")
        return self

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def expected_calibration_error(probability, is_right, bins=10):
    """How far a stated probability sits from the rate it actually comes true, averaged over the rows.

    Quantile bins, not equal width. Every confidence here is above 0.83 and most are above 0.999, so
    equal-width bins would drop nearly every row into the top bin and report almost no error no matter
    what the mapping did.
    """
    probability = xp.asarray(probability, dtype=xp.float64)
    right = xp.asarray(is_right, dtype=xp.float64)
    if len(probability) != len(right):
        raise ValueError(f"{len(probability)} probabilities against {len(right)} labels")
    order = xp.argsort(probability, kind="stable")
    error = 0.0
    for group in xp.array_split(order, min(bins, len(order))):
        if len(group):
            error += len(group) * abs(probability[group].mean() - right[group].mean())
    return error / len(probability)


def brier(probability, is_right):
    """Mean squared error of a probability. Unlike calibration error it also punishes being unsure."""
    return float(xp.mean((xp.asarray(probability, dtype=xp.float64) - xp.asarray(is_right)) ** 2))


def ranking_auc(score, is_right):
    """The chance a right answer outscores a wrong one. 0.5 is a coin toss and the score is useless.

    Invariant to any monotone calibration, which is exactly why it is reported beside the calibration
    error: the two numbers can only be improved by different things.
    """
    score = xp.asarray(score, dtype=xp.float64)
    right = xp.asarray(is_right, dtype=bool)
    winners, losers = score[right], score[~right]
    if not len(winners) or not len(losers):
        raise ValueError("ranking needs at least one right and one wrong row to compare")
    above = winners[:, None] > losers[None, :]
    level = winners[:, None] == losers[None, :]
    return float((above.sum() + 0.5 * level.sum()) / (len(winners) * len(losers)))
