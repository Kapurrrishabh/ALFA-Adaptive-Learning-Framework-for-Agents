"""When to say nothing, solved from the log instead of picked by hand.

The hand-picked 0.9980 is a number with no meaning attached. It is a mean token probability, and nothing
about it states how often an answer above it turns out right — which is the only thing anyone deciding
whether to trust the answer wants to know. This states the bar in those units instead ("answer only when
it is at least 60% likely to be right") and solves for the confidence cut that delivers it, on rows the
cut was not chosen on.

What it cannot do is beat the hand-picked cut on the precision-coverage curve. A threshold on a
calibrated probability orders rows exactly as the raw confidence does, so both cuts live on one curve and
the curve is fixed by the model. What it buys is a defensible point on that curve and a number that
survives a new checkpoint: 0.9980 means nothing on a retrained model, 60% means the same thing always.
"""

import json
from pathlib import Path

from ..backend import xp

# Matching the floor in check_answers.report_confidence. Below a fifth of the rows, precision is
# estimated from too few answers to be a number rather than a coincidence.
COVERAGE_FLOOR = 0.2


def precision_at(confidence, is_right, cut):
    """(precision, coverage) if the model answered only where confidence reaches the cut."""
    answered = xp.asarray(confidence) >= cut
    coverage = float(answered.mean())
    if not answered.any():
        return float("nan"), 0.0
    return float(xp.asarray(is_right, dtype=bool)[answered].mean()), coverage


class Abstainer:
    """A confidence cut, chosen so the answers above it are right at least as often as asked."""

    def __init__(self, cut, wanted=0.0, expected=float("nan"), coverage=float("nan")):
        self.cut = float(cut)
        self.wanted = float(wanted)
        # What the fitting rows said this cut would deliver. Kept so a served threshold can be compared
        # against what actually happens, which is the only way drift becomes visible.
        self.expected = float(expected)
        self.coverage = float(coverage)

    @classmethod
    def fit(cls, confidence, is_right, wanted, floor=COVERAGE_FLOOR):
        """The lowest cut whose precision here meets `wanted`, so coverage is as high as the bar allows.

        Lowest rather than best: the highest-precision cut is almost always the one answering three rows,
        and a threshold fitted to three rows is a coincidence quoted as a policy.
        """
        confidence = xp.asarray(confidence, dtype=xp.float64)
        scored = [(precision, coverage, cut)
                  for precision, coverage, cut in
                  ((*precision_at(confidence, is_right, cut), cut) for cut in xp.unique(confidence))
                  if coverage >= floor]
        if not scored:
            raise ValueError(
                f"cannot fit a cut on {len(confidence)} rows with a coverage floor of {floor}; "
                f"either lower the floor or collect more feedback"
            )
        met = [row for row in scored if row[0] >= wanted]
        # Nobody clearing the bar is a real answer: take the most precise cut the floor permits and let
        # `expected` report that it fell short. Returning the bar itself would claim a precision never seen.
        precision, coverage, cut = min(met, key=lambda row: row[2]) if met else max(scored)
        return cls(cut, wanted, precision, coverage)

    def answers(self, confidence):
        """True where the model should speak, False where it should refuse."""
        return xp.asarray(confidence) >= self.cut

    def save(self, path):
        Path(path).write_text(json.dumps(
            {"cut": self.cut, "wanted": self.wanted, "expected": self.expected,
             "coverage": self.coverage}) + "\n")
        return self

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
