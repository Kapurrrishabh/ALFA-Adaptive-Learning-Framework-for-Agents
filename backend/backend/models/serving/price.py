"""The week-ahead risk outlook, and which of the two things that can produce it gets served.

The generator never computes this: it only phrases what arrives in the evidence, so whatever is behind
`outlook calm ; outlook_confidence 44%` decides whether that line is worth reading at all. Two candidates
come out of one artifact, and the numbers recorded in that artifact pick between them:

  `PriceHead`     the trained GRU over a 128-bar window, 202,755 parameters, 49.8% held out
  `Persistence`   one line of arithmetic -- bucket the trailing 20-day volatility, 47.8%

`load` serves the head only when the file says it beat the rule by more than the noise in its own
measurement. On the served checkpoint it does: **49.8% against 47.8%** on all 28,676 held-out windows,
a gap of 6.8 standard errors. The transformer tower, measured on the same windows, scores 46.3% on eight
times the parameters -- below the arithmetic -- and averaging both towers with the rule scores 48.9%, so
neither is served. The margin rule stays either way: the one line of arithmetic is what answers until a
head clears it by more than the measurement's own noise.

**The point-in-time rule.** A window is built from bars up to the as-of date and no further, by
`selfagent.data.prices.window_at`, which slices `bars[end - window : end + 1]` and standardises inside
that slice. So a served vector and one rebuilt later from a truncated file are the same array, exactly --
the property `tests/test_agent.py` pins against both cheap ways to break it, standardising over the whole
series and taking the scale from `bars[-1]`, because either still produces a plausible number.

**Why both confidences are hit rates.** `Persistence` quotes the row-normalised training confusion of its
own buckets, so "44%" means that bucket was right 44% of the time in the training period. The head's raw
softmax is not that, so `confidence` runs it through the Platt fit recorded in the artifact -- an
uncalibrated number in the evidence is one the answer will quote verbatim. It is a separate method from
`probabilities` on purpose: a calibrated top entry is no longer a distribution, and rescaling the rest to
fit around it lets a runner-up overtake the class the head actually chose.
"""

import math

from backend.models.core import pretrained
from backend.models.core.autograd import functional as F
from backend.models.core.autograd import no_grad
from backend.models.core.backend import xp
from backend.models.data import advisory, prices
from backend.models.learning import Calibrator
from backend.models.networks import PriceWindowClassifier

# The tertile order the classes come in: lowest forward volatility is the calm one.
CLASSES = advisory.RISK_NAMES

TRAILING_WINDOW = 20

# The two towers `train_price_head.py` compares. Named here because the loader has to rebuild the one
# that was saved, and the name in the file is the only record of which it was.
RECURRENT_TOWER = "gru"
TOWERS = (RECURRENT_TOWER, "transformer")

# What a served file has to carry. Without any one of these the artifact cannot be served at all rather
# than being served with a guess: edges decide which class a value is, and the two accuracies and the
# count they were measured on decide which candidate answers.
REQUIRED = ("tower", "target", "horizon", "edges", "accuracy", "evaluated", "persistence",
            "trailing_edges", "table")


def price_bands(close, edges, horizon):
    """How far from `close` each volatility class puts one standard deviation, `horizon` bars out.

    A class is a band of daily log-return spread, so the distance it implies is that spread widened by the
    square root of the horizon. The top class is open above -- its edge is a floor, not a range -- and
    `open` says so instead of closing the band with an invented ceiling.
    """
    spreads = [float(edge) * horizon ** 0.5 for edge in edges]
    spreads.append(spreads[-1])
    return [{"name": name, "sigma": spread, "low": close * math.exp(-spread),
             "high": close * math.exp(spread), "open": index == len(CLASSES) - 1}
            for index, (name, spread) in enumerate(zip(CLASSES, spreads))]


def beats(accuracy, rule, evaluated):
    """Whether a head's held-out accuracy clears the arithmetic by more than its own measurement noise.

    One standard error of a proportion: at 49% on 28,676 windows that is 0.3 points, and the measured gap
    is 1.4. Without the margin the served advisor would flip on two windows in a thousand -- which is what
    the first comparison, on a 1,920-window prefix, was deciding on.
    """
    return accuracy - rule > (accuracy * (1.0 - accuracy) / evaluated) ** 0.5


class Persistence:
    """Bucket the trailing volatility, quote how often that bucket was right.

    `edges` bucket the trailing value and `table` maps each bucket to its measured distribution over the
    forward classes, both from the training period alone. Taken over all history they would carry the
    future's volatility into a figure the answer states.
    """

    name = "persistence"

    def __init__(self, edges, table, measured, window=TRAILING_WINDOW):
        self.edges = xp.asarray(edges, dtype=xp.float64)
        self.table = xp.asarray(table, dtype=xp.float64)
        if self.table.shape != (len(self.edges) + 1, len(CLASSES)):
            raise ValueError(f"table is {self.table.shape}, expected "
                             f"{(len(self.edges) + 1, len(CLASSES))} for {len(CLASSES)} classes")
        self.measured = measured
        self.window = window

    @property
    def version(self):
        return f"{self.name}@{self.window}d"

    @property
    def bars_needed(self):
        return self.window + 1

    def probabilities(self, bars, end):
        value = prices.trailing_volatility(bars, end, self.window)
        return self.table[int(prices.to_tertile([value], self.edges)[0])]

    def confidence(self, probabilities):
        """The table row's own peak. It is already the rate that bucket came true, so nothing rescales it."""
        return float(max(probabilities))

    def describe(self):
        """The card a client shows in place of trusting the deployment. The table is already a hit rate,
        so this rule has nothing left to calibrate."""
        return {**self.measured, "serving": self.name, "version": self.version, "parameters": 0,
                "calibrated": True}


class PriceHead:
    """The trained tower, pinned to the config that shaped it and the edges it was labelled with."""

    def __init__(self, model, config, tower, measured, calibrator=None):
        self.model = model
        self.config = config
        self.name = tower
        self.measured = measured
        self.calibrator = calibrator

    @classmethod
    def of(cls, config, weights, tower, recurrent, measured, calibrator=None):
        model = PriceWindowClassifier(config, recurrent=recurrent)
        model.load_state_dict(weights)
        model.eval()
        return cls(model, config, tower, measured, calibrator)

    @property
    def version(self):
        """Which weights answered, so a logged turn can be traced to one file rather than to a name."""
        return f"{self.name}@{self.config.fingerprint}"

    @property
    def bars_needed(self):
        # One extra bar behind the window because differencing costs a row.
        return self.config.price_window + 2

    def probabilities(self, bars, end):
        window = prices.window_at(bars, end, self.config.price_window)
        if window.shape[-1] != self.config.price_channels:
            raise ValueError(f"the feature builder produces {window.shape[-1]} channels and this head "
                             f"was trained on {self.config.price_channels}; retrain or rebuild")
        with no_grad():
            return F.softmax(self.model(window[None, ...])).data[0]

    def confidence(self, probabilities):
        """How often the top class comes true, rather than how sure the softmax was, once a fit exists.

        Kept apart from `probabilities` rather than folded into it: a calibrated top entry is no longer a
        distribution, and rescaling the others to fit around it lets one of them overtake the top, which
        hands the answer a different class than the head chose.
        """
        raw = float(max(probabilities))
        return raw if self.calibrator is None else float(self.calibrator(raw))

    def describe(self):
        """The card a client shows in place of trusting the deployment: what is answering, how it is
        shaped, and what it scored. Straight out of the artifact, so the page cannot quote a stale figure."""
        return {**self.measured, "serving": self.name, "version": self.version,
                "parameters": int(sum(p.size for p in self.model.parameters())),
                "dim": self.config.dim, "calibrated": self.calibrator is not None}


def load(path):
    """The advisor for `path`: the head when the file records it beating the rule, otherwise the rule.

    Both candidates come from one file so the comparison cannot be made against a number from somewhere
    else, and the choice is the recorded measurement rather than a flag a caller can set wrongly.
    """
    config, weights = pretrained.load(path)
    recorded = pretrained.metadata(path)
    missing = [name for name in REQUIRED if name not in recorded]
    if missing:
        raise ValueError(f"{path} records no {', '.join(missing)}; re-save it with "
                         f"scripts/train_price_head.py --save, which writes what serving needs")
    if not beats(recorded["accuracy"], recorded["persistence"], recorded["evaluated"]):
        return Persistence(recorded["trailing_edges"], recorded["table"], recorded)
    if recorded["tower"] not in TOWERS:
        raise ValueError(f"{path} was written by tower {recorded['tower']!r}, which this loader cannot "
                         f"rebuild; expected one of {TOWERS}")
    # An artifact written before the fit carries no `calibration`, and then the confidence is the raw
    # softmax exactly as it was. `describe` reports which of the two it is rather than leaving it implied.
    fit = recorded.get("calibration")
    return PriceHead.of(config, weights, recorded["tower"], recorded["tower"] == RECURRENT_TOWER,
                        recorded, Calibrator(**fit) if fit else None)
