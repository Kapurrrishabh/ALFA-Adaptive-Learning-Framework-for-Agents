"""The week-ahead risk outlook, and which of the two things that can produce it gets served.

The generator never computes this: it only phrases what arrives in the evidence, so whatever is behind
`outlook calm ; outlook_confidence 44%` decides whether that line is worth reading at all. Two candidates
come out of one artifact, and the numbers recorded in that artifact pick between them:

  `PriceHead`     the trained GRU over a 128-bar window, 202,755 parameters, 49.1% held out
  `Persistence`   one line of arithmetic -- bucket the trailing 20-day volatility, 47.8%

`load` serves the head only when the file says it beat the rule by more than the noise in its own
measurement. On the served checkpoint it does: **49.1% against 47.8%** on all 28,676 held-out windows,
a gap of 4.6 standard errors. The earlier reading that the head only *ties* arithmetic came from scoring
it on the first 1,920 held-out windows, which is 7 of 101 tickers, while the rule was scored on a sample
of all of them (§5.14). The margin rule stays either way: the one line of arithmetic is what answers
until a head clears it by more than the measurement's own noise.

**The point-in-time rule.** A window is built from bars up to the as-of date and no further, by
`selfagent.data.prices.window_at`, which slices `bars[end - window : end + 1]` and standardises inside
that slice. So a served vector and one rebuilt later from a truncated file are the same array, exactly --
the property `tests/test_agent.py` pins against both cheap ways to break it, standardising over the whole
series and taking the scale from `bars[-1]`, because either still produces a plausible number.

**Why the rule's confidence is a table rather than a softmax.** `Persistence` quotes the row-normalised
training confusion of its own buckets, so "44%" means that bucket was right 44% of the time in the
training period. The head's softmax is uncalibrated -- B3 measured the generator's needing a Platt fit to
be readable at all -- and an uncalibrated number in the evidence is one the answer will quote verbatim.
"""

from selfagent import pretrained
from selfagent.autograd import functional as F
from selfagent.autograd import no_grad
from selfagent.backend import xp
from selfagent.data import advisory, prices
from selfagent.models import PriceWindowClassifier

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

    def __init__(self, edges, table, window=TRAILING_WINDOW):
        self.edges = xp.asarray(edges, dtype=xp.float64)
        self.table = xp.asarray(table, dtype=xp.float64)
        if self.table.shape != (len(self.edges) + 1, len(CLASSES)):
            raise ValueError(f"table is {self.table.shape}, expected "
                             f"{(len(self.edges) + 1, len(CLASSES))} for {len(CLASSES)} classes")
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


class PriceHead:
    """The trained tower, pinned to the config that shaped it and the edges it was labelled with."""

    def __init__(self, model, config, tower):
        self.model = model
        self.config = config
        self.name = tower

    @classmethod
    def of(cls, config, weights, tower, recurrent):
        model = PriceWindowClassifier(config, recurrent=recurrent)
        model.load_state_dict(weights)
        model.eval()
        return cls(model, config, tower)

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
        return Persistence(recorded["trailing_edges"], recorded["table"])
    if recorded["tower"] not in TOWERS:
        raise ValueError(f"{path} was written by tower {recorded['tower']!r}, which this loader cannot "
                         f"rebuild; expected one of {TOWERS}")
    return PriceHead.of(config, weights, recorded["tower"], recorded["tower"] == RECURRENT_TOWER)
