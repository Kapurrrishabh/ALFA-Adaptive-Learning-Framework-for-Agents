"""Daily returns as tokens, so a language model can learn their distribution and write new ones.

This is the Chronos recipe -- scale, quantise, train with cross-entropy, sample -- with two changes, both
forced by what a return series is rather than chosen.

**The bins are uniform in asinh space, not in the return.** Chronos spaces 4,094 bins uniformly over
[-15s, 15s], which suits sales or energy and fails on returns: a daily return is peaked near zero and has
tails a normal distribution puts at odds of billions to one, so uniform bins either blur the centre or
cut off the crashes. asinh is linear near zero and logarithmic in the tails -- the mu-law compander WaveNet
uses on audio -- so BINS uniform bins in asinh(x) give a resolution of 0.03 scale units at the centre and
still reach LIMIT scale units at the edges.

**The scale is measured strictly before the window.** Chronos divides by the mean absolute value of the
context. A decoder trained on every position of its window would then be divided by a number that already
contains the returns it is asked to predict, which leaks their size into every input. Here the scale is the
mean absolute return over the SCALE_BARS bars before the window's first return, so every token in a window
is predicted from bars that happened before it, in training, in evaluation and in serving alike.

The log score in return units is exact rather than approximate. A bin's probability spread uniformly over
its width is a proper density, so `density` turns a categorical prediction into one that can be compared
directly against a Gaussian or a GARCH model on the same returns -- which is the comparison that says
whether the model has learned anything.
"""

from backend.models.core.backend import xp

# Vocabulary size. 256 bins put 0.032 units of asinh between neighbours: 3% of a scale unit at the centre,
# 16% at five scale units out. Provisional -- the measure is whether halving it moves held-out log score.
BINS = 256

# Largest scaled return a bin can hold. Measured over the 101 tickers on file, a scale unit is the mean
# absolute daily return, around 1.3%, so 30 units is a 39% day. Beyond it a return is clipped into the edge
# bin, and `clipped` counts how often that happens so the loss is visible rather than silent.
LIMIT = 30.0

# Bars behind a window that its scale is measured over. A year of trading days: long enough that one bad
# week does not set the unit, short enough to follow a stock whose character has changed.
SCALE_BARS = 250

_EDGES = xp.linspace(-xp.arcsinh(LIMIT), xp.arcsinh(LIMIT), BINS + 1)
# The same edges in scaled-return units. The width between neighbours is what turns a bin probability into
# a density.
_BOUNDS = xp.sinh(_EDGES)


def log_returns(closes):
    """Daily log returns, one fewer than the closes. Rejects a non-positive close rather than guessing."""
    closes = xp.asarray(closes, dtype=xp.float64)
    if (closes <= 0).any():
        raise ValueError(f"log returns need positive closes; got {closes[closes <= 0][:3].tolist()}")
    return xp.diff(xp.log(closes))


def scale_before(returns, start):
    """The unit a window starting at `start` is measured in: mean |return| over the SCALE_BARS before it.

    Raises when fewer than SCALE_BARS precede `start`, or when every one of them is zero -- a scale of
    zero would divide every later return into infinity, and a shorter history would make the unit mean
    something different for the first year of every ticker.
    """
    if start < SCALE_BARS:
        raise ValueError(f"a window starting at {start} has only {start} returns behind it; the scale "
                         f"needs {SCALE_BARS}")
    scale = float(xp.abs(returns[start - SCALE_BARS : start]).mean())
    if not scale > 0.0:
        raise ValueError(f"the {SCALE_BARS} returns before {start} are all zero, so there is no unit to "
                         f"measure the window in")
    return scale


def encode(returns, scale):
    """Token ids, one per return. Values past LIMIT scale units land in the edge bins."""
    code = xp.arcsinh(xp.asarray(returns, dtype=xp.float64) / scale)
    ids = xp.searchsorted(_EDGES, code, side="right") - 1
    return xp.clip(ids, 0, BINS - 1).astype(xp.int64)


def clipped(returns, scale):
    """How many returns fall outside what the bins can represent."""
    return int((xp.abs(xp.asarray(returns, dtype=xp.float64) / scale) > LIMIT).sum())


def decode(ids, scale, rng):
    """One return per token, drawn uniformly across its bin.

    Uniform in the return and not the centre of the bin, because that is the density `density` scores:
    a sampler that always emitted the centre would generate a lattice the model never claimed.
    """
    ids = xp.asarray(ids, dtype=xp.int64)
    low, high = _BOUNDS[ids], _BOUNDS[ids + 1]
    return (low + (high - low) * rng.random(ids.shape)) * scale


def density(probabilities, returns, scale):
    """The density a categorical prediction puts on each realised return, in return units.

    `probabilities` is (..., BINS) and `returns` matches its leading shape. The piecewise-uniform density
    is what makes a bin model and a Gaussian comparable at all: its log is the log score, which is strictly
    proper, so it is maximised in expectation only by the true distribution and cannot be won by hedging.
    """
    ids = encode(returns, scale)
    widths = (_BOUNDS[ids + 1] - _BOUNDS[ids]) * scale
    return xp.take_along_axis(xp.asarray(probabilities), ids[..., None], axis=-1)[..., 0] / widths


def cdf(probabilities, returns, scale):
    """The forecast's cumulative probability at each realised return: the PIT, uniform when calibrated.

    Read under the same piecewise-uniform density `density` scores, so the calibration check and the log
    score describe one forecast.
    """
    ids = encode(returns, scale)
    x = xp.clip(xp.asarray(returns, dtype=xp.float64) / scale, _BOUNDS[0], _BOUNDS[-1])
    below = xp.cumsum(probabilities, axis=-1) - probabilities
    inside = (x - _BOUNDS[ids]) / (_BOUNDS[ids + 1] - _BOUNDS[ids])
    taken = xp.take_along_axis(xp.asarray(probabilities), ids[..., None], axis=-1)[..., 0]
    return xp.take_along_axis(below, ids[..., None], axis=-1)[..., 0] + inside * taken
