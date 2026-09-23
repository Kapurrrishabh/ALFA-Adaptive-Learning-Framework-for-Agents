"""Price history into windows and direction labels. Reading the CSVs is the only I/O here.

Two things about this task leak the answer if done the obvious way, so both are handled here rather
than left to each caller. Windows are split by date and never at random: two windows a day apart
share all but one bar, so a random split puts near-duplicates on both sides and reports an accuracy
that will never be seen again. And the class boundaries are measured on the training period alone,
because quantiles taken over all history carry the future's volatility into the labels.

Each window ends at the bar being judged from and the label looks only forward of it, so nothing a
window can see includes its own answer.
"""

import csv

from ..backend import xp
from .features import forward_return, to_features

# yfinance's close is already split-adjusted and its OHLC are mutually consistent. adj_close removes
# dividends too, but only from the close, which would break the high >= close >= low relationship.
_COLUMNS = ("open", "high", "low", "close", "volume")


def load_bars(path):
    """(dates, (timesteps, 5) OHLCV) for one ticker, oldest first.

    Rows with a non-positive price are dropped here because log returns are undefined on them, and
    features.to_features asks its callers to clean rather than guess.
    """
    dates, rows = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                values = [float(row[name]) for name in _COLUMNS]
            except (KeyError, ValueError):
                continue
            if min(values[:4]) <= 0.0 or values[4] < 0.0:
                continue
            dates.append(row["date"])
            rows.append(values)
    if not rows:
        raise ValueError(f"{path} held no usable bars; every row was blank or non-positive")
    return dates, xp.asarray(rows, dtype=xp.float64)


def sample_ends(count, window, horizon, stride=1):
    """End indices with both a full window behind them and a full horizon ahead.

    One extra bar is needed behind the window because differencing costs a row, hence the start at
    `window` rather than `window - 1`.
    """
    return list(range(window, count - horizon, stride))


def window_at(bars, end, window):
    """The `window` standardised feature rows ending at bar `end`, plus the window's own scale.

    Standardising inside a window divides out its return spread, and that spread is trailing
    volatility: the one quantity measured to predict the forward kind (r = +0.42 held out). So the
    scale that standardising removed comes back as a last channel, logged so a quiet window and a
    wild one sit the same distance apart in either direction.
    """
    features = to_features(bars[end - window : end + 1])
    spread = xp.diff(xp.log(bars[end - window : end + 1, 3])).std()
    scale = xp.log(spread + 1e-8)
    return xp.concatenate([features, xp.full((features.shape[0], 1), scale)], axis=-1)


def index_on_or_before(dates, as_of):
    """The last bar dated on or before `as_of`, or the last bar on file when `as_of` is None.

    The as-of rule in one place. Two callers each writing their own search is how a served figure and a
    rebuilt one come to disagree about which bar "that date" means.
    """
    if as_of is None:
        return len(dates) - 1
    for index in range(len(dates) - 1, -1, -1):
        if dates[index] <= as_of:
            return index
    raise ValueError(f"no bar on or before {as_of}; the earliest on file is {dates[0]}")


def trailing_volatility(bars, end, window=20):
    """Standard deviation of the daily log returns over the `window` bars up to and including `end`.

    The one line of arithmetic the price head has to beat, and the same computation on both sides of
    that comparison: bucketing by this alone scored 47.0% held out where the GRU scored 46.9%.
    """
    closes = xp.asarray(bars[max(0, end - window) : end + 1, 3], dtype=xp.float64)
    if len(closes) < window + 1:
        raise ValueError(f"need {window + 1} bars up to {end}, got {len(closes)}")
    return float(xp.diff(xp.log(closes)).std(ddof=1))


def label_at(bars, end, horizon):
    """Forward log return from bar `end` to `horizon` bars later."""
    closes = bars[end : end + horizon + 1, 3]
    return float(forward_return(closes, horizon)[0])


def forward_volatility(bars, end, horizon):
    """Standard deviation of the daily log returns over the `horizon` bars after `end`.

    Measured against direction on 51,070 samples, this is the part of the future a price window
    actually carries: trailing volatility explains 22.1% of it, where trailing return explains 0.3%
    of the next return. So this is the label worth learning, and direction is the one to abstain on.
    """
    closes = xp.asarray(bars[end : end + horizon + 1, 3], dtype=xp.float64)
    if len(closes) < horizon + 1:
        raise ValueError(f"need {horizon + 1} bars after {end}, got {len(closes)}")
    return float(xp.diff(xp.log(closes)).std(ddof=1))


def tertile_edges(values, quantiles=(1.0 / 3.0, 2.0 / 3.0)):
    """Tertile boundaries, so the three classes are balanced instead of arbitrary.

    Pass training-period values only. A threshold like +/-1% would make the classes whatever the
    era's volatility happens to imply, and the majority-class baseline then flatters the model.
    """
    return xp.quantile(xp.asarray(values, dtype=xp.float64), xp.asarray(quantiles))


def to_tertile(values, edges):
    """Low, middle, high class against boundaries from tertile_edges."""
    return xp.searchsorted(xp.asarray(edges), xp.asarray(values, dtype=xp.float64)).astype(xp.int64)
