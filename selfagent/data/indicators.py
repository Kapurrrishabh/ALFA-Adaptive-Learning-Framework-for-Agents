"""Point-in-time technical indicators. Each takes history and returns its value as of the last bar.

Returning one number for the end of the series, instead of a column aligned to every bar, is
deliberate. An aligned column invites the off-by-one that reads tomorrow's value into today's row,
and that is the one bug here that improves every score while invalidating all of them. A caller that
wants the value as of day t passes the history up to day t and cannot express anything else.

Pure functions, no I/O. Short history raises rather than returning a partial figure, because an
indicator quietly computed over fewer bars than it names is a wrong number with a right label.
"""

from ..backend import xp

TRADING_DAYS_PER_YEAR = 252


def _checked(values, needed, name):
    values = xp.asarray(values, dtype=xp.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} needs a single series, got shape {values.shape}")
    if len(values) < needed:
        raise ValueError(f"{name} needs {needed} bars, got {len(values)}")
    return values


def simple_moving_average(closes, window):
    closes = _checked(closes, window, f"a {window}-bar moving average")
    return float(closes[-window:].mean())


def total_return(closes, window):
    """Simple return over the last `window` bars, as a fraction."""
    closes = _checked(closes, window + 1, f"a {window}-bar return")
    return float(closes[-1] / closes[-window - 1] - 1.0)


def realized_volatility(closes, window=20, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Annualised standard deviation of daily log returns over the last `window` bars."""
    closes = _checked(closes, window + 1, f"{window}-bar volatility")
    returns = xp.diff(xp.log(closes[-window - 1 :]))
    return float(returns.std(ddof=1) * xp.sqrt(periods_per_year))


def max_drawdown(closes, window=60):
    """Worst peak-to-trough fall inside the window, as a negative fraction."""
    closes = _checked(closes, window, f"a {window}-bar drawdown")
    recent = closes[-window:]
    peak = xp.maximum.accumulate(recent)
    return float((recent / peak - 1.0).min())


def relative_strength_index(closes, window=14):
    """Wilder's RSI, 0-100. Above 70 is conventionally overbought, below 30 oversold."""
    closes = _checked(closes, window + 1, f"a {window}-bar RSI")
    change = xp.diff(closes)
    gain = xp.maximum(change, 0.0)
    loss = xp.maximum(-change, 0.0)
    average_gain = float(gain[:window].mean())
    average_loss = float(loss[:window].mean())
    for step in range(window, len(change)):
        average_gain = (average_gain * (window - 1) + float(gain[step])) / window
        average_loss = (average_loss * (window - 1) + float(loss[step])) / window
    if average_loss == 0.0:
        # Every bar in the window rose. RSI is 100 by definition; dividing would be a nan.
        return 100.0 if average_gain > 0.0 else 50.0
    return 100.0 - 100.0 / (1.0 + average_gain / average_loss)


def average_true_range(highs, lows, closes, window=14):
    """Wilder's ATR in price units: typical bar range, allowing for gaps between bars."""
    highs = _checked(highs, window + 1, f"a {window}-bar ATR")
    lows = _checked(lows, window + 1, f"a {window}-bar ATR")
    closes = _checked(closes, window + 1, f"a {window}-bar ATR")
    previous_close = closes[:-1]
    true_range = xp.maximum(
        highs[1:] - lows[1:],
        xp.maximum(xp.abs(highs[1:] - previous_close), xp.abs(lows[1:] - previous_close)),
    )
    average = float(true_range[:window].mean())
    for step in range(window, len(true_range)):
        average = (average * (window - 1) + float(true_range[step])) / window
    return average
