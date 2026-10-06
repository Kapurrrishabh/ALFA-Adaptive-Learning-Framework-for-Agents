"""Deterministic technical indicators. Pure functions of price/volume series;
no I/O, no state, so every value is unit-testable against known references.
RSI and ATR use Wilder smoothing (SMA-seeded) to match standard charting
platforms value for value.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def wma(s: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    return s.rolling(window).apply(lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)


def returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    return close.pct_change(window) * 100.0


def wilder(s: pd.Series, window: int) -> pd.Series:
    """Wilder smoothing: seeded with the simple mean of the first `window`
    values, then avg_t = (avg_{t-1} * (window - 1) + x_t) / window."""
    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < window:
        return pd.Series(out, index=s.index)
    start = valid[0] + window - 1
    out[start] = x[valid[0]:start + 1].mean()
    for i in range(start + 1, len(x)):
        out[i] = (out[i - 1] * (window - 1) + x[i]) / window
    return pd.Series(out, index=s.index)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = wilder(delta.clip(lower=0.0), window)
    avg_loss = wilder((-delta).clip(lower=0.0), window)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # All-gain windows have zero loss -> RS undefined -> RSI is 100 by definition.
    out = out.where(avg_loss != 0.0, 100.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_window).min()
    highest = high.rolling(k_window).max()
    k = 100.0 * (close - lowest) / (highest - lowest).replace(0.0, np.nan)
    return k, k.rolling(d_window).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(),
                      (low - prev_close).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), window)


def bollinger(close: pd.Series, window: int = 20,
              num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    std = close.rolling(window).std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def historical_volatility(close: pd.Series, window: int = 20,
                          annualization: int = 252) -> pd.Series:
    return log_returns(close).rolling(window).std(ddof=1) * np.sqrt(annualization)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series,
                              volume: pd.Series) -> pd.Series:
    rng = (high - low).replace(0.0, np.nan)
    clv = ((close - low) - (high - close)) / rng
    return (clv.fillna(0.0) * volume).cumsum()


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Today's volume vs its trailing average (excluding today)."""
    avg = volume.shift(1).rolling(window).mean()
    return volume / avg.replace(0.0, np.nan)


def slope(s: pd.Series, window: int = 20) -> pd.Series:
    """Rolling normalized least-squares slope (per bar, as fraction of level)."""
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float((x ** 2).sum())

    def _fit(y: np.ndarray) -> float:
        level = y.mean()
        if level == 0:
            return 0.0
        return float(np.dot(x, y - y.mean()) / denom / level)

    return s.rolling(window).apply(_fit, raw=True)
