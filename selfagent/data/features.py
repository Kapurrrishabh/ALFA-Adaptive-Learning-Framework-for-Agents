"""Turning raw OHLCV bars into model inputs. Pure functions, no I/O.

Raw price levels are non-stationary. A model fed them learns the price range of its training
era rather than market behaviour, and collapses on any window outside that range. So the
model never sees a price: it sees log returns, standardised inside each window.
"""

from ..backend import xp

CLOSE_COLUMN = 3


def to_features(bars):
    """(..., timesteps, 5) raw OHLCV -> (..., timesteps - 1, 5) standardised features.

    Differencing costs one row: to feed a window of N timesteps, pass N + 1 raw bars.
    """
    bars = xp.asarray(bars, dtype=xp.float64)
    if bars.shape[-1] != 5:
        raise ValueError(f"expected 5 OHLCV columns, got {bars.shape[-1]}")
    if bars.shape[-2] < 2:
        raise ValueError("need at least 2 bars to compute a return")

    prices = bars[..., :4]
    volume = bars[..., 4:]
    if (prices <= 0).any():
        raise ValueError(
            "found a non-positive price; log returns are undefined. "
            "Clean or drop these bars in the loader rather than here."
        )
    if (volume < 0).any():
        raise ValueError("found a negative volume")

    returns = xp.diff(xp.log(prices), axis=-2)
    volume_change = xp.diff(xp.log1p(volume), axis=-2)
    return standardize(xp.concatenate([returns, volume_change], axis=-1))


def standardize(features, eps=1e-8):
    """Per-window z-scoring, so a $5 stock and a $500 stock look the same to the model."""
    mean = features.mean(axis=-2, keepdims=True)
    deviation = features.std(axis=-2, keepdims=True)
    return (features - mean) / (deviation + eps)


def forward_return(closes, horizon):
    """Log return from each bar to `horizon` bars later. Trailing rows have no label."""
    closes = xp.asarray(closes, dtype=xp.float64)
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 bar, got {horizon}")
    log_close = xp.log(closes)
    return log_close[horizon:] - log_close[:-horizon]
