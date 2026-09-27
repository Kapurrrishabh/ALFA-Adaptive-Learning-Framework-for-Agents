"""Synthetic market data for tests and offline demos ONLY.

Every frame produced here is labelled by the provider name "synthetic" so it
can never be mistaken for real market data in an analysis or report.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .provider import DataUnavailable, Provider


def ohlcv(n: int = 800, seed: int = 0, drift: float = 0.0004, vol: float = 0.015,
          start_price: float = 1000.0, end: Optional[str] = None,
          regime_switch: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sig = np.full(n, vol)
    if regime_switch:
        # Alternate calm/turbulent stretches so regime and risk code have signal.
        block = rng.integers(60, 160)
        turbulent = False
        i = 0
        while i < n:
            sig[i:i + block] = vol * (2.2 if turbulent else 0.8)
            i += block
            turbulent = not turbulent
            block = int(rng.integers(60, 160))
    r = rng.normal(drift, sig)
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.concatenate([[start_price], close[:-1]]) * np.exp(rng.normal(0, sig * 0.3))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, sig * 0.5)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, sig * 0.5)))
    volume = rng.lognormal(13, 0.35, n) * (1 + 8 * np.abs(r))
    idx = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": volume}, index=idx)


class SyntheticProvider(Provider):
    """Offline provider: deterministic synthetic bars per symbol, no
    fundamentals or news (those domains correctly report unavailable)."""

    name = "synthetic"

    def __init__(self, n: int = 1000, end: Optional[str] = None):
        self.n, self.end = n, end

    def _seed(self, symbol: str) -> int:
        return sum(ord(ch) * (i + 1) for i, ch in enumerate(symbol)) % (2 ** 31)

    def ohlcv(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        seed = self._seed(symbol)
        index_like = symbol.startswith("^")
        return ohlcv(self.n, seed=seed, vol=0.010 if index_like else 0.017,
                     drift=0.0003 + (seed % 7 - 3) * 0.0001, end=self.end,
                     start_price=20000.0 if index_like else 200.0 + seed % 3000)

    def info(self, symbol: str) -> dict:
        return {"longName": f"Synthetic {symbol.split('.')[0]}", "currency": "INR",
                "sector": "Synthetic"}

    def financials(self, symbol: str):
        raise DataUnavailable("synthetic provider has no financial statements")

    def news(self, symbol: str, limit: int = 25):
        raise DataUnavailable("synthetic provider has no news")

    def upcoming_events(self, symbol: str):
        return []
