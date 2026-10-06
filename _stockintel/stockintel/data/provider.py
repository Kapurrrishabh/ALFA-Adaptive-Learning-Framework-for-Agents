"""Provider abstraction: every data source implements this interface.

OHLCV frames are normalized to a DatetimeIndex with lowercase columns
open/high/low/close/volume, split/dividend adjusted. Missing data raises
DataUnavailable — providers never fill in substitutes.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..evidence import Provenance, utcnow_iso

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataUnavailable(Exception):
    """Raised when a provider cannot supply real data for a request."""


@dataclass
class NewsItem:
    headline: str
    summary: str
    published_at: str            # ISO timestamp
    source: str
    url: str = ""
    symbol: Optional[str] = None
    provenance: Provenance = field(default_factory=lambda: Provenance(source="unknown", as_of=utcnow_iso()))


class Provider(abc.ABC):
    """A market data source. All methods raise DataUnavailable on failure."""

    name: str = "abstract"

    @abc.abstractmethod
    def ohlcv(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Adjusted daily bars, oldest first."""

    @abc.abstractmethod
    def info(self, symbol: str) -> Dict[str, Any]:
        """Company profile / snapshot fields (sector, market cap, ratios...)."""

    @abc.abstractmethod
    def financials(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """{'income': df, 'balance': df, 'cashflow': df}; columns = periods."""

    @abc.abstractmethod
    def news(self, symbol: str, limit: int = 25) -> List[NewsItem]:
        """Recent articles for the symbol."""

    def intraday(self, symbol: str, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
        """Intraday bars, exchange-local time, oldest first."""
        raise DataUnavailable(f"{self.name} provider has no intraday data")

    def upcoming_events(self, symbol: str) -> List[str]:
        """Scheduled events (earnings, ex-dividend) after today, as sentences."""
        raise DataUnavailable(f"{self.name} provider has no event calendar")

    def provenance(self, symbol: str, as_of: Optional[str] = None,
                   period: Optional[str] = None) -> Provenance:
        return Provenance(source=f"{self.name}:{symbol}", as_of=as_of or utcnow_iso(),
                          period=period)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase the OHLCV columns, sort chronologically, drop dead rows."""
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    if "adj_close" in out.columns and "close" in out.columns:
        # Adjust the whole bar, not just the close, or bars straddle a split
        # with close outside [low, high] and open-to-open returns jump.
        factor = out["adj_close"] / out["close"]
        for col in ("open", "high", "low"):
            if col in out.columns:
                out[col] = out[col] * factor
        out["close"] = out["adj_close"]
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise DataUnavailable(f"OHLCV frame is missing columns {missing}; got {list(out.columns)}")
    out = out[OHLCV_COLUMNS].sort_index()
    out = out.dropna(subset=["close"])
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out
