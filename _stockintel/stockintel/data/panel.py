"""Cross-sectional price panel: many stocks, aligned dates, cached on disk.

Cross-sectional signals (ranking stocks against each other) need the whole
universe at once. Downloads run in small sequential batches to keep memory
low; prices are stored as float32.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .provider import DataUnavailable

log = logging.getLogger("stockintel.panel")

INDEX_LISTS = {
    "nifty50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "nifty200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "midcap150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
}
CACHE_DIR = Path.home() / ".stockintel" / "cache"
CACHE_TTL_S = 12 * 3600
BATCH = 50


@dataclass
class Panel:
    close: pd.DataFrame      # dates x symbols, split/dividend adjusted
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    sectors: Dict[str, str]
    source: str

    @property
    def symbols(self) -> List[str]:
        return list(self.close.columns)

    def traded_value(self) -> pd.DataFrame:
        return self.close * self.volume


def index_constituents(index: str = "nifty500") -> pd.DataFrame:
    """Today's constituents (symbol, sector). Using today's list for history
    is survivorship-biased; callers must say so."""
    if index not in INDEX_LISTS:
        raise ValueError(f"unknown index {index!r}; choose from {sorted(INDEX_LISTS)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{index}.csv"
    if not path.exists() or time.time() - path.stat().st_mtime > 7 * 86400:
        import httpx
        resp = httpx.get(INDEX_LISTS[index], headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
                         follow_redirects=True)
        if resp.status_code == 200 and "Symbol" in resp.text[:300]:
            path.write_text(resp.text)
        elif path.exists():
            log.warning("constituent refresh failed (HTTP %s); using cached list", resp.status_code)
        else:
            raise DataUnavailable(f"could not download {index} constituents: HTTP {resp.status_code}")
    df = pd.read_csv(path)
    return pd.DataFrame({"symbol": df["Symbol"].str.strip(), "sector": df["Industry"].str.strip()})


def load_panel(symbols: List[str], sectors: Optional[Dict[str, str]] = None, period: str = "10y",
               suffix: str = ".NS", refresh: bool = False) -> Panel:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{sorted(symbols)}|{period}|{suffix}".encode()).hexdigest()[:12]
    path = CACHE_DIR / f"panel_{digest}.pkl"
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < CACHE_TTL_S:
        return pd.read_pickle(path)
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailable("yfinance is not installed") from exc
    fields = {f: [] for f in ("Open", "High", "Low", "Close", "Volume")}
    for i in range(0, len(symbols), BATCH):
        chunk = [s + suffix for s in symbols[i:i + BATCH]]
        raw = yf.download(chunk, period=period, auto_adjust=True, progress=False, threads=False,
                          group_by="column")
        if raw is None or raw.empty:
            continue
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        for f in fields:
            part = raw[f].astype("float32")
            part.columns = [c[:-len(suffix)] if c.endswith(suffix) else c for c in part.columns]
            fields[f].append(part)
        del raw
    if not fields["Close"]:
        raise DataUnavailable("batch download returned no data")
    merged = {f: pd.concat(v, axis=1).sort_index() for f, v in fields.items()}
    close = merged["Close"].dropna(how="all", axis=1)
    keep = list(close.columns)
    panel = Panel(close=close, open=merged["Open"][keep], high=merged["High"][keep],
                  low=merged["Low"][keep], volume=merged["Volume"][keep],
                  sectors={s: (sectors or {}).get(s, "Unknown") for s in keep}, source="yahoo")
    missing = sorted(set(symbols) - set(keep))
    if missing:
        log.warning("%d symbols had no data: %s", len(missing), missing[:20])
    pd.to_pickle(panel, path)
    return panel


def universe_panel(index: str = "nifty500", period: str = "10y", refresh: bool = False) -> Panel:
    cons = index_constituents(index)
    return load_panel(list(cons["symbol"]), dict(zip(cons["symbol"], cons["sector"])), period,
                      refresh=refresh)


def panel_from_frames(frames: Dict[str, pd.DataFrame], sectors: Optional[Dict[str, str]] = None) -> Panel:
    """Build a panel from per-symbol OHLCV frames (tests, offline use)."""
    def stack(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: f[col] for s, f in frames.items()}).sort_index()
    return Panel(close=stack("close"), open=stack("open"), high=stack("high"), low=stack("low"),
                 volume=stack("volume"), sectors={s: (sectors or {}).get(s, "Unknown") for s in frames},
                 source="frames")
