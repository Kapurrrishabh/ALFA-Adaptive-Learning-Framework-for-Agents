"""Cross-sectional price panel: many stocks, aligned dates, cached on disk.

Cross-sectional signals (ranking stocks against each other) need the whole
universe at once. Downloads run in small sequential batches to keep memory
low; prices are stored as float32.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import pandas as pd

from backend.database.sources.provider import DataUnavailable
from backend.paths import DATA

log = logging.getLogger("stockintel.panel")

INDEX_LISTS = {
    "nifty50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "nifty200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "midcap150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
}
NSE_LISTS = {
    **INDEX_LISTS,
    "equity_l": "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    "etf": "https://archives.nseindia.com/content/equities/eq_etfseclist.csv",
    "sme": "https://archives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv",
}
# (symbol, ISIN) columns, read in this order: an ISIN found earlier is kept, so a company that moved up from
# SME keeps its main-board symbol. nifty500 is here for the REITs, which EQUITY_L leaves out
ISIN_COLUMNS = {"equity_l": ("SYMBOL", "ISIN NUMBER"), "etf": ("Symbol", "ISINNumber"),
                "sme": ("SYMBOL", "ISIN_NUMBER"), "nifty500": ("Symbol", "ISIN Code")}
CACHE_DIR = DATA / "cache"
CACHE_TTL_S = 12 * 3600
# provisional: three days spans a weekend the VM is off. How often the Nifty 200 buy list differs between panels
# that many days apart, over past dates, would confirm it
STALE_LIMIT_S = 3 * 86400
BATCH = 50
_downloading = threading.Lock()
_refreshes_ended = 0


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
    df = _constituents_file(index, INDEX_LISTS)
    return pd.DataFrame({"symbol": df["Symbol"].str.strip(), "sector": df["Industry"].str.strip()})


def listed_shares() -> pd.DataFrame:
    """Every share on NSE's main board (symbol, name), from NSE's own list."""
    df = _constituents_file("equity_l", NSE_LISTS)
    return pd.DataFrame({"symbol": df["SYMBOL"].str.strip(), "name": df["NAME OF COMPANY"].str.strip()})


def company_names(index: str = "nifty500") -> Dict[str, str]:
    """symbol -> the company's name as people write it: "Infosys Ltd." becomes "Infosys"."""
    df = _constituents_file(index, INDEX_LISTS)
    return {sym.strip(): plain_name(name) for sym, name in zip(df["Symbol"], df["Company Name"])}


def plain_name(name: str) -> str:
    """Any case, because NSE's list of shares writes some names as "360 ONE WAM LIMITED"."""
    return re.sub(r"\s+(ltd\.?|limited)$", "", name.strip(), flags=re.I)


class Listing(NamedTuple):
    symbol: str
    kind: str        # the NSE list it was found in: a key of ISIN_COLUMNS


def isin_directory() -> Dict[str, Listing]:
    """ISIN -> NSE symbol, for every listed share, ETF, SME share and REIT."""
    directory: Dict[str, Listing] = {}
    for kind, (symbol, isin) in ISIN_COLUMNS.items():
        df = _constituents_file(kind, NSE_LISTS)
        missing = sorted({symbol, isin} - set(df.columns))
        if missing:
            raise DataUnavailable(f"NSE's {kind} list has no {missing} column; its header is {list(df.columns)}. "
                                  "The format changed: update panel.ISIN_COLUMNS.")
        for s, i in zip(df[symbol].str.strip(), df[isin].str.strip()):
            directory.setdefault(i, Listing(s, kind))
    return directory


def _constituents_file(name: str, lists: Dict[str, str]) -> pd.DataFrame:
    """One of NSE's `lists`, refreshed weekly and kept in the cache."""
    if name not in lists:
        raise ValueError(f"unknown NSE list {name!r}; choose from {sorted(lists)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.csv"
    if not path.exists() or time.time() - path.stat().st_mtime > 7 * 86400:
        import httpx
        resp = httpx.get(lists[name], headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
                         follow_redirects=True)
        if resp.status_code == 200 and "symbol" in resp.text[:300].lower():
            path.write_text(resp.text)
        elif path.exists():
            log.warning("NSE %s list refresh failed (HTTP %s); using cached list", name, resp.status_code)
        else:
            raise DataUnavailable(f"could not download NSE's {name} list: HTTP {resp.status_code}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()      # EQUITY_L writes " ISIN NUMBER"
    return df


def load_panel(symbols: List[str], sectors: Optional[Dict[str, str]] = None, period: str = "10y",
               suffix: str = ".NS", refresh: bool = False, serve_stale: bool = False) -> Panel:
    """The cached panel, or a new download once it is CACHE_TTL_S old. With `serve_stale`, for the server, one up to
    STALE_LIMIT_S old is served at once while one background download replaces it: a one-off job would exit before
    that download ends, and use the stale panel without a word."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{sorted(symbols)}|{period}|{suffix}".encode()).hexdigest()[:12]
    path = CACHE_DIR / f"panel_{digest}.pkl"
    download = (path, symbols, sectors, period, suffix)
    age = _age(path)
    if not refresh and (age < CACHE_TTL_S or serve_stale and age < STALE_LIMIT_S):
        if age >= CACHE_TTL_S:
            threading.Thread(target=_refresh, args=download, daemon=True, name="panel-refresh").start()
        return _with_closes(pd.read_pickle(path))      # one cached by older code can end on a day with no closes
    with _downloading:
        if not refresh and _age(path) < CACHE_TTL_S:      # downloaded by another caller while this one waited
            return _with_closes(pd.read_pickle(path))
        return _download(*download)


def refreshes_ended() -> int:
    """How many background refreshes have ended, so a holder of a stale panel knows when to read it again."""
    return _refreshes_ended


def _age(path: Path) -> float:
    return time.time() - path.stat().st_mtime if path.exists() else float("inf")


def _refresh(path: Path, symbols: List[str], sectors: Optional[Dict[str, str]], period: str, suffix: str) -> None:
    """Replace the stale panel at `path`, unless a download is running already. One that fails leaves it serving."""
    global _refreshes_ended
    if not _downloading.acquire(blocking=False):
        return
    try:
        if _age(path) >= CACHE_TTL_S:      # a download that ended after the caller looked has replaced it already
            _download(path, symbols, sectors, period, suffix)
    except Exception:
        log.exception("refreshing the price panel %s failed, so its copy saved %.1f hours ago is still served. The next "
                      "read tries again; past %.0f hours old it is refused and the reader waits for a download",
                      path.name, _age(path) / 3600, STALE_LIMIT_S / 3600)
    finally:
        _downloading.release()
        _refreshes_ended += 1      # after the release, so a holder that reads again can start the next try


def _download(path: Path, symbols: List[str], sectors: Optional[Dict[str, str]], period: str, suffix: str) -> Panel:
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
    panel = _with_closes(Panel(close=merged["Close"], open=merged["Open"], high=merged["High"], low=merged["Low"],
                               volume=merged["Volume"], source="yahoo",
                               sectors={s: (sectors or {}).get(s, "Unknown") for s in merged["Close"].columns}))
    missing = sorted(set(symbols) - set(panel.symbols))
    if missing:
        log.warning("%d symbols had no data: %s", len(missing), missing[:20])
    # a name for each writer: the app and the prices job share this folder, each with its own lock, and as they run
    # in separate containers their process ids can be the same
    part = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    try:
        pd.to_pickle(panel, part)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)
    return panel


def _with_closes(p: Panel) -> Panel:
    """`p` without a share or a day that has no close at all."""
    # Yahoo sent 16 Nov 2026 with no closes, and ranked on that row no share of the Nifty 200 had a rank
    close = p.close.dropna(how="all", axis=1).dropna(how="all")
    keep, days = list(close.columns), close.index
    return Panel(close=close, open=p.open.loc[days, keep], high=p.high.loc[days, keep], low=p.low.loc[days, keep],
                 volume=p.volume.loc[days, keep], sectors={s: p.sectors[s] for s in keep}, source=p.source)


def universe_panel(index: str = "nifty500", period: str = "10y", refresh: bool = False,
                   serve_stale: bool = False) -> Panel:
    cons = index_constituents(index)
    return load_panel(list(cons["symbol"]), dict(zip(cons["symbol"], cons["sector"])), period,
                      refresh=refresh, serve_stale=serve_stale)


def panel_from_frames(frames: Dict[str, pd.DataFrame], sectors: Optional[Dict[str, str]] = None) -> Panel:
    """Build a panel from per-symbol OHLCV frames (tests, offline use)."""
    def stack(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: f[col] for s, f in frames.items()}).sort_index()
    return Panel(close=stack("close"), open=stack("open"), high=stack("high"), low=stack("low"),
                 volume=stack("volume"), sectors={s: (sectors or {}).get(s, "Unknown") for s in frames},
                 source="frames")
