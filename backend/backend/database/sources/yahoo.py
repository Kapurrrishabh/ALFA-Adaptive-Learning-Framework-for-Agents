"""Yahoo Finance provider (yfinance). Free tier: good NSE/BSE coverage via
.NS/.BO suffixes; fundamentals depth varies by company. yfinance is imported
lazily so the rest of the platform works offline without it installed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from backend.database.sources.provider import DataUnavailable, NewsItem, Provider, normalize_ohlcv


def _yf():
    try:
        import yfinance
        return yfinance
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailable("yfinance is not installed; `pip install yfinance` "
                              "or use LocalProvider") from exc


class YahooProvider(Provider):
    name = "yahoo"

    def ohlcv(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        df = _yf().Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            raise DataUnavailable(f"yahoo returned no OHLCV for {symbol}; check the "
                                  f"symbol (NSE symbols need the .NS suffix)")
        return normalize_ohlcv(df)

    def info(self, symbol: str) -> Dict[str, Any]:
        try:
            info = _yf().Ticker(symbol).info or {}
        except Exception as exc:
            raise DataUnavailable(f"yahoo info fetch failed for {symbol}: {exc}") from exc
        if not info or info.get("regularMarketPrice") is None and "longName" not in info:
            raise DataUnavailable(f"yahoo has no profile for {symbol}")
        return info

    def financials(self, symbol: str) -> Dict[str, pd.DataFrame]:
        t = _yf().Ticker(symbol)
        out: Dict[str, pd.DataFrame] = {}
        try:
            for key, df in (("income", t.income_stmt), ("balance", t.balance_sheet),
                            ("cashflow", t.cashflow),
                            ("income_quarterly", t.quarterly_income_stmt)):
                if df is not None and not df.empty:
                    out[key] = df
        except Exception as exc:
            raise DataUnavailable(f"yahoo statements fetch failed for {symbol}: {exc}") from exc
        if not out:
            raise DataUnavailable(f"yahoo has no financial statements for {symbol}")
        return out

    def intraday(self, symbol: str, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
        df = _yf().Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            raise DataUnavailable(f"yahoo returned no {interval} bars for {symbol} (market holiday or bad symbol)")
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return normalize_ohlcv(df)

    def upcoming_events(self, symbol: str) -> List[str]:
        try:
            cal = _yf().Ticker(symbol).calendar or {}
        except Exception as exc:
            raise DataUnavailable(f"yahoo calendar fetch failed for {symbol}: {exc}") from exc
        today = datetime.now(timezone.utc).date()
        out: List[str] = []
        for label, key in (("Earnings", "Earnings Date"), ("Ex-dividend", "Ex-Dividend Date")):
            value = cal.get(key)
            for d in (value if isinstance(value, list) else [value]):
                if isinstance(d, datetime):
                    d = d.date()
                if hasattr(d, "year") and d >= today:
                    out.append(f"{label} date {d} (source: yahoo calendar)")
        return out

    def news(self, symbol: str, limit: int = 25) -> List[NewsItem]:
        try:
            raw = _yf().Ticker(symbol).news or []
        except Exception as exc:
            raise DataUnavailable(f"yahoo news fetch failed for {symbol}: {exc}") from exc
        items: List[NewsItem] = []
        for r in raw[:limit]:
            content = r.get("content", r)  # yfinance >=0.2.5x nests under 'content'
            title = content.get("title") or ""
            if not title:
                continue
            ts = content.get("pubDate") or content.get("providerPublishTime")
            if isinstance(ts, (int, float)):
                published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
            else:
                published = str(ts or "")
            prov = content.get("provider") or {}
            source = prov.get("displayName") if isinstance(prov, dict) else str(prov)
            url = ""
            link = content.get("canonicalUrl") or content.get("clickThroughUrl") or content.get("link")
            if isinstance(link, dict):
                url = link.get("url", "")
            elif isinstance(link, str):
                url = link
            items.append(NewsItem(
                headline=title, summary=content.get("summary") or "",
                published_at=published, source=source or "yahoo", url=url, symbol=symbol,
                provenance=self.provenance(symbol, as_of=published or None)))
        if not items:
            raise DataUnavailable(f"yahoo returned no news for {symbol}")
        return items
