"""Offline provider backed by CSV/JSON files on disk.

Used by tests and for cached research data. Layout under root_dir:
    <SYMBOL>.csv                 OHLCV with a date index column
    <SYMBOL>.info.json           profile fields
    <SYMBOL>.income.csv / .balance.csv / .cashflow.csv
    <SYMBOL>.news.json           list of news dicts
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.database.sources.provider import DataUnavailable, NewsItem, Provider, normalize_ohlcv


class LocalProvider(Provider):
    name = "local"

    def __init__(self, root_dir: str, frames: Optional[Dict[str, pd.DataFrame]] = None):
        self.root = Path(root_dir)
        # In-memory frames let tests inject synthetic data without touching disk.
        self.frames = dict(frames or {})
        self.infos: Dict[str, Dict[str, Any]] = {}
        self.news_items: Dict[str, List[NewsItem]] = {}
        self.statements: Dict[str, Dict[str, pd.DataFrame]] = {}

    def _key(self, symbol: str) -> str:
        return symbol.upper()

    def ohlcv(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        key = self._key(symbol)
        if key in self.frames:
            return normalize_ohlcv(self.frames[key])
        path = self.root / f"{key}.csv"
        if not path.exists():
            raise DataUnavailable(f"no local OHLCV for {symbol} at {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return normalize_ohlcv(df)

    def info(self, symbol: str) -> Dict[str, Any]:
        key = self._key(symbol)
        if key in self.infos:
            return self.infos[key]
        path = self.root / f"{key}.info.json"
        if not path.exists():
            raise DataUnavailable(f"no local info for {symbol} at {path}")
        return json.loads(path.read_text())

    def financials(self, symbol: str) -> Dict[str, pd.DataFrame]:
        key = self._key(symbol)
        if key in self.statements:
            return self.statements[key]
        out: Dict[str, pd.DataFrame] = {}
        for kind in ("income", "balance", "cashflow"):
            path = self.root / f"{key}.{kind}.csv"
            if path.exists():
                out[kind] = pd.read_csv(path, index_col=0)
        if not out:
            raise DataUnavailable(f"no local financial statements for {symbol}")
        return out

    def news(self, symbol: str, limit: int = 25) -> List[NewsItem]:
        key = self._key(symbol)
        if key in self.news_items:
            return self.news_items[key][:limit]
        path = self.root / f"{key}.news.json"
        if not path.exists():
            raise DataUnavailable(f"no local news for {symbol}")
        raw = json.loads(path.read_text())
        return [NewsItem(headline=r["headline"], summary=r.get("summary", ""),
                         published_at=r["published_at"], source=r.get("source", "local"),
                         url=r.get("url", ""), symbol=symbol,
                         provenance=self.provenance(symbol, as_of=r["published_at"]))
                for r in raw[:limit]]
