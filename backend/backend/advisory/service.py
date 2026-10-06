"""Application service: fetch -> validate -> analyze -> fuse.

This is the single entry point the orchestrator, API, CLI and alerts use, so
every surface gets identical evidence. Network fetches run in parallel with a
timeout; a fetch that fails or times out marks its domain unavailable with
the reason (visible in the decision's key uncertainties) rather than
stopping the whole analysis or being papered over.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from backend.advisory.statistics import analogues, candlesticks, forecast, patterns, quant, regime, technical
from backend.advisory.fundamental import fundamentals
from backend.advisory.sentiment import news
from backend.config import MARKETS, resolve_symbol
from backend.database.sources.provider import DataUnavailable, NewsItem, Provider
from backend.database.sources.quality import DataQualityError, check_ohlcv
from backend.database.sources import rss
from backend.advisory.fusion.evidence import DomainResult, unavailable, utcnow_iso
from backend.advisory.fusion.fusion import Decision, fuse

log = logging.getLogger("stockintel.service")
FETCH_TIMEOUT_S = 25.0
CACHE_TTL_S = 900


@dataclass
class StockAnalysis:
    symbol: str
    provider_symbol: str
    market: str
    name: str
    currency: str
    price: Optional[float]
    analysis_timestamp: str
    quality: Dict[str, Any]
    results: Dict[str, DomainResult]
    decision: Decision
    held: bool = False
    upcoming_events: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "provider_symbol": self.provider_symbol,
                "market": self.market, "name": self.name, "currency": self.currency,
                "price": self.price, "analysis_timestamp": self.analysis_timestamp, "held": self.held,
                "quality": self.quality, "upcoming_events": self.upcoming_events,
                "results": {k: v.to_dict() for k, v in self.results.items()},
                "decision": self.decision.to_dict(), "timings_ms": self.timings_ms}


def company_terms(symbol: str, name: str) -> List[str]:
    base = symbol.split(".")[0]
    clean = re.sub(r"\b(limited|ltd\.?|inc\.?|corporation|corp\.?|plc)\b", "", name,
                   flags=re.I).strip(" ,.")
    terms = [base, clean]
    first = clean.split()[0] if clean else ""
    if len(first) > 3:
        terms.append(first)
    return [t for t in terms if t]


class AnalysisService:
    def __init__(self, provider: Provider, market: str = "NSE", use_rss_news: bool = True,
                 history_period: str = "5y", now: Optional[Callable[[], datetime]] = None,
                 forecast_models: Sequence[str] = ("climatology", "logistic"),
                 pattern_priors: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None):
        if market not in MARKETS:
            raise ValueError(f"unknown market {market!r}; configured: {list(MARKETS)}")
        self.provider = provider
        self.market = MARKETS[market]
        self.use_rss_news = use_rss_news
        self.history_period = history_period
        self.now = now or datetime.now
        self.forecast_models = tuple(forecast_models)
        self.pattern_priors = pattern_priors or {}
        self._cache: Dict[tuple, tuple] = {}
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="fetch")

    # --- data access with a short TTL cache ----------------------------------
    def _cached(self, key: tuple, fn: Callable[[], Any]) -> Any:
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL_S:
            return hit[1]
        value = fn()
        self._cache[key] = (time.time(), value)
        return value

    def history(self, symbol: str) -> pd.DataFrame:
        sym = resolve_symbol(symbol, self.market.code)
        return self._cached(("ohlcv", sym), lambda: self.provider.ohlcv(sym, self.history_period))

    def benchmark(self) -> Optional[pd.DataFrame]:
        try:
            return self._cached(("ohlcv", self.market.benchmark),
                                lambda: self.provider.ohlcv(self.market.benchmark, self.history_period))
        except DataUnavailable as exc:
            log.warning("benchmark unavailable: %s", exc)
            return None

    def info(self, symbol: str) -> Dict[str, Any]:
        sym = resolve_symbol(symbol, self.market.code)
        return self._cached(("info", sym), lambda: self.provider.info(sym))

    def financials(self, symbol: str) -> Dict[str, pd.DataFrame]:
        sym = resolve_symbol(symbol, self.market.code)
        return self._cached(("stmt", sym), lambda: self.provider.financials(sym))

    def _submit(self, fn: Callable[[], Any]):
        return self._pool.submit(fn)

    def _result(self, fut, what: str) -> Any:
        try:
            return fut.result(timeout=FETCH_TIMEOUT_S)
        except FutureTimeout:
            raise DataUnavailable(f"{what} timed out after {FETCH_TIMEOUT_S:.0f}s") from None

    def _news_items(self, sym: str, name: str) -> List[NewsItem]:
        items: List[NewsItem] = []
        errors: List[str] = []
        try:
            items.extend(self.provider.news(sym))
        except DataUnavailable as exc:
            errors.append(str(exc))
        if self.use_rss_news and name:
            terms = company_terms(sym, name)
            query = f"\"{terms[1]}\" share" if len(terms) > 1 else f"{terms[0]} share"
            try:
                items.extend(rss.google_news(query, sym))
            except DataUnavailable as exc:
                errors.append(str(exc))
        if not items:
            raise DataUnavailable("; ".join(errors) or "no news sources returned articles")
        return items

    # --- full analysis --------------------------------------------------------
    def analyze(self, symbol: str, held: bool = False, horizon: str = "medium") -> StockAnalysis:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        sym = resolve_symbol(symbol, self.market.code)
        df = self.history(symbol)
        report = check_ohlcv(df, now=self.now())
        timings["ohlcv"] = (time.perf_counter() - t0) * 1000

        f_info = self._submit(lambda: self.info(symbol))
        f_stmt = self._submit(lambda: self.financials(symbol))
        f_bench = self._submit(self.benchmark)
        f_events = self._submit(lambda: self.provider.upcoming_events(sym))

        results: Dict[str, DomainResult] = {}

        def run(domain: str, fn: Callable[[], DomainResult]) -> None:
            t = time.perf_counter()
            try:
                results[domain] = fn()
            except (DataUnavailable, DataQualityError) as exc:
                results[domain] = unavailable(domain, str(exc))
            timings[domain] = round((time.perf_counter() - t) * 1000, 1)

        run("technical", lambda: technical.analyze(df))
        tech = results["technical"]
        run("candlestick", lambda: candlesticks.analyze(
            df, tech.details.get("support_levels", []), tech.details.get("resistance_levels", []),
            self.pattern_priors.get("candlestick")))
        run("pattern", lambda: patterns.analyze(df, self.pattern_priors.get("chart")))

        try:
            info = self._result(f_info, "company profile")
        except DataUnavailable:
            info = {}
        name = info.get("longName") or info.get("shortName") or sym
        f_news = self._submit(lambda: self._news_items(sym, name))
        try:
            bench = self._result(f_bench, "benchmark history")
        except DataUnavailable:
            bench = None

        run("risk", lambda: quant.analyze(df, bench, self.market.risk_free_rate))
        run("regime", lambda: regime.analyze(df, bench))
        run("forecast", lambda: forecast.analyze(df, bench, models=self.forecast_models))
        run("historical", lambda: analogues.analyze(df))

        def fund() -> DomainResult:
            try:
                statements, stmt_error = self._result(f_stmt, "financial statements"), None
            except DataUnavailable as exc:
                if not info:
                    raise
                statements, stmt_error = {}, str(exc)
            out = fundamentals.analyze(sym, info, statements, source=self.provider.name)
            if stmt_error and not out.available:
                return unavailable("fundamental", stmt_error)
            if stmt_error:
                out.details["statement_error"] = stmt_error
            return out
        run("fundamental", fund)
        run("news_sentiment", lambda: news.analyze(sym, company_terms(sym, name),
                                                   self._result(f_news, "news")))
        try:
            events = self._result(f_events, "event calendar")
        except DataUnavailable:
            events = []

        decision = fuse(results, held=held, horizon=horizon, stale=report.is_stale,
                        upcoming_events=events)
        for w in report.warnings:
            decision.key_uncertainties.append(f"Data quality: {w}")
        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return StockAnalysis(
            symbol=symbol.upper(), provider_symbol=sym, market=self.market.code, name=name,
            currency=info.get("currency") or self.market.currency,
            price=float(df["close"].iloc[-1]), analysis_timestamp=utcnow_iso(),
            quality={"rows": report.rows, "first_date": report.first_date,
                     "last_date": report.last_date, "is_stale": report.is_stale,
                     "warnings": report.warnings, "source": self.provider.name},
            results=results, decision=decision, held=held, upcoming_events=events,
            timings_ms=timings)
