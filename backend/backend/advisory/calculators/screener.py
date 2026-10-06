"""Budget-aware research screener: "I have ₹50,000 — what should I research?"

Stage 1 applies hard constraints (affordability within the position limit,
liquidity, volatility ceiling, sector preferences); every exclusion keeps its
reason. Stage 2 scores survivors with the same engines and the same fusion
used for single-stock analysis — there is no second scoring system. Price
level never enters the ranking, so cheap shares get no preference.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from backend.advisory.fundamental import fundamentals
from backend.advisory.statistics import quant, technical
from backend.config import RISK_PROFILES, resolve_symbol
from backend.database.sources.provider import DataUnavailable
from backend.database.sources.quality import DataQualityError, check_ohlcv
from backend.advisory.fusion.evidence import DomainResult, unavailable, utcnow_iso
from backend.advisory.fusion.fusion import fuse
from backend.user.portfolio.portfolio import Position
from backend.advisory.service import AnalysisService

UNIVERSE_FILE = Path(__file__).parent / "resources" / "universe_india.csv"


def load_universe(path: Path = UNIVERSE_FILE) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def _median_traded_value(df: pd.DataFrame) -> float:
    return float((df["close"] * df["volume"]).iloc[-20:].median())


def score_symbol(service: AnalysisService, symbol: str, df: pd.DataFrame,
                 with_fundamentals: bool) -> Dict[str, DomainResult]:
    """Technical, risk and (optionally) fundamental results for one stock."""
    results: Dict[str, DomainResult] = {
        "technical": technical.analyze(df),
        "risk": quant.analyze(df, service.benchmark(), service.market.risk_free_rate),
        # Declared, not omitted, so confidence reflects the evidence a screen skips.
        "news_sentiment": unavailable("news_sentiment", "not fetched during screening; run a full analysis"),
    }
    if with_fundamentals:
        sym = resolve_symbol(symbol, service.market.code)
        try:
            results["fundamental"] = fundamentals.analyze(
                sym, service.info(symbol), service.financials(symbol), source=service.provider.name)
        except DataUnavailable as e:
            results["fundamental"] = unavailable("fundamental", str(e))
    return results


def screen(service: AnalysisService, budget: float, risk_profile: str = "moderate",
           horizon: str = "medium", positions: Optional[Dict[str, Position]] = None,
           include_sectors: Optional[Iterable[str]] = None,
           exclude_sectors: Optional[Iterable[str]] = None,
           universe: Optional[pd.DataFrame] = None, max_candidates: int = 8,
           with_fundamentals: bool = True) -> Dict[str, object]:
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if risk_profile not in RISK_PROFILES:
        raise ValueError(f"risk_profile must be one of {list(RISK_PROFILES)}, got {risk_profile!r}")
    profile = RISK_PROFILES[risk_profile]
    uni = universe if universe is not None else load_universe()
    inc = {s.lower() for s in include_sectors or []}
    exc = {s.lower() for s in exclude_sectors or []}
    held = {s.split(".")[0].upper(): p for s, p in (positions or {}).items() if p.quantity > 0}
    held_sectors: Dict[str, float] = {}
    for p in held.values():
        if p.sector:
            held_sectors[p.sector] = held_sectors.get(p.sector, 0.0) + p.cost_basis
    held_total = sum(held_sectors.values())
    position_limit = budget * profile.max_position_weight

    excluded: List[Dict[str, str]] = []
    survivors: List[Dict[str, object]] = []

    def stage1(row) -> Optional[Dict[str, object]]:
        sym, sector = row.symbol, row.sector
        if inc and sector.lower() not in inc:
            return {"symbol": sym, "reason": f"sector filter: {sector} not in requested sectors"}
        if sector.lower() in exc:
            return {"symbol": sym, "reason": f"sector filter: {sector} excluded by you"}
        try:
            df = service.history(sym)
            check_ohlcv(df, min_rows=250, now=service.now())
        except (DataUnavailable, DataQualityError) as e:
            return {"symbol": sym, "reason": f"data unavailable: {e}"}
        price = float(df["close"].iloc[-1])
        if price > position_limit:
            return {"symbol": sym, "reason": f"affordability: one share ({price:,.0f}) exceeds the "
                                             f"{profile.max_position_weight:.0%} position limit ({position_limit:,.0f})"}
        mtv = _median_traded_value(df)
        if mtv < profile.min_median_traded_value:
            return {"symbol": sym, "reason": f"liquidity: median traded value {mtv:,.0f}/day below "
                                             f"{profile.min_median_traded_value:,.0f}"}
        vol = quant.ann_vol(df["close"].pct_change().dropna().iloc[-252:])
        if vol > profile.max_holding_vol:
            return {"symbol": sym, "reason": f"volatility: {vol:.0%} above the {profile.name} "
                                             f"ceiling of {profile.max_holding_vol:.0%}"}
        return {"symbol": sym, "sector": sector, "df": df, "price": price, "mtv": mtv, "vol": vol}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for res in pool.map(stage1, uni.itertuples(index=False)):
            (survivors if "df" in res else excluded).append(res)

    def stage2(c: Dict[str, object]) -> Dict[str, object]:
        results = score_symbol(service, c["symbol"], c["df"], with_fundamentals)
        fund = results.get("fundamental")
        if fund is not None and fund.available and fund.details["sector"] != "Data unavailable":
            c["sector"] = fund.details["sector"]
        d = fuse(results, held=c["symbol"].upper() in held, horizon=horizon)
        reasons = [f"affordable: one share is {c['price'] / budget:.1%} of the budget",
                   f"liquid: median traded value {c['mtv'] / 1e7:,.0f} crore/day",
                   f"volatility {c['vol']:.0%} within the {profile.max_holding_vol:.0%} ceiling",
                   f"technical {results['technical'].state}, risk {results['risk'].state}"
                   + (f", fundamentals {results['fundamental'].state}" if "fundamental" in results else "")]
        top = d.supporting_evidence[:2]
        reasons += [f"evidence: {e['claim']}" for e in top]
        overlap_note = None
        if c["symbol"].upper() in held:
            overlap_note = "already held — adding increases concentration"
        elif held_total and held_sectors.get(c["sector"], 0) / held_total > profile.max_sector_weight:
            overlap_note = f"your portfolio is already over the {profile.max_sector_weight:.0%} limit in {c['sector']}"
        if overlap_note:
            reasons.append(f"portfolio overlap: {overlap_note}")
        penalty = 0.15 if overlap_note else 0.0
        return {"symbol": c["symbol"], "sector": c["sector"], "price": round(c["price"], 2),
                "label": d.decision_support_label, "score": d.score, "confidence": d.confidence,
                "rank_score": d.score * (0.5 + 0.5 * d.confidence) - penalty,
                "max_shares": int(math.floor(position_limit / c["price"])),
                "position_limit_pct": round(profile.max_position_weight * 100, 1),
                "reasons": reasons, "uncertainties": d.key_uncertainties[:3]}

    with ThreadPoolExecutor(max_workers=6) as pool:
        scored = list(pool.map(stage2, survivors))
    ranked = sorted(scored, key=lambda s: s["rank_score"], reverse=True)
    candidates = [s for s in ranked if s["label"] not in ("AVOID", "SELL", "INSUFFICIENT_EVIDENCE")]
    return {"timestamp": utcnow_iso(), "budget": budget, "risk_profile": risk_profile,
            "horizon": horizon, "universe_size": len(uni),
            "candidates": candidates[:max_candidates], "also_scored": ranked[max_candidates:],
            "excluded": excluded,
            "assumptions": [
                "Universe is a curated large-cap list; it excludes delisted names (survivorship bias) "
                "and small caps.",
                f"Position limit {profile.max_position_weight:.0%} of the budget per stock "
                f"({profile.name} profile); whole shares only.",
                "Ranking uses the fused evidence score weighted by confidence; share price is not a factor.",
                "News is not fetched during screening (speed); run a full analysis on any candidate.",
            ]}


def _fund(r: Dict[str, DomainResult]) -> Optional[DomainResult]:
    f = r.get("fundamental")
    return f if f is not None and f.available else None


def _metric_value(r: Dict[str, DomainResult], key: str) -> Optional[float]:
    f = _fund(r)
    return None if f is None else f.details["metrics"].get(key, {}).get("value")


# name -> (trigger phrases, predicate over score_symbol results, plain description)
CRITERIA = {
    "fundamentally_strong": (("fundamentally strong", "strong fundamentals", "quality compan"),
                             lambda r: _fund(r) is not None and _fund(r).state in ("strong", "sound"),
                             "fundamental state strong or sound"),
    "improving_earnings": (("improving earnings", "earnings growth", "growing earnings", "earnings improving"),
                           lambda r: (_metric_value(r, "eps_growth") or _metric_value(r, "net_income_growth") or -1) > 0,
                           "EPS or net income grew year on year"),
    "fundamentals_improving": (("fundamentals are improving", "improving fundamentals", "fundamentals improving"),
                               lambda r: _fund(r) is not None and _fund(r).details["trend"] == "improving",
                               "fundamental trend improving"),
    "positive_momentum": (("positive momentum", "strong momentum", "uptrend", "momentum"),
                          lambda r: r["technical"].available and r["technical"].details.get("return_20d_pct", 0) > 0
                          and r["technical"].state != "bearish",
                          "positive 20-day return and technical state not bearish"),
    "high_volume": (("unusually high volume", "high volume", "volume spike"),
                    lambda r: r["technical"].available and (r["technical"].details.get("volume_vs_20d_avg") or 0) >= 2,
                    "volume at least 2x its 20-day average"),
    "breakout": (("breakout", "breaking out"),
                 lambda r: r["technical"].available and r["technical"].details.get("breakout_status") == "above 20d high",
                 "close above the prior 20-day high"),
    "low_volatility": (("low volatility", "low risk", "less volatile", "stable stock"),
                       lambda r: r["risk"].available and r["risk"].details["annualized_vol_pct"] < 25,
                       "annualized volatility below 25%"),
    "price_not_recovered": (("not yet recovered", "price has not recovered", "price lagging", "beaten down"),
                            lambda r: r["risk"].available and r["risk"].details["current_drawdown_pct"] <= -15,
                            "price at least 15% below its peak in the loaded history"),
}
UNSUPPORTED = {
    "historical valuation": "valuation-range screens need point-in-time valuation history, which the "
                            "current free data source does not provide (Data unavailable)",
    "sentiment": "news sentiment is not scored during universe scans; ask about specific stocks instead",
}


def parse_criteria(text: str) -> Dict[str, List[str]]:
    t = text.lower()
    found = [name for name, (phrases, _, _) in CRITERIA.items() if any(p in t for p in phrases)]
    if "fundamentals_improving" in found and "fundamentally_strong" in found and "fundamentally strong" not in t:
        found.remove("fundamentally_strong")
    unsupported = [msg for key, msg in UNSUPPORTED.items() if key in t]
    return {"criteria": found, "unsupported": unsupported}


def research_query(service: AnalysisService, text: str, universe: Optional[pd.DataFrame] = None,
                   limit: int = 10) -> Dict[str, object]:
    parsed = parse_criteria(text)
    if not parsed["criteria"]:
        return {"timestamp": utcnow_iso(), "matches": [], **parsed,
                "error": "no supported screening criteria recognised in the question"}
    uni = universe if universe is not None else load_universe()
    needs_fund = any(c in parsed["criteria"] for c in
                     ("fundamentally_strong", "improving_earnings", "fundamentals_improving"))

    def evaluate(symbol: str) -> Optional[Dict[str, object]]:
        try:
            df = service.history(symbol)
            check_ohlcv(df, min_rows=250, now=service.now())
        except (DataUnavailable, DataQualityError):
            return None
        r = score_symbol(service, symbol, df, needs_fund)
        if not all(CRITERIA[c][1](r) for c in parsed["criteria"]):
            return None
        d = fuse(r)
        return {"symbol": symbol, "label": d.decision_support_label, "score": d.score,
                "technical": r["technical"].state, "risk": r["risk"].state,
                "fundamental": r["fundamental"].state if "fundamental" in r else "not evaluated",
                "why": [CRITERIA[c][2] for c in parsed["criteria"]]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        matches = [m for m in pool.map(evaluate, uni["symbol"]) if m]
    matches.sort(key=lambda m: m["score"], reverse=True)
    return {"timestamp": utcnow_iso(), "universe_size": len(uni), "matches": matches[:limit],
            "criteria_applied": {c: CRITERIA[c][2] for c in parsed["criteria"]}, **parsed}
