"""Quantitative / statistical risk engine. Deterministic formulas only.

Conventions: simple daily returns, 252-day annualization, historical (not
parametric) VaR/CVaR at 95%, Jensen alpha annualized. The risk-domain score
is positive for a benign profile and negative for elevated risk; it is a
summary of facts, not a forecast.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from backend.config import ANNUALIZATION_DAYS as A
from backend.advisory.fusion.evidence import DomainResult, Evidence, Provenance, clamp, unavailable
from backend.advisory.statistics import indicators as ind
from backend.advisory.statistics.stats import variance_ratio

DOMAIN = "risk"


def cagr(close: pd.Series) -> float:
    years = len(close) / A
    return float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * math.sqrt(A))


def downside_deviation(r: pd.Series, mar: float = 0.0) -> float:
    d = np.minimum(r - mar, 0.0)
    return float(math.sqrt((d ** 2).mean()) * math.sqrt(A))


def max_drawdown(close: pd.Series) -> Dict[str, object]:
    peak = close.cummax()
    dd = close / peak - 1.0
    trough = dd.idxmin()
    peak_date = close.loc[:trough].idxmax()
    after = close.loc[trough:]
    recovered = after[after >= close.loc[peak_date]]
    return {"max_drawdown_pct": round(float(dd.min()) * 100, 2),
            "peak_date": str(peak_date.date()), "trough_date": str(trough.date()),
            "recovery_date": str(recovered.index[0].date()) if not recovered.empty else None,
            "current_drawdown_pct": round(float(dd.iloc[-1]) * 100, 2)}


def sharpe(r: pd.Series, rf: float) -> float:
    ex = r - rf / A
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(A)) if sd > 1e-12 else float("nan")


def sortino(r: pd.Series, rf: float) -> float:
    dd = downside_deviation(r, rf / A)
    return float((r.mean() * A - rf) / dd) if dd > 1e-12 else float("nan")


def calmar(close: pd.Series) -> float:
    mdd = abs(max_drawdown(close)["max_drawdown_pct"]) / 100
    return float(cagr(close) / mdd) if mdd > 0 else float("nan")


def var_cvar(r: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    q = float(np.quantile(r.dropna(), 1 - level))
    tail = r[r <= q]
    return q, float(tail.mean()) if len(tail) else q


def beta_alpha(r: pd.Series, b: pd.Series, rf: float) -> Dict[str, float]:
    joined = pd.concat([r, b], axis=1, join="inner").dropna()
    if len(joined) < 60:
        return {}
    x, y = joined.iloc[:, 1], joined.iloc[:, 0]
    var_b = x.var(ddof=1)
    beta = float(joined.cov().iloc[0, 1] / var_b) if var_b > 0 else float("nan")
    rf_d = rf / A
    alpha_d = (y.mean() - rf_d) - beta * (x.mean() - rf_d)
    corr = float(y.corr(x))
    return {"beta": round(beta, 3), "alpha_annual_pct": round(alpha_d * A * 100, 2),
            "correlation": round(corr, 3), "r_squared": round(corr ** 2, 3),
            "observations": int(len(joined))}


def period_returns(close: pd.Series) -> Dict[str, Optional[float]]:
    def back(n: int) -> Optional[float]:
        return round(float(close.iloc[-1] / close.iloc[-1 - n] - 1) * 100, 2) if len(close) > n else None
    return {"1d_pct": back(1), "1w_pct": back(5), "1m_pct": back(21), "3m_pct": back(63),
            "6m_pct": back(126), "1y_pct": back(252)}


def analyze(df: pd.DataFrame, bench: Optional[pd.DataFrame], rf: float) -> DomainResult:
    if df is None or len(df) < 120:
        return unavailable(DOMAIN, "need at least 120 bars for risk statistics")
    close = df["close"]
    r = close.pct_change().dropna()
    window = r.iloc[-A:]
    as_of = str(df.index[-1].date())
    span = f"{window.index[0].date()} to {window.index[-1].date()}"
    prov = lambda name: Provenance(source=f"computed:{name}", as_of=as_of, period=span)

    vol = ann_vol(window)
    dd = max_drawdown(close)
    var95, cvar95 = var_cvar(window)
    lr = np.log(close).diff().dropna()
    vr, vr_z = variance_ratio(lr.iloc[-A:].to_numpy(), 5)
    stats: Dict[str, object] = {
        "period": span, "returns": period_returns(close),
        "cagr_pct": round(cagr(close) * 100, 2), "annualized_vol_pct": round(vol * 100, 2),
        "downside_deviation_pct": round(downside_deviation(window) * 100, 2),
        "sharpe": round(sharpe(window, rf), 2), "sortino": round(sortino(window, rf), 2),
        "calmar": round(calmar(close), 2), "var_95_1d_pct": round(var95 * 100, 2),
        "cvar_95_1d_pct": round(cvar95 * 100, 2), **dd,
        "skew": round(float(window.skew()), 2), "excess_kurtosis": round(float(window.kurt()), 2),
        "autocorr_lag1": round(float(window.autocorr(1)), 3),
        "vol_clustering_acf_abs": round(float(window.abs().autocorr(1)), 3),
        "variance_ratio_5": None if np.isnan(vr) else round(vr, 3),
        "variance_ratio_z": None if np.isnan(vr_z) else round(vr_z, 2),
        "risk_free_rate_used": rf,
    }
    ev = []

    def add(claim: str, score: float, name: str, value: float):
        ev.append(Evidence(domain=DOMAIN, claim=claim, direction=int(np.sign(score)),
                           strength=abs(score), provenance=prov(name), value=value))

    vol_score = 0.4 if vol < 0.20 else 0.1 if vol < 0.30 else -0.3 if vol < 0.45 else -0.7
    add(f"Annualized volatility is {vol * 100:.1f}% over the last year.", vol_score,
        "annualized_vol", vol * 100)
    mdd = dd["max_drawdown_pct"]
    add(f"Maximum drawdown in the loaded history is {mdd:.1f}% "
        f"({dd['peak_date']} to {dd['trough_date']}).",
        0.2 if mdd > -20 else -0.1 if mdd > -35 else -0.4, "max_drawdown", mdd)
    cur = dd["current_drawdown_pct"]
    if cur < -20:
        add(f"Price is {abs(cur):.1f}% below its peak in the loaded history.", -0.3,
            "current_drawdown", cur)
    add(f"Historical 1-day 95% CVaR is {cvar95 * 100:.2f}% (average loss on the worst 5% of days).",
        -0.3 if cvar95 < -0.04 else 0.0 if cvar95 < -0.025 else 0.2, "cvar_95", cvar95 * 100)

    if bench is not None and len(bench) > 60:
        br = bench["close"].pct_change().dropna()
        ba = beta_alpha(window, br.iloc[-A:], rf)
        stats["benchmark"] = ba
        if ba:
            beta = ba["beta"]
            add(f"Beta to the benchmark is {beta:.2f} (correlation {ba['correlation']:.2f}).",
                -0.3 if beta > 1.3 else 0.2 if beta < 0.8 else 0.0, "beta", beta)
            roll = r.rolling(60).corr(br).dropna()
            if len(roll) >= 63:
                stats["rolling_corr_60d_now"] = round(float(roll.iloc[-1]), 3)
                stats["rolling_corr_60d_3m_ago"] = round(float(roll.iloc[-63]), 3)
    else:
        stats["benchmark"] = "Data unavailable"

    hv = ind.historical_volatility(close).dropna()
    if len(hv) > 120:
        pct = float((hv < hv.iloc[-1]).mean() * 100)
        stats["vol_percentile_now"] = round(pct, 0)
        if pct > 80:
            add(f"Current 20-day volatility is at percentile {pct:.0f} of its history.", -0.3,
                "vol_percentile", pct)

    score = clamp(sum(e.direction * e.strength for e in ev) / max(1.0, len(ev) * 0.5))
    state = ("low" if score > 0.25 else "moderate" if score > -0.1 else
             "elevated" if score > -0.4 else "high")
    return DomainResult(domain=DOMAIN, state=state, score=score, evidence=ev,
                        details=stats, as_of=as_of)
