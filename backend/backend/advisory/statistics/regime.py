"""Market / stock regime detection.

Three independent, testable lenses rather than one opaque label:
  trend       close vs a rising/falling 200-day average
  volatility  current realized vol percentile in its own history
  dependence  Lo-MacKinlay variance ratio: trending vs mean-reverting
plus a 2-state Gaussian HMM on returns whose *filtered* (forward-only)
probability of the high-volatility state is reported — filtering never uses
bars after t, so the same code is safe inside backtests.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from backend.config import THRESHOLDS
from backend.advisory.fusion.evidence import DomainResult, Evidence, Provenance, unavailable
from backend.advisory.statistics import indicators as ind
from backend.advisory.statistics.stats import variance_ratio

DOMAIN = "regime"


def trend_regime(close: pd.Series) -> str:
    if len(close) < 220:
        return "undetermined"
    sma200 = ind.sma(close, 200)
    rising = sma200.iloc[-1] > sma200.iloc[-21]
    above = close.iloc[-1] > sma200.iloc[-1]
    if above and rising:
        return "bullish"
    if not above and not rising:
        return "bearish"
    return "sideways/transition"


def vol_regime(close: pd.Series) -> Tuple[str, Optional[float]]:
    hv = ind.historical_volatility(close).dropna()
    if len(hv) < 120:
        return "undetermined", None
    pct = float((hv < hv.iloc[-1]).mean() * 100)
    if pct >= THRESHOLDS.high_vol_percentile:
        return "high volatility", pct
    if pct <= THRESHOLDS.low_vol_percentile:
        return "low volatility", pct
    return "normal volatility", pct


def dependence_regime(close: pd.Series, q: int = 10) -> Dict[str, object]:
    lr = np.log(close).diff().dropna().iloc[-252:].to_numpy()
    vr, z = variance_ratio(lr, q)
    if np.isnan(vr):
        return {"state": "undetermined"}
    state = ("trending" if z > 1.96 else "mean-reverting" if z < -1.96
             else "no significant serial dependence")
    return {"state": state, "variance_ratio": round(vr, 3), "z": round(z, 2), "q": q}


def fit_hmm(x: np.ndarray, n_iter: int = 60, tol: float = 1e-6) -> Dict[str, np.ndarray]:
    """Two-state Gaussian HMM via Baum-Welch with scaling. State 1 = high vol."""
    x = np.asarray(x, float)
    T = len(x)
    cut = np.quantile(np.abs(x), 0.7)
    lo, hi = x[np.abs(x) <= cut], x[np.abs(x) > cut]
    mu = np.array([lo.mean(), hi.mean()])
    sd = np.array([max(lo.std(), 1e-6), max(hi.std(), 1e-6)])
    A_ = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    prev_ll = -np.inf
    for _ in range(n_iter):
        B = np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        B = np.maximum(B, 1e-300)
        alpha = np.empty((T, 2))
        c = np.empty(T)
        alpha[0] = pi * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A_) * B[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        beta = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            beta[t] = (A_ @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * A_[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        pi = gamma[0]
        A_ = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        A_ /= A_.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mu = (gamma * x[:, None]).sum(axis=0) / w
        sd = np.sqrt((gamma * (x[:, None] - mu) ** 2).sum(axis=0) / w)
        sd = np.maximum(sd, 1e-6)
        ll = float(np.log(c).sum())
        if abs(ll - prev_ll) < tol * abs(ll):
            break
        prev_ll = ll
    if sd[0] > sd[1]:
        order = [1, 0]
        mu, sd, pi, A_, alpha = mu[order], sd[order], pi[order], A_[np.ix_(order, order)], alpha[:, order]
    return {"mu": mu, "sd": sd, "A": A_, "pi": pi, "filtered": alpha, "loglik": ll}


def analyze(df: pd.DataFrame, bench: Optional[pd.DataFrame] = None) -> DomainResult:
    if df is None or len(df) < 120:
        return unavailable(DOMAIN, "need at least 120 bars for regime detection")
    close = df["close"]
    as_of = str(df.index[-1].date())
    details: Dict[str, object] = {"trend": trend_regime(close)}
    details["volatility"], details["vol_percentile"] = vol_regime(close)
    details["dependence"] = dependence_regime(close)
    r = close.pct_change().dropna().to_numpy()
    if len(r) >= 250:
        hmm = fit_hmm(r)
        p_high = float(hmm["filtered"][-1, 1])
        details["hmm"] = {"p_high_vol_state": round(p_high, 3),
                          "state_ann_vol_pct": [round(float(s) * math.sqrt(252) * 100, 1)
                                                for s in hmm["sd"]],
                          "persistence": [round(float(hmm["A"][i, i]), 3) for i in range(2)]}
    ev = []
    prov = Provenance(source="computed:regime", as_of=as_of)
    trend = details["trend"]
    if trend in ("bullish", "bearish"):
        ev.append(Evidence(domain=DOMAIN, claim=f"Stock trend regime is {trend} "
                           "(price vs a " + ("rising" if trend == "bullish" else "falling")
                           + " 200-day average).", direction=1 if trend == "bullish" else -1,
                           strength=0.5, provenance=prov))
    if details["volatility"] == "high volatility":
        ev.append(Evidence(domain=DOMAIN, claim=f"Stock is in a high-volatility regime "
                           f"(percentile {details['vol_percentile']:.0f}).", direction=-1,
                           strength=0.4, provenance=prov))
    market = None
    if bench is not None and len(bench) >= 220:
        bt, (bv, bp) = trend_regime(bench["close"]), vol_regime(bench["close"])
        risk_on = bt == "bullish" and bv != "high volatility"
        risk_off = bt == "bearish" or bv == "high volatility"
        market = {"trend": bt, "volatility": bv, "vol_percentile": bp,
                  "state": "risk-on" if risk_on else "risk-off" if risk_off else "neutral"}
        if market["state"] != "neutral":
            ev.append(Evidence(domain=DOMAIN, claim=f"Benchmark regime is {market['state']} "
                               f"({bt} trend, {bv}).", direction=1 if risk_on else -1,
                               strength=0.4, provenance=Provenance(source="computed:regime:benchmark",
                                                                   as_of=as_of)))
    details["market"] = market or "Data unavailable"
    score = sum(e.direction * e.strength for e in ev) / max(1.0, len(ev) * 0.5) if ev else 0.0
    state = f"{trend}, {details['volatility']}"
    return DomainResult(domain=DOMAIN, state=state, score=max(-1.0, min(1.0, score)),
                        evidence=ev, details=details, as_of=as_of,
                        confidence=1.0 if ev else 0.0)
