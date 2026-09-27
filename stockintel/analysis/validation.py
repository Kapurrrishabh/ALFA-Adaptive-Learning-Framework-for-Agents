"""Historical validation of signals: does a pattern actually precede returns
that differ from the stock's unconditional behaviour?

A signal observed at the close of day t is scored on close[t+h]/close[t]-1,
so the outcome is strictly after the information. Events closer together than
the horizon are de-clustered (only the first counts) so overlapping windows do
not inflate the sample size. The verdict is conservative on purpose.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .stats import two_sided_p

MIN_EVENTS = 20       # below this, no verdict is issued
SIGNIFICANCE_Z = 2.0  # roughly p < 0.05, before multiple-testing caution
# A null result on this many events is evidence of no edge, not just "unknown".
# 48 NSE large caps x 5y gave 50-4,100 events per candle pattern with none
# significant after Bonferroni (`stockintel validate-patterns --universe`).
WELL_POWERED = 200


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def decluster(mask: pd.Series, spacing: int) -> pd.Series:
    out = pd.Series(False, index=mask.index)
    last = -spacing - 1
    for i, flag in enumerate(mask.to_numpy()):
        if flag and i - last > spacing:
            out.iloc[i] = True
            last = i
    return out


def event_study(close: pd.Series, signal: pd.Series, horizon: int,
                direction: int) -> Dict[str, object]:
    """Compare forward returns after signal days against all days.

    direction: +1 if the signal claims bullish, -1 bearish, 0 informational.
    """
    fwd = forward_returns(close, horizon)
    known = fwd.notna()
    events = decluster(signal.fillna(False).astype(bool) & known, horizon)
    ev_ret = fwd[events].to_numpy()
    base = fwd[known].to_numpy()
    n = len(ev_ret)
    out: Dict[str, object] = {
        "horizon_days": horizon, "events": n,
        "baseline_mean_pct": round(float(base.mean()) * 100, 3) if len(base) else None,
        "baseline_hit_rate": round(float((base > 0).mean()), 3) if len(base) else None,
    }
    if n == 0:
        out["verdict"] = "no historical occurrences"
        return out
    out["mean_pct"] = round(float(ev_ret.mean()) * 100, 3)
    out["hit_rate"] = round(float((ev_ret > 0).mean()), 3)
    if n < MIN_EVENTS:
        out["verdict"] = f"insufficient history ({n} < {MIN_EVENTS} events)"
        return out
    excess = ev_ret - base.mean()
    se = excess.std(ddof=1) / math.sqrt(n)
    z = float(excess.mean() / se) if se > 0 else 0.0
    out["excess_mean_pct"] = round(float(excess.mean()) * 100, 3)
    out["z"] = round(z, 2)
    out["p_value"] = round(two_sided_p(z), 4)
    if abs(z) < SIGNIFICANCE_Z:
        out["verdict"] = "no statistically significant edge"
    elif direction != 0 and np.sign(z) != direction:
        out["verdict"] = "significant but opposite to the textbook direction"
    else:
        out["verdict"] = "historically significant in the claimed direction"
    return out


def edge_multiplier(stats: Dict[str, object]) -> float:
    """How much a textbook signal is allowed to count, given its history.

    Validated signals keep full weight; unproven ones are heavily discounted;
    contradicted ones are zeroed. PROVISIONAL factors.
    """
    verdict = str(stats.get("verdict", ""))
    if verdict.startswith("historically significant"):
        return 1.0
    if verdict.startswith("significant but opposite"):
        return 0.0
    tested = stats.get("pooled") or stats
    if verdict.startswith("no statistically significant") and tested.get("events", 0) >= WELL_POWERED:
        return 0.1
    return 0.35


def pooled_event_study(pairs, horizon: int, direction: int,
                       z_threshold: float = SIGNIFICANCE_Z) -> Dict[str, object]:
    """Event study across many stocks. Each event's return is measured in
    excess of its own stock's unconditional mean, so high-drift stocks do not
    masquerade as pattern edge. pairs: iterable of (close, signal)."""
    excess, n_stocks = [], 0
    for close, signal in pairs:
        fwd = forward_returns(close, horizon)
        known = fwd.notna()
        if known.sum() < 2 * horizon:
            continue
        events = decluster(signal.reindex(close.index).fillna(False).astype(bool) & known, horizon)
        if events.any():
            n_stocks += 1
            mu = float(fwd[known].mean())
            excess.extend((fwd[events] - mu).tolist())
    n = len(excess)
    out: Dict[str, object] = {"horizon_days": horizon, "events": n, "stocks": n_stocks}
    if n < MIN_EVENTS:
        out["verdict"] = f"insufficient history ({n} < {MIN_EVENTS} events)"
        return out
    x = np.asarray(excess)
    se = x.std(ddof=1) / math.sqrt(n)
    z = float(x.mean() / se) if se > 0 else 0.0
    out.update({"excess_mean_pct": round(float(x.mean()) * 100, 3), "z": round(z, 2),
                "p_value": round(two_sided_p(z), 4), "z_threshold": round(z_threshold, 2)})
    if abs(z) < z_threshold:
        out["verdict"] = "no statistically significant edge"
    elif direction != 0 and np.sign(z) != direction:
        out["verdict"] = "significant but opposite to the textbook direction"
    else:
        out["verdict"] = "historically significant in the claimed direction"
    return out


def with_prior(own: Dict[str, object], prior: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Use the stock's own history when it has enough events; otherwise fall
    back to the pooled universe result, marked as such."""
    if own.get("events", 0) >= MIN_EVENTS or not prior:
        return own
    return {**own, "verdict": prior["verdict"], "pooled": prior}
