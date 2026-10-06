"""Historical-analogue engine: "what happened after setups like today's?"

The current state is a vector of standardized price features. Neighbours are
past bars whose h-day outcome is already realized and which are not in the
last EXCLUDE_RECENT bars (those are trivially similar to today). Neighbours
are de-clustered so one historical episode cannot fill the sample. Scaling
uses only the candidate history, never the current bar. Results are
historical observations, not predictions.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from backend.advisory.fusion.evidence import DomainResult, Evidence, Provenance, unavailable
from backend.advisory.statistics.forecast import features
from backend.advisory.statistics.stats import summarize, two_sided_p

DOMAIN = "historical"
STATE_FEATURES = ["r5", "r20", "r60", "rsi", "dist50", "vol20", "volume"]
K = 30
EXCLUDE_RECENT = 60
HORIZONS = (5, 20)


def find_analogues(df: pd.DataFrame, horizon: int, k: int = K) -> Dict[str, object]:
    F = features(df)[STATE_FEATURES]
    close = df["close"]
    n = len(df)
    if F.iloc[-1].isna().any():
        return {"error": "latest bar has incomplete features"}
    fwd = (close.shift(-horizon) / close - 1.0).to_numpy()
    last_candidate = n - 1 - max(horizon, EXCLUDE_RECENT)
    cand = [i for i in range(last_candidate + 1)
            if not F.iloc[i].isna().any() and not np.isnan(fwd[i])]
    if len(cand) < k * 3:
        return {"error": f"only {len(cand)} historical candidates; need {k * 3}"}
    X = F.iloc[cand].to_numpy()
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = (X - mu) / sd
    now = (F.iloc[-1].to_numpy() - mu) / sd
    dist = np.sqrt(((Z - now) ** 2).sum(axis=1))
    chosen: List[int] = []
    for j in np.argsort(dist):
        i = cand[j]
        if all(abs(i - c) >= horizon for c in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    outcomes = fwd[chosen]
    base = fwd[cand]
    s = summarize(outcomes)
    excess = outcomes - base.mean()
    z = float(excess.mean() / (excess.std(ddof=1) / math.sqrt(len(excess)))) if len(excess) > 1 and excess.std(ddof=1) > 0 else 0.0
    return {"horizon_days": horizon, "analogues": len(chosen),
            "dates": [str(df.index[i].date()) for i in sorted(chosen)][-10:],
            "outcome": {k2: (round(v * 100, 2) if k2 not in ("n", "hit_rate") else round(v, 3))
                        for k2, v in s.items()},
            "baseline": {"mean_pct": round(float(base.mean()) * 100, 2),
                         "hit_rate": round(float((base > 0).mean()), 3)},
            "z_vs_baseline": round(z, 2), "p_value": round(two_sided_p(z), 3),
            "median_distance": round(float(np.median(dist[np.argsort(dist)[:k]])), 2)}


def analyze(df: pd.DataFrame, horizons: Sequence[int] = HORIZONS) -> DomainResult:
    if df is None or len(df) < 400:
        return unavailable(DOMAIN, "need at least 400 bars for a meaningful analogue search")
    as_of = str(df.index[-1].date())
    ev: List[Evidence] = []
    details: Dict[str, object] = {}
    directional = []
    for h in horizons:
        res = find_analogues(df, h)
        details[f"{h}d"] = res
        if "error" in res:
            continue
        o, b = res["outcome"], res["baseline"]
        significant = abs(res["z_vs_baseline"]) >= 2.0
        direction = int(np.sign(o["mean"] - b["mean_pct"])) if significant else 0
        strength = 0.4 if significant else 0.0
        directional.append(direction * strength)
        ev.append(Evidence(
            domain=DOMAIN, direction=direction, strength=strength,
            claim=(f"{res['analogues']} historical setups resembled today's. Over the next {h} "
                   f"days they returned {o['mean']:+.2f}% on average (median {o['median']:+.2f}%, "
                   f"{o['hit_rate']:.0%} positive, 10th–90th pct {o['p10']:+.1f}% to {o['p90']:+.1f}%) "
                   f"vs an unconditional {b['mean_pct']:+.2f}% ({b['hit_rate']:.0%} positive)"
                   + (" — a statistically significant difference." if significant
                      else " — not a statistically significant difference.")),
            value=o["mean"], provenance=Provenance(source=f"computed:analogues:h{h}", as_of=as_of)))
    if not ev:
        return unavailable(DOMAIN, "; ".join(str(v.get("error")) for v in details.values()))
    active = [d for d in directional if d != 0]
    score = float(np.mean(active)) if active else 0.0
    return DomainResult(domain=DOMAIN, state="significant analogue skew" if active else
                        "analogues match the base rate", score=score, evidence=ev,
                        details=details, as_of=as_of, confidence=1.0 if active else 0.0)
