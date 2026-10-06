"""Live model monitoring and data-drift checks.

Predictions logged at analysis time are scored once their horizon has passed,
giving a live Brier score that can be compared with the walk-forward estimate
the model was promoted on. Data drift uses the population stability index
(PSI) between a stock's recent and reference return distributions.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.advisory.statistics.stats import brier
from backend.database.storage import Store

PSI_ALERT = 0.25    # common rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 major shift
RECENT, REFERENCE = 60, 500


def resolve_predictions(store: Store, closes: Dict[str, pd.Series]) -> int:
    """Fill in realized outcomes for matured predictions. Returns the count resolved."""
    done = 0
    for p in store.predictions(unresolved=True):
        close = closes.get(p["symbol"])
        if close is None:
            continue
        ts = pd.Timestamp(p["as_of"])
        if ts not in close.index:
            continue
        i = close.index.get_loc(ts)
        if i + p["horizon"] >= len(close):
            continue
        store.resolve_prediction(p["id"], float(close.iloc[i + p["horizon"]] > close.iloc[i]))
        done += 1
    return done


def live_skill(store: Store) -> Dict[str, Dict[str, Any]]:
    rows = [p for p in store.predictions() if p["realized"] is not None]
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted({(p["model"], p["horizon"]) for p in rows}):
        sel = [p for p in rows if (p["model"], p["horizon"]) == key]
        y = np.array([p["realized"] for p in sel])
        prob = np.array([p["output"] for p in sel])
        out[f"{key[0]}_h{key[1]}"] = {"n": len(sel), "brier": round(brier(y, prob), 4),
                                      "brier_if_0.5": round(brier(y, np.full(len(y), 0.5)), 4),
                                      "hit_rate": round(float(y.mean()), 3),
                                      "mean_predicted": round(float(prob.mean()), 3)}
    return out


def psi(reference: np.ndarray, recent: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(recent, edges)[0] / len(recent)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def data_drift(closes: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
    out = []
    for sym, close in closes.items():
        r = close.pct_change().dropna().to_numpy()
        if len(r) < RECENT + REFERENCE // 2:
            continue
        value = psi(r[-(RECENT + REFERENCE):-RECENT], r[-RECENT:])
        out.append({"symbol": sym, "returns_psi": round(value, 3), "drift": value > PSI_ALERT})
    return sorted(out, key=lambda d: -d["returns_psi"])
