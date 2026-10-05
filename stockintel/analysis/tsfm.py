"""Open-source time-series foundation model (Amazon Chronos-Bolt) as a forecast candidate.

Chronos-Bolt returns quantiles of future prices directly, so it competes with the
EWMA volatility band on the job that band does (the range), and with the base rate
on direction. It is loaded lazily and only used where `evaluate` shows it earns it.
Every origin uses only prices up to that date; the model never sees the future.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .forecast import ewma_sigma
from .stats import norm_ppf

log = logging.getLogger("stockintel.tsfm")
MODEL_ID = "amazon/chronos-bolt-small"
CONTEXT = 512
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
_PIPE = None


def pipeline(model_id: str = MODEL_ID):
    global _PIPE
    if _PIPE is None:
        import torch
        from chronos import BaseChronosPipeline
        torch.set_num_threads(2)            # keep the laptop responsive; inference is small
        _PIPE = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    return _PIPE


def forecast_quantiles(closes: Sequence[np.ndarray], horizon: int,
                       quantiles: Sequence[float] = QUANTILES) -> np.ndarray:
    """(n_series, horizon, n_quantiles) price quantiles for each context series."""
    import torch
    ctx = [torch.tensor(np.asarray(c[-CONTEXT:], dtype=np.float32)) for c in closes]
    q, _ = pipeline().predict_quantiles(ctx, prediction_length=horizon, quantile_levels=list(quantiles))
    return q.numpy()


def _pinball(y: float, qs: np.ndarray, levels: Sequence[float]) -> float:
    return float(np.mean([max(l * (y - q), (l - 1) * (y - q)) for q, l in zip(qs, levels)]))


def evaluate(closes: Dict[str, pd.Series], horizons: Sequence[int] = (5, 20), start: str = "2022-01-01",
             step: int = 21) -> Dict[str, object]:
    """Walk-forward comparison against the EWMA band and the base rate, on log-return scale."""
    z = {l: float(norm_ppf(l)) for l in QUANTILES}
    rows: Dict[int, List[dict]] = {h: [] for h in horizons}
    for sym, c in closes.items():
        c = c.dropna()
        idx = [i for i in range(CONTEXT, len(c) - max(horizons), step) if c.index[i] >= pd.Timestamp(start)]
        if not idx:
            continue
        sig = ewma_sigma(c).to_numpy()
        for h in horizons:
            qs = forecast_quantiles([c.to_numpy()[: i + 1] for i in idx], h)
            for k, i in enumerate(idx):
                p0, y = float(c.iloc[i]), float(np.log(c.iloc[i + h] / c.iloc[i]))
                qc = np.log(qs[k, -1, :] / p0)                      # Chronos, log-return quantiles at day h
                s = sig[i] * math.sqrt(h)
                qe = np.array([z[l] * s for l in QUANTILES])         # EWMA normal, zero drift
                past = np.log(c.iloc[h: i + 1].to_numpy() / c.iloc[: i + 1 - h].to_numpy())
                rows[h].append({"symbol": sym, "y": y, "chronos_median": float(qc[2]),
                                "chronos_in80": bool(qc[0] <= y <= qc[-1]), "ewma_in80": bool(qe[0] <= y <= qe[-1]),
                                "chronos_pinball": _pinball(y, qc, QUANTILES), "ewma_pinball": _pinball(y, qe, QUANTILES),
                                "base_up": float((past > 0).mean())})
    out: Dict[str, object] = {"model": MODEL_ID, "horizons": []}
    for h, rs in rows.items():
        if not rs:
            continue
        df = pd.DataFrame(rs)
        up = df["y"] > 0
        out["horizons"].append({
            "horizon": h, "n": len(df), "stocks": int(df["symbol"].nunique()),
            "chronos_direction_acc": round(float(((df["chronos_median"] > 0) == up).mean()), 4),
            "base_rate_acc": round(float(((df["base_up"] > 0.5) == up).mean()), 4),
            "chronos_cover80": round(float(df["chronos_in80"].mean()), 4),
            "ewma_cover80": round(float(df["ewma_in80"].mean()), 4),
            "chronos_pinball": round(float(df["chronos_pinball"].mean()), 5),
            "ewma_pinball": round(float(df["ewma_pinball"].mean()), 5),
            "pinball_skill": round(1 - float(df["chronos_pinball"].mean()) / float(df["ewma_pinball"].mean()), 4)})
    return out


def fan(close: pd.Series, horizon: int = 20) -> Optional[Dict[str, list]]:
    """Today's Chronos quantile fan for one stock, as prices per future day."""
    c = close.dropna()
    if len(c) < 64:
        return None
    q = forecast_quantiles([c.to_numpy()], horizon)[0]
    return {f"q{int(l * 100)}": [round(float(v), 2) for v in q[:, j]] for j, l in enumerate(QUANTILES)}
