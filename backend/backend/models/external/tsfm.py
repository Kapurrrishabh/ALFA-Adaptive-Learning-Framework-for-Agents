"""Open-source time-series foundation model (Amazon Chronos-Bolt) as a forecast candidate.

Chronos-Bolt returns quantiles of future prices directly, so it competes with the
EWMA volatility band on the job that band does (the range), and with the base rate
on direction. It is loaded lazily and only used where `evaluate` shows it earns it.
Every origin uses only prices up to that date; the model never sees the future.
"""
from __future__ import annotations

import functools
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backend.config import FAN_LEVELS, TORCH_DEVICE
from backend.paths import ARTIFACTS
from backend.database.sources.provider import DataUnavailable
from backend.models.learning.protocol import direction_edge
from backend.advisory.statistics.forecast import ewma_sigma
from backend.advisory.statistics.stats import norm_ppf

log = logging.getLogger("stockintel.tsfm")
MODEL_ID = "amazon/chronos-bolt-small"
TUNED = ARTIFACTS / "external" / "chronos-2-nse"
# our fine-tuned weights when present, else the base model; every fan's note names which one it is
SERVED_MODEL = os.environ.get("STOCKINTEL_CHRONOS_MODEL", str(TUNED) if TUNED.exists() else MODEL_ID)
CONTEXT = 512
QUANTILES = FAN_LEVELS
# denser levels for scoring, so P(up) can be read off the forecast distribution
EVAL_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
RECORD_FILE = "stockintel_record.json"     # test record saved next to fine-tuned weights
_PIPE = None


@functools.lru_cache(maxsize=None)
def model_record(model_id: str) -> Optional[dict]:
    """The test record saved with fine-tuned weights (local folder or Hub repo); None for a base model."""
    local = Path(model_id) / RECORD_FILE
    if local.exists():
        return json.loads(local.read_text())
    if Path(model_id).exists():
        return None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        return json.loads(Path(hf_hub_download(model_id, RECORD_FILE)).read_text())
    except EntryNotFoundError:
        return None


def pipeline(model_id: str = SERVED_MODEL):
    global _PIPE
    if _PIPE is None:
        import torch
        from chronos import BaseChronosPipeline
        torch.set_num_threads(2)            # keep the laptop responsive; inference is small
        _PIPE = BaseChronosPipeline.from_pretrained(model_id, device_map=TORCH_DEVICE, torch_dtype=torch.float32)
    return _PIPE


def forecast_quantiles(closes: Sequence[np.ndarray], horizon: int,
                       quantiles: Sequence[float] = QUANTILES, pipe=None) -> np.ndarray:
    """(n_series, horizon, n_quantiles) price quantiles for each context series."""
    import torch
    ctx = [torch.tensor(np.asarray(c[-CONTEXT:], dtype=np.float32)) for c in closes]
    q, _ = (pipe or pipeline()).predict_quantiles(ctx, prediction_length=horizon, quantile_levels=list(quantiles))
    if isinstance(q, list):                 # Chronos-2 returns one (variates, horizon, levels) tensor per series
        q = torch.stack([x[0] for x in q])
    return q.cpu().numpy()


def _pinball(y: float, qs: np.ndarray, levels: Sequence[float]) -> float:
    return float(np.mean([max(l * (y - q), (l - 1) * (y - q)) for q, l in zip(qs, levels)]))


def forecast_rows(closes: Dict[str, pd.Series], horizons: Sequence[int] = (5, 20), start: str = "2022-01-01",
                  end: Optional[str] = None, step: int = 21, pipe=None) -> List[dict]:
    """One row per (stock, origin, horizon), on log-return scale. Origins share one calendar,
    every `step` trading days from `start`; with `end`, every outcome lands on or before it."""
    levels = EVAL_LEVELS
    z = np.array([norm_ppf(l) for l in levels])
    lo, mid, hi = levels.index(0.1), levels.index(0.5), levels.index(0.9)
    cal = pd.DatetimeIndex(sorted(set().union(*[c.dropna().index for c in closes.values()])))
    origins = set(cal[cal >= pd.Timestamp(start)][::step])
    rows: List[dict] = []
    for sym, c in closes.items():
        c = c.dropna()
        idx = [i for i in range(CONTEXT, len(c) - max(horizons)) if c.index[i] in origins
               and (end is None or c.index[i + max(horizons)] <= pd.Timestamp(end))]
        if not idx:
            continue
        sig = ewma_sigma(c).to_numpy()
        for h in horizons:
            qs = forecast_quantiles([c.to_numpy()[: i + 1] for i in idx], h, levels, pipe)
            for k, i in enumerate(idx):
                p0, y = float(c.iloc[i]), float(np.log(c.iloc[i + h] / c.iloc[i]))
                qc = np.sort(np.log(qs[k, -1, :] / p0))            # model, log-return quantiles at day h
                qe = z * sig[i] * math.sqrt(h)                      # EWMA normal, zero drift
                past = np.log(c.iloc[h: i + 1].to_numpy() / c.iloc[: i + 1 - h].to_numpy())
                rows.append({"symbol": sym, "date": str(c.index[i].date()), "horizon": h, "y": y,
                             "median": float(qc[mid]), "p_up": float(1 - np.interp(0.0, qc, levels)),
                             "in80": bool(qc[lo] <= y <= qc[hi]), "ewma_in80": bool(qe[lo] <= y <= qe[hi]),
                             "pinball": _pinball(y, qc, levels), "ewma_pinball": _pinball(y, qe, levels),
                             "base_up": float((past > 0).mean())})
    return rows


def summarize(rows: List[dict], model: str) -> Dict[str, object]:
    out: Dict[str, object] = {"model": model, "horizons": []}
    df = pd.DataFrame(rows)
    for h, g in (df.groupby("horizon") if len(df) else []):
        up = g["y"] > 0
        edge_lo, edge_hi = direction_edge(g["date"], g["median"] > 0, up, g["base_up"] > 0.5)
        out["horizons"].append({
            "horizon": int(h), "n": len(g), "stocks": int(g["symbol"].nunique()), "origins": int(g["date"].nunique()),
            "chronos_direction_acc": round(float(((g["median"] > 0) == up).mean()), 4),
            "base_rate_acc": round(float(((g["base_up"] > 0.5) == up).mean()), 4),
            "up_rate": round(float(up.mean()), 4),
            "chronos_cover80": round(float(g["in80"].mean()), 4),
            "ewma_cover80": round(float(g["ewma_in80"].mean()), 4),
            "chronos_pinball": round(float(g["pinball"].mean()), 5),
            "ewma_pinball": round(float(g["ewma_pinball"].mean()), 5),
            "pinball_skill": round(1 - float(g["pinball"].mean()) / float(g["ewma_pinball"].mean()), 4),
            "direction_edge_lo": edge_lo, "direction_edge_hi": edge_hi})
    return out


def evaluate(closes: Dict[str, pd.Series], horizons: Sequence[int] = (5, 20), start: str = "2022-01-01",
             step: int = 21, end: Optional[str] = None, pipe=None, model: str = MODEL_ID) -> Dict[str, object]:
    """Walk-forward comparison against the EWMA band and the base rate."""
    return summarize(forecast_rows(closes, horizons, start, end, step, pipe), model)


MIN_CONTEXT = 64


def fan(close: pd.Series, horizon: int = 20) -> Dict[str, list]:
    """Today's Chronos quantile fan for one stock, as prices per future day."""
    c = close.dropna()
    if len(c) < MIN_CONTEXT:
        raise DataUnavailable(f"only {len(c)} closes; Chronos needs at least {MIN_CONTEXT}")
    q = forecast_quantiles([c.to_numpy()], horizon)[0]
    return {**{f"q{int(l * 100)}": [round(float(v), 2) for v in q[:, j]] for j, l in enumerate(QUANTILES)},
            "served_model": SERVED_MODEL, "served_record": model_record(SERVED_MODEL)}
