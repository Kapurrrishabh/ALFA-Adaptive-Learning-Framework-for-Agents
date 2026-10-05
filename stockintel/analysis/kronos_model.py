"""Kronos (open-source candlestick foundation model, MIT) as a next-candles candidate.

Kronos reads OHLCV candles and generates the next candles. The code lives outside
the package (it is not on PyPI): set STOCKINTEL_KRONOS_PATH or clone
https://github.com/shiyu-coder/Kronos into ~/.stockintel/vendor/kronos.
Loaded lazily; nothing else in the app depends on it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..data.provider import DataUnavailable

PATH = Path(os.environ.get("STOCKINTEL_KRONOS_PATH", Path.home() / ".stockintel" / "vendor" / "kronos"))
MODEL_ID, TOKENIZER_ID = "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK = 256
_PRED = None


def available() -> bool:
    return (PATH / "model" / "kronos.py").exists()


def predictor():
    global _PRED
    if _PRED is None:
        if not available():
            raise DataUnavailable(f"Kronos code not found at {PATH}; clone https://github.com/shiyu-coder/Kronos there")
        import torch
        torch.set_num_threads(2)
        sys.path.insert(0, str(PATH))
        from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402
        _PRED = KronosPredictor(Kronos.from_pretrained(MODEL_ID), KronosTokenizer.from_pretrained(TOKENIZER_ID),
                                device="cpu", max_context=512)
    return _PRED


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    f = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    f["amount"] = f["volume"] * f["close"]
    return f.reset_index(drop=True)


def predict_batch(windows: Sequence[pd.DataFrame], pred_len: int = 5, samples: int = 5,
                  seed: int = 0) -> List[pd.DataFrame]:
    """Mean of `samples` generated paths for each window (all windows the same length)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    xs, ys = [], []
    for w in windows:
        xs.append(pd.Series(w.index))
        ys.append(pd.Series(pd.bdate_range(w.index[-1], periods=pred_len + 1)[1:]))
    out = predictor().predict_batch([_frame(w) for w in windows], xs, ys, pred_len=pred_len, T=1.0,
                                    top_p=0.9, sample_count=samples, verbose=False)
    return [o.set_index(y) for o, y in zip(out, ys)]


def next_candles(df: pd.DataFrame, pred_len: int = 5, samples: int = 8) -> Dict[str, list]:
    """Today's projected candles for one stock, as lists the chart can draw."""
    w = df.iloc[-LOOKBACK:]
    p = predict_batch([w], pred_len, samples)[0]
    return {"dates": [str(d.date()) for d in p.index], **{c: [round(float(v), 2) for v in p[c]]
                                                          for c in ("open", "high", "low", "close")}}


def evaluate(frames: Dict[str, pd.DataFrame], pred_len: int = 5, start: str = "2023-01-01",
             step: int = 21, samples: int = 5) -> Dict[str, object]:
    """Walk-forward: does the projected close beat the base rate on direction and a
    no-change forecast on error? Only candles up to each origin are given to the model."""
    rows = []
    for sym, df in frames.items():
        idx = [i for i in range(LOOKBACK, len(df) - pred_len, step) if df.index[i] >= pd.Timestamp(start)]
        if not idx:
            continue
        preds = predict_batch([df.iloc[i - LOOKBACK + 1: i + 1] for i in idx], pred_len, samples)
        c = df["close"].to_numpy()
        for i, p in zip(idx, preds):
            actual = c[i + pred_len] / c[i] - 1
            pred = float(p["close"].iloc[-1]) / c[i] - 1
            past = c[pred_len: i + 1] / c[: i + 1 - pred_len] - 1
            rows.append({"symbol": sym, "actual": actual, "pred": pred, "base_up": float((past > 0).mean())})
    d = pd.DataFrame(rows)
    up = d["actual"] > 0
    return {"model": MODEL_ID, "horizon": pred_len, "n": len(d), "stocks": int(d["symbol"].nunique()),
            "direction_acc": round(float(((d["pred"] > 0) == up).mean()), 4),
            "base_rate_acc": round(float(((d["base_up"] > 0.5) == up).mean()), 4),
            "mae_pct": round(float((d["pred"] - d["actual"]).abs().mean()) * 100, 3),
            "no_change_mae_pct": round(float(d["actual"].abs().mean()) * 100, 3),
            "corr": round(float(np.corrcoef(d["pred"], d["actual"])[0, 1]), 4)}
