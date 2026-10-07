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
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backend.config import TORCH_DEVICE
from backend.paths import ARTIFACTS
from backend.database.sources.provider import DataUnavailable
from backend.models.external.tsfm import model_record

PATH = Path(os.environ.get("STOCKINTEL_KRONOS_PATH", Path.home() / ".stockintel" / "vendor" / "kronos"))
# Kronos is not on PyPI, so its code is fetched at the commit the app was tested with
CODE_REPO, CODE_COMMIT = "shiyu-coder/Kronos", "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL_ID, TOKENIZER_ID = "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"
TUNED = ARTIFACTS / "external" / "kronos-nse"
# our fine-tuned predictor when present, else the base one; the note names which one it is
SERVED_MODEL = os.environ.get("STOCKINTEL_KRONOS_MODEL", str(TUNED) if TUNED.exists() else MODEL_ID)
LOOKBACK = 256
_PRED = None


def available() -> bool:
    return (PATH / "model" / "kronos.py").exists()


def fetch_code(path: Path = PATH) -> None:
    """Put the Kronos code at CODE_COMMIT into `path`; nothing to do when it is already there."""
    if (path / "model" / "kronos.py").exists():
        return
    import io
    import tarfile
    import urllib.request
    with urllib.request.urlopen(f"https://github.com/{CODE_REPO}/archive/{CODE_COMMIT}.tar.gz", timeout=120) as resp:
        archive = tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz")
    top = archive.getmembers()[0].name.split("/")[0] + "/"
    path.mkdir(parents=True, exist_ok=True)
    for member in archive.getmembers():
        if member.name.startswith(top):
            member.name = member.name[len(top):]
            archive.extract(member, path, filter="data")


def kronos_code():
    """The Kronos package (model.Kronos, model.KronosTokenizer, model.KronosPredictor)."""
    if not available():
        raise DataUnavailable(f"Kronos code not found at {PATH}; clone https://github.com/shiyu-coder/Kronos there")
    if str(PATH) not in sys.path:
        sys.path.insert(0, str(PATH))
    import model  # noqa: E402
    return model


def load(model_id: str = SERVED_MODEL):
    m = kronos_code()
    return m.KronosPredictor(m.Kronos.from_pretrained(model_id), m.KronosTokenizer.from_pretrained(TOKENIZER_ID),
                             device=TORCH_DEVICE, max_context=512)


def predictor():
    global _PRED
    if _PRED is None:
        import torch
        torch.set_num_threads(2)
        _PRED = load()
    return _PRED


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    f = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    f["amount"] = f["volume"] * f["close"]
    return f.reset_index(drop=True)


def predict_batch(windows: Sequence[pd.DataFrame], pred_len: int = 5, samples: int = 5,
                  seed: int = 0, pred=None) -> List[pd.DataFrame]:
    """Mean of `samples` generated paths for each window (all windows the same length)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    xs, ys = [], []
    for w in windows:
        xs.append(pd.Series(w.index))
        ys.append(pd.Series(pd.bdate_range(w.index[-1], periods=pred_len + 1)[1:]))
    out = (pred or predictor()).predict_batch([_frame(w) for w in windows], xs, ys, pred_len=pred_len, T=1.0,
                                    top_p=0.9, sample_count=samples, verbose=False)
    return [o.set_index(y) for o, y in zip(out, ys)]


def next_candles(df: pd.DataFrame, pred_len: int = 5, samples: int = 8) -> Dict[str, list]:
    """Today's projected candles for one stock, as lists the chart can draw."""
    w = df.iloc[-LOOKBACK:]
    p = predict_batch([w], pred_len, samples)[0]
    return {"dates": [str(d.date()) for d in p.index], **{c: [round(float(v), 2) for v in p[c]] for c in ("open", "high", "low", "close")},
            "served_model": SERVED_MODEL, "served_record": model_record(SERVED_MODEL)}


def forecast_rows(frames: Dict[str, pd.DataFrame], pred_len: int = 5, start: str = "2023-01-01",
                  end: Optional[str] = None, step: int = 21, samples: int = 5, pred=None) -> List[dict]:
    """One row per (stock, origin): the projected and actual return over `pred_len` days.
    Origins share one calendar; only candles up to each origin are given to the model."""
    cal = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    origins = set(cal[cal >= pd.Timestamp(start)][::step])
    rows = []
    for sym, df in frames.items():
        idx = [i for i in range(LOOKBACK, len(df) - pred_len) if df.index[i] in origins
               and (end is None or df.index[i + pred_len] <= pd.Timestamp(end))]
        if not idx:
            continue
        preds = predict_batch([df.iloc[i - LOOKBACK + 1: i + 1] for i in idx], pred_len, samples, pred=pred)
        c = df["close"].to_numpy()
        for i, p in zip(idx, preds):
            past = c[pred_len: i + 1] / c[: i + 1 - pred_len] - 1
            rows.append({"symbol": sym, "date": str(df.index[i].date()), "horizon": pred_len,
                         "y": c[i + pred_len] / c[i] - 1, "pred": float(p["close"].iloc[-1]) / c[i] - 1,
                         "base_up": float((past > 0).mean())})
    return rows


def summarize(rows: List[dict], model: str = MODEL_ID) -> Dict[str, object]:
    d = pd.DataFrame(rows)
    up = d["y"] > 0
    return {"model": model, "horizon": int(d["horizon"].iloc[0]), "n": len(d), "stocks": int(d["symbol"].nunique()),
            "origins": int(d["date"].nunique()),
            "direction_acc": round(float(((d["pred"] > 0) == up).mean()), 4),
            "base_rate_acc": round(float(((d["base_up"] > 0.5) == up).mean()), 4),
            "mae_pct": round(float((d["pred"] - d["y"]).abs().mean()) * 100, 3),
            "no_change_mae_pct": round(float(d["y"].abs().mean()) * 100, 3),
            "corr": round(float(np.corrcoef(d["pred"], d["y"])[0, 1]), 4)}


def evaluate(frames: Dict[str, pd.DataFrame], pred_len: int = 5, start: str = "2023-01-01",
             step: int = 21, samples: int = 5, end: Optional[str] = None, pred=None, model: str = MODEL_ID) -> Dict[str, object]:
    """Walk-forward: does the projected close beat the base rate on direction and a
    no-change forecast on error?"""
    return summarize(forecast_rows(frames, pred_len, start, end, step, samples, pred), model)
