"""Fine-tune the Kronos predictor on NSE daily candles (tokenizer frozen), tested like Chronos-2.

Protocol, fixed before any result was seen:
- train on candles up to TRAIN_END for each learning rate in RATES; choose the rate with the
  lowest token loss on a fixed set of windows lying wholly inside VAL_START..VAL_END;
- retrain the chosen rate on candles up to VAL_END and score it once from TEST_START,
  next to the untuned model and a no-change forecast;
- "up" calls fire when the projected return clears a cut set on validation to fire 10% of
  the time; their precision is reported on the test window against the base rate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import kronos_model as K
from .tsfm import RECORD_FILE
from .tsfm_train import TEST_START, TRAIN_END, VAL_END, VAL_START, selective

log = logging.getLogger("stockintel.kronos_train")
PRED_LEN = 5
WINDOW = K.LOOKBACK + 10        # a full inference context plus the stretch after it
BATCH, STEPS = 16, 1000
RATES = (4e-5, 1e-5)            # 4e-5 is the Kronos authors' own fine-tuning rate
VAL_WINDOWS = 256
CLIP = 5.0                      # same clip as KronosPredictor at inference


def _arrays(frames: Dict[str, pd.DataFrame], start: str, end: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per stock: features (open, high, low, close, volume, amount) and time stamps in [start, end]."""
    out = []
    for df in frames.values():
        d = df.loc[start:end].dropna()
        if len(d) < WINDOW:
            continue
        x = np.column_stack([d[c] for c in ("open", "high", "low", "close", "volume")] + [d["volume"] * d["close"]])
        ix = d.index
        stamp = np.column_stack([np.zeros(len(ix)), np.zeros(len(ix)), ix.weekday, ix.day, ix.month])
        out.append((x.astype(np.float32), stamp.astype(np.float32)))
    return out


def _windows(arrays, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """n random windows, each normalised on its context part only, exactly as at inference."""
    xs, ss = [], []
    for _ in range(n):
        x, s = arrays[rng.integers(len(arrays))]
        i = rng.integers(0, len(x) - WINDOW + 1)
        w = x[i:i + WINDOW]
        mu, sd = w[:K.LOOKBACK].mean(0), w[:K.LOOKBACK].std(0)
        xs.append(np.clip((w - mu) / (sd + 1e-5), -CLIP, CLIP))
        ss.append(s[i:i + WINDOW])
    return np.stack(xs), np.stack(ss)


def _loss(model, tokenizer, x: np.ndarray, stamp: np.ndarray):
    import torch
    x, stamp = torch.from_numpy(x), torch.from_numpy(stamp)
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x, half=True)
    logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1, :])
    return model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])[0]


def train(arrays, lr: float, steps: int = STEPS, seed: int = 0):
    import torch
    m = K.kronos_code()
    torch.manual_seed(seed)
    model, tokenizer = m.Kronos.from_pretrained(K.MODEL_ID), m.KronosTokenizer.from_pretrained(K.TOKENIZER_ID)
    tokenizer.eval()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.03, div_factor=10)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        loss = _loss(model, tokenizer, *_windows(arrays, rng, BATCH))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            log.info("lr %g step %d loss %.4f", lr, step, loss.item())
    model.eval()
    return model, tokenizer


def val_loss(model, tokenizer, arrays, n: int = VAL_WINDOWS) -> float:
    import torch
    x, s = _windows(arrays, np.random.default_rng(12345), n)
    with torch.no_grad():
        return float(np.mean([_loss(model, tokenizer, x[i:i + 32], s[i:i + 32]).item() for i in range(0, n, 32)]))


def run(frames: Dict[str, pd.DataFrame], eval_frames: Dict[str, pd.DataFrame], out_dir: Path) -> Dict[str, object]:
    m = K.kronos_code()
    first = min(f.index[0] for f in frames.values()).strftime("%Y-%m-%d")
    val_arrays = _arrays(frames, VAL_START, VAL_END)
    selection = []
    for lr in RATES:
        model, tok = train(_arrays(frames, first, TRAIN_END), lr)
        selection.append({"learning_rate": lr, "val_loss": round(val_loss(model, tok, val_arrays), 4)})
        log.info("lr %g: validation loss %.4f", lr, selection[-1]["val_loss"])
    untuned = m.Kronos.from_pretrained(K.MODEL_ID).eval()
    tok = m.KronosTokenizer.from_pretrained(K.TOKENIZER_ID).eval()
    selection.append({"learning_rate": 0.0, "val_loss": round(val_loss(untuned, tok, val_arrays), 4)})
    best = min((s for s in selection if s["learning_rate"]), key=lambda s: s["val_loss"])["learning_rate"]
    final, tok = train(_arrays(frames, first, VAL_END), best)
    results: Dict[str, object] = {"protocol": __doc__, "base_model": K.MODEL_ID, "selection": selection, "chosen_lr": best,
                                  "test": {}, "precision_calls": {}}
    for name, model in (("kronos-nse", final), ("kronos-untuned", untuned)):
        pred = m.KronosPredictor(model, tok, device="cpu", max_context=512)
        val_rows = K.forecast_rows(eval_frames, PRED_LEN, VAL_START, VAL_END, pred=pred)
        test_rows = K.forecast_rows(eval_frames, PRED_LEN, TEST_START, pred=pred)
        results["test"][name] = K.summarize(test_rows, name)
        results["precision_calls"][name] = selective(val_rows, test_rows, key="pred", horizons=(PRED_LEN,))
        log.info("%s test: %s", name, results["test"][name])
    final.save_pretrained(str(out_dir / "kronos-nse"))
    (out_dir / "kronos-nse" / RECORD_FILE).write_text(json.dumps({
        "model": "kronos-nse", "base_model": K.MODEL_ID, "tokenizer": K.TOKENIZER_ID, "chosen_lr": best, "protocol": __doc__,
        "selection": selection, "test": results["test"]["kronos-nse"], "precision_calls": results["precision_calls"]["kronos-nse"],
        "compared_on_same_test": {"kronos-untuned": results["test"]["kronos-untuned"]}}, indent=2, default=float))
    return results
