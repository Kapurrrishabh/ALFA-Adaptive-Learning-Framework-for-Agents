"""Fine-tune Chronos-2 on NSE prices, and test it the way every model here is tested.

Protocol, fixed before any result was seen:
- choose among CONFIGS with weights trained on prices up to TRAIN_END, scored on origins
  whose outcomes fall in VAL_START..VAL_END (pinball loss: a proper score of the whole range);
- retrain the chosen config on prices up to VAL_END and score it once on TEST_START onward,
  next to the untuned model, Chronos-Bolt and the EWMA band;
- "up" calls fire when P(up) clears a cut set on the validation window to fire 10% of the
  time; their precision is reported on the test window against the base rate, with a 95%
  range from resampling whole origin dates (stocks on one date move together).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from backend.models.external import tsfm
from backend.models.learning.protocol import TEST_START, TRAIN_END, VAL_END, VAL_START, selective

log = logging.getLogger("stockintel.tsfm_train")
BASE_MODEL = "autogluon/chronos-2-small"
HORIZONS = (5, 20)


@dataclass(frozen=True)
class Config:
    name: str
    mode: str              # "lora" trains small adapters; "full" trains every weight
    learning_rate: float
    steps: int


# LoRA wants ~10x the full-tuning rate (chronos docs); two rates each, one step budget
CONFIGS = (Config("lora-1e-5", "lora", 1e-5, 1000), Config("lora-1e-4", "lora", 1e-4, 1000),
           Config("full-1e-6", "full", 1e-6, 1000), Config("full-1e-5", "full", 1e-5, 1000))


def training_series(close: pd.DataFrame, end: str, min_len: int = tsfm.CONTEXT) -> List[np.ndarray]:
    """Each stock's closes up to `end`, long enough to fill one context."""
    out = []
    for s in close.columns:
        c = close[s].loc[:end].dropna()
        if len(c) >= min_len:
            out.append(c.to_numpy(dtype=np.float32))
    return out


def eval_universe(close: pd.DataFrame, symbols: Sequence[str], start: str) -> Dict[str, pd.Series]:
    """Stocks with a full context of history before the first origin."""
    return {s: close[s].dropna() for s in symbols
            if s in close.columns and close[s].loc[:start].dropna().size > tsfm.CONTEXT}


def finetune(series: List[np.ndarray], cfg: Config, out_dir: Path, base: str = BASE_MODEL):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(base, device_map="cpu")
    return pipe.fit(series, prediction_length=max(HORIZONS), finetune_mode=cfg.mode, learning_rate=cfg.learning_rate,
                    num_steps=cfg.steps, batch_size=32, context_length=tsfm.CONTEXT,
                    output_dir=str(out_dir / cfg.name), report_to=[], logging_steps=100, save_strategy="no")


def run(close: pd.DataFrame, symbols: Sequence[str], out_dir: Path) -> Dict[str, object]:
    """The whole protocol. Returns everything measured; the chosen model is saved under out_dir."""
    from chronos import BaseChronosPipeline
    universe = eval_universe(close, symbols, VAL_START)
    score = lambda rows: float(np.mean([h["pinball_skill"] for h in tsfm.summarize(rows, "")["horizons"]]))
    selection = []
    for cfg in CONFIGS:
        pipe = finetune(training_series(close, TRAIN_END), cfg, out_dir / "selection")
        rows = tsfm.forecast_rows(universe, HORIZONS, VAL_START, VAL_END, pipe=pipe)
        selection.append({**asdict(cfg), "val_pinball_skill": round(score(rows), 4)})
        log.info("config %s: validation pinball skill %.4f", cfg.name, selection[-1]["val_pinball_skill"])
    best = max(CONFIGS, key=lambda c: next(s["val_pinball_skill"] for s in selection if s["name"] == c.name))
    final = finetune(training_series(close, VAL_END), best, out_dir / "final")
    untuned = BaseChronosPipeline.from_pretrained(BASE_MODEL, device_map="cpu")
    bolt = BaseChronosPipeline.from_pretrained(tsfm.MODEL_ID, device_map="cpu")
    results: Dict[str, object] = {"protocol": __doc__, "base_model": BASE_MODEL, "selection": selection, "chosen": best.name,
                                  "stocks": len(universe), "test": {}, "precision_calls": {}}
    for name, pipe in (("chronos-2-nse", final), ("chronos-2-untuned", untuned), ("chronos-bolt", bolt)):
        val_rows = tsfm.forecast_rows(universe, HORIZONS, VAL_START, VAL_END, pipe=pipe)
        test_rows = tsfm.forecast_rows(universe, HORIZONS, TEST_START, pipe=pipe)
        results["test"][name] = tsfm.summarize(test_rows, name)
        results["precision_calls"][name] = selective(val_rows, test_rows, HORIZONS)
        log.info("%s test: %s", name, results["test"][name]["horizons"])
    final.save_pretrained(str(out_dir / "chronos-2-nse"))
    # the record travels with the weights, so whoever serves them can show what they scored
    (out_dir / "chronos-2-nse" / tsfm.RECORD_FILE).write_text(json.dumps({
        "model": "chronos-2-nse", "base_model": BASE_MODEL, "chosen": best.name, "protocol": __doc__, "selection": selection,
        "test": results["test"]["chronos-2-nse"], "precision_calls": results["precision_calls"]["chronos-2-nse"],
        "compared_on_same_test": {k: v for k, v in results["test"].items() if k != "chronos-2-nse"}}, indent=2, default=float))
    return results
