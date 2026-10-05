"""The open-source forecast models served from a Hugging Face Space (deploy/hf-models-space),
so a small server without torch can still show them. Set STOCKINTEL_MODEL_SPACE to the
Space id ("you/stockintel-models") or its URL, and HF_TOKEN if the Space is private.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict

import pandas as pd

from ..data.provider import DataUnavailable
from .kronos_model import LOOKBACK
from .tsfm import CONTEXT

_CLIENT = None
_LOCK = threading.Lock()


def space() -> str:
    return os.environ.get("STOCKINTEL_MODEL_SPACE", "")


def _client():
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from gradio_client import Client
            _CLIENT = Client(space(), token=os.environ.get("HF_TOKEN") or None, verbose=False)
        return _CLIENT


def _call(api_name: str, *args: Any) -> Dict[str, list]:
    try:
        return json.loads(_client().predict(*args, api_name=api_name))
    except Exception as exc:         # any client or Space failure means the forecast is unavailable
        raise DataUnavailable(f"model Space {space()!r} failed on {api_name}: {exc}") from exc


def fan(close: pd.Series, horizon: int) -> Dict[str, list]:
    c = close.dropna().iloc[-CONTEXT:]
    return _call("/chronos_fan", json.dumps([round(float(x), 4) for x in c]), horizon)


def next_candles(df: pd.DataFrame, pred_len: int = 5, samples: int = 8) -> Dict[str, list]:
    w = df.iloc[-LOOKBACK:]
    payload = {"dates": [str(d.date()) for d in w.index],
               **{c: [round(float(x), 4) for x in w[c]] for c in ("open", "high", "low", "close", "volume")}}
    return _call("/kronos_candles", json.dumps(payload), pred_len, samples)
