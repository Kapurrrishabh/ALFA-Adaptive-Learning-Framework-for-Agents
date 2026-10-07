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

from backend.database.sources.provider import DataUnavailable
from backend.models.external.kronos_model import LOOKBACK
from backend.models.external.tsfm import CONTEXT

_CLIENT = None
_LOCK = threading.Lock()


# Everything the Space serves. A host that can run some of them itself names the rest in
# STOCKINTEL_SPACE_MODELS: a free Render instance sends all four, an Oracle VM only the writer.
SPACE_MODELS = ("chronos", "kronos", "alfa", "writer")


def space() -> str:
    return os.environ.get("STOCKINTEL_MODEL_SPACE", "")


def on_space(model: str) -> bool:
    """Whether `model` is served by the model Space rather than by this process."""
    if model not in SPACE_MODELS:
        raise ValueError(f"{model!r} is not served by the Space; it serves {', '.join(SPACE_MODELS)}")
    named = [m.strip() for m in os.environ.get("STOCKINTEL_SPACE_MODELS", ",".join(SPACE_MODELS)).split(",") if m.strip()]
    unknown = sorted(set(named) - set(SPACE_MODELS))
    if unknown:
        raise ValueError(f"STOCKINTEL_SPACE_MODELS names {', '.join(unknown)}; choose from {', '.join(SPACE_MODELS)}")
    return bool(space()) and model in named


def writer():
    """The language model that phrases notes and answers: the Space's when it serves the writer, else local."""
    from backend.models.external import language
    return RemoteWriter() if on_space("writer") else language.OpenLM()


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


def alfa_fan(close: pd.Series, steps: int, count: int) -> Dict[str, Any]:
    return _call("/return_paths", json.dumps([round(float(x), 4) for x in close.dropna()]), steps, count)


class RemoteWriter:
    """The Space's language model behind the LLMClient interface (`create`, `text`)."""

    def create(self, system: str, messages, tools=None):
        from backend.models.external.language import Reply
        if tools:
            raise DataUnavailable("the Space writer does not take tools")
        out = _call("/write", system, json.dumps(messages))
        return Reply(out["model"], out["text"])

    @staticmethod
    def text(resp) -> str:
        return resp.content


def next_candles(df: pd.DataFrame, pred_len: int = 5, samples: int = 8) -> Dict[str, list]:
    w = df.iloc[-LOOKBACK:]
    payload = {"dates": [str(d.date()) for d in w.index],
               **{c: [round(float(x), 4) for x in w[c]] for c in ("open", "high", "low", "close", "volume")}}
    return _call("/kronos_candles", json.dumps(payload), pred_len, samples)
