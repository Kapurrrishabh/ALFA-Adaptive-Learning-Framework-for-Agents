"""Model Space for the ALFA app, called by backend/models/external/remote_models.py:
- Chronos price ranges and Kronos candles on free ZeroGPU, with fine-tuned weights pulled from the
  Hub repos named in STOCKINTEL_CHRONOS_MODEL / STOCKINTEL_KRONOS_MODEL;
- the language model that rewrites the analyst note (backend/models/external/language.py), on ZeroGPU;
- ALFA's return generator (NumPy, CPU), its weights from ALFA_MODELS_REPO.
"""
import json
import os

os.environ["STOCKINTEL_KRONOS_PATH"] = "/tmp/kronos"
os.environ.setdefault("STOCKINTEL_TORCH_DEVICE", "cuda" if os.environ.get("SPACES_ZERO_GPU") else "cpu")

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
import spaces  # noqa: E402
from backend.models.external import kronos_model, language, tsfm  # noqa: E402
from backend.models.serving import alfa_fan as alfa_model, weights  # noqa: E402

kronos_model.fetch_code()
if weights.repo():
    weights.pull([alfa_model.WEIGHTS])
# ZeroGPU wants the models placed at import time, not inside the GPU functions
tsfm.pipeline()
kronos_model.predictor()
WRITER = language.OpenLM()
WRITER.load()


@spaces.GPU(duration=30)
def chronos_fan(closes: str, horizon: float) -> str:
    return json.dumps(tsfm.fan(pd.Series(json.loads(closes), dtype=float), int(horizon)))


@spaces.GPU(duration=60)
def kronos_candles(ohlcv: str, pred_len: float, samples: float) -> str:
    d = json.loads(ohlcv)
    df = pd.DataFrame({c: d[c] for c in ("open", "high", "low", "close", "volume")}, index=pd.to_datetime(d["dates"]))
    return json.dumps(kronos_model.next_candles(df, int(pred_len), int(samples)))


@spaces.GPU(duration=30)
def write(system: str, messages: str) -> str:
    reply = WRITER.create(system, json.loads(messages))
    return json.dumps({"model": reply.model, "text": reply.content})


def return_paths(closes: str, steps: float, count: float) -> str:
    if not alfa_model.available():
        raise gr.Error("ALFA return model not installed: set ALFA_MODELS_REPO and restart the Space")
    return json.dumps(alfa_model.fan(pd.Series(json.loads(closes), dtype=float), int(steps), int(count)), default=float)


with gr.Blocks(title="ALFA models") as demo:
    gr.Markdown("API for the ALFA app: `/chronos_fan`, `/kronos_candles`, `/return_paths` and `/write`.")
    closes, horizon, fan_out = gr.Textbox(label="closes, JSON list"), gr.Number(10, label="horizon (days)"), gr.Textbox(label="quantiles, JSON")
    gr.Button("Chronos").click(chronos_fan, [closes, horizon], fan_out, api_name="chronos_fan")
    ohlcv, pred_len, samples, candles_out = (gr.Textbox(label="OHLCV, JSON"), gr.Number(5, label="days"),
                                             gr.Number(8, label="samples"), gr.Textbox(label="candles, JSON"))
    gr.Button("Kronos").click(kronos_candles, [ohlcv, pred_len, samples], candles_out, api_name="kronos_candles")
    rcloses, rsteps, rcount, rout = (gr.Textbox(label="closes, JSON list"), gr.Number(10, label="days"),
                                     gr.Number(500, label="paths"), gr.Textbox(label="fan, JSON"))
    gr.Button("ALFA return model").click(return_paths, [rcloses, rsteps, rcount], rout, api_name="return_paths")
    system, messages, note = gr.Textbox(label="system prompt"), gr.Textbox(label="messages, JSON"), gr.Textbox(label="reply, JSON")
    gr.Button("Write").click(write, [system, messages], note, api_name="write")

demo.launch()
