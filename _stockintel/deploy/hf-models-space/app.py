"""StockIntel model Space, called by the StockIntel server (stockintel/analysis/remote_models.py):
- Chronos price ranges and Kronos candles on free ZeroGPU, with fine-tuned weights pulled from the
  Hub repos named in STOCKINTEL_CHRONOS_MODEL / STOCKINTEL_KRONOS_MODEL;
- the ALFA return generator (NumPy, CPU), its code under alfa/ and weights from ALFA_RETURNS_REPO.
"""
import json
import os
import subprocess

KRONOS_DIR, KRONOS_COMMIT = "/tmp/kronos", "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
if not os.path.exists(KRONOS_DIR):         # Kronos is not on PyPI; pin the commit the app was tested with
    subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos", KRONOS_DIR], check=True)
    subprocess.run(["git", "-C", KRONOS_DIR, "checkout", KRONOS_COMMIT], check=True)
os.environ["STOCKINTEL_KRONOS_PATH"] = KRONOS_DIR
os.environ["STOCKINTEL_ALFA_PATH"] = os.path.abspath("alfa")
if os.environ.get("ALFA_RETURNS_REPO"):
    from huggingface_hub import hf_hub_download
    hf_hub_download(os.environ["ALFA_RETURNS_REPO"], "returns.npz", local_dir="alfa/artifacts")
os.environ.setdefault("STOCKINTEL_TORCH_DEVICE", "cuda" if os.environ.get("SPACES_ZERO_GPU") else "cpu")

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
import spaces  # noqa: E402
from stockintel.analysis import alfa_model, kronos_model, tsfm  # noqa: E402

# ZeroGPU wants the models placed at import time, not inside the GPU functions
tsfm.pipeline()
kronos_model.predictor()
ALFA_ON = alfa_model.available()


@spaces.GPU(duration=30)
def chronos_fan(closes: str, horizon: float) -> str:
    return json.dumps(tsfm.fan(pd.Series(json.loads(closes), dtype=float), int(horizon)))


@spaces.GPU(duration=60)
def kronos_candles(ohlcv: str, pred_len: float, samples: float) -> str:
    d = json.loads(ohlcv)
    df = pd.DataFrame({c: d[c] for c in ("open", "high", "low", "close", "volume")}, index=pd.to_datetime(d["dates"]))
    return json.dumps(kronos_model.next_candles(df, int(pred_len), int(samples)))


def return_paths(closes: str, steps: float, count: float) -> str:
    if not ALFA_ON:
        raise gr.Error("ALFA return model not installed: upload alfa/ and set ALFA_RETURNS_REPO")
    return json.dumps(alfa_model.fan(pd.Series(json.loads(closes), dtype=float), int(steps), int(count)), default=float)


with gr.Blocks(title="StockIntel models") as demo:
    gr.Markdown("API for the StockIntel app: `/chronos_fan` (closes → price quantiles) and `/kronos_candles` (OHLCV → next candles).")
    closes, horizon, fan_out = gr.Textbox(label="closes, JSON list"), gr.Number(10, label="horizon (days)"), gr.Textbox(label="quantiles, JSON")
    gr.Button("Chronos").click(chronos_fan, [closes, horizon], fan_out, api_name="chronos_fan")
    ohlcv, pred_len, samples, candles_out = (gr.Textbox(label="OHLCV, JSON"), gr.Number(5, label="days"),
                                             gr.Number(8, label="samples"), gr.Textbox(label="candles, JSON"))
    gr.Button("Kronos").click(kronos_candles, [ohlcv, pred_len, samples], candles_out, api_name="kronos_candles")
    rcloses, rsteps, rcount, rout = (gr.Textbox(label="closes, JSON list"), gr.Number(10, label="days"),
                                     gr.Number(500, label="paths"), gr.Textbox(label="fan, JSON"))
    gr.Button("ALFA return model").click(return_paths, [rcloses, rsteps, rcount], rout, api_name="return_paths")

demo.launch()
