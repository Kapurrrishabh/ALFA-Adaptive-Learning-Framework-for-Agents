"""The chart's GRU layer: models/networks/path_gru.py, served with the record it was saved with.

The median path is the forecast. The sampled paths show how prices could zigzag around it, each day's
swing drawn at the size the GRU forecast for that day; they are possibilities, not predictions.
"""
from backend.database.sources.provider import DataUnavailable
from backend.models.core.backend import xp
from backend.models.data import returns
from backend.models.networks import path_gru as G
from backend.models.training.calibrate_path_gru import METHOD
from backend.paths import ARTIFACTS

WEIGHTS = ARTIFACTS / "path_gru.npz"
# Three of SIMULATED paths, each picked by a different measure: the one ending nearest the 10th, the 50th and the
# 90th percentile of where all of them end. Random draws would be three unlabelled lines that mean the same thing.
SIMULATED = 1000
SHOWN = ((0.1, "low", "1 in 10 simulated paths ends lower"),
         (0.5, "middle", "half end lower, half higher"),
         (0.9, "high", "1 in 10 simulated paths ends higher"))
_LOADED = (None, None)
# the two-sided 80% multiple of a normal, which the calibration replaces with what 80% of known outcomes needed
NORMAL_Z80 = 1.2816


def available() -> bool:
    return WEIGHTS.exists()


def _loaded():
    """The model and its record, reloaded when the file changes: the VM recalibrates it every day."""
    global _LOADED
    if not available():
        raise DataUnavailable(f"GRU weights not found at {WEIGHTS}; train them with "
                              "python -m backend.models.training.train_path_gru")
    stamp = WEIGHTS.stat().st_mtime_ns
    if _LOADED[0] != stamp:
        _LOADED = (stamp, G.load(WEIGHTS))
    return _LOADED[1]


def line(closes, highs, lows, market):
    """Median and example paths for the next HORIZON days, from aligned daily arrays (stock OHLC, Nifty close)."""
    model, record = _loaded()
    c, m = xp.asarray(closes, dtype=float), xp.asarray(market, dtype=float)
    if len(c) < returns.SCALE_BARS + G.WINDOW + 1:
        raise DataUnavailable(f"only {len(c)} closes; the GRU needs {returns.SCALE_BARS + G.WINDOW + 1}")
    r, rm = returns.log_returns(c), returns.log_returns(m)
    ranges = xp.log(xp.asarray(highs, dtype=float) / xp.asarray(lows, dtype=float))[1:]
    window, scale = G.window_at(r, rm, ranges, len(r) - 1)
    mean, log_sd = (t.data[0] for t in model(window[None].astype("float32")))
    # Trained on likelihood, its 80% range was too wide; scaled by the multiple the calibration fitted on outcomes the
    # weights never trained on (models/training/calibrate_path_gru.py), "1 in 10 ends lower" is true of the low path.
    method = record.get("calibration", {}).get("method")
    if method != METHOD:
        raise DataUnavailable(f"{WEIGHTS} carries the calibration {method!r}, not {METHOD!r}; rerun "
                              "python -m backend.models.training.calibrate_path_gru")
    z = record["interval_z80"][str(G.HORIZON)]
    paths = G.sample_paths(c[-1], mean, log_sd + xp.log(z / NORMAL_Z80), scale, SIMULATED)
    return {"close": [round(float(v), 2) for v in G.median_prices(c[-1], mean, scale)],
            "paths": labelled_paths(paths), "record": record}


def labelled_paths(simulated):
    """The simulated paths ending nearest each SHOWN percentile, with what each one stands for."""
    ends = simulated[:, -1]
    picked = []
    for q, name, meaning in SHOWN:
        target = float(xp.quantile(ends, q))
        path = simulated[int(xp.argmin(xp.abs(ends - target)))]
        picked.append({"name": name, "percentile": int(q * 100), "meaning": meaning,
                       "close": [round(float(v), 2) for v in path]})
    return picked
