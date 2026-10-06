"""The chart's dashed GRU line: models/networks/path_gru.py, served with the record it was saved with."""
from backend.database.sources.provider import DataUnavailable
from backend.models.core.backend import xp
from backend.models.data import returns
from backend.models.networks import path_gru as G
from backend.paths import ARTIFACTS

WEIGHTS = ARTIFACTS / "path_gru.npz"
_LOADED = None


def available() -> bool:
    return WEIGHTS.exists()


def _loaded():
    global _LOADED
    if _LOADED is None:
        if not available():
            raise DataUnavailable(f"GRU weights not found at {WEIGHTS}; train them with "
                                  "python -m backend.models.training.train_path_gru")
        _LOADED = G.load(WEIGHTS)
    return _LOADED


def line(closes, market):
    """Projected prices for the next HORIZON days from aligned close arrays (stock, Nifty)."""
    model, record = _loaded()
    c, m = xp.asarray(closes, dtype=float), xp.asarray(market, dtype=float)
    if len(c) < returns.SCALE_BARS + G.WINDOW + 1:
        raise DataUnavailable(f"only {len(c)} closes; the GRU needs {returns.SCALE_BARS + G.WINDOW + 1}")
    r, rm = returns.log_returns(c), returns.log_returns(m)
    window, scale = G.window_at(r, rm, len(r) - 1)
    predicted = model(window[None]).data[0]
    return {"close": [round(float(v), 2) for v in G.to_prices(c[-1], predicted, scale)], "record": record}
