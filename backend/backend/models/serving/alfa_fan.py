"""The chart's fan from ALFA's return generator: a NumPy transformer that draws possible
daily-return paths. ALFA's loader (serving/scenarios.py) serves it only while its saved record
shows it beating GARCH(1,1)-t, and otherwise lets GARCH draw.
"""
from __future__ import annotations

from typing import Any, Dict

from backend.config import FAN_LEVELS, FORECAST_DAYS
from backend.database.sources.provider import DataUnavailable
from backend.models.core.backend import default_rng, xp
from backend.paths import ARTIFACTS

WEIGHTS = ARTIFACTS / "returns.npz"
STEPS, PATHS = FORECAST_DAYS, 500
_LOADED = None


def available() -> bool:
    return WEIGHTS.exists()


def _loaded():
    global _LOADED
    if _LOADED is None:
        if not available():
            raise DataUnavailable(f"ALFA return model not found at {WEIGHTS}; train it with "
                                  "models/training/train_return_generator.py")
        from backend.models.serving import scenarios
        _LOADED = scenarios, scenarios.load(str(WEIGHTS))
    return _LOADED


def fan(close: "pd.Series", steps: int = STEPS, count: int = PATHS, seed: int = 0) -> Dict[str, Any]:
    """Price quantiles per day ahead from `count` sampled paths (the chart's fan levels), with
    the drawer's own record."""
    scenarios, drawer = _loaded()
    c = xp.asarray(close.dropna(), dtype=float)
    if len(c) < drawer.bars_needed:
        raise DataUnavailable(f"only {len(c)} closes; the return model needs {drawer.bars_needed}")
    drawn = drawer.paths(c, steps, count, default_rng(seed))
    prices = scenarios.priced(float(c[-1]), drawn)
    return {**{f"q{int(l * 100)}": [round(float(v), 2) for v in xp.quantile(prices, l, axis=0)] for l in FAN_LEVELS},
            "drawer": drawer.describe(), "direction_note": scenarios.DIRECTION}
