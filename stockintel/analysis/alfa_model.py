"""The return generator from the ALFA project: a NumPy transformer that draws possible
daily-return paths. ALFA's own loader (backend/models/scenarios.py) serves it only while its
saved record shows it beating GARCH(1,1)-t, and otherwise lets GARCH draw.

Set STOCKINTEL_ALFA_PATH to the ALFA project folder (it needs backend/, selfagent/ and
artifacts/returns.npz). The model Space carries the same three under alfa/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..data.provider import DataUnavailable
from .tsfm import QUANTILES

PATH = os.environ.get("STOCKINTEL_ALFA_PATH", "")
WEIGHTS = "artifacts/returns.npz"
STEPS, PATHS = 10, 500
_LOADED = None


def available() -> bool:
    return bool(PATH) and (Path(PATH) / WEIGHTS).exists()


def _loaded():
    global _LOADED
    if _LOADED is None:
        if not available():
            raise DataUnavailable(f"ALFA return model not found: set STOCKINTEL_ALFA_PATH to the ALFA folder "
                                  f"(now {PATH!r}, needs {WEIGHTS})")
        if PATH not in sys.path:
            sys.path.insert(0, PATH)
        from backend.models import scenarios
        _LOADED = scenarios, scenarios.load(str(Path(PATH) / WEIGHTS))
    return _LOADED


def fan(close: pd.Series, steps: int = STEPS, count: int = PATHS, seed: int = 0) -> Dict[str, Any]:
    """Price quantiles per day ahead from `count` sampled paths (the chart's fan levels), with
    the drawer's own record."""
    scenarios, drawer = _loaded()
    c = close.dropna().to_numpy(dtype=float)
    if len(c) < drawer.bars_needed:
        raise DataUnavailable(f"only {len(c)} closes; the return model needs {drawer.bars_needed}")
    drawn = drawer.paths(c, steps, count, np.random.default_rng(seed))
    prices = scenarios.priced(float(c[-1]), drawn)
    return {**{f"q{int(l * 100)}": [round(float(v), 2) for v in np.quantile(prices, l, axis=0)] for l in QUANTILES},
            "drawer": drawer.describe(), "direction_note": scenarios.DIRECTION}
