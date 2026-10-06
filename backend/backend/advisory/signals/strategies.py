"""Pre-registered strategies (parameters fixed from NSE methodology and the
research reports before any backtest was run; see docs/research/).

Selection with a buffer: a holding is kept while it ranks within `keep`;
empty slots are filled from the top ranks. This is NSE's own device for
cutting turnover (Momentum 30 keeps names within rank 45).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.database.sources.panel import Panel
from backend.advisory.signals import factors as F

Ranker = Callable[[pd.Timestamp, pd.Index], pd.DataFrame]


def buffered_pick(ranked: pd.Index, held: List[str], n: int, keep: int) -> List[str]:
    keep_set = set(ranked[:keep])
    kept = [h for h in held if h in keep_set]
    kept = sorted(kept, key=lambda s: ranked.get_loc(s))[:n]
    for s in ranked:
        if len(kept) >= n:
            break
        if s not in kept:
            kept.append(s)
    return kept


def equal_weight(names: List[str]) -> Dict[str, float]:
    return {s: 1.0 / len(names) for s in names} if names else {}


def inverse_vol_weight(names: List[str], sigma: pd.Series) -> Dict[str, float]:
    inv = (1 / sigma.reindex(names)).dropna()
    return (inv / inv.sum()).to_dict() if len(inv) else {}


class Factory:
    """Builds selectors bound to one panel and benchmark."""

    def __init__(self, panel: Panel, bench: pd.Series, universe_size: int = 200):
        self.panel, self.bench, self.universe_size = panel, bench, universe_size
        self._uni: Dict[pd.Timestamp, pd.Index] = {}

    def universe(self, asof: pd.Timestamp) -> pd.Index:
        if asof not in self._uni:
            self._uni[asof] = F.liquid_universe(self.panel.close, self.panel.volume, asof,
                                                top=self.universe_size)
        return self._uni[asof]

    def momentum(self, asof: pd.Timestamp) -> pd.DataFrame:
        return F.momentum_score(self.panel.close, asof, self.universe(asof))

    def lowvol(self, asof: pd.Timestamp) -> pd.DataFrame:
        return F.lowvol_score(self.panel.close, asof, self.universe(asof))

    def equal_weight_universe(self) -> Callable:
        return lambda asof, held: equal_weight(list(self.universe(asof)))

    def top_n(self, ranker: Ranker, n: int, keep: int, weighting: str = "equal") -> Callable:
        def select(asof: pd.Timestamp, held: List[str]) -> Dict[str, float]:
            ranked = ranker(asof)
            names = buffered_pick(ranked.index, held, n, keep)
            if weighting == "inverse_vol":
                return inverse_vol_weight(names, F.volatility(self.panel.close, asof))
            return equal_weight(names)
        return select

    def blend(self, parts: List[tuple]) -> Callable:
        """parts: [(selector, weight)]; held names are shared across sleeves."""
        def select(asof: pd.Timestamp, held: List[str]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for sel, w in parts:
                for s, x in sel(asof, held).items():
                    out[s] = out.get(s, 0.0) + w * x
            return out
        return select


# --- exposure overlays -----------------------------------------------------------

def trend_filter(bench: pd.Series, months: int = 10) -> Callable[[pd.Timestamp], float]:
    """Faber: invested when the benchmark's month-end close is above its 10-month SMA."""
    me = bench.loc[F.month_ends(bench.index)]
    sma = me.rolling(months).mean()

    def expo(asof: pd.Timestamp) -> float:
        past = me.loc[:asof]
        if len(past) < months:
            return 1.0
        return 1.0 if past.iloc[-1] > sma.loc[past.index[-1]] else 0.0
    return expo


def vol_target(bench: pd.Series, target: float = 0.15, half_life: int = 20,
               cap: float = 1.0) -> Callable[[pd.Timestamp], float]:
    """Harvey et al.: exposure = target / EWMA vol of the benchmark, capped."""
    r = np.log(bench / bench.shift(1))
    sigma = np.sqrt((r ** 2).ewm(halflife=half_life).mean() * F.YEAR)

    def expo(asof: pd.Timestamp) -> float:
        s = sigma.loc[:asof].iloc[-1]
        return float(min(cap, target / s)) if s > 0 else 1.0
    return expo


def momentum_crash_guard(bench: pd.Series, cut: float = 0.5) -> Callable[[pd.Timestamp], float]:
    """Daniel–Moskowitz: momentum crashes come in rebounds after bear markets.
    Cut exposure when the benchmark's 24-month return is negative and its
    realised volatility is in the top quintile of its history so far."""
    r = np.log(bench / bench.shift(1))
    vol = r.rolling(126).std() * np.sqrt(F.YEAR)

    def expo(asof: pd.Timestamp) -> float:
        b = bench.loc[:asof]
        v = vol.loc[:asof].dropna()
        if len(b) < 2 * F.YEAR or len(v) < F.YEAR:
            return 1.0
        bear = b.iloc[-1] / b.iloc[-2 * F.YEAR] - 1 < 0
        high_vol = v.iloc[-1] >= v.quantile(0.8)
        return cut if bear and high_vol else 1.0
    return expo


def combine(*overlays: Callable[[pd.Timestamp], float]) -> Callable[[pd.Timestamp], float]:
    return lambda asof: float(np.prod([o(asof) for o in overlays]))


def schedule(index: pd.DatetimeIndex, months: Optional[List[int]] = None) -> List[pd.Timestamp]:
    """Month-end signal dates, optionally restricted to given calendar months."""
    me = F.month_ends(index)
    return [d for d in me if months is None or d.month in months]
