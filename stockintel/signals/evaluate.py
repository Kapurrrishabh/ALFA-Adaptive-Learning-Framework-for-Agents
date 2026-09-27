"""Pre-registered evaluation of the cross-sectional strategies.

The variant list is fixed here, before results are seen, so the comparison
is not a search over parameters. Every variant runs on the same dates.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..data.panel import Panel
from . import strategies as S
from .simulate import SimResult, metrics, simulate, yearly

SEMIANNUAL = [5, 11]       # NSE rebalances Jun/Dec on end-May/end-Nov data
QUARTERLY = [2, 5, 8, 11]


def variants(fac: S.Factory, bench: pd.Series, n: int = 20) -> Dict[str, dict]:
    mom = fac.top_n(fac.momentum, n, 2 * n)
    low = fac.top_n(fac.lowvol, n, 2 * n, weighting="inverse_vol")
    half_mom = fac.top_n(fac.momentum, n // 2, n)
    half_low = fac.top_n(fac.lowvol, n // 2, n)
    overlay = S.combine(S.trend_filter(bench), S.vol_target(bench))
    return {
        "equal_weight_universe": dict(sel=fac.equal_weight_universe(), months=None),
        "momentum_semiannual": dict(sel=mom, months=SEMIANNUAL),
        "momentum_monthly": dict(sel=mom, months=None),
        "lowvol_quarterly": dict(sel=low, months=QUARTERLY),
        "mom_lowvol_50_50": dict(sel=fac.blend([(half_mom, 0.5), (half_low, 0.5)]), months=SEMIANNUAL),
        "momentum_crash_guard": dict(sel=mom, months=SEMIANNUAL, exposure=S.momentum_crash_guard(bench)),
        "momentum_trend_vol_overlay": dict(sel=mom, months=None, exposure=overlay),
        "momentum_stop10": dict(sel=mom, months=SEMIANNUAL, stop=0.10),
    }


def run(panel: Panel, bench: pd.Series, start: str, end: Optional[str] = None, n: int = 20,
        only: Optional[List[str]] = None, cost_bps: float = 16.0) -> Dict[str, SimResult]:
    fac = S.Factory(panel, bench)
    out: Dict[str, SimResult] = {}
    for name, v in variants(fac, bench, n).items():
        if only and name not in only:
            continue
        dates = S.schedule(panel.close.index, v["months"])
        out[name] = simulate(panel, name, v["sel"], dates, pd.Timestamp(start),
                             pd.Timestamp(end) if end else None, exposure=v.get("exposure"),
                             stop_loss=v.get("stop"), cost_bps=cost_bps)
    return out


def table(results: Dict[str, SimResult], benchmarks: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    first = next(iter(results.values())).equity
    for name, s in benchmarks.items():
        e = s.reindex(first.index).ffill().dropna()
        rows.append({"strategy": name, **metrics(e / e.iloc[0]), "turnover_yr": 0.0})
    for name, r in results.items():
        rows.append({"strategy": name, **metrics(r.equity), "turnover_yr": round(r.turnover, 2),
                     "st_gain_share": None if r.short_term_gain_share is None
                     else round(r.short_term_gain_share, 2)})
    return pd.DataFrame(rows).set_index("strategy")


def yearly_table(results: Dict[str, SimResult], benchmarks: Dict[str, pd.Series]) -> pd.DataFrame:
    first = next(iter(results.values())).equity
    cols = {n: yearly(s.reindex(first.index).ffill().dropna()) for n, s in benchmarks.items()}
    cols.update({n: yearly(r.equity) for n, r in results.items()})
    return pd.DataFrame(cols).round(1)
