"""Event study: do promoter market purchases predict excess returns on NSE?

Entry is the OPEN of the first trading day after the broadcast day (filings
land mostly after the close). Exit at the close h trading days later. Excess
return = stock return minus the equal-weight average of every stock in the
panel over the identical open→close window, so market moves and the panel's
survivorship tilt largely cancel. Events on the same stock within h days are
de-clustered. Significance uses month-clustered standard errors: events in
one month are averaged first, because they share market shocks.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from ..analysis.stats import two_sided_p
from ..data.panel import Panel

HORIZONS = (5, 20, 60)
ROUND_TRIP_COST = 0.0032   # 16 bps per side, as in the portfolio simulator


def event_returns(panel: Panel, ev: pd.DataFrame, horizon: int) -> pd.DataFrame:
    close, opens = panel.close, panel.open.where(panel.open.notna(), panel.close.shift(1))
    days = close.index
    col = {s: i for i, s in enumerate(close.columns)}
    C, O = close.to_numpy(dtype=float), opens.to_numpy(dtype=float)
    out, last_used = [], {}
    for r in ev.sort_values("day").itertuples(index=False):
        j = col.get(r.symbol)
        if j is None:
            continue
        e = days.searchsorted(r.day, side="right")          # first trading day after the broadcast day
        x = e + horizon - 1
        if x >= len(days):
            continue
        if r.symbol in last_used and e - last_used[r.symbol] < horizon:
            continue
        o, c = O[e, j], C[x, j]
        if not (o > 0 and c > 0):
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            all_r = C[x] / O[e] - 1
        mkt = np.nanmean(all_r[np.isfinite(all_r)])
        last_used[r.symbol] = e
        out.append({"symbol": r.symbol, "day": r.day, "entry": days[e], "ret": c / o - 1,
                    "excess": c / o - 1 - mkt, "value": r.value})
    return pd.DataFrame(out)


def summarize(er: pd.DataFrame, horizon: int) -> Dict[str, object]:
    n = len(er)
    if n < 20:
        return {"horizon": horizon, "events": n, "verdict": "insufficient events"}
    monthly = er.groupby(er["entry"].dt.to_period("M"))["excess"].mean()
    m = len(monthly)
    se = monthly.std(ddof=1) / math.sqrt(m) if m > 1 else float("nan")
    t = float(monthly.mean() / se) if se and se > 0 else 0.0
    by_year = er.groupby(er["entry"].dt.year)["excess"].agg(["mean", "count"])
    return {"horizon": horizon, "events": n, "months": m,
            "mean_excess_pct": round(float(er["excess"].mean()) * 100, 2),
            "median_excess_pct": round(float(er["excess"].median()) * 100, 2),
            "hit_rate": round(float((er["excess"] > 0).mean()), 3),
            "net_of_costs_pct": round((float(er["excess"].mean()) - ROUND_TRIP_COST) * 100, 2),
            "t_month_clustered": round(t, 2), "p_value": round(two_sided_p(t), 4),
            "years_positive": f"{int((by_year['mean'] > 0).sum())}/{len(by_year)}",
            "by_year_pct": {int(y): round(float(v) * 100, 2) for y, v in by_year["mean"].items()}}


def study(panel: Panel, groups: Dict[str, pd.DataFrame], horizons=HORIZONS) -> List[Dict[str, object]]:
    rows = []
    for name, ev in groups.items():
        for h in horizons:
            rows.append({"group": name, **summarize(event_returns(panel, ev, h), h)})
    return rows
