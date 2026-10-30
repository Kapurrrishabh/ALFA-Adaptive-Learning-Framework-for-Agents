"""A mutual fund's record from its NAV history: trailing returns, volatility, worst fall, and a line saying them.

Trailing returns are measured between calendar dates, as fund returns are quoted, not by counting NAVs:
funds publish about 246 NAVs a year (scheme 120599, 2013-2025), so 252 of them reach back a year and a week.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from backend.advisory.statistics.quant import ann_vol, calendar_cagr, max_drawdown

PERIODS = {"1M": (1, False), "6M": (6, False), "1Y": (12, False), "3Y": (36, True), "5Y": (60, True)}
WORDS = {"1M": "month", "6M": "6 months", "1Y": "year", "3Y": "3 years", "5Y": "5 years"}


def trailing_returns(nav: pd.Series) -> Dict[str, Optional[Dict[str, Any]]]:
    """Return over each period ending on the last NAV, from the last NAV on or before its start;
    3 and 5 years as a yearly rate. None where the history does not reach back that far."""
    end = nav.index[-1]
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for label, (months, annualised) in PERIODS.items():
        start = nav.loc[:end - pd.DateOffset(months=months)]
        if start.empty:
            out[label] = None
            continue
        held = nav.loc[start.index[-1]:]
        rate = calendar_cagr(held) if annualised else float(held.iloc[-1] / held.iloc[0]) - 1
        out[label] = {"pct": round(rate * 100, 2), "annualised": annualised, "from": str(start.index[-1].date())}
    return out


def analytics(nav: pd.Series, is_growth: bool) -> Dict[str, Any]:
    """`is_growth`: the NAV keeps the fund's income, so its change is the whole return. Otherwise payouts are left out."""
    end = nav.index[-1]
    year = nav.loc[end - pd.DateOffset(years=1):]
    covers_year = nav.index[0] <= end - pd.DateOffset(years=1)
    return {"as_of": str(end.date()), "since": str(nav.index[0].date()), "returns": trailing_returns(nav),
            "is_total_return": is_growth,
            "volatility_1y_pct": round(ann_vol(year.pct_change().dropna()) * 100, 2) if covers_year else None,
            "max_drawdown": max_drawdown(nav)}


def summary(a: Dict[str, Any]) -> str:
    """Plain sentences from `analytics` output: the longest return it has, the worst fall, and any payouts left out."""
    covered = [p for p in PERIODS if a["returns"][p]]
    dd = a["max_drawdown"]
    fall = (f"its worst fall since {a['since']} was {abs(dd['max_drawdown_pct'])}%, from {dd['peak_date']} to "
            f"{dd['trough_date']}" if dd["max_drawdown_pct"] < 0 else f"it has not fallen below an earlier high since {a['since']}")
    payouts = "" if a["is_total_return"] else (" AMFI does not list this as a Growth option, so any payouts are left "
                                               "out of these figures and show as falls.")
    if not covered:
        return f"Its NAV history starts on {a['since']}, too soon for a 1-month return, and {fall}.{payouts}"
    r = a["returns"][covered[-1]]
    moved = "grew" if r["pct"] >= 0 else "fell"
    rate = " a year" if r["annualised"] else ""
    return f"Over the last {WORDS[covered[-1]]} its NAV {moved} {abs(r['pct'])}%{rate}, and {fall}.{payouts}"
