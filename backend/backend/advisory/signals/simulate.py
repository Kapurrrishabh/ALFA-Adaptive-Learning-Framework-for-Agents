"""Share-level portfolio simulator for cross-sectional strategies.

Timing: a strategy sees prices up to a signal date s (a month-end close) and
returns target weights; trades fill at the OPEN of the next trading day. Each
day the portfolio is marked at the close. Costs are charged per side on
traded value. Uninvested cash earns a money-market yield. Optional per-stock
stop-loss: a holding whose low breaches entry × (1 − stop) is sold at
min(open, stop level) and stays in cash until the next rebalance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.database.sources.panel import Panel

YEAR = 252
# PROVISIONAL, from Zerodha's charges page: STT 0.1% + stamp/exchange ≈ 0.11% per side,
# plus ~5 bps spread/impact for liquid large and mid caps.
COST_BPS_PER_SIDE = 16.0
CASH_YIELD = 0.065        # liquid-fund / T-bill proxy; PROVISIONAL

Selector = Callable[[pd.Timestamp, List[str]], Dict[str, float]]
Exposure = Callable[[pd.Timestamp], float]


@dataclass
class SimResult:
    name: str
    equity: pd.Series
    turnover: float                      # average one-way turnover per year, fraction of NAV
    trades: int
    stops: int
    holdings: Dict[pd.Timestamp, Dict[str, float]] = field(default_factory=dict)
    exposure: Dict[pd.Timestamp, float] = field(default_factory=dict)
    short_term_gain_share: Optional[float] = None


def simulate(panel: Panel, name: str, selector: Selector, signal_dates: List[pd.Timestamp],
             start: pd.Timestamp, end: Optional[pd.Timestamp] = None, exposure: Optional[Exposure] = None,
             stop_loss: Optional[float] = None, cost_bps: float = COST_BPS_PER_SIDE,
             cash_yield: float = CASH_YIELD, capital: float = 1.0) -> SimResult:
    close = panel.close.ffill()
    opens = panel.open.where(panel.open.notna(), close.shift(1))
    lows = panel.low.where(panel.low.notna(), close)
    days = close.index[(close.index >= start) & ((close.index <= end) if end is not None else True)]
    execute: Dict[pd.Timestamp, pd.Timestamp] = {}
    for s in signal_dates:
        nxt = close.index[close.index > s]
        if len(nxt) and nxt[0] in days:
            execute[nxt[0]] = s
    cash, shares = capital, {}
    entry: Dict[str, float] = {}
    lots: Dict[str, List[list]] = {}          # [qty, price, date] FIFO for tax analysis
    st_gain = lt_gain = 0.0
    stopped: set = set()
    unit = cost_bps / 1e4
    daily_cash = (1 + cash_yield) ** (1 / YEAR) - 1
    curve, traded_total, n_trades, n_stops = [], 0.0, 0, 0
    held_log, expo_log = {}, {}

    def realise(sym: str, qty: float, px: float, day: pd.Timestamp) -> None:
        nonlocal st_gain, lt_gain
        q = qty
        while q > 1e-12 and lots.get(sym):
            lot = lots[sym][0]
            take = min(lot[0], q)
            gain = take * (px - lot[1])
            if (day - lot[2]).days > 365:
                lt_gain += gain
            else:
                st_gain += gain
            lot[0] -= take
            q -= take
            if lot[0] <= 1e-12:
                lots[sym].pop(0)

    for d in days:
        o, c_, lo = opens.loc[d], close.loc[d], lows.loc[d]
        if d in execute:
            s = execute[d]
            held = [k for k, v in shares.items() if v > 0]
            target = selector(s, held)
            expo = 1.0 if exposure is None else float(np.clip(exposure(s), 0.0, 1.0))
            nav = cash + sum(q * o[k] for k, q in shares.items())
            want = {k: nav * expo * w / o[k] for k, w in target.items() if w > 0 and o[k] > 0}
            for k in set(shares) | set(want):
                delta = want.get(k, 0.0) - shares.get(k, 0.0)
                if abs(delta) * o[k] < 1e-9 * nav:
                    continue
                value = delta * o[k]
                cost = abs(value) * unit
                cash -= value + cost
                traded_total += abs(value) / nav
                n_trades += 1
                if delta > 0:
                    lots.setdefault(k, []).append([delta, o[k], d])
                    entry[k] = o[k] if shares.get(k, 0.0) <= 0 else entry.get(k, o[k])
                else:
                    realise(k, -delta, o[k], d)
                shares[k] = shares.get(k, 0.0) + delta
                if shares[k] <= 1e-12:
                    shares.pop(k)
                    entry.pop(k, None)
            stopped = set()
            held_log[s] = dict(target)
            expo_log[s] = expo
        if stop_loss:
            for k in list(shares):
                level = entry[k] * (1 - stop_loss)
                if lo[k] <= level:
                    px = min(o[k], level)
                    qty = shares.pop(k)
                    value = qty * px
                    cash += value - value * unit
                    traded_total += value / max(1e-12, cash + sum(q * c_[j] for j, q in shares.items()))
                    realise(k, qty, px, d)
                    entry.pop(k, None)
                    stopped.add(k)
                    n_stops += 1
        cash *= 1 + daily_cash
        curve.append(cash + sum(q * c_[k] for k, q in shares.items()))
    equity = pd.Series(curve, index=days) / capital
    years = len(days) / YEAR
    total_gain = st_gain + lt_gain
    return SimResult(name=name, equity=equity, turnover=traded_total / years / 2 if years else 0.0,
                     trades=n_trades, stops=n_stops, holdings=held_log, exposure=expo_log,
                     short_term_gain_share=(st_gain / total_gain) if total_gain > 0 else None)


def metrics(equity: pd.Series, rf: float = CASH_YIELD) -> Dict[str, float]:
    r = equity.pct_change().dropna()
    years = len(r) / YEAR
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else float("nan")
    vol = float(r.std(ddof=1) * math.sqrt(YEAR))
    ex = r - ((1 + rf) ** (1 / YEAR) - 1)
    dd = equity / equity.cummax() - 1
    down = r[r < 0]
    return {"cagr_pct": round(cagr * 100, 2), "vol_pct": round(vol * 100, 2),
            "sharpe": round(float(ex.mean() / ex.std(ddof=1) * math.sqrt(YEAR)), 2) if ex.std() > 0 else 0.0,
            "sortino": round(float(ex.mean() * YEAR / (down.std(ddof=1) * math.sqrt(YEAR))), 2) if len(down) > 1 else 0.0,
            "max_dd_pct": round(float(dd.min()) * 100, 2),
            "calmar": round(cagr / abs(float(dd.min())), 2) if dd.min() < 0 else 0.0,
            "worst_month_pct": round(float(equity.resample("ME").last().pct_change().min()) * 100, 2)}


def yearly(equity: pd.Series) -> pd.Series:
    ye = equity.resample("YE").last()
    first = equity.iloc[0]
    return (ye / ye.shift(1).fillna(first) - 1).rename(lambda t: t.year) * 100
