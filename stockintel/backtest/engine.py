"""Backtesting engine.

Execution model: a target exposure decided at the close of day t is filled at
the open of t+1 and held open-to-open, so a signal can never earn the return
of the bar that produced it. Costs (brokerage/taxes + slippage) are charged
on every change in exposure. Optional stop-loss is checked against each
day's low (high for shorts); a gap through the stop fills at the open, and
after a stop the strategy stays flat until its own target returns to zero.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ANNUALIZATION_DAYS as A, SLIPPAGE_BPS, TRANSACTION_COST_BPS


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = TRANSACTION_COST_BPS
    slippage_bps: float = SLIPPAGE_BPS
    stop_loss_pct: Optional[float] = None   # e.g. 0.08 = exit 8% beyond entry
    capital: float = 100_000.0
    max_adv_participation: float = 0.01     # flag days above 1% of median traded value
    risk_free_rate: float = 0.0


@dataclass
class BacktestResult:
    name: str
    start: str
    end: str
    metrics: Dict[str, float]
    trades: List[Dict[str, object]]
    warnings: List[str] = field(default_factory=list)
    equity: Optional[pd.Series] = None

    def to_dict(self, with_equity: bool = False) -> Dict[str, object]:
        d = asdict(self)
        d["equity"] = ({str(k.date()): round(float(v), 5) for k, v in self.equity.items()}
                       if with_equity and self.equity is not None else None)
        return d


def run(df: pd.DataFrame, target: pd.Series, name: str, cfg: BacktestConfig = BacktestConfig(),
        start: Optional[str] = None) -> BacktestResult:
    target = target.reindex(df.index).fillna(0.0).clip(-1.0, 1.0)
    desired_all = target.shift(1).fillna(0.0)       # decided at close t-1, filled at open t
    oo_all = df["open"].shift(-1) / df["open"] - 1.0
    keep = np.ones(len(df), dtype=bool) if start is None else (df.index >= pd.Timestamp(start))
    keep[-1] = False                                # last bar has no next open
    idx = df.index[keep]
    if len(idx) < 2:
        raise ValueError(f"backtest window for {name} has {len(idx)} bars; need at least 2")
    desired = desired_all[keep].to_numpy()
    oo = oo_all[keep].to_numpy()
    opens, lows, highs = (df[c][keep].to_numpy() for c in ("open", "low", "high"))
    unit_cost = (cfg.cost_bps + cfg.slippage_bps) / 1e4

    net = np.zeros(len(idx))
    held_arr = np.zeros(len(idx))
    turnover = np.zeros(len(idx))
    trades: List[Dict[str, object]] = []
    held, entry_px, stopped = 0.0, None, False
    trade_start, trade_growth = None, 1.0
    for i in range(len(idx)):
        want = desired[i]
        if stopped:
            if want == 0:
                stopped = False
            want = 0.0
        turn = abs(want - held)
        if want != 0 and (held == 0 or np.sign(want) != np.sign(held)):
            entry_px = opens[i]
        held = want
        ret = held * oo[i]
        stop_hit = False
        if cfg.stop_loss_pct and held != 0 and entry_px is not None:
            level = entry_px * (1 - cfg.stop_loss_pct * np.sign(held))
            if (held > 0 and lows[i] <= level) or (held < 0 and highs[i] >= level):
                fill = min(opens[i], level) if held > 0 else max(opens[i], level)
                ret = held * (fill / opens[i] - 1.0)
                turn += abs(held)
                stop_hit = True
        net[i] = ret - turn * unit_cost
        turnover[i] = turn
        held_arr[i] = held
        if held != 0 and trade_start is None:
            trade_start, trade_growth = i, 1.0
        if trade_start is not None:
            trade_growth *= 1.0 + net[i]
            next_want = desired[i + 1] if i + 1 < len(idx) else 0.0
            if stop_hit or i == len(idx) - 1 or next_want == 0 or np.sign(next_want) != np.sign(held):
                trades.append({"entry": str(idx[trade_start].date()), "exit": str(idx[i].date()),
                               "bars": i - trade_start + 1,
                               "return_pct": round((trade_growth - 1) * 100, 3),
                               "stopped": stop_hit})
                trade_start = None
        if stop_hit:
            held, entry_px, stopped = 0.0, None, True

    equity = pd.Series(np.cumprod(1.0 + net), index=idx)
    net_s = pd.Series(net, index=idx)
    years = len(net) / A
    total = float(equity.iloc[-1])
    rf_d = cfg.risk_free_rate / A
    sd = float(net_s.std(ddof=1))
    downside = float(np.sqrt((np.minimum(net - rf_d, 0) ** 2).mean()) * math.sqrt(A))
    dd = float((equity / equity.cummax() - 1).min())
    cagr = total ** (1 / years) - 1 if total > 0 else -1.0
    wins = [t["return_pct"] for t in trades if t["return_pct"] > 0]
    losses = [t["return_pct"] for t in trades if t["return_pct"] <= 0]
    metrics = {
        "cagr_pct": round(cagr * 100, 2), "total_return_pct": round((total - 1) * 100, 2),
        "sharpe": round((net.mean() - rf_d) / sd * math.sqrt(A), 3) if sd > 1e-12 else 0.0,
        "sortino": round((net.mean() - rf_d) * A / downside, 3) if downside > 1e-12 else 0.0,
        "max_drawdown_pct": round(dd * 100, 2),
        "calmar": round(cagr / abs(dd), 3) if dd < 0 else 0.0,
        "volatility_pct": round(sd * math.sqrt(A) * 100, 2),
        "win_rate": round(len(wins) / len(trades), 3) if trades else 0.0,
        "profit_factor": (round(sum(wins) / abs(sum(losses)), 3) if losses and sum(losses) < 0
                          else None),
        "avg_trade_return_pct": round(float(np.mean([t["return_pct"] for t in trades])), 3)
        if trades else 0.0,
        "trade_count": len(trades),
        "turnover_per_year": round(float(turnover.sum()) / years, 2),
        "exposure_pct": round(float((held_arr != 0).mean()) * 100, 1),
        "costs_paid_pct": round(float(turnover.sum() * unit_cost) * 100, 3),
    }
    warnings: List[str] = []
    adv = (df["close"] * df["volume"]).rolling(20).median()[keep].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        part = cfg.capital * np.abs(held_arr) / adv
    over = int(np.nansum(part > cfg.max_adv_participation))
    if over:
        warnings.append(f"position size exceeded {cfg.max_adv_participation:.0%} of median daily "
                        f"traded value on {over} days; fills may be unrealistic")
    return BacktestResult(name=name, start=str(idx[0].date()), end=str(idx[-1].date()),
                          metrics=metrics, trades=trades, warnings=warnings, equity=equity)
