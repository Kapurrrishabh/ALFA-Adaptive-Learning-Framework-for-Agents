"""Strategies expressed as target-exposure series decided at each close.

Baselines use fixed textbook parameters; nothing is tuned on the test data,
so comparing them involves no parameter snooping. The engine-replay
strategies call the live analysis code on truncated histories, so the
backtested logic is exactly the logic the assistant uses today.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..analysis import forecast, indicators as ind, technical
from ..evidence import DomainResult
from .engine import BacktestConfig, BacktestResult, run

Strategy = Callable[[pd.DataFrame], pd.Series]


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    c = df["close"]
    return (ind.sma(c, fast) > ind.sma(c, slow)).astype(float)


def time_series_momentum(df: pd.DataFrame, lookback: int = 126) -> pd.Series:
    return (df["close"].pct_change(lookback) > 0).astype(float)


def _stateful(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    pos, out = 0.0, []
    for e, x in zip(enter.to_numpy(), exit_.to_numpy()):
        if pos == 0 and e:
            pos = 1.0
        elif pos == 1 and x:
            pos = 0.0
        out.append(pos)
    return pd.Series(out, index=enter.index)


def rsi_mean_reversion(df: pd.DataFrame, entry: float = 30, exit_: float = 55) -> pd.Series:
    r = ind.rsi(df["close"])
    return _stateful(r < entry, r > exit_)


def bollinger_reversion(df: pd.DataFrame) -> pd.Series:
    upper, mid, lower = ind.bollinger(df["close"])
    return _stateful(df["close"] < lower, df["close"] > mid)


def replay(df: pd.DataFrame, engine: Callable[[pd.DataFrame], DomainResult], step: int = 5,
           warmup: int = 250) -> pd.Series:
    """Score an engine point-in-time: each value uses only bars up to that date."""
    scores = pd.Series(np.nan, index=df.index)
    for t in range(warmup, len(df), step):
        res = engine(df.iloc[:t + 1])
        scores.iloc[t] = res.score if res.available else 0.0
    return scores.ffill().fillna(0.0)


def technical_engine(df: pd.DataFrame, threshold: float = 0.25, step: int = 5) -> pd.Series:
    return (replay(df, technical.analyze, step) > threshold).astype(float)


def forecast_model(df: pd.DataFrame, horizon: int = 5, model: str = "logistic",
                   threshold: float = 0.55, bench: Optional[pd.DataFrame] = None) -> pd.Series:
    wf = forecast.walk_forward(df, horizon, ("climatology", model), bench)
    if "error" in wf:
        raise ValueError(f"forecast walk-forward failed: {wf['error']}")
    p = wf["predictions"][model].reindex(df.index)
    # Outside the out-of-sample window there is no honest prediction: stay flat.
    return (p > threshold).astype(float).where(p.notna(), 0.0)


BASELINES: Dict[str, Strategy] = {
    "buy_and_hold": buy_and_hold,
    "sma_50_200": sma_crossover,
    "momentum_126d": time_series_momentum,
    "rsi_mean_reversion": rsi_mean_reversion,
    "bollinger_reversion": bollinger_reversion,
}


def compare(df: pd.DataFrame, strategies: Dict[str, Strategy], start: Optional[str] = None,
            cfg: BacktestConfig = BacktestConfig()) -> Dict[str, object]:
    """Run every strategy over the same window; rank by Sharpe."""
    results: List[BacktestResult] = []
    for name, fn in strategies.items():
        results.append(run(df, fn(df), name, cfg, start=start))
    table = sorted(({"strategy": r.name, **r.metrics, "warnings": r.warnings} for r in results),
                   key=lambda row: row["sharpe"], reverse=True)
    bh = next((r for r in results if r.name == "buy_and_hold"), None)
    return {"window": f"{results[0].start} to {results[0].end}", "table": table,
            "benchmark": "buy_and_hold" if bh else None,
            "caveat": (f"{len(results)} strategies compared on one path; the best of N is biased "
                       "upward (data snooping). Treat differences under ~0.3 Sharpe as noise "
                       "unless they hold across stocks and sub-periods."),
            "results": results}


def price_fusion(df: pd.DataFrame, bench: Optional[pd.DataFrame] = None, step: int = 5,
                 warmup: int = 300) -> pd.Series:
    """Replay the fused score of the price-derived domains point-in-time and
    hold the stock while it clears the WATCH band. This is the part of the
    decision engine that can be tested historically without look-ahead."""
    from ..analysis import candlesticks, patterns, quant, regime
    from ..config import DECISION
    from ..fusion import fuse

    target = pd.Series(np.nan, index=df.index)
    for t in range(warmup, len(df), step):
        view = df.iloc[:t + 1]
        bview = None if bench is None else bench.loc[:view.index[-1]]
        tech = technical.analyze(view)
        results = {"technical": tech,
                   "candlestick": candlesticks.analyze(view, tech.details.get("support_levels", []),
                                                       tech.details.get("resistance_levels", [])),
                   "pattern": patterns.analyze(view),
                   "risk": quant.analyze(view, bview, 0.0),
                   "regime": regime.analyze(view, bview)}
        target.iloc[t] = float(fuse(results).score >= DECISION.mild)
    return target.ffill().fillna(0.0)
