"""Cross-sectional factor scores, computed point-in-time.

Formulas follow NSE's published index methodology (docs/research/01_factor_evidence.md),
so parameters are fixed in advance rather than tuned on our own backtest:
  momentum  NSE Nifty200 Momentum 30: z-scores of R12/σ and R6/σ, 50/50, then 1+Z or 1/(1−Z)
  low vol   NSE Nifty100 Low Volatility 30: 1-year σ of daily log returns (lower is better)
  alpha     NSE Nifty Alpha 50: 1-year Jensen alpha against the benchmark
Every function takes `asof` and uses only prices on or before it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

YEAR = 252


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in the index."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def _window(close: pd.DataFrame, asof: pd.Timestamp, n: int) -> pd.DataFrame:
    return close.loc[:asof].iloc[-n:]


def volatility(close: pd.DataFrame, asof: pd.Timestamp, n: int = YEAR) -> pd.Series:
    w = _window(close, asof, n + 1)
    lr = np.log(w / w.shift(1)).iloc[1:]
    enough = lr.notna().sum() >= int(0.9 * n)
    return (lr.std(ddof=1) * np.sqrt(YEAR)).where(enough)


def _zscore(x: pd.Series) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd and sd > 0 else x * 0.0


def nse_transform(z: pd.Series) -> pd.Series:
    return pd.Series(np.where(z >= 0, 1 + z, 1 / (1 - z)), index=z.index)


def momentum_score(close: pd.DataFrame, asof: pd.Timestamp,
                   universe: Optional[pd.Index] = None) -> pd.DataFrame:
    """NSE momentum score on month-end prices ending at the month before `asof`'s month."""
    hist = close.loc[:asof]
    me = month_ends(hist.index)
    if len(me) < 14:
        return pd.DataFrame(columns=["r6", "r12", "sigma", "z", "score"])
    # NSE uses P(M−1)/P(M−13): the latest completed month-end is M−1.
    p1, p7, p13 = (hist.loc[me[-1]], hist.loc[me[-7]], hist.loc[me[-13]])
    sigma = volatility(close, asof)
    df = pd.DataFrame({"r6": p1 / p7 - 1, "r12": p1 / p13 - 1, "sigma": sigma})
    if universe is not None:
        df = df.reindex(universe)
    df = df.dropna()
    df = df[df["sigma"] > 0]
    if df.empty:
        return df.assign(z=[], score=[])
    z = 0.5 * _zscore(df["r12"] / df["sigma"]) + 0.5 * _zscore(df["r6"] / df["sigma"])
    df["z"] = z
    df["score"] = nse_transform(z)
    return df.sort_values("score", ascending=False)


def lowvol_score(close: pd.DataFrame, asof: pd.Timestamp,
                 universe: Optional[pd.Index] = None) -> pd.DataFrame:
    sigma = volatility(close, asof)
    if universe is not None:
        sigma = sigma.reindex(universe)
    df = pd.DataFrame({"sigma": sigma.dropna()})
    df = df[df["sigma"] > 0]
    df["score"] = 1 / df["sigma"]
    return df.sort_values("score", ascending=False)


def alpha_score(close: pd.DataFrame, bench: pd.Series, asof: pd.Timestamp, rf: float = 0.065,
                universe: Optional[pd.Index] = None) -> pd.DataFrame:
    w = _window(close, asof, YEAR + 1)
    r = w.pct_change().iloc[1:]
    b = bench.loc[:asof].pct_change().reindex(r.index)
    rf_d = rf / YEAR
    bx = b - rf_d
    var_b = bx.var(ddof=1)
    rows = {}
    cols = universe if universe is not None else r.columns
    for s in cols:
        if s not in r.columns:
            continue
        y = r[s]
        ok = y.notna() & bx.notna()
        if ok.sum() < int(0.9 * YEAR) or var_b == 0:
            continue
        yx = y[ok] - rf_d
        beta = float(np.cov(yx, bx[ok], ddof=1)[0, 1] / bx[ok].var(ddof=1))
        rows[s] = {"beta": beta, "alpha": float((yx.mean() - beta * bx[ok].mean()) * YEAR)}
    df = pd.DataFrame.from_dict(rows, orient="index")
    if df.empty:
        return df
    df["score"] = df["alpha"]
    return df.sort_values("score", ascending=False)


def liquid_universe(close: pd.DataFrame, volume: pd.DataFrame, asof: pd.Timestamp, top: int = 200,
                    min_history: int = YEAR + 30) -> pd.Index:
    """Point-in-time proxy for an index universe: the `top` most-traded stocks by
    6-month median daily traded value among those with at least a year of history.
    Uses only data up to `asof`, so no future index membership leaks in."""
    hist = close.loc[:asof]
    listed = hist.notna().sum() >= min_history
    traded = (hist * volume.loc[:asof]).iloc[-126:].median()
    traded = traded[listed & hist.iloc[-1].notna()]
    return traded.sort_values(ascending=False).index[:top]
