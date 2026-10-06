"""Candlestick intelligence: vectorized formation detection plus context.

A formation alone never becomes a signal. Its weight depends on the prior
trend (built into each definition), relative volume, proximity to support or
resistance, subsequent confirmation, and — most importantly — whether the
formation has had any measurable edge on this stock's own history.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import THRESHOLDS
from ..evidence import DomainResult, Evidence, Provenance, clamp, unavailable
from . import indicators as ind
from .validation import edge_multiplier, event_study, with_prior

DOMAIN = "candlestick"
VALIDATION_HORIZON = 5
LOOKBACK_BARS = 3

DIRECTION: Dict[str, int] = {
    "hammer": 1, "inverted_hammer": 1, "bullish_engulfing": 1, "bullish_harami": 1,
    "morning_star": 1, "piercing_line": 1, "three_white_soldiers": 1,
    "hanging_man": -1, "shooting_star": -1, "bearish_engulfing": -1, "bearish_harami": -1,
    "evening_star": -1, "dark_cloud_cover": -1, "three_black_crows": -1,
    "doji": 0,
}

# Bars in each formation, so trend context is measured before the first bar.
LENGTH: Dict[str, int] = {name: 1 for name in DIRECTION}
LENGTH.update({"bullish_engulfing": 2, "bearish_engulfing": 2, "bullish_harami": 2,
               "bearish_harami": 2, "piercing_line": 2, "dark_cloud_cover": 2,
               "morning_star": 3, "evening_star": 3,
               "three_white_soldiers": 3, "three_black_crows": 3})


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: one column per formation, True on the formation's last bar."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0.0, np.nan)
    top, bottom = np.maximum(o, c), np.minimum(o, c)
    upper, lower = h - top, bottom - l
    bull, bear = c > o, c < o
    avg_body = body.rolling(10).mean().shift(1)
    large = body > avg_body
    small = body < 0.5 * avg_body
    sma10 = ind.sma(c, 10)

    def down_before(k: int) -> pd.Series:
        ref = c.shift(k)
        return (ref < c.shift(k + 5)) & (ref < sma10.shift(k))

    def up_before(k: int) -> pd.Series:
        ref = c.shift(k)
        return (ref > c.shift(k + 5)) & (ref > sma10.shift(k))

    mid_prev = (o.shift(1) + c.shift(1)) / 2.0
    mid_prev2 = (o.shift(2) + c.shift(2)) / 2.0
    hammer_shape = ((lower >= 2 * body) & (upper <= 0.15 * rng)
                    & (body >= 0.05 * rng) & (lower >= 0.55 * rng))
    inverted_shape = ((upper >= 2 * body) & (lower <= 0.15 * rng)
                      & (body >= 0.05 * rng) & (upper >= 0.55 * rng))

    out = pd.DataFrame(index=df.index)
    out["doji"] = (body <= THRESHOLDS.doji_body_ratio * rng)
    out["hammer"] = hammer_shape & down_before(1)
    out["hanging_man"] = hammer_shape & up_before(1)
    out["inverted_hammer"] = inverted_shape & down_before(1)
    out["shooting_star"] = inverted_shape & up_before(1)
    out["bullish_engulfing"] = (bear.shift(1, fill_value=False) & bull & (o <= c.shift(1))
                                & (c >= o.shift(1)) & (body > body.shift(1)) & down_before(2))
    out["bearish_engulfing"] = (bull.shift(1, fill_value=False) & bear & (o >= c.shift(1))
                                & (c <= o.shift(1)) & (body > body.shift(1)) & up_before(2))
    out["bullish_harami"] = (bear.shift(1, fill_value=False) & large.shift(1, fill_value=False)
                             & bull & (top <= o.shift(1)) & (bottom >= c.shift(1))
                             & (body < 0.6 * body.shift(1)) & down_before(2))
    out["bearish_harami"] = (bull.shift(1, fill_value=False) & large.shift(1, fill_value=False)
                             & bear & (top <= c.shift(1)) & (bottom >= o.shift(1))
                             & (body < 0.6 * body.shift(1)) & up_before(2))
    out["piercing_line"] = (bear.shift(1, fill_value=False) & large.shift(1, fill_value=False)
                            & bull & (o < c.shift(1)) & (c > mid_prev) & (c < o.shift(1))
                            & down_before(2))
    out["dark_cloud_cover"] = (bull.shift(1, fill_value=False) & large.shift(1, fill_value=False)
                               & bear & (o > c.shift(1)) & (c < mid_prev) & (c > o.shift(1))
                               & up_before(2))
    star_top = np.maximum(o.shift(1), c.shift(1))
    star_bottom = np.minimum(o.shift(1), c.shift(1))
    out["morning_star"] = (bear.shift(2, fill_value=False) & large.shift(2, fill_value=False)
                           & small.shift(1, fill_value=False) & (star_top < c.shift(2) + 0.1 * body.shift(2))
                           & bull & (c > mid_prev2) & down_before(3))
    out["evening_star"] = (bull.shift(2, fill_value=False) & large.shift(2, fill_value=False)
                           & small.shift(1, fill_value=False) & (star_bottom > c.shift(2) - 0.1 * body.shift(2))
                           & bear & (c < mid_prev2) & up_before(3))
    solid = body >= 0.5 * avg_body
    soldier = bull & solid & (upper <= 0.3 * body)
    crow = bear & solid & (lower <= 0.3 * body)
    out["three_white_soldiers"] = (soldier & soldier.shift(1, fill_value=False)
                                   & soldier.shift(2, fill_value=False)
                                   & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
                                   & (o > o.shift(1)) & (o < c.shift(1))
                                   & (o.shift(1) > o.shift(2)) & (o.shift(1) < c.shift(2)))
    out["three_black_crows"] = (crow & crow.shift(1, fill_value=False)
                                & crow.shift(2, fill_value=False)
                                & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
                                & (o < o.shift(1)) & (o > c.shift(1))
                                & (o.shift(1) < o.shift(2)) & (o.shift(1) > c.shift(2)))
    return out.fillna(False).astype(bool)


def _near(level_list: List[float], price: float, atr_now: float) -> Optional[float]:
    for lvl in level_list:
        if abs(price - lvl) <= max(atr_now, price * THRESHOLDS.sr_cluster_pct / 100):
            return lvl
    return None


def analyze(df: pd.DataFrame, supports: List[float], resistances: List[float],
            priors: Optional[Dict[str, Dict[str, object]]] = None) -> DomainResult:
    if df is None or len(df) < 60:
        return unavailable(DOMAIN, "need at least 60 bars for candlestick context")
    det = detect_all(df)
    close, high, low = df["close"], df["high"], df["low"]
    vol_ratio = ind.volume_ratio(df["volume"])
    atr_now = float(ind.atr(high, low, close).iloc[-1])
    n = len(df)
    ev: List[Evidence] = []
    found: List[Dict[str, object]] = []

    for pos in range(n - LOOKBACK_BARS, n):
        date = df.index[pos]
        for name in det.columns:
            if not det[name].iloc[pos]:
                continue
            direction = DIRECTION[name]
            vr = vol_ratio.iloc[pos]
            vol_confirms = bool(not np.isnan(vr) and vr >= 1.5)
            if direction > 0:
                level = _near(supports, float(low.iloc[pos]), atr_now)
            elif direction < 0:
                level = _near(resistances, float(high.iloc[pos]), atr_now)
            else:
                level = None
            after = df.iloc[pos + 1:]
            if after.empty:
                confirmation = "awaiting next session"
            elif direction > 0:
                confirmation = ("confirmed" if (after["close"] > high.iloc[pos]).any()
                                else "not confirmed")
            elif direction < 0:
                confirmation = ("confirmed" if (after["close"] < low.iloc[pos]).any()
                                else "not confirmed")
            else:
                confirmation = "n/a (indecision)"
            hist = with_prior(event_study(close, det[name], VALIDATION_HORIZON, direction),
                              (priors or {}).get(name))
            context = 0.4 + 0.2 * vol_confirms + 0.2 * (level is not None) \
                + 0.2 * (confirmation == "confirmed") - 0.2 * (confirmation == "not confirmed")
            strength = max(0.0, context) * edge_multiplier(hist)
            weak = []
            if not vol_confirms:
                weak.append("volume did not increase")
            if direction != 0 and level is None:
                weak.append("it did not occur near a meaningful "
                            + ("support" if direction > 0 else "resistance") + " zone")
            label = name.replace("_", " ")
            text = f"{label.capitalize()} detected on {date.date()}"
            if direction != 0:
                text += (f"; confirmation is weak because {' and '.join(weak)}" if weak
                         else "; volume and location support it")
            text += f" ({confirmation})."
            if hist.get("events"):
                text += (f" On this stock's history: {hist['events']} occurrences, "
                         f"{VALIDATION_HORIZON}d mean {hist.get('mean_pct', 0):+.2f}% vs "
                         f"baseline {hist['baseline_mean_pct']:+.2f}%")
                pooled = hist.get("pooled")
                text += (f"; across {pooled['stocks']} stocks ({pooled['events']} events): {hist['verdict']}."
                         if pooled else f" — {hist['verdict']}.")
            ev.append(Evidence(domain=DOMAIN, claim=text, direction=direction,
                               strength=round(strength, 3) if direction else 0.0,
                               provenance=Provenance(source=f"computed:candle:{name}",
                                                     as_of=str(date.date()))))
            found.append({"pattern": name, "date": str(date.date()), "direction": direction,
                          "volume_ratio": None if np.isnan(vr) else round(float(vr), 2),
                          "near_level": None if level is None else round(level, 2),
                          "confirmation": confirmation, "historical": hist})

    as_of = str(df.index[-1].date())
    directional = [e for e in ev if e.direction != 0]
    if not directional:
        state = "indecision" if ev else "no recent formation"
        return DomainResult(domain=DOMAIN, state=state, score=0.0, evidence=ev,
                            details={"formations": found}, as_of=as_of, confidence=0.0)
    total = sum(e.strength for e in directional)
    score = clamp(sum(e.direction * e.strength for e in directional) / max(total, 1e-9)) \
        * min(1.0, total)
    state = "bullish formation" if score > 0 else "bearish formation" if score < 0 else "mixed"
    return DomainResult(domain=DOMAIN, state=state, score=score, evidence=ev,
                        details={"formations": found}, as_of=as_of,
                        confidence=min(1.0, total))
