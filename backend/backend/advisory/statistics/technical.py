"""Technical analysis engine: turns raw indicator values into structured,
directional observations with provenance. Every claim carries the number it
was derived from; the summary score is a weighted vote of the observations.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backend.config import THRESHOLDS
from backend.advisory.fusion.evidence import DomainResult, Evidence, Provenance, clamp, unavailable
from backend.advisory.statistics import indicators as ind

DOMAIN = "technical"


def _prov(name: str, as_of: str) -> Provenance:
    return Provenance(source=f"computed:{name}", as_of=as_of)


def support_resistance(df: pd.DataFrame, order: int = None,
                       cluster_pct: float = None) -> Tuple[List[float], List[float]]:
    """Cluster pivot lows/highs into support/resistance zones near the price."""
    order = order or THRESHOLDS.pivot_order
    cluster_pct = cluster_pct or THRESHOLDS.sr_cluster_pct
    highs, lows = df["high"], df["low"]
    piv_hi, piv_lo = [], []
    values_hi, values_lo = highs.to_numpy(), lows.to_numpy()
    n = len(df)
    for i in range(order, n - order):
        window_hi = values_hi[i - order:i + order + 1]
        window_lo = values_lo[i - order:i + order + 1]
        if values_hi[i] == window_hi.max():
            piv_hi.append(float(values_hi[i]))
        if values_lo[i] == window_lo.min():
            piv_lo.append(float(values_lo[i]))

    def cluster(levels: List[float]) -> List[float]:
        zones: List[List[float]] = []
        for lvl in sorted(levels):
            if zones and abs(lvl - zones[-1][-1]) / zones[-1][-1] * 100 <= cluster_pct:
                zones[-1].append(lvl)
            else:
                zones.append([lvl])
        # A level touched more often is a stronger zone; keep multi-touch zones first.
        zones.sort(key=len, reverse=True)
        return [float(np.mean(z)) for z in zones[:5]]

    price = float(df["close"].iloc[-1])
    supports = sorted([z for z in cluster(piv_lo) if z < price], reverse=True)[:3]
    resistances = sorted([z for z in cluster(piv_hi) if z > price])[:3]
    return supports, resistances


def trend_structure(close: pd.Series, order: int = None) -> str:
    """Classify by the last two swing highs/lows: HH+HL=uptrend, LH+LL=downtrend."""
    order = order or THRESHOLDS.pivot_order
    values = close.to_numpy()
    n = len(values)
    piv_hi_idx = [i for i in range(order, n - order)
                  if values[i] == values[i - order:i + order + 1].max()]
    piv_lo_idx = [i for i in range(order, n - order)
                  if values[i] == values[i - order:i + order + 1].min()]
    if len(piv_hi_idx) < 2 or len(piv_lo_idx) < 2:
        return "undetermined"
    hh = values[piv_hi_idx[-1]] > values[piv_hi_idx[-2]]
    hl = values[piv_lo_idx[-1]] > values[piv_lo_idx[-2]]
    if hh and hl:
        return "uptrend"
    if not hh and not hl:
        return "downtrend"
    return "sideways"


def analyze(df: pd.DataFrame) -> DomainResult:
    if df is None or len(df) < 60:
        return unavailable(DOMAIN, "need at least 60 bars for technical analysis")

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    as_of = str(df.index[-1].date())
    price = float(close.iloc[-1])
    ev: List[Evidence] = []
    details: Dict[str, object] = {"price": price, "as_of": as_of}

    def add(name: str, claim: str, direction: int, strength: float, value=None):
        ev.append(Evidence(domain=DOMAIN, claim=claim, direction=direction,
                           strength=strength, provenance=_prov(name, as_of),
                           value=None if value is None else float(value)))

    # Trend: price vs long moving averages plus swing structure.
    sma50 = ind.sma(close, 50)
    sma200 = ind.sma(close, 200) if len(df) >= 200 else None
    structure = trend_structure(close)
    details["trend_structure"] = structure
    above50 = price > float(sma50.iloc[-1])
    slope50 = float(ind.slope(sma50.dropna(), 20).iloc[-1]) if sma50.notna().sum() >= 20 else 0.0
    if structure == "uptrend":
        add("swing_structure", "Swing structure shows higher highs and higher lows.", 1, 0.8)
    elif structure == "downtrend":
        add("swing_structure", "Swing structure shows lower highs and lower lows.", -1, 0.8)
    add("sma(50)", f"Price is {'above' if above50 else 'below'} the 50-day average "
        f"({sma50.iloc[-1]:,.2f}) and the average is {'rising' if slope50 > 0 else 'falling'}.",
        1 if above50 else -1, 0.6, sma50.iloc[-1])
    if sma200 is not None and not np.isnan(sma200.iloc[-1]):
        above200 = price > float(sma200.iloc[-1])
        add("sma(200)", f"Price is {'above' if above200 else 'below'} the 200-day average "
            f"({sma200.iloc[-1]:,.2f}).", 1 if above200 else -1, 0.7, sma200.iloc[-1])
        golden = float(sma50.iloc[-1]) > float(sma200.iloc[-1])
        details["ma_structure"] = "50d above 200d" if golden else "50d below 200d"

    # Momentum
    rsi_now = float(ind.rsi(close).iloc[-1])
    details["rsi_14"] = round(rsi_now, 1)
    if rsi_now >= THRESHOLDS.rsi_overbought:
        add("rsi(14)", f"RSI is {rsi_now:.1f} — momentum strong but statistically overbought.",
            -1, 0.5, rsi_now)
    elif rsi_now <= THRESHOLDS.rsi_oversold:
        add("rsi(14)", f"RSI is {rsi_now:.1f} — oversold territory.", 1, 0.5, rsi_now)
    else:
        add("rsi(14)", f"RSI is {rsi_now:.1f} — neutral momentum zone.",
            1 if rsi_now > 50 else -1, 0.3, rsi_now)

    macd_line, macd_sig, macd_hist = ind.macd(close)
    hist_now = float(macd_hist.iloc[-1])
    details["macd_histogram"] = round(hist_now, 4)
    add("macd(12,26,9)",
        f"MACD histogram is {'positive' if hist_now > 0 else 'negative'} ({hist_now:+.2f}).",
        1 if hist_now > 0 else -1, 0.5, hist_now)

    mom20 = float(close.pct_change(20).iloc[-1])
    details["return_20d_pct"] = round(mom20 * 100, 2)
    add("momentum(20d)", f"20-day return is {mom20 * 100:+.1f}%.",
        1 if mom20 > 0 else -1, min(0.8, abs(mom20) * 8), mom20 * 100)

    # Volatility context
    vol_series = ind.historical_volatility(close).dropna()
    if len(vol_series) > 60:
        vol_now = float(vol_series.iloc[-1])
        vol_pct = float((vol_series < vol_now).mean() * 100)
        details["annualized_vol_pct"] = round(vol_now * 100, 1)
        details["vol_percentile"] = round(vol_pct, 0)
        if vol_pct >= THRESHOLDS.high_vol_percentile:
            add("realized_vol(20d)", f"Realized volatility ({vol_now * 100:.0f}% ann.) is at "
                f"percentile {vol_pct:.0f} of its history — unusually volatile.", -1, 0.4, vol_now * 100)

    upper, mid_bb, lower = ind.bollinger(close)
    if price > float(upper.iloc[-1]):
        add("bollinger(20,2)", "Price closed above the upper Bollinger band — extended move.",
            -1, 0.3, price)
    elif price < float(lower.iloc[-1]):
        add("bollinger(20,2)", "Price closed below the lower Bollinger band — stretched down.",
            1, 0.3, price)

    # Volume behaviour
    vr = ind.volume_ratio(volume)
    vr_now = float(vr.iloc[-1]) if not np.isnan(vr.iloc[-1]) else None
    if vr_now is not None:
        details["volume_vs_20d_avg"] = round(vr_now, 2)
        if vr_now >= THRESHOLDS.volume_spike_mult:
            day_ret = float(close.pct_change().iloc[-1])
            add("volume_ratio(20d)",
                f"Volume is {vr_now:.1f}x its 20-day average on a {day_ret * 100:+.1f}% day.",
                1 if day_ret > 0 else -1, 0.5, vr_now)

    # Support / resistance and breakout status
    supports, resistances = support_resistance(df)
    details["support_levels"] = [round(s, 2) for s in supports]
    details["resistance_levels"] = [round(r, 2) for r in resistances]
    prior_high = float(high.iloc[-21:-1].max())
    breakout = price > prior_high
    details["breakout_status"] = "above 20d high" if breakout else "inside recent range"
    if breakout:
        confirmed = vr_now is not None and vr_now >= 1.5
        add("breakout(20d)", "Price broke above its prior 20-day high"
            + (" with elevated volume." if confirmed else ", but volume did not confirm."),
            1, 0.7 if confirmed else 0.35, prior_high)

    # Extension risk: distance from the 50-day mean in ATR units.
    atr_now = float(ind.atr(high, low, close).iloc[-1])
    if atr_now > 0 and not np.isnan(sma50.iloc[-1]):
        extension = (price - float(sma50.iloc[-1])) / atr_now
        details["extension_atr"] = round(extension, 1)
        if abs(extension) > 4:
            add("extension(atr)", f"Price is {extension:+.1f} ATRs from its 50-day average — "
                "mean-reversion risk elevated.", -1 if extension > 0 else 1, 0.4, extension)

    score = clamp(sum(e.direction * e.strength for e in ev) /
                  max(1.0, sum(e.strength for e in ev)))
    state = ("bullish" if score > 0.25 else "bearish" if score < -0.25 else "neutral")
    return DomainResult(domain=DOMAIN, state=state, score=score, evidence=ev,
                        details=details, as_of=as_of)
