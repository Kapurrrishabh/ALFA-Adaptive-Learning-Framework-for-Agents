"""Chart / price-pattern recognition from the numeric price series.

Detection works on alternating swing pivots. A pivot at bar i is only known
once `order` further bars exist, so running `detect` on a truncated history
is causal — `scan_history` relies on that to measure each pattern's past edge
without look-ahead. Vision-based detection is a research track (docs/DESIGN.md
§8); it must beat this detector on the same labelled set before it is added.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.config import THRESHOLDS
from backend.advisory.fusion.evidence import DomainResult, Evidence, Provenance, clamp, unavailable
from backend.advisory.statistics import indicators as ind
from backend.advisory.statistics.validation import edge_multiplier, event_study, with_prior

DOMAIN = "pattern"
WINDOW = 120
VALIDATION_HORIZON = 10
SCAN_STEP = 5
FLAT_SLOPE = 0.0006   # per-bar slope (fraction of price) below which a line is "flat"; PROVISIONAL

Pivot = Tuple[int, float, str]  # (bar position, price, "H" | "L")


def pivots(df: pd.DataFrame, order: int = None) -> List[Pivot]:
    order = order or THRESHOLDS.pivot_order
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    raw: List[Pivot] = []
    for i in range(order, len(df) - order):
        is_h = h[i] == h[i - order:i + order + 1].max()
        is_l = l[i] == l[i - order:i + order + 1].min()
        if is_h and is_l:
            first_h = bool(raw) and raw[-1][2] == "L"
            pair = [(i, float(h[i]), "H"), (i, float(l[i]), "L")]
            raw.extend(pair if first_h else pair[::-1])
        elif is_h:
            raw.append((i, float(h[i]), "H"))
        elif is_l:
            raw.append((i, float(l[i]), "L"))
    alt: List[Pivot] = []
    for p in raw:
        if alt and alt[-1][2] == p[2]:
            if (p[2] == "H" and p[1] > alt[-1][1]) or (p[2] == "L" and p[1] < alt[-1][1]):
                alt[-1] = p
        else:
            alt.append(p)
    return alt


def _pct_diff(a: float, b: float) -> float:
    return abs(a - b) / ((a + b) / 2.0) * 100.0


def _pt(df: pd.DataFrame, i: int, price: float, label: str = "") -> list:
    return [str(df.index[int(i)].date()), round(float(price), 2), label]


def _line(df: pd.DataFrame, i0: int, p0: float, i1: int, p1: float, label: str, style: str = "solid") -> dict:
    return {"from": _pt(df, i0, p0), "to": _pt(df, i1, p1), "label": label, "style": style}


def _record(name: str, direction: int, df: pd.DataFrame, start: int, end: int,
            confidence: float, features: List[str], status: str,
            level: Optional[float], target: Optional[float] = None,
            geometry: Optional[Dict[str, list]] = None) -> Dict[str, object]:
    """`geometry` holds what a chart should draw: `points` (the swing points the
    pattern was built from, joined in order) and `lines` (necklines, trendlines)."""
    return {"pattern": name, "direction": direction,
            "geometry": geometry or {"points": [], "lines": []},
            "start_date": str(df.index[start].date()), "end_date": str(df.index[end].date()),
            "confidence": round(float(min(0.95, max(0.05, confidence))), 2),
            "supporting_features": features, "confirmation_status": status,
            "breakout_level": None if level is None else round(float(level), 2),
            "target": None if target is None else round(float(target), 2)}


def _tops_bottoms(df: pd.DataFrame, piv: List[Pivot]) -> List[Dict[str, object]]:
    tol = THRESHOLDS.pattern_tolerance_pct
    close = df["close"].to_numpy()
    n = len(df)
    price = close[-1]
    out: List[Dict[str, object]] = []
    # Only patterns whose last extreme is recent are "current".
    recent = [k for k in range(len(piv)) if piv[k][0] >= n - 40]
    for kind, direction, extreme in (("H", -1, "top"), ("L", 1, "bottom")):
        idx = [k for k in recent if piv[k][2] == kind]
        if not idx:
            continue
        k = idx[-1]
        # Triple first: three extremes within tolerance.
        if k >= 4:
            e1, t1, e2, t2, e3 = piv[k - 4:k + 1]
            extremes = [e1[1], e2[1], e3[1]]
            broken = (price > max(extremes) * (1 + tol / 100) if kind == "H"
                      else price < min(extremes) * (1 - tol / 100))
            if (max(_pct_diff(e1[1], e2[1]), _pct_diff(e2[1], e3[1]), _pct_diff(e1[1], e3[1])) <= tol
                    and e3[0] - e1[0] >= 20 and not broken):
                neck = min(t1[1], t2[1]) if kind == "H" else max(t1[1], t2[1])
                depth = _pct_diff(np.mean([e1[1], e2[1], e3[1]]), neck)
                if depth >= 2 * tol:
                    done = price < neck if kind == "H" else price > neck
                    height = abs(e2[1] - neck)
                    lab = "Top" if kind == "H" else "Bottom"
                    geo = {"points": [_pt(df, e1[0], e1[1], f"{lab} 1"), _pt(df, t1[0], t1[1]),
                                      _pt(df, e2[0], e2[1], f"{lab} 2"), _pt(df, t2[0], t2[1]),
                                      _pt(df, e3[0], e3[1], f"{lab} 3")],
                           "lines": [_line(df, e1[0], neck, n - 1, neck, "Neckline")]}
                    out.append(_record(
                        f"triple_{extreme}", direction, df, e1[0], n - 1,
                        0.55 + 0.25 * done, [f"three {extreme}s within {tol}%",
                                             f"depth {depth:.1f}%"],
                        "confirmed (neckline broken)" if done else "forming (neckline intact)",
                        neck, neck - height if kind == "H" else neck + height, geo))
                    continue
        if k >= 2:
            e1, t, e2 = piv[k - 2:k + 1]
            diff = _pct_diff(e1[1], e2[1])
            depth = _pct_diff((e1[1] + e2[1]) / 2, t[1])
            lead = max(0, e1[0] - 20)
            rose_into = close[lead] < t[1] if kind == "H" else close[lead] > t[1]
            invalid = (close[e2[0]:] > max(e1[1], e2[1]) * (1 + tol / 100)).any() if kind == "H" \
                else (close[e2[0]:] < min(e1[1], e2[1]) * (1 - tol / 100)).any()
            if diff <= tol and depth >= 2 * tol and e2[0] - e1[0] >= 10 and rose_into and not invalid:
                neck = t[1]
                done = price < neck if kind == "H" else price > neck
                height = abs((e1[1] + e2[1]) / 2 - neck)
                lab = "Top" if kind == "H" else "Bottom"
                geo = {"points": [_pt(df, lead, close[lead], "Start"), _pt(df, e1[0], e1[1], f"{lab} 1"),
                                  _pt(df, t[0], t[1], "Neckline"), _pt(df, e2[0], e2[1], f"{lab} 2")],
                       "lines": [_line(df, t[0], neck, n - 1, neck, "Neckline")]}
                out.append(_record(
                    f"double_{extreme}", direction, df, e1[0], n - 1,
                    0.5 + 0.2 * (1 - diff / tol) + 0.25 * done,
                    [f"{extreme}s {diff:.1f}% apart", f"depth {depth:.1f}%",
                     f"prior {'advance' if kind == 'H' else 'decline'} into first {extreme}"],
                    "confirmed (neckline broken)" if done else "forming (neckline intact)",
                    neck, neck - height if kind == "H" else neck + height, geo))
    return out


def _head_shoulders(df: pd.DataFrame, piv: List[Pivot]) -> List[Dict[str, object]]:
    tol = THRESHOLDS.pattern_tolerance_pct
    n = len(df)
    price = float(df["close"].iloc[-1])
    out: List[Dict[str, object]] = []
    for kind, name, direction in (("H", "head_and_shoulders", -1),
                                  ("L", "inverse_head_and_shoulders", 1)):
        idx = [k for k in range(len(piv)) if piv[k][2] == kind and piv[k][0] >= n - 40]
        if not idx or idx[-1] < 4:
            continue
        k = idx[-1]
        ls, t1, head, t2, rs = piv[k - 4:k + 1]
        sign = 1 if kind == "H" else -1
        head_prominent = (sign * (head[1] - ls[1]) / ls[1] * 100 >= tol
                          and sign * (head[1] - rs[1]) / rs[1] * 100 >= tol)
        shoulders_match = _pct_diff(ls[1], rs[1]) <= 1.5 * tol
        if not (head_prominent and shoulders_match):
            continue
        slope = (t2[1] - t1[1]) / max(1, t2[0] - t1[0])
        neck_now = t1[1] + slope * (n - 1 - t1[0])
        done = price < neck_now if kind == "H" else price > neck_now
        height = abs(head[1] - (t1[1] + slope * (head[0] - t1[0])))
        geo = {"points": [_pt(df, ls[0], ls[1], "Left shoulder"), _pt(df, t1[0], t1[1]),
                          _pt(df, head[0], head[1], "Head"), _pt(df, t2[0], t2[1]),
                          _pt(df, rs[0], rs[1], "Right shoulder")],
               "lines": [_line(df, t1[0], t1[1], n - 1, neck_now, "Neckline")]}
        out.append(_record(
            name, direction, df, ls[0], n - 1, 0.55 + 0.3 * done,
            [f"head {abs(head[1] / ls[1] - 1) * 100:.1f}% beyond left shoulder",
             f"shoulders {_pct_diff(ls[1], rs[1]):.1f}% apart", "sloped neckline through troughs"
             if abs(slope) > 0 else "flat neckline"],
            "confirmed (neckline broken)" if done else "forming (neckline intact)",
            neck_now, neck_now - sign * height, geo))
    return out


def _fit(points: List[Pivot]) -> Tuple[float, float, float]:
    x = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 and len(points) > 2 else 0.6
    return float(slope), float(intercept), r2


def _boundaries(df: pd.DataFrame, piv: List[Pivot]) -> List[Dict[str, object]]:
    n = len(df)
    price = float(df["close"].iloc[-1])
    recent = [p for p in piv if p[0] >= n - 90]
    highs = [p for p in recent if p[2] == "H"][-4:]
    lows = [p for p in recent if p[2] == "L"][-4:]
    if len(highs) < 2 or len(lows) < 2:
        return []
    start = min(highs[0][0], lows[0][0])
    if n - 1 - start < 15:
        return []
    su, iu, r2u = _fit(highs)
    sl, il, r2l = _fit(lows)
    level = float(df["close"].iloc[start:].mean())
    nu, nl = su / level, sl / level
    flat_u, flat_l = abs(nu) < FLAT_SLOPE, abs(nl) < FLAT_SLOPE
    upper_now, lower_now = su * (n - 1) + iu, sl * (n - 1) + il
    if upper_now <= lower_now:
        return []
    if flat_u and flat_l:
        name, bias = "rectangle", 0
    elif flat_u and nl > 0:
        name, bias = "ascending_triangle", 1
    elif flat_l and nu < 0:
        name, bias = "descending_triangle", -1
    elif nu < 0 < nl:
        name, bias = "symmetric_triangle", 0
    elif nu > 0 and nl < 0:
        name, bias = "broadening_formation", 0
    elif nu > 0 and nl > 0:
        parallel = abs(nu - nl) / max(abs(nu), abs(nl)) < 0.5
        name, bias = ("rising_channel", 1) if parallel else (
            ("rising_wedge", -1) if nl > nu else ("rising_channel", 1))
    elif nu < 0 and nl < 0:
        parallel = abs(nu - nl) / max(abs(nu), abs(nl)) < 0.5
        name, bias = ("falling_channel", -1) if parallel else (
            ("falling_wedge", 1) if nu < nl else ("falling_channel", -1))
    else:
        return []
    if price > upper_now * 1.005:
        status, direction, level_hit = "breakout above upper boundary", 1, upper_now
    elif price < lower_now * 0.995:
        status, direction, level_hit = "breakdown below lower boundary", -1, lower_now
    else:
        status, direction, level_hit = "inside pattern (unconfirmed)", bias, (
            upper_now if bias >= 0 else lower_now)
    touches = len(highs) + len(lows)
    conf = 0.3 + 0.08 * min(touches, 8) / 2 + 0.2 * (r2u + r2l) / 2 \
        + 0.15 * (not status.startswith("inside"))
    features = [f"{len(highs)} swing highs, {len(lows)} swing lows",
                f"upper slope {nu * 100:+.3f}%/bar", f"lower slope {nl * 100:+.3f}%/bar",
                f"fit R² {r2u:.2f}/{r2l:.2f}"]
    geo = {"points": [], "touches": [_pt(df, p_[0], p_[1]) for p_ in highs + lows],
           "lines": [_line(df, start, su * start + iu, n - 1, upper_now, "Upper trendline"),
                     _line(df, start, sl * start + il, n - 1, lower_now, "Lower trendline")]}
    return [_record(name, direction, df, start, n - 1, conf, features, status, level_hit, None, geo)]


def _flag(df: pd.DataFrame) -> List[Dict[str, object]]:
    n = len(df)
    close, high, low = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    atr = ind.atr(df["high"], df["low"], df["close"]).to_numpy()
    best = None
    for p in range(max(15, n - 25), n - 5):
        if np.isnan(atr[p]) or atr[p] <= 0:
            continue
        move = close[p] - close[p - 10]
        if abs(move) / atr[p] < 5.0:   # pole must be >= 5 ATR in 10 bars; PROVISIONAL
            continue
        seg_hi, seg_lo = high[p + 1:].max(), low[p + 1:].min()
        if seg_hi - seg_lo > 0.5 * abs(move):
            continue
        if move > 0 and seg_lo < close[p] - 0.5 * move:
            continue
        if move < 0 and seg_hi > close[p] - 0.5 * move:
            continue
        if best is None or abs(move) > abs(best[1]):
            best = (p, move)
    if best is None:
        return []
    p, move = best
    xs = np.arange(p + 1, n, dtype=float)
    hs, _ = np.polyfit(xs, high[p + 1:], 1)
    ls_, _ = np.polyfit(xs, low[p + 1:], 1)
    converging = hs < 0 < ls_
    name = ("bull_" if move > 0 else "bear_") + ("pennant" if converging else "flag")
    direction = 1 if move > 0 else -1
    # Compare with the consolidation range before today, or today's own bar
    # would always define the boundary it is tested against.
    prior_boundary = high[p + 1:n - 1].max() if move > 0 else low[p + 1:n - 1].min()
    done = close[-1] > prior_boundary if move > 0 else close[-1] < prior_boundary
    hi_fit = np.polyfit(xs, high[p + 1:], 1)
    lo_fit = np.polyfit(xs, low[p + 1:], 1)
    geo = {"points": [_pt(df, p - 10, close[p - 10], "Pole start"), _pt(df, p, close[p], "Pole top" if move > 0 else "Pole bottom")],
           "lines": [_line(df, p + 1, np.polyval(hi_fit, p + 1), n - 1, np.polyval(hi_fit, n - 1), "Flag upper"),
                     _line(df, p + 1, np.polyval(lo_fit, p + 1), n - 1, np.polyval(lo_fit, n - 1), "Flag lower")]}
    return [_record(name, direction, df, p - 10, n - 1, 0.5 + 0.3 * done,
                    [f"pole {move / close[p - 10] * 100:+.1f}% in 10 bars",
                     f"{n - 1 - p}-bar consolidation within half the pole"],
                    "breakout (continuation confirmed)" if done else "forming",
                    prior_boundary, close[-1] + move if done else None, geo)]


def _rounding(df: pd.DataFrame) -> List[Dict[str, object]]:
    n = len(df)
    w = min(100, n)
    y = df["close"].to_numpy()[-w:]
    x = np.linspace(0, 1, w)
    a, b, c = np.polyfit(x, y, 2)
    fit = a * x ** 2 + b * x + c
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - fit) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    vertex = -b / (2 * a) if a != 0 else -1
    depth = (y.max() - y.min()) / y.max() * 100
    if r2 < 0.75 or not 0.3 <= vertex <= 0.7 or depth < 10:
        return []
    start = n - w
    samples = np.linspace(0, w - 1, 14).astype(int)
    geo = {"points": [], "curve": [_pt(df, start + i, fit[i]) for i in samples],
           "lines": [_line(df, start, y[0], n - 1, y[0], "Rim")]}
    if a > 0:
        rim = y[0]
        done = y[-1] >= rim
        return [_record("rounding_bottom", 1, df, start, n - 1, 0.4 + 0.3 * r2 + 0.2 * done,
                        [f"U-shaped fit R² {r2:.2f}", f"depth {depth:.1f}%"],
                        "confirmed (back at left rim)" if done else "forming", rim, None, geo)]
    rim = y[0]
    done = y[-1] <= rim
    return [_record("rounding_top", -1, df, start, n - 1, 0.4 + 0.3 * r2 + 0.2 * done,
                    [f"inverted-U fit R² {r2:.2f}", f"height {depth:.1f}%"],
                    "confirmed (back at left rim)" if done else "forming", rim, None, geo)]


def structure_events(df: pd.DataFrame) -> List[Dict[str, object]]:
    close, high, low = df["close"], df["high"], df["low"]
    n = len(df)
    events: List[Dict[str, object]] = []
    if n < 70:
        return events
    prior_hi = high.shift(1).rolling(60).max()
    prior_lo = low.shift(1).rolling(60).min()
    today = str(df.index[-1].date())
    if close.iloc[-1] > prior_hi.iloc[-1]:
        events.append({"event": "breakout", "date": today, "level": round(float(prior_hi.iloc[-1]), 2)})
    elif close.iloc[-1] < prior_lo.iloc[-1]:
        events.append({"event": "breakdown", "date": today, "level": round(float(prior_lo.iloc[-1]), 2)})
    for j in range(n - 6, n - 1):
        if close.iloc[j] > prior_hi.iloc[j] and close.iloc[-1] < prior_hi.iloc[j]:
            events.append({"event": "false_breakout", "date": str(df.index[j].date()),
                           "level": round(float(prior_hi.iloc[j]), 2)})
            break
        if close.iloc[j] < prior_lo.iloc[j] and close.iloc[-1] > prior_lo.iloc[j]:
            events.append({"event": "false_breakdown", "date": str(df.index[j].date()),
                           "level": round(float(prior_lo.iloc[j]), 2)})
            break
    atr_now = float(ind.atr(high, low, close).iloc[-1])
    range20 = float(high.iloc[-20:].max() - low.iloc[-20:].min())
    if atr_now > 0 and range20 / atr_now < 4.0:   # PROVISIONAL tightness cut-off
        events.append({"event": "consolidation", "date": today,
                       "range_atr": round(range20 / atr_now, 1)})
    return events


def detect(df: pd.DataFrame) -> List[Dict[str, object]]:
    """All current chart patterns on the most recent WINDOW bars."""
    view = df.iloc[-WINDOW:]
    piv = pivots(view)
    found = _tops_bottoms(view, piv) + _head_shoulders(view, piv) + _boundaries(view, piv) \
        + _flag(view) + _rounding(view)
    return found


def _key(p: Dict[str, object]) -> str:
    d = int(p["direction"])
    return f"{p['pattern']}:{'up' if d > 0 else 'down' if d < 0 else 'flat'}"


def _is_completed(p: Dict[str, object]) -> bool:
    s = str(p["confirmation_status"])
    return s.startswith(("confirmed", "breakout", "breakdown"))


def scan_history(df: pd.DataFrame, step: int = SCAN_STEP) -> Dict[str, pd.Series]:
    """Replay `detect` on every step-th truncated history; mark completions."""
    signals: Dict[str, pd.Series] = {}
    start = max(WINDOW // 2, 60)
    for t in range(start, len(df), step):
        for p in detect(df.iloc[:t + 1]):
            if _is_completed(p):
                key = _key(p)
                if key not in signals:
                    signals[key] = pd.Series(False, index=df.index)
                signals[key].iloc[t] = True
    return signals


def analyze(df: pd.DataFrame, priors: Optional[Dict[str, Dict[str, object]]] = None) -> DomainResult:
    if df is None or len(df) < 80:
        return unavailable(DOMAIN, "need at least 80 bars for chart-pattern detection")
    current = detect(df)
    events = structure_events(df)
    history = scan_history(df)
    close = df["close"]
    as_of = str(df.index[-1].date())
    ev: List[Evidence] = []
    for p in current:
        key = _key(p)
        sig = history.get(key)
        own = (event_study(close, sig, VALIDATION_HORIZON, int(p["direction"]))
               if sig is not None else {"events": 0, "verdict": "no historical occurrences"})
        hist = with_prior(own, (priors or {}).get(key))
        p["historical_occurrence_count"] = hist.get("events", 0)
        p["historical_outcomes"] = hist
        status_factor = 1.0 if _is_completed(p) else 0.4
        strength = float(p["confidence"]) * status_factor * edge_multiplier(hist)
        name = str(p["pattern"]).replace("_", " ")
        claim = (f"{name.capitalize()} ({p['start_date']} to {p['end_date']}), "
                 f"{p['confirmation_status']}, confidence {p['confidence']:.2f}.")
        pooled = hist.get("pooled")
        if pooled:
            claim += (f" Too few completions on this stock ({hist['events']}); across {pooled['stocks']} "
                      f"stocks ({pooled['events']} events): {hist['verdict']}.")
        elif hist.get("events"):
            claim += (f" Past completions on this stock: {hist['events']}, "
                      f"{VALIDATION_HORIZON}d mean {hist.get('mean_pct', 0):+.2f}% — {hist['verdict']}.")
        else:
            claim += " No past completions on this stock to validate against."
        ev.append(Evidence(domain=DOMAIN, claim=claim, direction=int(p["direction"]),
                           strength=round(strength, 3),
                           provenance=Provenance(source=f"computed:pattern:{p['pattern']}",
                                                 as_of=as_of),
                           value=p["breakout_level"]))
    for e in events:
        if e["event"] in ("false_breakout", "false_breakdown"):
            bearish = e["event"] == "false_breakout"
            ev.append(Evidence(domain=DOMAIN,
                               claim=f"{e['event'].replace('_', ' ').capitalize()} at "
                                     f"{e['level']:,.2f} on {e['date']}; price has since reversed.",
                               direction=-1 if bearish else 1, strength=0.35,
                               provenance=Provenance(source="computed:structure", as_of=as_of),
                               value=e["level"]))
    details = {"patterns": current, "structure_events": events}
    directional = [e for e in ev if e.direction != 0 and e.strength > 0]
    if not directional:
        return DomainResult(domain=DOMAIN, state="no actionable pattern", score=0.0,
                            evidence=ev, details=details, as_of=as_of, confidence=0.0)
    total = sum(e.strength for e in directional)
    score = clamp(sum(e.direction * e.strength for e in directional) / total) * min(1.0, total)
    state = "bullish structure" if score > 0.1 else "bearish structure" if score < -0.1 else "mixed"
    return DomainResult(domain=DOMAIN, state=state, score=score, evidence=ev,
                        details=details, as_of=as_of, confidence=min(1.0, total))
