import numpy as np
import pandas as pd
import pytest

from stockintel.analysis import candlesticks, patterns
from stockintel.analysis.validation import decluster, event_study, forward_returns

from .conftest import bars


def _frame(rows):
    idx = pd.bdate_range(end="2026-09-25", periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx, dtype=float)


def _downtrend(n=12, start=120.0):
    rows, p = [], start
    for _ in range(n):
        rows.append([p, p + 0.5, p - 2.5, p - 2.0, 1e6])
        p -= 2.0
    return rows, p


def _uptrend(n=12, start=80.0):
    rows, p = [], start
    for _ in range(n):
        rows.append([p, p + 2.5, p - 0.5, p + 2.0, 1e6])
        p += 2.0
    return rows, p


def test_hammer_is_detected_after_a_decline():
    rows, p = _downtrend()
    rows.append([p, p + 0.2, p - 6.0, p + 0.5, 1e6])  # long lower wick, small body near top
    assert candlesticks.detect_all(_frame(rows))["hammer"].iloc[-1]


def test_same_shape_after_an_advance_is_a_hanging_man_not_a_hammer():
    rows, p = _uptrend()
    rows.append([p, p + 0.2, p - 6.0, p - 0.5, 1e6])
    det = candlesticks.detect_all(_frame(rows))
    assert det["hanging_man"].iloc[-1] and not det["hammer"].iloc[-1]


def test_bullish_engulfing_requires_the_new_body_to_cover_the_old():
    rows, p = _downtrend()
    rows.append([p, p + 0.3, p - 1.3, p - 1.0, 1e6])          # small bearish
    rows.append([p - 1.5, p + 1.8, p - 1.6, p + 1.5, 1e6])    # larger bullish covering it
    assert candlesticks.detect_all(_frame(rows))["bullish_engulfing"].iloc[-1]


def test_doji_has_almost_no_body():
    rows, p = _uptrend()
    rows.append([p, p + 3.0, p - 3.0, p + 0.1, 1e6])
    assert candlesticks.detect_all(_frame(rows))["doji"].iloc[-1]


def test_three_white_soldiers():
    rows, p = _downtrend(12)
    for _ in range(3):
        rows.append([p, p + 3.1, p - 0.1, p + 3.0, 1e6])
        p += 2.0
    det = candlesticks.detect_all(_frame(rows))
    assert det["three_white_soldiers"].iloc[-1]


def test_unvalidated_formation_is_discounted_and_explained(random_walk):
    tech_levels = ([], [])
    res = candlesticks.analyze(random_walk, *tech_levels)
    for e in res.evidence:
        if e.direction:
            assert "confirmation" in e.claim
            assert e.strength <= 0.35 + 1e-9 or "historically significant" in e.claim


def test_forward_returns_align_signal_with_the_future_only():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    fwd = forward_returns(close, 1)
    assert fwd.iloc[0] == pytest.approx(0.1) and np.isnan(fwd.iloc[-1])


def test_decluster_keeps_only_the_first_signal_in_each_horizon():
    m = pd.Series([True, True, True, False, False, True])
    assert decluster(m, 2).tolist() == [True, False, False, True, False, False] or \
        decluster(m, 2).tolist() == [True, False, False, False, False, True]


def test_random_signals_show_no_significant_edge():
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000))))
    sig = pd.Series(rng.random(3000) < 0.05)
    res = event_study(close, sig, 5, 1)
    assert res["events"] >= 20 and res["verdict"] == "no statistically significant edge"


def test_a_signal_that_precedes_rises_is_validated():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.005, 3000)
    sig = np.zeros(3000, bool)
    sig[::50] = True
    for i in np.where(sig)[0]:
        r[i + 1:i + 6] += 0.01   # five up days follow each signal
    close = pd.Series(100 * np.exp(np.cumsum(r)))
    res = event_study(close, pd.Series(sig), 5, 1)
    assert res["verdict"] == "historically significant in the claimed direction"


def test_double_top_is_detected_and_confirmed_below_the_neckline():
    up = np.linspace(100, 130, 30)
    fall = np.linspace(130, 115, 12)
    rise = np.linspace(115, 129.5, 12)
    drop = np.linspace(129.5, 110, 14)
    closes = np.concatenate([np.linspace(95, 100, 60), up, fall[1:], rise[1:], drop[1:]])
    found = patterns.detect(bars(closes, spread=0.004))
    tops = [p for p in found if p["pattern"] == "double_top"]
    assert tops and tops[0]["confirmation_status"].startswith("confirmed")


def test_pattern_scan_is_causal():
    df = bars(np.concatenate([np.linspace(95, 100, 60), np.linspace(100, 130, 30),
                              np.linspace(130, 115, 12), np.linspace(115, 129.5, 12),
                              np.linspace(129.5, 110, 14)]), spread=0.004)
    full = patterns.scan_history(df, step=1)
    # Truncating the future must not change any signal already emitted.
    cut = len(df) - 10
    early = patterns.scan_history(df.iloc[:cut], step=1)
    for key, sig in early.items():
        assert (full[key].iloc[:cut].to_numpy() == sig.to_numpy()).all()
