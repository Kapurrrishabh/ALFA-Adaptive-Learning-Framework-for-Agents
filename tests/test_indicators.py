import numpy as np
import pandas as pd
import pytest

from stockintel.analysis import indicators as ind

# StockCharts' published 14-period RSI worked example.
STOCKCHARTS_CLOSES = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
                      46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25,
                      45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13]
STOCKCHARTS_RSI = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38, 54.71,
                   50.42, 39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77]


def test_rsi_matches_hand_computed_wilder_values():
    # First 14 changes: gains sum 3.34, losses sum 1.40. Next change -0.28.
    first = 100 - 100 / (1 + 3.34 / 1.40)
    ag, al = (3.34 / 14 * 13) / 14, (1.40 / 14 * 13 + 0.28) / 14
    second = 100 - 100 / (1 + ag / al)
    r = ind.rsi(pd.Series(STOCKCHARTS_CLOSES)).dropna()
    assert r.iloc[0] == pytest.approx(first, abs=1e-9)
    assert r.iloc[1] == pytest.approx(second, abs=1e-9)


def test_rsi_tracks_published_reference_within_its_rounding():
    # StockCharts rounds intermediate averages, which shifts its table by < 0.1.
    r = ind.rsi(pd.Series(STOCKCHARTS_CLOSES)).dropna().tolist()
    assert r == pytest.approx(STOCKCHARTS_RSI, abs=0.1)


def test_rsi_is_undefined_before_the_window_fills():
    r = ind.rsi(pd.Series(STOCKCHARTS_CLOSES))
    assert r.iloc[:14].isna().all() and not np.isnan(r.iloc[14])


def test_rsi_of_a_series_that_only_rises_is_100():
    assert ind.rsi(pd.Series(np.arange(1, 40, dtype=float))).iloc[-1] == 100.0


def test_rsi_of_a_series_that_only_falls_is_0():
    assert ind.rsi(pd.Series(np.arange(40, 1, -1, dtype=float))).iloc[-1] == 0.0


def test_ema_uses_recursive_smoothing():
    assert ind.ema(pd.Series([1, 2, 3, 4, 5], dtype=float), 3).tolist() == [1, 1.5, 2.25, 3.125, 4.0625]


def test_wma_weights_recent_values_more():
    assert ind.wma(pd.Series([1, 2, 3], dtype=float), 3).iloc[-1] == pytest.approx((1 + 4 + 9) / 6)


def test_macd_line_is_fast_minus_slow_ema():
    s = pd.Series(np.linspace(100, 130, 80))
    line, sig, hist = ind.macd(s)
    assert line.iloc[-1] == pytest.approx(ind.ema(s, 12).iloc[-1] - ind.ema(s, 26).iloc[-1])
    assert hist.iloc[-1] == pytest.approx(line.iloc[-1] - sig.iloc[-1])


def test_atr_of_constant_range_bars_equals_the_range():
    n = 30
    close = pd.Series(np.full(n, 100.0))
    atr = ind.atr(close + 2, close - 2, close)
    assert atr.iloc[-1] == pytest.approx(4.0)
    assert atr.iloc[:13].isna().all()


def test_true_range_includes_gaps_from_previous_close():
    tr = ind.true_range(pd.Series([10.0, 21.0]), pd.Series([9.0, 20.0]), pd.Series([9.5, 20.5]))
    assert tr.iloc[1] == pytest.approx(21.0 - 9.5)


def test_bollinger_bands_collapse_on_a_constant_series():
    up, mid, lo = ind.bollinger(pd.Series(np.full(30, 50.0)))
    assert up.iloc[-1] == mid.iloc[-1] == lo.iloc[-1] == 50.0


def test_obv_adds_volume_on_up_days_and_subtracts_on_down_days():
    obv = ind.obv(pd.Series([10.0, 11, 10, 10, 12]), pd.Series([100.0, 200, 300, 400, 500]))
    assert obv.tolist() == [0, 200, -100, -100, 400]


def test_volume_ratio_excludes_today_from_its_own_average():
    v = pd.Series([100.0] * 20 + [300.0])
    assert ind.volume_ratio(v).iloc[-1] == pytest.approx(3.0)


def test_historical_volatility_annualizes_daily_std():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000))))
    hv = ind.historical_volatility(close, 250).iloc[-1]
    assert hv == pytest.approx(0.01 * np.sqrt(252), rel=0.15)
