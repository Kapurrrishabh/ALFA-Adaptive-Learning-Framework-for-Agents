import numpy as np
import pandas as pd
import pytest

from stockintel.analysis import analogues, forecast, quant, regime
from stockintel.analysis.stats import auc, norm_cdf, norm_ppf, variance_ratio

from .conftest import bars


def test_max_drawdown_of_a_known_path():
    close = pd.Series([100, 120, 90, 95, 130, 117], index=pd.bdate_range("2026-01-01", periods=6), dtype=float)
    dd = quant.max_drawdown(close)
    assert dd["max_drawdown_pct"] == pytest.approx(-25.0)
    assert dd["peak_date"] == "2026-01-02" and dd["trough_date"] == "2026-01-05"
    assert dd["recovery_date"] == "2026-01-07"
    assert dd["current_drawdown_pct"] == pytest.approx(-10.0)


def test_beta_of_a_series_against_itself_is_one():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300), index=pd.bdate_range("2025-01-01", periods=300))
    ba = quant.beta_alpha(r, r, 0.0)
    assert ba["beta"] == pytest.approx(1.0) and ba["correlation"] == pytest.approx(1.0)


def test_beta_of_a_levered_series_is_the_leverage():
    b = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300), index=pd.bdate_range("2025-01-01", periods=300))
    assert quant.beta_alpha(2 * b, b, 0.0)["beta"] == pytest.approx(2.0)


def test_historical_var_and_cvar():
    r = pd.Series(np.arange(-50, 50) / 1000.0)
    var, cvar = quant.var_cvar(r, 0.95)
    assert var == pytest.approx(np.quantile(r, 0.05))
    assert cvar == pytest.approx(r[r <= var].mean())


def test_sharpe_of_constant_positive_excess_returns_is_infinite_or_nan_safe():
    r = pd.Series([0.001] * 100)
    assert np.isnan(quant.sharpe(r, 0.0))  # zero volatility: undefined, not a huge number


def test_stats_helpers():
    assert norm_cdf(0) == pytest.approx(0.5)
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_variance_ratio_separates_trending_from_mean_reverting():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 0.01, 4000)
    trending = np.convolve(e, np.ones(5) / 5, mode="same")        # positively autocorrelated
    reverting = e - 0.6 * np.concatenate([[0], e[:-1]])           # negatively autocorrelated
    vr_t, z_t = variance_ratio(trending, 5)
    vr_r, z_r = variance_ratio(reverting, 5)
    assert vr_t > 1 and z_t > 2 and vr_r < 1 and z_r < -2


def test_hmm_recovers_a_high_volatility_stretch():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0, 0.005, 400), rng.normal(0, 0.03, 200), rng.normal(0, 0.005, 400)])
    hmm = regime.fit_hmm(x)
    p_high = hmm["filtered"][:, 1]
    assert p_high[450:580].mean() > 0.8 and p_high[100:380].mean() < 0.2
    assert hmm["sd"][1] > hmm["sd"][0]


def test_walk_forward_never_trains_on_labels_that_overlap_the_test_block(random_walk):
    h = 10
    wf = forecast.walk_forward(random_walk, h)
    assert wf["audit"]
    for block in wf["audit"]:
        assert block["label_end"] < block["test_first"]


def test_forecast_on_a_random_walk_reports_no_edge_and_gets_zero_weight(random_walk):
    res = forecast.analyze(random_walk)
    assert res.confidence == 0.0 and res.score == 0.0
    for h in res.details["horizons"].values():
        assert h["model"] == "climatology" and not h["skill_demonstrated"]


def test_forecast_finds_skill_when_the_future_is_predictable():
    rng = np.random.default_rng(7)
    n = 1200
    r = rng.normal(0, 0.01, n)
    for i in range(1, n):          # strong momentum: tomorrow repeats part of today's sign
        r[i] += 0.8 * r[i - 1]
    df = bars(100 * np.exp(np.cumsum(r)), spread=0.002)
    wf = forecast.walk_forward(df, 1)
    assert wf["metrics"]["logistic"]["brier_skill"] > 0.05
    assert wf["metrics"]["logistic"]["auc"] > 0.6


def test_expected_range_is_calibrated_on_a_random_walk(random_walk):
    wf = forecast.walk_forward(random_walk, 5)
    assert 0.72 <= wf["interval_coverage"] <= 0.88


def test_features_use_only_past_bars(random_walk):
    full = forecast.features(random_walk)
    cut = forecast.features(random_walk.iloc[:600])
    pd.testing.assert_frame_equal(full.iloc[:600], cut)


def test_analogues_exclude_recent_bars_and_unrealized_outcomes(random_walk):
    res = analogues.find_analogues(random_walk, 20)
    last_allowed = random_walk.index[len(random_walk) - 1 - analogues.EXCLUDE_RECENT]
    assert all(pd.Timestamp(d) <= last_allowed for d in res["dates"])
    assert res["analogues"] == analogues.K


def test_analogues_on_a_random_walk_do_not_claim_significance(random_walk):
    res = analogues.analyze(random_walk)
    assert res.available and all(e.direction == 0 for e in res.evidence) or res.confidence == 1.0


@pytest.mark.parametrize("n", [121, 122, 125])
def test_short_histories_with_a_benchmark_do_not_crash(n, bench):
    from stockintel.data import synthetic
    df = synthetic.ohlcv(n, seed=4, end="2026-09-25")
    assert quant.analyze(df, bench, 0.0).available
