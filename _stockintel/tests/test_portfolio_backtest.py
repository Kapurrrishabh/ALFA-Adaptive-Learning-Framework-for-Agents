import numpy as np
import pandas as pd
import pytest

from stockintel.backtest import strategies as S
from stockintel.backtest.engine import BacktestConfig, run
from stockintel.portfolio import Transaction, analyze, build_positions, risk_decomposition, what_if
from stockintel.registry import ModelRegistry, PromotionError

from .conftest import bars


def tx(sym, side, q, px, d, sector=None, fees=0.0):
    return Transaction(sym, side, q, px, d, fees, sector)


def test_fifo_realized_pl_and_remaining_cost_basis():
    pos = build_positions([tx("A", "BUY", 10, 100, "2024-01-01"), tx("A", "BUY", 10, 120, "2024-02-01"),
                           tx("A", "SELL", 15, 130, "2024-03-01")])["A"]
    assert pos.realized_pl == pytest.approx(10 * 30 + 5 * 10)
    assert pos.quantity == 5 and pos.avg_cost == pytest.approx(120)


def test_fees_reduce_realized_pl():
    pos = build_positions([tx("A", "BUY", 10, 100, "2024-01-01", fees=10),
                           tx("A", "SELL", 10, 110, "2024-02-01", fees=10)])["A"]
    assert pos.realized_pl == pytest.approx(100 - 20)


def test_selling_more_than_held_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        build_positions([tx("A", "BUY", 5, 100, "2024-01-01"), tx("A", "SELL", 6, 100, "2024-01-02")])


def test_invalid_transaction_side_is_rejected():
    with pytest.raises(ValueError):
        tx("A", "HOLD", 1, 1, "2024-01-01")


def _prices(seed_a=1, seed_b=2, n=300):
    rng = np.random.default_rng(seed_a)
    idx = pd.bdate_range(end="2026-09-25", periods=n)
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=idx)
    b = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(seed_b).normal(0, 0.01, n))), index=idx)
    return {"A": a, "B": b}


def test_portfolio_values_weights_and_pl():
    prices = _prices()
    pos = build_positions([tx("A", "BUY", 10, 50, "2024-01-01", "Tech"), tx("B", "BUY", 20, 80, "2024-01-01", "Bank")])
    rep = analyze(pos, 1000.0, prices)
    va, vb = 10 * prices["A"].iloc[-1], 20 * prices["B"].iloc[-1]
    assert rep.invested_value == pytest.approx(va + vb, abs=0.01)
    assert rep.total_value == pytest.approx(va + vb + 1000, abs=0.01)
    assert rep.unrealized_pl == pytest.approx(va + vb - 500 - 1600, abs=0.01)
    assert sum(h["weight_pct"] for h in rep.holdings) + 1000 / rep.total_value * 100 == pytest.approx(100, abs=0.05)
    assert set(rep.sector_exposure_pct) == {"Tech", "Bank"}


def test_risk_contributions_sum_to_100_percent():
    rets = pd.DataFrame(_prices()).pct_change().dropna()
    rd = risk_decomposition(pd.Series({"A": 0.6, "B": 0.4}), rets)
    assert sum(rd["risk_contribution_pct"].values()) == pytest.approx(100, abs=0.05)
    assert rd["diversification_ratio"] >= 1.0


def test_volatile_holding_is_flagged_as_the_risk_driver():
    prices = _prices()
    pos = build_positions([tx("A", "BUY", 10, 100, "2024-01-01"), tx("B", "BUY", 10, 100, "2024-01-01")])
    rep = analyze(pos, 0.0, prices)
    top = max(rep.risk["risk_contribution_pct"], key=rep.risk["risk_contribution_pct"].get)
    assert top == "A"


def test_concentration_limit_is_flagged():
    prices = _prices()
    pos = build_positions([tx("A", "BUY", 100, 100, "2024-01-01", "Tech"), tx("B", "BUY", 1, 100, "2024-01-01", "Bank")])
    rep = analyze(pos, 0.0, prices, "conservative")
    assert any("position limit" in f for f in rep.flags) and any("sector limit" in f for f in rep.flags)


def test_missing_prices_fail_loudly():
    pos = build_positions([tx("C", "BUY", 1, 10, "2024-01-01")])
    with pytest.raises(ValueError, match="no price history"):
        analyze(pos, 0.0, _prices())


def test_what_if_adding_to_the_low_vol_stock_lowers_portfolio_vol():
    prices = _prices()
    pos = build_positions([tx("A", "BUY", 50, 100, "2024-01-01")])
    res = what_if(pos, 0.0, prices, {"into B": {"B": 20000}, "into A": {"A": 20000}})
    assert res["scenarios"]["into B"]["portfolio_vol_pct"] < res["scenarios"]["into A"]["portfolio_vol_pct"]
    assert res["scenarios"]["into B"]["shares_bought"]["B"] == int(20000 // prices["B"].iloc[-1])


def test_backtest_signal_cannot_earn_the_bar_that_produced_it():
    closes = np.full(30, 100.0)
    closes[20:] = 200.0               # jump on day 20
    df = bars(closes, spread=0.0)
    df["open"] = df["close"]          # the jump happens at the open of day 20 as well
    target = pd.Series(0.0, index=df.index)
    target.iloc[20:] = 1.0            # signal first appears at the close of day 20
    res = run(df, target, "late", BacktestConfig(cost_bps=0, slippage_bps=0))
    assert res.metrics["total_return_pct"] == pytest.approx(0.0, abs=1e-9)


def test_backtest_charges_costs_on_turnover():
    df = bars(np.full(60, 100.0), spread=0.0)
    df["open"] = df["close"]
    flip = pd.Series([1.0, 0.0] * 30, index=df.index)
    res = run(df, flip, "churn", BacktestConfig(cost_bps=10, slippage_bps=0))
    assert res.metrics["total_return_pct"] < 0 and res.metrics["costs_paid_pct"] > 0


def test_buy_and_hold_tracks_the_underlying_open_to_open():
    df = bars(np.linspace(100, 150, 100), spread=0.0)
    res = run(df, S.buy_and_hold(df), "bh", BacktestConfig(cost_bps=0, slippage_bps=0))
    expected = df["open"].iloc[-1] / df["open"].iloc[1] - 1
    assert res.metrics["total_return_pct"] == pytest.approx(expected * 100, abs=0.01)


def test_stop_loss_exits_at_the_stop_level():
    closes = np.concatenate([np.full(10, 100.0), np.full(10, 80.0)])
    df = bars(closes, spread=0.0)
    df["open"] = df["close"]
    df.iloc[10, df.columns.get_loc("open")] = 100.0
    df.iloc[10, df.columns.get_loc("low")] = 80.0
    res = run(df, S.buy_and_hold(df), "bh", BacktestConfig(cost_bps=0, slippage_bps=0, stop_loss_pct=0.10))
    assert res.trades[0]["stopped"]
    assert res.metrics["total_return_pct"] == pytest.approx(-10.0, abs=1e-6)


def test_forecast_strategy_stays_flat_outside_out_of_sample_window(random_walk):
    pos = S.forecast_model(random_walk)
    assert (pos.iloc[:250] == 0).all()


def test_comparison_runs_all_strategies_on_the_same_window(random_walk):
    res = S.compare(random_walk, S.BASELINES, start=str(random_walk.index[300].date()))
    starts = {r.start for r in res["results"]}
    assert len(starts) == 1 and "data snooping" in res["caveat"]


def test_registry_requires_improvement_and_named_approval(store):
    reg = ModelRegistry(store)
    reg.register("forecast", "v1", training_period="t", features=[], hyperparameters={"models": ["climatology"]},
                 metrics={"brier_skill": 0.01}, backtest=None, limitations="")
    with pytest.raises(PromotionError, match="named human"):
        reg.approve("forecast", "v1", "  ")
    with pytest.raises(PromotionError, match="approved before"):
        reg.deploy("forecast", "v1")
    reg.approve("forecast", "v1", "rishabh")
    reg.deploy("forecast", "v1")
    reg.register("forecast", "v2", training_period="t", features=[], hyperparameters={"models": ["gbm"]},
                 metrics={"brier_skill": 0.005}, backtest=None, limitations="")
    with pytest.raises(PromotionError, match="does not beat"):
        reg.approve("forecast", "v2", "rishabh")
    assert reg.deployed("forecast")["version"] == "v1"


def test_predictions_are_resolved_against_realized_prices(store):
    from stockintel.monitoring import live_skill, resolve_predictions
    idx = pd.bdate_range(end="2026-09-25", periods=10)
    close = pd.Series(np.arange(100, 110, dtype=float), index=idx)
    store.log_prediction("logistic", "v1", "X.NS", 5, str(idx[2].date()), {"a": 1}, 0.7)
    store.log_prediction("logistic", "v1", "X.NS", 5, str(idx[8].date()), {"a": 2}, 0.7)  # not matured
    assert resolve_predictions(store, {"X.NS": close}) == 1
    m = live_skill(store)["logistic_h5"]
    assert m["n"] == 1 and m["hit_rate"] == 1.0 and m["brier"] == pytest.approx(0.09)


def test_psi_flags_a_volatility_regime_change():
    from stockintel.monitoring import psi
    rng = np.random.default_rng(0)
    calm, wild = rng.normal(0, 0.01, 500), rng.normal(0, 0.04, 60)
    assert psi(calm, rng.normal(0, 0.01, 60)) < 0.25 < psi(calm, wild)


def test_registry_rejects_a_candidate_that_does_not_beat_the_base_rate(store):
    reg = ModelRegistry(store)
    reg.register("forecast", "neg", training_period="t", features=[], hyperparameters={"models": ["gbm"]},
                 metrics={"brier_skill": -0.05}, backtest=None, limitations="")
    with pytest.raises(PromotionError, match="does not beat the baseline"):
        reg.approve("forecast", "neg", "rishabh")
