from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.database.sources import synthetic
from backend.database.sources.panel import panel_from_frames
from backend.advisory.signals import factors as F
from backend.advisory.signals import strategies as S
from backend.advisory.signals.advisor import Holding, _tax_note, exposure_for, plan
from backend.advisory.signals.simulate import metrics, simulate

END = "2026-09-25"


@pytest.fixture(scope="module")
def panel():
    frames = {f"S{i:02d}": synthetic.ohlcv(700, seed=i, drift=0.0, vol=0.015, end=END, start_price=100 + i)
              for i in range(25)}
    frames["WIN"] = synthetic.ohlcv(700, seed=99, drift=0.005, vol=0.010, end=END, start_price=100)
    frames["LOSE"] = synthetic.ohlcv(700, seed=98, drift=-0.005, vol=0.010, end=END, start_price=100)
    return panel_from_frames(frames)


def test_nse_momentum_formula_matches_hand_computation(panel):
    asof = panel.close.index[-1]
    m = F.momentum_score(panel.close, asof)
    me = F.month_ends(panel.close.loc[:asof].index)
    c = panel.close
    r12 = c.loc[me[-1]] / c.loc[me[-13]] - 1
    r6 = c.loc[me[-1]] / c.loc[me[-7]] - 1
    lr = np.log(c / c.shift(1)).iloc[-252:]
    sig = lr.std(ddof=1) * np.sqrt(252)
    z = lambda x: (x - x.mean()) / x.std(ddof=1)  # noqa: E731
    zz = 0.5 * z(r12 / sig) + 0.5 * z(r6 / sig)
    expected = np.where(zz >= 0, 1 + zz, 1 / (1 - zz))
    got = m["score"].reindex(zz.index)
    assert np.allclose(got.values, expected, rtol=1e-6)
    assert m.index[0] == "WIN" and m.index[-1] == "LOSE"


def test_momentum_uses_only_data_up_to_asof(panel):
    asof = panel.close.index[-200]
    a = F.momentum_score(panel.close, asof)
    b = F.momentum_score(panel.close.loc[:asof], asof)
    pd.testing.assert_frame_equal(a, b)


def test_liquid_universe_excludes_stocks_without_a_year_of_history(panel):
    close = panel.close.copy()
    close.iloc[:-100, close.columns.get_loc("S00")] = np.nan
    uni = F.liquid_universe(close, panel.volume, close.index[-1], top=100)
    assert "S00" not in uni and "WIN" in uni


def test_buffer_keeps_held_names_within_the_keep_band():
    ranked = pd.Index([f"s{i}" for i in range(10)])
    assert S.buffered_pick(ranked, ["s6"], 3, 7) == ["s6", "s0", "s1"]
    assert S.buffered_pick(ranked, ["s8"], 3, 7) == ["s0", "s1", "s2"]


def test_simulator_fills_at_next_open_not_at_the_signal_close():
    idx = pd.bdate_range(end=END, periods=60)
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e6}, index=idx)
    jump = flat.copy()
    cut = idx[40]  # month-end signal day? use a day we control below
    jump.loc[idx[41]:, ["open", "high", "low", "close"]] = 200.0
    p = panel_from_frames({"J": jump, "F": flat})
    res = simulate(p, "t", lambda s, h: {"J": 1.0}, [cut], idx[0], cost_bps=0, cash_yield=0.0)
    # Filled at the open of idx[41], already 200: the jump is not captured.
    assert res.equity.iloc[-1] == pytest.approx(1.0)


def test_simulator_charges_costs_per_side():
    idx = pd.bdate_range(end=END, periods=30)
    flat = pd.DataFrame({"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 1e6}, index=idx)
    p = panel_from_frames({"A": flat, "B": flat})
    res = simulate(p, "t", lambda s, h: {"A": 1.0} if len(h) == 0 or h == ["B"] else {"B": 1.0},
                   [idx[5], idx[15]], idx[0], cost_bps=10, cash_yield=0.0)
    # buy A (10bps), then sell A + buy B (20bps)
    assert res.equity.iloc[-1] == pytest.approx(1 - 0.001 - 0.001 * 0.999 - 0.001 * 0.998, rel=1e-3)


def test_stop_loss_exits_at_the_stop_level():
    idx = pd.bdate_range(end=END, periods=30)
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e6}, index=idx)
    df.loc[idx[10], ["low", "close"]] = [80.0, 85.0]
    df.loc[idx[11]:, ["open", "high", "low", "close"]] = 85.0
    p = panel_from_frames({"A": df})
    res = simulate(p, "t", lambda s, h: {"A": 1.0}, [idx[2]], idx[0], stop_loss=0.10, cost_bps=0, cash_yield=0.0)
    assert res.stops == 1 and res.equity.iloc[-1] == pytest.approx(0.90)


def test_trend_filter_goes_to_cash_below_the_10_month_average():
    idx = pd.bdate_range(end=END, periods=400)
    up = pd.Series(np.linspace(100, 200, 400), index=idx)
    down = pd.Series(np.linspace(200, 100, 400), index=idx)
    assert S.trend_filter(up)(idx[-1]) == 1.0 and S.trend_filter(down)(idx[-1]) == 0.0


def test_exposure_modes():
    m = {"trend": "down", "vol_target_exposure": 0.8, "momentum_crash_risk": True}
    assert exposure_for("full", m) == 1.0 and exposure_for("balanced", m) == 0.5
    assert exposure_for("defensive", m) == 0.0
    with pytest.raises(ValueError):
        exposure_for("yolo", m)


def test_plan_buys_top_ranked_sells_dropouts_and_sizes_to_capital(panel):
    bench = panel.close["S01"]
    holdings = [Holding("LOSE", 100, 100.0, "2026-01-02")]
    p = plan(panel, bench, 100_000, holdings, n=4, today=date(2026, 9, 27), measured=[])
    buys = [a for a in p.actions if a.action == "BUY"]
    assert "WIN" in [a.symbol for a in buys]
    sell = next(a for a in p.actions if a.symbol == "LOSE")
    assert sell.action == "SELL" and sell.shares == 100
    assert sum(a.value for a in buys) <= p.capital + 1e-6
    assert all(a.stop_level == pytest.approx(round(a.price * 0.9, 2)) for a in buys)


def test_plan_skips_stocks_whose_single_share_exceeds_the_slot(panel):
    frames = {s: pd.DataFrame({c: getattr(panel, c)[s] for c in ("open", "high", "low", "close", "volume")})
              for s in panel.symbols}
    frames["WIN"] = frames["WIN"] * [100, 100, 100, 100, 1]
    p2 = panel_from_frames(frames)
    p = plan(p2, p2.close["S01"], 40_000, [], n=2, today=date(2026, 9, 27), measured=[])
    assert "WIN" not in [a.symbol for a in p.actions]
    assert any("WIN" in w for w in p.warnings)


def test_tax_note_defers_profitable_exits_near_long_term_status():
    h = Holding("A", 10, 100.0, "2025-11-01")
    assert _tax_note(h, 150.0, date(2026, 9, 27)).startswith("DEFER")
    assert "loss" in _tax_note(Holding("A", 10, 100.0, "2026-06-01"), 90.0, date(2026, 9, 27))
    assert "long-term loss" in _tax_note(Holding("A", 10, 100.0, "2024-01-01"), 90.0, date(2026, 9, 27))


def test_metrics_on_a_known_path():
    idx = pd.bdate_range(end=END, periods=253)
    eq = pd.Series(np.linspace(1.0, 1.2, 253), index=idx)
    m = metrics(eq)
    assert m["cagr_pct"] == pytest.approx(20.0, abs=0.5) and m["max_dd_pct"] == 0.0


def test_insider_flags_are_informational_and_do_not_change_selection(panel):
    bench = panel.close["S01"]
    asof = panel.close.index[-1]
    trades = pd.DataFrame({"broadcast": [asof - pd.Timedelta(days=10)], "symbol": ["WIN"],
                           "category": ["Promoter Group"], "direction": [1], "value": [5e7], "pid": ["x"]})
    with_flags = plan(panel, bench, 100_000, [], n=4, today=date(2026, 9, 27), insider_trades=trades, measured=[])
    without = plan(panel, bench, 100_000, [], n=4, today=date(2026, 9, 27), measured=[])
    assert [a.symbol for a in with_flags.actions] == [a.symbol for a in without.actions]
    win = next(a for a in with_flags.actions if a.symbol == "WIN")
    assert any("promoters bought ₹5.00 cr" in r for r in win.reasons)
