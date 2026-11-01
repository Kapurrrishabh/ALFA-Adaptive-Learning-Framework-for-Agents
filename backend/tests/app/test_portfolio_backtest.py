import io
import json
import logging
import re
import signal
import subprocess
import sys
import time
import tracemalloc
import zipfile
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.advisory.calculators.backtest import strategies as S
from backend.advisory.calculators.backtest.engine import BacktestConfig, run
from backend.database.sources import panel
from backend.database.sources.amfi import Scheme
from backend.database.sources.panel import Listing
from backend.user.portfolio import cas, cas_worker, importers
from backend.user.portfolio.portfolio import Transaction, analyze, build_positions, risk_decomposition, what_if
from backend.advisory.statistics.model_registry import ModelRegistry, PromotionError

from .conftest import bars


def tx(sym, side, q, px, d, sector=None, fees=0.0):
    return Transaction(sym, side, q, px, d, fees, sector)


def test_fifo_realized_pl_and_remaining_cost_basis():
    pos = build_positions([tx("A", "BUY", 10, 100, "2024-01-01"), tx("A", "BUY", 10, 120, "2024-02-01"),
                           tx("A", "SELL", 15, 130, "2024-03-01")])["A"]
    assert pos.realized_pl == pytest.approx(10 * 30 + 5 * 10)
    assert pos.quantity == 5 and pos.avg_cost == pytest.approx(120)


def test_open_lots_keep_the_dates_of_the_shares_still_held():
    pos = build_positions([tx("A", "BUY", 10, 100, "2024-01-01"), tx("A", "SELL", 10, 110, "2024-08-01"),
                           tx("A", "BUY", 10, 120, "2026-08-20"), tx("A", "BUY", 5, 130, "2026-09-01"),
                           tx("A", "SELL", 3, 125, "2026-09-10")])["A"]
    assert pos.lots == [("2026-08-20", 7, 120), ("2026-09-01", 5, 130)]


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
    from backend.advisory.statistics.monitoring import live_skill, resolve_predictions
    idx = pd.bdate_range(end="2026-09-25", periods=10)
    close = pd.Series(np.arange(100, 110, dtype=float), index=idx)
    store.log_prediction("logistic", "v1", "X.NS", 5, str(idx[2].date()), {"a": 1}, 0.7)
    store.log_prediction("logistic", "v1", "X.NS", 5, str(idx[8].date()), {"a": 2}, 0.7)  # not matured
    assert resolve_predictions(store, {"X.NS": close}) == 1
    m = live_skill(store)["logistic_h5"]
    assert m["n"] == 1 and m["hit_rate"] == 1.0 and m["brier"] == pytest.approx(0.09)


def test_psi_flags_a_volatility_regime_change():
    from backend.advisory.statistics.monitoring import psi
    rng = np.random.default_rng(0)
    calm, wild = rng.normal(0, 0.01, 500), rng.normal(0, 0.04, 60)
    assert psi(calm, rng.normal(0, 0.01, 60)) < 0.25 < psi(calm, wild)


def test_registry_rejects_a_candidate_that_does_not_beat_the_base_rate(store):
    reg = ModelRegistry(store)
    reg.register("forecast", "neg", training_period="t", features=[], hyperparameters={"models": ["gbm"]},
                 metrics={"brier_skill": -0.05}, backtest=None, limitations="")
    with pytest.raises(PromotionError, match="does not beat the baseline"):
        reg.approve("forecast", "neg", "rishabh")


# Laid out as NSE's files are: EQUITY_L's capitals and spaced headers, the ETF list's CRLF, SME's trailing comma
NSE_FILES = {
    "EQUITY_L.csv":
        "SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE\n"
        "RELIANCE,Reliance Industries Limited,EQ,29-NOV-1995,10,1,INE002A01018,10\n"
        "MOVEDUP,Moved Up Limited,EQ,01-JAN-2024,10,1,INE111A01011,10\n",
    "eq_etfseclist.csv":
        "Symbol,Underlying Asset,SecurityName,DateofListing,MarketLot,ISINNumber,FaceValue,ETF Underlying,"
        "Underlying Key\r\nNIFTYBEES,Nifty 50,NIPINDETFNIFTYBEES,08-Jan-02,1,INF204KB14I2,1,EQUITY,Nifty 50\r\n",
    "SME_EQUITY_L.csv":
        "SYMBOL,NAME_OF_COMPANY,SERIES,DATE_OF_LISTING,PAID_UP_VALUE,ISIN_NUMBER,FACE_VALUE,\n"
        "MOVEDUPSM,Moved Up Limited,SM,01-JAN-2020,10,INE111A01011,10,\n",
    "ind_nifty500list.csv":
        "Company Name,Industry,Symbol,Series,ISIN Code\n"
        "Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018\n"
        "Embassy Office Parks REIT,Realty,EMBASSY,RR,INE041025011\n",
}


@pytest.fixture
def nse_directory(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(panel, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(httpx, "get",
                        lambda url, **_: SimpleNamespace(status_code=200, text=NSE_FILES[url.rsplit("/", 1)[1]]))
    return panel.isin_directory()


def test_equity_l_is_read_despite_its_capitals_and_spaced_headers(nse_directory):
    assert nse_directory["INE002A01018"] == Listing("RELIANCE", "equity_l")


def test_an_etf_isin_maps_to_the_listed_etf(nse_directory):
    assert nse_directory["INF204KB14I2"] == Listing("NIFTYBEES", "etf")


def test_a_company_that_moved_up_from_sme_keeps_its_main_board_symbol(nse_directory):
    assert nse_directory["INE111A01011"] == Listing("MOVEDUP", "equity_l")


def test_reits_come_from_the_nifty_500_list(nse_directory):
    assert nse_directory["INE041025011"] == Listing("EMBASSY", "nifty500")


def test_index_constituents_refuses_a_list_that_is_not_an_index(nse_directory):
    with pytest.raises(ValueError, match="unknown NSE list 'equity_l'"):
        panel.index_constituents("equity_l")


def workbook(sheets):
    """An .xlsx built here, so no real statement or personal data enters the repo."""
    import openpyxl
    book = openpyxl.Workbook()
    book.remove(book.active)
    for title, rows in sheets.items():
        sheet = book.create_sheet(title)
        for row in rows:
            sheet.append(row)
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


ZERODHA_COLUMNS = ["Symbol", "ISIN", "Sector", "Quantity Available", "Quantity Discrepant", "Quantity Long Term",
                   "Quantity Pledged (Margin)", "Quantity Pledged (Loan)", "Average Price", "Previous Closing Price"]
HOLDING = {"Symbol": "SYNTHA", "ISIN": "INE000D01019", "Sector": "SYNTHETIC", "Quantity Available": 14,
           "Quantity Discrepant": 0, "Quantity Long Term": 0, "Quantity Pledged (Margin)": 0,
           "Quantity Pledged (Loan)": 0, "Average Price": 412.39, "Previous Closing Price": 455.5}
ZERODHA_FUND_COLUMNS = ["Symbol", "ISIN", "Instrument Type", "Quantity Available", "Quantity Discrepant",
                        "Quantity Pledged (Margin)", "Quantity Pledged (Loan)", "Average Price",
                        "Previous Closing Price", "Unrealized P&L", "Unrealized P&L Pct."]
FUND_HOLDING = {"Symbol": "SYNTHETIC MULTICAP FUND - DIRECT PLAN", "ISIN": "INF000S01011", "Quantity Available": 12.345,
                "Quantity Discrepant": 0, "Quantity Pledged (Margin)": 0, "Quantity Pledged (Loan)": 0,
                "Average Price": 810.0, "Previous Closing Price": 800.5}


def zerodha_holdings(*holdings, funds=()):
    """Zerodha's holdings workbook as third-party descriptions lay it out: Equity, Mutual Funds and Combined sheets,
    each with a client line, a dated title and a summary above its table."""
    def sheet(kind, columns, rows):
        return [["Client ID", "ZZ0000"], [], [f"{kind} Holdings Statement as on 2026-06-30"], [], ["Summary"],
                ["Invested Value", 1000.0], [], columns] + [[h.get(c) for c in columns] for h in rows]
    combined = ZERODHA_COLUMNS[:3] + ["Instrument Type"] + ZERODHA_COLUMNS[3:]
    return workbook({"Equity": sheet("Equity", ZERODHA_COLUMNS, holdings),
                     "Mutual Funds": sheet("Mutual Funds", ZERODHA_FUND_COLUMNS, funds),
                     "Combined": sheet("Combined", combined, [{**h, "Instrument Type": "-"} for h in holdings])})


def rewritten(data, change, compression=zipfile.ZIP_DEFLATED, added=()):
    """The workbook `data` with each part's bytes passed through change(name, part), plus the `added` (name, part)s."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w", compression) as dst:
        for entry in src.infolist():
            dst.writestr(entry.filename, change(entry.filename, src.read(entry)))
        for name, part in added:
            dst.writestr(name, part)
    return out.getvalue()


def test_holdings_rows_leave_the_buy_date_empty():
    rows = importers.parse(zerodha_holdings(HOLDING)).rows
    assert [(r.symbol, r.quantity, r.trade_date) for r in rows] == [("SYNTHA", 14, None)]


def test_a_sheet_that_understates_its_size_is_still_read_in_full():
    understated = rewritten(zerodha_holdings(HOLDING, HOLDING),
                            lambda _, part: re.sub(rb'<dimension ref="[^"]*"/>', b'<dimension ref="A1"/>', part))
    assert len(importers.parse(understated).rows) == 2


def test_the_sheets_left_unread_are_named():
    assert importers.parse(zerodha_holdings(HOLDING)).warnings == [
        "Only the 'Equity' and 'Mutual Funds' sheets were read, not 'Combined'."]


def test_pledged_shares_are_shown_in_the_note_not_added_to_the_quantity():
    row, = importers.parse(zerodha_holdings({**HOLDING, "Quantity Pledged (Margin)": 2})).rows
    assert row.quantity == 14 and "2 pledged for margin" in row.note


def test_a_groww_statement_is_dated_by_its_title():
    statement = importers.parse(workbook({"Sheet1": [
        ["Unique Client Code", "X1"], ["Holdings statement for stocks as on 01-09-2026"], [],
        ["Stock Name", "ISIN", "Quantity", "Average buy price", "Closing price"],
        ["RELIANCE INDUSTRIES LTD", "INE002A01018", 5, 1300.5, 1400]]}))
    assert (statement.source, statement.as_of) == ("Groww stocks", date(2026, 9, 1))


def test_an_unknown_layout_is_refused_with_the_layouts_we_read():
    with pytest.raises(ValueError, match="Zerodha holdings: .*Groww stocks: .*Zerodha tradebook: .*CSV template: "):
        importers.parse(workbook({"Sheet1": [["Name", "Shares"], ["Infosys", 3]]}))


def test_a_quantity_that_is_not_a_number_is_refused_with_its_row():
    with pytest.raises(ValueError, match="Row 9: Quantity Available is 'abc'"):
        importers.parse(zerodha_holdings({**HOLDING, "Quantity Available": "abc"}))


def test_a_number_too_long_for_a_float_is_refused_with_its_row():
    too_long = b"<v>" + b"9" * 400 + b"</v>"         # openpyxl cannot write a number this long, so it goes in by hand
    huge = rewritten(zerodha_holdings(HOLDING), lambda _, part: part.replace(b"<v>14</v>", too_long))
    with pytest.raises(ValueError, match="Row 9: Quantity Available is 9999"):
        importers.parse(huge)


def test_an_excel_file_that_unpacks_past_the_limit_is_refused():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        with archive.open("xl/worksheets/sheet1.xml", "w") as part:
            for _ in range(importers.MAX_TEXT_BYTES // 2**20 + 1):
                part.write(bytes(2**20))
    with pytest.raises(ValueError, match="unpacks to"):
        importers.parse(out.getvalue())


def test_a_wide_sparse_sheet_is_read_without_its_empty_cells():
    import openpyxl
    book = openpyxl.Workbook()
    for row in range(1, 2001):
        book.active.cell(row=row, column=16384, value=1)        # column XFD, Excel's last
    out = io.BytesIO()
    book.save(out)
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="doesn't match a layout"):
            importers.parse(out.getvalue())
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 20e6          # padding each row out to column XFD peaked at 263 MB


def test_a_sheet_that_skips_to_a_far_row_is_refused():
    import openpyxl
    book = openpyxl.Workbook()
    book.active.cell(row=importers.MAX_ROWS + 1, column=1, value="x")
    out = io.BytesIO()
    book.save(out)
    with pytest.raises(ValueError, match=f"more than {importers.MAX_ROWS:,} rows"):
        importers.parse(out.getvalue())


def test_a_workbook_with_more_sheets_than_we_read_is_refused():
    sheets = {f"Sheet{i}": [["a"]] for i in range(importers.MAX_SHEETS + 1)}
    with pytest.raises(ValueError, match=f"has {importers.MAX_SHEETS + 1} sheets"):
        importers.parse(workbook(sheets))


def chart_workbook():
    import openpyxl
    book = openpyxl.Workbook()
    book.create_chartsheet("Chart")
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def test_a_workbook_with_a_chart_sheet_is_refused():
    with pytest.raises(ValueError, match="'Chart' is a chart"):
        importers.parse(chart_workbook())


def test_two_sheets_naming_one_part_are_refused():
    twice = rewritten(workbook({"A": [["a"]]}), lambda _, part: re.sub(rb"<sheet .*?/>", lambda m: m[0] * 2, part))
    with pytest.raises(ValueError, match="two of its sheets name one part"):
        importers.parse(twice)


def test_many_cells_sharing_one_long_text_do_not_each_get_a_copy():
    """Every cell points at one text in the shared strings part, which openpyxl never writes, so it is put in here."""
    texts = (f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>'
             f'{"a " * 10_000}</t></si></sst>').encode()
    row = b"<row>" + b'<c t="s"><v>0</v></c>' * importers.MAX_COLUMNS + b"</row>"

    def share(name, part):
        if name == "[Content_Types].xml":
            return part.replace(b"</Types>", b'<Override PartName="/xl/sharedStrings.xml" ContentType="application/'
                                b'vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>')
        return re.sub(rb"<sheetData>.*</sheetData>", b"<sheetData>" + row * 100 + b"</sheetData>", part)
    shared = rewritten(workbook({"Sheet1": [["a"]]}), share, added=[("xl/sharedStrings.xml", texts)])
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="doesn't match a layout"):
            importers.parse(shared)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 20e6          # a tidied copy for each of the 2,600 cells peaked at 52 MB; one shared text, 1.1 MB


@pytest.mark.parametrize("sheet", ["Combined", "Equity"], ids=["unread sheet", "above the header"])
def test_a_long_cell_outside_the_rows_read_is_not_refused(sheet):
    import openpyxl
    book = openpyxl.load_workbook(io.BytesIO(zerodha_holdings(HOLDING)))
    book[sheet]["A2"] = "Disclaimer: " + "x " * importers.MAX_CELL_CHARS
    out = io.BytesIO()
    book.save(out)
    assert len(importers.parse(out.getvalue()).rows) == 1


def test_a_long_cell_in_a_row_read_is_refused_with_its_row_and_column():
    too_long = importers.MAX_CELL_CHARS + 1
    with pytest.raises(ValueError, match=f"Row 9: Symbol has {too_long} characters"):
        importers.parse(zerodha_holdings({**HOLDING, "Symbol": "S" * too_long}))


def not_a_workbook(_):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("notes.txt", "not a workbook")
    return out.getvalue()


def encrypted(good, part="xl/workbook.xml"):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(good)) as src, zipfile.ZipFile(out, "w") as archive:
        for entry in src.infolist():
            archive.writestr(entry.filename, src.read(entry))
        archive.getinfo(part).flag_bits |= 1     # zipfile writes this flag when it closes
    return out.getvalue()


LONG = "N" * 10_000
TOO_MANY_DIGITS = b"<v>" + b"9" * (sys.get_int_max_str_digits() + 1) + b"</v>"


@pytest.mark.parametrize("damage", [
    lambda good: rewritten(good, lambda name, part: part[:-20] if name.endswith("sheet1.xml") else part),
    lambda good: rewritten(good, lambda _, part: part, zipfile.ZIP_STORED).replace(b"SYNTHA", b"SYNTHZ", 1),
    not_a_workbook, encrypted,
    lambda good: rewritten(good, lambda _, part: part.replace(b"<v>14</v>", TOO_MANY_DIGITS))],
    ids=["broken XML", "bad checksum", "not a workbook", "encrypted part", "number past Python's digit limit"])
def test_a_damaged_workbook_is_refused_as_a_bad_file(damage):
    with pytest.raises(ValueError, match="damaged or is not a workbook"):
        importers.parse(damage(zerodha_holdings(HOLDING)))


@pytest.mark.parametrize("named", [
    lambda: rewritten(chart_workbook(), lambda _, part: part.replace(b'name="Chart"', f'name="{LONG}"'.encode())),
    lambda: encrypted(rewritten(zerodha_holdings(HOLDING), lambda _, part: part, added=[(LONG, b"")]), LONG),
    lambda: rewritten(zerodha_holdings(HOLDING),
                      lambda _, part: part.replace(b'"/xl/workbook.xml"', f'"/xl/{LONG}.xml"'.encode())),
    lambda: workbook({"Sheet1": [[f"Holdings statement for stocks as on {LONG}"],
                                 ["Stock Name", "ISIN", "Quantity", "Average buy price"]]}),
    lambda: rewritten(workbook({"A": [CSV_HEADER.strip().split(",")], "B": [CSV_HEADER.strip().split(",")]}),
                      lambda _, part: part.replace(b'name="A"', f'name="{LONG}"'.encode())),
    lambda: rewritten(zerodha_holdings(HOLDING), lambda _, part: part.replace(b"<v>14</v>", TOO_MANY_DIGITS)
                      .replace(b'name="Equity"', f'name="{LONG}"'.encode()))],
    ids=["chart sheet", "encrypted part", "missing part", "title", "sheet of a second layout", "sheet of a bad cell"])
def test_a_long_name_from_the_file_is_cut_short_in_the_refusal(named):
    with pytest.raises(ValueError) as refused:
        importers.parse(named())
    assert len(str(refused.value)) < 1_000


@pytest.mark.parametrize("sheet", [b"Equity", b"Combined"], ids=["sheet read", "sheet left unread"])
def test_a_long_sheet_name_is_cut_short_in_the_warning(sheet):
    renamed = rewritten(zerodha_holdings(HOLDING),
                        lambda _, part: part.replace(b'name="%s"' % sheet, f'name="{LONG}"'.encode()))
    warning, = importers.parse(renamed).warnings
    assert len(warning) < 1_000


def test_a_pdf_statement_is_unsupported_rather_than_misread():
    with pytest.raises(importers.ImportUnsupported):
        importers.parse(b"%PDF-1.7\n")


def test_the_csv_template_reads_back_what_was_written():
    text = ("\ufeffsymbol_or_isin,side,quantity,price,trade_date\n"      # Excel starts a UTF-8 CSV with a BOM
            "INE009A01021,BUY,10,1500.5,2024-01-15\ntcs.ns,sell,2,,\n")
    rows = importers.parse(text.encode()).rows
    assert [(r.isin, r.symbol, r.side, r.quantity, r.price, r.trade_date) for r in rows] == [
        ("INE009A01021", None, "BUY", 10, 1500.5, date(2024, 1, 15)), (None, "TCS.NS", "SELL", 2, None, None)]


CSV_HEADER = "symbol_or_isin,side,quantity,price,trade_date\n"


def test_a_csv_file_over_the_size_limit_is_refused():
    text = CSV_HEADER + "INFY,BUY,1,,\n" * (importers.MAX_TEXT_BYTES // 12)
    with pytest.raises(ValueError, match="This CSV file is"):
        importers.parse(text.encode() + b"\xff")      # not UTF-8 either: the size is checked before decoding


def test_a_csv_field_too_long_to_read_is_refused_with_its_line():
    text = CSV_HEADER + '"' + "a" * 200_000 + '",BUY,1,,\n'
    with pytest.raises(ValueError, match="Line 2 of this CSV file cannot be read"):
        importers.parse(text.encode())


def test_a_tradebook_keeps_each_trades_side_and_date():
    text = ("symbol,isin,trade_date,exchange,segment,series,trade_type,auction,quantity,price,trade_id,order_id,"
            "order_execution_time\n"
            "INFY,INE009A01021,2024-03-01,NSE,EQ,EQ,sell,false,5,1600.25,1,2,2024-03-01T10:00:00\n")
    row, = importers.parse(text.encode()).rows
    assert (row.side, row.trade_date) == ("SELL", date(2024, 3, 1))


DIRECTORY = {"INE002A01018": Listing("RELIANCE", "equity_l"), "INE000E01027": Listing("SYNTHB", "equity_l"),
             "INE155A01022": Listing("TATAMOTORS", "equity_l"), "INE081A01020": Listing("TATASTEEL", "equity_l"),
             "INE245A01021": Listing("TATAPOWER", "equity_l"), "INE151A01013": Listing("TATACOMM", "equity_l")}
NAMES = {"RELIANCE": "Reliance Industries", "TATAMOTORS": "Tata Motors", "TATASTEEL": "Tata Steel",
         "TATAPOWER": "Tata Power Co.", "TATACOMM": "Tata Communications"}


def file_row(isin=None, symbol=None, name=None, quantity=1.0):
    return importers.Row(line=2, name=name, isin=isin, symbol=symbol, quantity=quantity, price=None,
                         trade_date=None, side="BUY", note=None, fees=None, scheme_code=None, doubt=None)


def test_an_isin_maps_to_its_listed_symbol():
    found, = importers.match([file_row(isin="INE002A01018")], DIRECTORY, NAMES)
    assert (found.state, found.listing) == ("matched", Listing("RELIANCE", "equity_l"))


def test_the_trading_series_is_stripped_from_a_symbol():
    found, = importers.match([file_row(symbol="SYNTHB-BE")], DIRECTORY, NAMES)
    assert found.listing == Listing("SYNTHB", "equity_l")


def test_a_symbol_whose_isin_the_lists_lack_is_only_offered():
    found, = importers.match([file_row(isin="INE999Z01019", symbol="SYNTHB")], DIRECTORY, NAMES)
    assert (found.state, found.listing, found.choices) == ("check", None, [Listing("SYNTHB", "equity_l")])


def test_a_holding_with_no_free_shares_is_only_offered():
    found, = importers.match([file_row(isin="INE002A01018", quantity=0.0)], DIRECTORY, NAMES)
    assert (found.state, found.choices) == ("check", [Listing("RELIANCE", "equity_l")])


def test_an_unknown_isin_is_not_found():
    found, = importers.match([file_row(isin="INE999Z01019", symbol="GONE")], DIRECTORY, NAMES)
    assert found.state == "not_found"


def test_a_name_that_fits_one_company_is_still_only_offered():
    found, = importers.match([file_row(isin="INE999Z01019", name="TATA MOTORS LTD")], DIRECTORY, NAMES)
    assert (found.state, found.listing, found.choices) == ("check", None, [Listing("TATAMOTORS", "equity_l")])


def test_a_name_offers_at_most_three_choices():
    found, = importers.match([file_row(name="TATA LTD")], DIRECTORY, NAMES)
    assert len(found.choices) == 3


def test_every_row_is_counted_in_one_state():
    rows = [file_row(isin="INE002A01018"), file_row(name="Tata Steel Limited"), file_row(symbol="GONE")]
    assert importers.counts(importers.match(rows, DIRECTORY, NAMES)) == {"matched": 1, "check": 1, "not_found": 1}


def amfi_scheme(code, isin, plan):
    return Scheme(code=code, isin_growth=isin, isin_reinvestment=None, fund="Synthetic Multicap Fund", plan=plan,
                  option="Growth", nav=800.5, nav_date=date(2026, 9, 30), scheme_type="Open Ended Schemes",
                  category="Equity Scheme - Multi Cap Fund", amc="Synthetic Mutual Fund")


DIRECT_GROWTH = amfi_scheme(100001, "INF000S01011", "Direct Plan")
REGULAR_GROWTH = amfi_scheme(100002, "INF000S01029", "Regular Plan")
SCHEMES = [DIRECT_GROWTH, REGULAR_GROWTH]
GROWW_ORDERS = [["Personal Details"], ["Name", "A. Holder"], ["PAN", "ABCDE1234F"], [],
                ["TRANSACTIONS FROM Jan 01 2025 TO Jun 30 2026"], [],
                ["Scheme Name", "Transaction Type", "Units", "NAV", "Amount", "Date"],
                ["Synthetic Multicap Fund Direct Growth", "PURCHASE", "10", "20", "200", "15 Jan 2026"],
                ["Synthetic Multicap Fund Direct Growth", "REDEEM", "4", "30", "120", "20 Mar 2026"]]


def test_a_zerodha_fund_sheet_isin_maps_to_its_scheme():
    fund, = importers.match_funds(importers.parse(zerodha_holdings(HOLDING, funds=[FUND_HOLDING])).funds, SCHEMES)
    assert (fund.state, fund.scheme) == ("matched", DIRECT_GROWTH)


def test_zerodha_sheets_dated_apart_are_refused_as_two_statements():
    title = b"Mutual Funds Holdings Statement as on 2026-06-"
    apart = rewritten(zerodha_holdings(HOLDING, funds=[FUND_HOLDING]),
                      lambda _, part: part.replace(title + b"30", title + b"29"))
    with pytest.raises(ValueError, match="Upload one statement at a time"):
        importers.parse(apart)


def test_a_groww_fund_order_history_keeps_each_orders_side_and_date():
    funds = importers.parse(workbook({"Transactions": GROWW_ORDERS})).funds
    assert [(r.side, r.trade_date) for r in funds] == [("BUY", date(2026, 1, 15)), ("SELL", date(2026, 3, 20))]


def test_groww_fund_rows_are_never_matched_automatically():
    found = importers.match_funds(importers.parse(workbook({"Transactions": GROWW_ORDERS})).funds, SCHEMES)
    assert [(m.state, m.choices) for m in found] == [("check", [DIRECT_GROWTH])] * 2


def test_a_file_naming_more_funds_than_we_search_for_is_refused(monkeypatch):
    monkeypatch.setattr(importers, "MAX_FUND_NAMES", 1)
    with pytest.raises(ValueError, match="names more than 1 funds"):
        importers.match_funds([file_row(name="Synthetic Multicap"), file_row(name="Synthetic Flexi Cap")], SCHEMES)


PDF, PASSWORD = b"%PDF-1.7 synthetic statement", "Pass-phrase-7"
HOLDER = {"name": "A. Holder", "email": "holder@example.com", "address": "1 Example Road", "mobile": "9800000000"}
CAMS_CAS = {    # laid out as casparser 1.4.1 writes its JSON, with the holder's details that must not pass on
    "statement_period": {"from": "01-Apr-2024", "to": "30-Sep-2026"}, "investor_info": HOLDER,
    "cas_type": "DETAILED", "file_type": "CAMS", "parse_warnings": ["unparseable header: 'Nominee: B. Holder'"],
    "folios": [{"folio": "1234567/89", "amc": "Synthetic Mutual Fund", "name": "A. Holder", "PAN": "ABCDE1234F",
                "KYC": "OK", "PANKYC": "OK", "schemes": [{
                    "scheme": "Synthetic Multicap Fund - Direct Plan - Growth", "advisor": "INA000000000",
                    "rta_code": "S01", "rta": "CAMS", "type": "EQUITY", "isin": "INF000S01011", "amfi": "100001",
                    "nominees": ["B. Holder"], "open": "0.000", "close": "12.345", "close_calculated": "12.345",
                    "valuation": {"date": "2026-09-30", "nav": "800.50", "cost": "9999.50", "value": "9882.17"},
                    "transactions": [
                        {"date": "2024-05-10", "description": "Purchase by A. Holder", "amount": "9999.50",
                         "units": "12.345", "nav": "810.00", "balance": "12.345", "type": "PURCHASE",
                         "dividend_rate": None, "gift_folio": None},
                        {"date": "2024-05-10", "description": "*** Stamp Duty ***", "amount": "0.50", "units": None,
                         "nav": None, "balance": None, "type": "STAMP_DUTY_TAX", "dividend_rate": None,
                         "gift_folio": None}]}]}]}
CAMS_KEPT = {
    "statement_period": {"from": "01-Apr-2024", "to": "30-Sep-2026"}, "cas_type": "DETAILED", "file_type": "CAMS",
    "parse_warning_count": 1, "folios": [{"schemes": [{
        "scheme": "Synthetic Multicap Fund - Direct Plan - Growth", "isin": "INF000S01011", "amfi": "100001",
        "open": "0.000", "close": "12.345", "transactions": [
            {"date": "2024-05-10", "type": "PURCHASE", "units": "12.345", "nav": "810.00", "amount": "9999.50"},
            {"date": "2024-05-10", "type": "STAMP_DUTY_TAX", "units": None, "nav": None, "amount": "0.50"}]}]}]}
NSDL_CAS = {
    "statement_period": {"from": "01-Sep-2026", "to": "30-Sep-2026"}, "investor_info": HOLDER, "file_type": "NSDL",
    "parse_warnings": [], "nps": {"pran": "110000000000", "nps_sp": "PROT", "value": "500.00", "schemes": [
        {"scheme": "Synthetic Pension Fund E - Tier I", "fund_manager": "Synthetic", "tier": "I", "asset_class": "E",
         "units": "10", "nav": "50", "value": "500.00"}]},
    "accounts": [{"name": "Synthetic Broking Limited", "type": "CDSL", "dp_id": "12345678", "client_id": "87654321",
                  "folios": 1, "balance": "16000.00", "owners": [{"name": "A. Holder", "PAN": "ABCDE1234F"}],
                  "equities": [{"name": "SYNTHETIC INDUSTRIES LTD", "isin": "INE000D01019", "num_shares": "5",
                                "price": "1400.00", "value": "7000.00", "symbol": "SYNTHA", "exchange": "NSE"}],
                  "mutual_funds": [{"name": "SYNTHETIC MULTICAP DIRECT GROWTH", "isin": "INF000S01011",
                                    "amfi": "100001", "type": "EQUITY", "balance": "10.000", "nav": "800.00",
                                    "value": "8000.00", "avg_cost": "700.00", "total_cost": "7000.00",
                                    "ucc": "MFUCC123", "folio": "1234567/89", "pnl": None, "return": None}],
                  "bonds": [{"name": "SYNTHETIC BOND", "isin": "INE000B07019", "num_bonds": "1", "value": "1000.00",
                             "face_value": None, "coupon_rate": None, "coupon_frequency": None,
                             "maturity_date": None, "market_price": None}]}]}
NSDL_KEPT = {
    "statement_period": {"from": "01-Sep-2026", "to": "30-Sep-2026"}, "file_type": "NSDL", "parse_warning_count": 0,
    "nps": {"schemes": [{"scheme": "Synthetic Pension Fund E - Tier I", "units": "10", "nav": "50",
                         "value": "500.00"}]},
    "accounts": [{"equities": [{"name": "SYNTHETIC INDUSTRIES LTD", "isin": "INE000D01019", "num_shares": "5",
                                "value": "7000.00"}],
                  "mutual_funds": [{"name": "SYNTHETIC MULTICAP DIRECT GROWTH", "isin": "INF000S01011",
                                    "amfi": "100001", "balance": "10.000", "nav": "800.00", "avg_cost": "700.00"}],
                  "bonds": [{"name": "SYNTHETIC BOND", "isin": "INE000B07019", "num_bonds": "1",
                             "value": "1000.00"}]}]}


@pytest.fixture
def casparser(tmp_path, monkeypatch):
    """Runs cas.read's worker with a stub casparser whose read_cas_pdf(file, password, output) runs the line `body`,
    or with no casparser when `body` is None. -I ignores PYTHONPATH, so a boot script puts the stub on its path."""
    def stub(body):
        boot = ["import runpy, sys"]
        if body is None:
            boot.append("sys.modules['casparser'] = None")     # as if not installed, even where it is
        else:
            (tmp_path / "casparser").mkdir()
            (tmp_path / "casparser" / "exceptions.py").write_text(
                "class CASParseError(Exception): pass\nclass IncorrectPasswordError(CASParseError): pass\n")
            (tmp_path / "casparser" / "__init__.py").write_text(
                f"import json, os, resource, sys, time\nfrom .exceptions import *\n"
                f"def read_cas_pdf(file, password, output):\n    {body}\n")
            boot.append(f"sys.path.insert(0, {str(tmp_path)!r})")
        boot.append(f"runpy.run_path({str(cas.WORKER)!r}, run_name='__main__')")
        (tmp_path / "boot.py").write_text("\n".join(boot))
        monkeypatch.setattr(cas, "WORKER", tmp_path / "boot.py")
    return stub


@pytest.fixture
def seen_in_cas_reader(casparser, tmp_path):
    """The JSON value of a Python expression evaluated inside the CAS reader."""
    def seen(expression):
        casparser(f"open({str(tmp_path / 'seen')!r}, 'w').write(json.dumps({expression})); raise CASParseError('x')")
        with pytest.raises(ValueError):
            cas.read(PDF, PASSWORD)
        return json.loads((tmp_path / "seen").read_text())
    return seen


@pytest.mark.parametrize("statement, kept", [(CAMS_CAS, CAMS_KEPT), (NSDL_CAS, NSDL_KEPT)], ids=["CAMS", "NSDL"])
def test_a_cas_keeps_its_holdings_but_not_who_holds_them(casparser, statement, kept):
    casparser(f"assert (file.read(), password, output) == ({PDF!r}, {PASSWORD!r}, 'json'); "
              f"return {json.dumps(statement)!r}")
    assert cas.read(PDF, PASSWORD) == kept


def test_a_wrong_cas_password_is_refused_in_plain_words(casparser):
    casparser("raise IncorrectPasswordError('Incorrect PDF password!')")
    with pytest.raises(ValueError, match="That password didn't open the file. .* usually your PAN in capitals"):
        cas.read(PDF, PASSWORD)


def test_a_pdf_that_is_not_a_cas_is_refused_with_casparsers_reason(casparser):
    casparser("raise CASParseError('Could not identify the CAS issuer.')")
    with pytest.raises(ValueError, match="isn't a CAS statement we can read: Could not identify the CAS issuer"):
        cas.read(PDF, PASSWORD)


def test_a_cas_that_reads_too_long_is_stopped_at_the_time_limit(casparser, monkeypatch):
    readers = []

    class RecordedPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            readers.append(self)
    casparser("time.sleep(60)")
    monkeypatch.setattr(cas, "CAS_TIMEOUT_S", 1)
    monkeypatch.setattr(subprocess, "Popen", RecordedPopen)
    started = time.monotonic()
    with pytest.raises(ValueError, match="took more than 1 seconds to read"):
        cas.read(PDF, PASSWORD)
    assert time.monotonic() - started < 5
    assert readers[0].returncode == -signal.SIGKILL     # stopped, not left running


def test_a_server_without_casparser_refuses_a_pdf_in_the_importers_words(casparser):
    casparser(None)
    with pytest.raises(importers.ImportUnsupported) as refused:
        cas.read(PDF, PASSWORD)
    with pytest.raises(importers.ImportUnsupported) as parsed:
        importers.parse(PDF)
    assert str(refused.value) == str(parsed.value)


def test_a_casparser_with_a_missing_dependency_is_a_server_error(casparser, tmp_path):
    casparser("pass")
    (tmp_path / "casparser" / "__init__.py").write_text("import casparser_missing_dependency\n")
    with pytest.raises(RuntimeError, match="ModuleNotFoundError: No module named 'casparser_missing_dependency'"):
        cas.read(PDF, PASSWORD)


def test_the_cas_password_is_never_on_the_command_line_or_in_the_logs(casparser, tmp_path, caplog):
    casparser(f"open({str(tmp_path / 'argv')!r}, 'w').write(json.dumps(sys.orig_argv)); "
              "raise IncorrectPasswordError('Incorrect PDF password!')")
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ValueError) as refused:
        cas.read(PDF, PASSWORD)
    assert PASSWORD not in (tmp_path / "argv").read_text() + caplog.text + str(refused.value)


def test_the_cas_reader_does_not_inherit_the_servers_environment(seen_in_cas_reader, monkeypatch):
    monkeypatch.setenv("STOCKINTEL_AUTH_SECRET", "not-for-the-reader")
    assert "STOCKINTEL_AUTH_SECRET" not in seen_in_cas_reader("dict(os.environ)")


def test_the_cas_reader_runs_in_pythons_isolated_mode(seen_in_cas_reader):
    assert seen_in_cas_reader("sys.flags.isolated") == 1


@pytest.mark.skipif(sys.platform != "linux", reason="macOS does not enforce RLIMIT_AS, so the reader sets it on Linux")
def test_the_cas_reader_has_a_memory_limit_on_linux(seen_in_cas_reader):
    assert seen_in_cas_reader("resource.getrlimit(resource.RLIMIT_AS)") == [cas_worker.MAX_MEMORY_BYTES] * 2


def test_a_cas_reader_failure_names_its_type_but_not_its_message(casparser):
    casparser("raise KeyError('ABCDE1234F')")
    with pytest.raises(RuntimeError, match="failed with KeyError") as failed:
        cas.read(PDF, PASSWORD)
    assert "ABCDE1234F" not in str(failed.value)


def test_a_cas_reader_that_dies_is_a_server_error(casparser):
    casparser("os._exit(3)")
    with pytest.raises(RuntimeError, match="exit code 3"):
        cas.read(PDF, PASSWORD)


def test_a_cas_reader_that_prints_is_a_server_error_not_a_bad_file(casparser):
    casparser(f"print('reading'); return {json.dumps(CAMS_CAS)!r}")
    with pytest.raises(RuntimeError, match="reply is not JSON"):
        cas.read(PDF, PASSWORD)


def test_a_cas_field_that_casparser_renamed_is_a_server_error_not_dropped(casparser):
    statement = json.loads(json.dumps(NSDL_CAS))
    equity, = statement["accounts"][0]["equities"]
    equity["shares"] = equity.pop("num_shares")
    casparser(f"return {json.dumps(statement)!r}")
    with pytest.raises(RuntimeError, match="failed with KeyError"):
        cas.read(PDF, PASSWORD)


def cams(transactions, close, opened="0.000", warnings=0, **scheme):
    """What cas.read gives for CAMS_CAS, with its one scheme's transactions, balances and other `scheme` keys set."""
    parsed = json.loads(json.dumps(CAMS_KEPT))
    parsed["folios"][0]["schemes"][0].update(transactions=transactions, open=opened, close=close, **scheme)
    return {**parsed, "parse_warning_count": warnings}


def cas_trade(kind, units, nav="810.00", amount=None):
    return {"date": "2024-05-10", "type": kind, "units": units, "nav": nav, "amount": amount}


@pytest.mark.parametrize("kind, units, side", [
    ("PURCHASE", "1.000", "BUY"), ("PURCHASE_SIP", "1.000", "BUY"), ("SWITCH_IN", "1.000", "BUY"),
    ("DIVIDEND_REINVEST", "1.000", "BUY"), ("REDEMPTION", "-1.000", "SELL"), ("SWITCH_OUT", "-1.000", "SELL")])
def test_a_cas_purchase_or_sale_becomes_a_buy_or_sell(kind, units, side):
    row, = cas.statement(cams([cas_trade(kind, units)], close="1.000")).funds
    assert (row.side, row.quantity, row.price, row.trade_date) == (side, 1.0, 810.0, date(2024, 5, 10))


def test_stamp_duty_becomes_the_fees_of_the_purchase_it_follows():
    row, = cas.statement(cams([cas_trade("PURCHASE", "12.345"), cas_trade("STAMP_DUTY_TAX", None, None, "0.50")],
                              close="12.345")).funds
    assert row.fees == 0.5


def test_cas_tax_rows_are_left_out_with_a_note():
    parsed = cas.statement(cams([cas_trade("PURCHASE", "1.000"), cas_trade("REDEMPTION", "-1.000"),
                                 cas_trade("STT_TAX", None, None, "0.01")], close="0.000"))
    assert ([r.side for r in parsed.funds], parsed.warnings) == (
        ["BUY", "SELL"], ["This statement's rows that are not purchases or sales are left out: stt tax (1)."])


def test_units_held_before_a_cas_began_are_a_row_with_no_date_or_cost():
    first, = cas.statement(cams([], opened="5.000", close="5.000")).funds
    assert (first.quantity, first.trade_date, first.price) == (5.0, None, None)


def test_a_cas_scheme_whose_isin_code_plan_and_units_agree_is_matched():
    funds = cas.statement(cams([cas_trade("PURCHASE", "12.345")], close="12.345")).funds
    assert [m.state for m in importers.match_funds(funds, SCHEMES)] == ["matched"]


def test_a_cas_units_mismatch_makes_the_scheme_check():
    funds = cas.statement(cams([cas_trade("PURCHASE", "10.000")], close="12.345")).funds
    assert [m.state for m in importers.match_funds(funds, SCHEMES)] == ["check"]


def test_casparsers_warnings_make_its_rows_check():
    funds = cas.statement(cams([cas_trade("PURCHASE", "12.345")], close="12.345", warnings=1)).funds
    assert [m.state for m in importers.match_funds(funds, SCHEMES)] == ["check"]


def test_casparsers_warnings_make_a_demat_share_check_too():
    shares = cas.statement({**NSDL_KEPT, "parse_warning_count": 1}).rows
    assert [m.state for m in importers.match(shares, {"INE000D01019": Listing("SYNTHA", "equity_l")}, {})] == ["check"]


def test_a_casparser_code_missing_from_amfis_list_is_only_offered():
    funds = cas.statement(cams([cas_trade("PURCHASE", "12.345")], close="12.345", amfi="999999")).funds
    found, = importers.match_funds(funds, SCHEMES)
    assert (found.state, found.choices) == ("check", [DIRECT_GROWTH])


def test_casparser_giving_a_regular_holding_the_direct_plan_offers_the_regular_one_first():
    # casparser-isin's README shows it: "Axis Long Term Equity Fund - Growth" got the Direct plan with a score of 100
    funds = cas.statement(cams([cas_trade("PURCHASE", "12.345")], close="12.345",
                               scheme="Synthetic Multicap Fund - Growth")).funds
    found, = importers.match_funds(funds, SCHEMES)
    assert (found.state, found.choices) == ("check", [REGULAR_GROWTH, DIRECT_GROWTH])


def test_a_summary_cas_is_refused_with_the_detailed_one_asked_for():
    with pytest.raises(ValueError, match="Ask CAMS or KFintech for the detailed one"):
        cas.statement({**cams([], close="0.000"), "cas_type": "SUMMARY"})


def test_demat_shares_come_with_no_cost_so_it_is_asked_for():
    share, = cas.statement(NSDL_KEPT).rows
    assert (share.isin, share.quantity, share.price) == ("INE000D01019", 5.0, None)


def test_a_demat_fund_is_matched_by_its_isin_and_code():
    fund, = importers.match_funds(cas.statement(NSDL_KEPT).funds, SCHEMES)
    assert (fund.state, fund.scheme, fund.row.quantity, fund.row.price) == ("matched", DIRECT_GROWTH, 10.0, 700.0)


def test_demat_bonds_and_pension_schemes_are_named_as_left_out():
    assert cas.statement(NSDL_KEPT).warnings == [
        "This statement's bonds (1) and NPS pension schemes (1) are left out: a portfolio here holds only shares and "
        "funds."]
