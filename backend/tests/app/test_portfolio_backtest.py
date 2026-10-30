import io
import re
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
from backend.database.sources.panel import Listing
from backend.user.portfolio import importers
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


def zerodha_holdings(*holdings):
    """Zerodha's holdings workbook as third-party descriptions lay it out: Equity, Mutual Funds and Combined sheets,
    each with a client line, a dated title and a summary above its table."""
    def sheet(kind, columns, rows):
        return [["Client ID", "ZZ0000"], [], [f"{kind} Holdings Statement as on 2026-06-30"], [], ["Summary"],
                ["Invested Value", 1000.0], [], columns] + [[h.get(c) for c in columns] for h in rows]
    combined = ZERODHA_COLUMNS[:3] + ["Instrument Type"] + ZERODHA_COLUMNS[3:]
    return workbook({"Equity": sheet("Equity", ZERODHA_COLUMNS, holdings),
                     "Mutual Funds": sheet("Mutual Funds", [], []),
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
        "Only the 'Equity' sheet was read, not 'Mutual Funds' or 'Combined'."]


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


def test_a_workbook_with_a_chart_sheet_is_refused():
    import openpyxl
    book = openpyxl.Workbook()
    book.create_chartsheet("Chart")
    out = io.BytesIO()
    book.save(out)
    with pytest.raises(ValueError, match="'Chart' is a chart"):
        importers.parse(out.getvalue())


def test_two_sheets_naming_one_part_are_refused():
    twice = rewritten(workbook({"A": [["a"]]}), lambda _, part: re.sub(rb"<sheet .*?/>", lambda m: m[0] * 2, part))
    with pytest.raises(ValueError, match="two of its sheets name one part"):
        importers.parse(twice)


def test_many_cells_sharing_one_long_text_are_refused():
    """Every cell points at one text in the shared strings part, which openpyxl never writes, so it is put in here."""
    texts = (f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>'
             f'{"a " * importers.MAX_CELL_CHARS}</t></si></sst>').encode()
    row = b"<row>" + b'<c t="s"><v>0</v></c>' * importers.MAX_COLUMNS + b"</row>"

    def share(name, part):
        if name == "[Content_Types].xml":
            return part.replace(b"</Types>", b'<Override PartName="/xl/sharedStrings.xml" ContentType="application/'
                                b'vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>')
        return re.sub(rb"<sheetData>.*</sheetData>", b"<sheetData>" + row * 100 + b"</sheetData>", part)
    shared = rewritten(workbook({"Sheet1": [["a"]]}), share, added=[("xl/sharedStrings.xml", texts)])
    with pytest.raises(ValueError, match=f"we read cells of up to {importers.MAX_CELL_CHARS}"):
        importers.parse(shared)


def not_a_workbook(_):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("notes.txt", "not a workbook")
    return out.getvalue()


def encrypted(good):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(good)) as src, zipfile.ZipFile(out, "w") as archive:
        for entry in src.infolist():
            archive.writestr(entry.filename, src.read(entry))
        archive.getinfo("xl/workbook.xml").flag_bits |= 1     # zipfile writes this flag when it closes
    return out.getvalue()


@pytest.mark.parametrize("damage", [
    lambda good: rewritten(good, lambda name, part: part[:-20] if name.endswith("sheet1.xml") else part),
    lambda good: rewritten(good, lambda _, part: part, zipfile.ZIP_STORED).replace(b"SYNTHA", b"SYNTHZ", 1),
    not_a_workbook, encrypted], ids=["broken XML", "bad checksum", "not a workbook", "encrypted part"])
def test_a_damaged_workbook_is_refused_as_a_bad_file(damage):
    with pytest.raises(ValueError, match="damaged or is not a workbook"):
        importers.parse(damage(zerodha_holdings(HOLDING)))


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
                         trade_date=None, side="BUY", note=None)


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
