from datetime import datetime, timezone

import pandas as pd
import pytest

from backend.advisory.fundamental import fundamentals
from backend.advisory.service import AnalysisService
from backend.advisory.sentiment import news
from backend.advisory.sentiment.sentiment import LexiconScorer

from .conftest import NOW, news_item, statements

INFO = {"sector": "Industrials", "industry": "Engineering", "currentPrice": 300.0, "marketCap": 3000.0}


def _stmts(**over):
    base = dict(revenue=[1200, 1000], net_income=[150, 120], eps=[15, 12], equity=[1000, 900],
                debt=[300, 300], assets=[2000, 1800], ebit=[220, 180], interest=[20, 20],
                ca=[600, 550], cl=[400, 380], fcf=[140, 110])
    base.update(over)
    return statements(**base)


def test_ratios_are_computed_exactly_from_line_items():
    r = fundamentals.analyze("X.NS", INFO, _stmts())
    m = r.details["metrics"]
    assert m["revenue_growth"]["value"] == pytest.approx(20.0)
    assert m["net_margin"]["value"] == pytest.approx(12.5)
    assert m["roe"]["value"] == pytest.approx(15.0)
    assert m["debt_to_equity"]["value"] == pytest.approx(0.3)
    assert m["interest_coverage"]["value"] == pytest.approx(11.0)
    assert m["current_ratio"]["value"] == pytest.approx(1.5)
    assert m["fcf_yield"]["value"] == pytest.approx(140 / 3000 * 100, abs=1e-3)


def test_every_metric_carries_its_period_and_source():
    r = fundamentals.analyze("X.NS", INFO, _stmts(), source="yahoo")
    rg = r.details["metrics"]["revenue_growth"]
    assert "2026-03-31" in rg["period"] and rg["source"].startswith("yahoo:")


def test_missing_line_items_are_reported_not_filled():
    stm = _stmts()
    stm["income"] = stm["income"].drop(index=["Diluted EPS"])
    r = fundamentals.analyze("X.NS", {"sector": "Industrials"}, stm)
    assert "eps_growth" in r.details["unavailable"]
    assert r.details["metrics"].get("eps_growth") is None


def test_growth_from_a_loss_is_not_reported_as_a_percentage():
    r = fundamentals.analyze("X.NS", INFO, _stmts(net_income=[50, -20]))
    assert "net_income_growth" in r.details["unavailable"]


def test_negative_equity_scores_as_a_red_flag():
    r = fundamentals.analyze("X.NS", INFO, _stmts(equity=[-100, 50]))
    de = [e for e in r.evidence if "equity is negative" in e.claim]
    assert de and de[0].direction == -1


def test_the_p_e_is_yahoos_over_the_last_twelve_months_the_stock_header_shows():
    """Price over the last FY's EPS read TCS 17.7 and NYKAA 515.5 beside the header's trailing 16.9 and 322.2."""
    pe = fundamentals.analyze("TCS.NS", {**INFO, "trailingPE": 16.86}, _stmts()).details["metrics"]["pe"]
    assert (pe["value"], pe["source"]) == (16.86, f"provider:{fundamentals.TRAILING_PE}")


def test_loss_making_company_gets_no_pe():
    r = fundamentals.analyze("X.NS", INFO, _stmts(eps=[-2, 1]))
    assert r.details["metrics"]["pe"]["value"] is None
    assert "not meaningful" in r.details["metrics"]["pe"]["note"]


def test_leverage_ratios_are_skipped_for_banks():
    r = fundamentals.analyze("BANK.NS", {**INFO, "sector": "Financial Services"}, _stmts(debt=[9000, 8000]))
    assert r.details["metrics"]["debt_to_equity"]["value"] is None


def test_no_data_at_all_is_unavailable():
    assert not fundamentals.analyze("X.NS", {}, {}).available


@pytest.mark.parametrize("text,sign", [
    ("Net profit falls 20% as costs rise", -1),
    ("Company beats estimates, revenue jumps", 1),
    ("Losses narrow sharply in the quarter", 1),
    ("Debt falls to a five-year low", 1),
    ("Revenue growth slows", -1),
    ("Results were not strong", -1),
])
def test_lexicon_handles_finance_phrasing(text, sign):
    s = LexiconScorer().score(text).score
    assert (s > 0) if sign > 0 else (s < 0)


def test_syndicated_copies_collapse_into_one_event():
    items = [news_item("Acme wins large order from railways", 2, "Economic Times"),
             news_item("Acme wins big order from railways", 2, "Moneycontrol"),
             news_item("Acme reports quarterly results", 1)]
    events = news.build_events("ACME.NS", ["Acme"], items, now=NOW)
    order = [e for e in events if e.event_type == "contract_order"]
    assert len(events) == 2 and order[0].syndicated_count == 2


def test_events_are_typed_and_scoped():
    ev = news.build_events("ACME.NS", ["Acme"], [news_item("RBI keeps repo rate unchanged", 1)], now=NOW)[0]
    assert ev.event_type == "macro" and ev.scope == "macro"
    ev = news.build_events("ACME.NS", ["Acme"], [news_item("Acme faces SEBI probe", 1)], now=NOW)[0]
    assert ev.scope == "company" and ev.event_type in ("regulatory", "legal")


def test_older_and_less_reliable_news_weighs_less():
    fresh = news.build_events("A.NS", ["Acme"], [news_item("Acme profit jumps", 0, "Reuters")], now=NOW)[0]
    old = news.build_events("A.NS", ["Acme"], [news_item("Acme profit jumps", 9, "Reuters")], now=NOW)[0]
    blog = news.build_events("A.NS", ["Acme"], [news_item("Acme profit jumps", 0, "some blog")], now=NOW)[0]
    assert fresh.weight > old.weight and fresh.weight > blog.weight


def test_sentiment_shift_is_detected():
    items = [news_item("Acme profit jumps, beats estimates", 1), news_item("Acme wins record order", 2),
             news_item("Acme shares plunge on fraud probe", 12), news_item("Acme misses estimates, weak outlook", 15)]
    r = news.analyze("A.NS", ["Acme"], items, now=NOW)
    assert r.details["shift"] == "positive"
    assert any("shifted materially positive" in e.claim for e in r.evidence)


def test_news_scores_are_labelled_model_outputs():
    r = news.analyze("A.NS", ["Acme"], [news_item("Acme profit jumps", 1)], now=NOW)
    assert r.evidence and all(e.is_model_output for e in r.evidence)


def test_no_articles_means_unavailable_not_neutral():
    assert not news.analyze("A.NS", ["Acme"], [], now=NOW).available


def test_mixed_currency_statements_do_not_produce_price_ratios():
    # Infosys-style: statements in USD, shares priced in INR.
    info = {**INFO, "currency": "INR", "financialCurrency": "USD", "trailingPE": 22.0}
    r = fundamentals.analyze("INFY.NS", info, _stmts())
    m = r.details["metrics"]
    assert m["pe"]["value"] == pytest.approx(22.0) and m["pe"]["source"].endswith("info.trailingPE")
    assert m["fcf_yield"]["value"] is None and "USD" in m["fcf_yield"]["note"]
    assert m["free_cash_flow"]["currency"] == "USD"


def test_interest_cover_reads_the_operating_margins_profit_line_not_yahoos_ebit():
    """Yahoo's "EBIT" is pretax income plus interest, one-off charges in: RHIM's cover read -7.7 beside a +6.8% margin."""
    stm = _stmts()
    stm["income"].loc["EBIT"] = [-50.0, -40.0]
    m = fundamentals.analyze("X.NS", INFO, stm).details["metrics"]
    assert m["operating_margin"]["value"] > 0 and m["interest_coverage"]["value"] == pytest.approx(11.0)


def test_interest_cover_is_not_taken_from_an_older_year_than_the_profit_it_is_dated_by():
    """SHOP gives no interest after 2022, and 2022's cover of -597 was shown as its FY 2025 one."""
    stm = _stmts(ebit=[220, -180])
    stm["income"].loc["Interest Expense", stm["income"].columns[0]] = float("nan")
    assert "interest_coverage" in fundamentals.analyze("X.NS", INFO, stm).details["unavailable"]


def test_roic_reads_the_operating_margins_profit_line_not_yahoos_ebit():
    """As interest cover did: yahoo's "EBIT" holds one-off charges, and it differs from Operating Income for 176 of the 186
    snapshot shares that give both."""
    stm = _stmts()
    stm["income"].loc["EBIT"] = [-50.0, -40.0]
    m = fundamentals.analyze("X.NS", INFO, stm).details["metrics"]
    assert m["roic"]["value"] == pytest.approx(220 / (300 + 1000 - 300 * 0.2) * (1 - 0.25) * 100, abs=1e-4)


@pytest.mark.parametrize("kind,item", [("balance", "Total Debt"), ("income", "Tax Provision")])
def test_roic_is_not_taken_from_an_older_year_than_the_profit_it_is_dated_by(kind, item):
    """NESTLEIND gives no FY 2026 tax or capital, and its FY 2025 ROIC of 69.5% was shown as its FY 2026 one."""
    stm = _stmts()
    stm[kind].loc[item, stm[kind].columns[0]] = float("nan")
    assert "roic" in fundamentals.analyze("X.NS", INFO, stm).details["unavailable"]


THREE_YEARS = dict(revenue=[1200, 1000, 900], net_income=[150, 120, 100], eps=[15, 12, 10], equity=[1000, 900, 800],
                   debt=[300, 300, 300], assets=[2000, 1800, 1600], ebit=[220, 180, 150], interest=[20, 20, 20],
                   ca=[600, 550, 500], cl=[400, 380, 360])


@pytest.mark.parametrize("key,kind,item", [
    ("gross_margin", "income", "Gross Profit"), ("operating_margin", "income", "Operating Income"),
    ("net_margin", "income", "Net Income"), ("net_margin_change", "income", "Net Income"),
    ("roe", "balance", "Stockholders Equity"), ("roa", "balance", "Total Assets"),
    ("debt_to_equity", "balance", "Total Debt"), ("current_ratio", "balance", "Current Liabilities")])
def test_a_ratio_one_line_lacks_the_newest_year_of_is_dated_by_the_older_year_it_is_computed_on(key, kind, item):
    """Each was dated by one line's newest year: SURANASOL gives no debt after 2023, and its 2023 debt/equity was shown
    as its FY 2026 one."""
    stm = statements(**THREE_YEARS)
    stm["income"].loc["Gross Profit"] = [600.0, 500.0, 450.0]
    stm[kind].loc[item, pd.Timestamp("2026-03-31")] = float("nan")
    period = fundamentals.analyze("X.NS", INFO, stm).details["metrics"][key]["period"]
    assert period.startswith("FY ending 2025-03-31")


def test_a_closed_analysis_values_the_share_at_its_last_closed_bar_not_the_live_quote(local_provider, random_walk):
    """At 10:00 IST the facts are dated by yesterday's close, so a P/E at the moving price was dated wrong."""
    ten_ist = datetime(2026, 9, 25, 4, 30, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)    # naive, as datetime.now
    closed, live = random_walk["close"].iloc[-2], random_walk["close"].iloc[-1]
    details = AnalysisService(local_provider, use_rss_news=False, now=lambda: ten_ist).analyze(
        "TEST", closed=True).results["fundamental"].details
    assert details["metrics"]["pe"]["value"] == pytest.approx(20.0 * closed / live, abs=1e-4)
    assert details["profile"]["market_cap"] == pytest.approx(5e11 * closed / live)


def test_a_quote_moved_to_a_close_drops_what_rests_on_a_price_it_does_not_give():
    assert "marketCap" not in fundamentals.priced_at({"marketCap": 5e11, "trailingPE": 20.0}, 100.0)


def test_a_quote_moved_to_a_close_moves_the_provider_multiples_and_ev_by_its_market_value_alone():
    """Dated by the last close, forward P/E, P/B, PEG and the EV multiples were still at the price they were fetched at."""
    info = {"currentPrice": 200.0, "marketCap": 1000.0, "enterpriseValue": 1100.0, "forwardPE": 20.0,
            "priceToBook": 4.0, "pegRatio": 2.0, "enterpriseToEbitda": 11.0, "enterpriseToRevenue": 2.2}
    moved = fundamentals.priced_at(info, 100.0)
    assert (moved["forwardPE"], moved["priceToBook"], moved["pegRatio"]) == (10.0, 2.0, 1.0)
    assert (moved["enterpriseToEbitda"], moved["enterpriseToRevenue"]) == pytest.approx((6.0, 1.2))   # EV 1100 -> 600


def test_ev_multiples_are_dropped_not_scaled_when_ev_cannot_be_split_into_market_value_and_the_rest():
    assert "enterpriseToEbitda" not in fundamentals.priced_at(
        {"currentPrice": 200.0, "marketCap": 1000.0, "enterpriseToEbitda": 11.0}, 100.0)


HCLTECH_QUARTERS = {pd.Timestamp(day): {"Diluted EPS": eps}
                    for day, eps in (("2026-09-30", 18.93), ("2026-06-30", 17.06), ("2026-03-31", 16.56),
                                     ("2025-12-31", 15.05))}


@pytest.mark.parametrize("trailing_eps,stated", [(67.59, "INR"), (67.59 * 1.16, "USD")],
                         ids=["adds up: HCLTECH", "16% off: ASML's euro to dollar"])
def test_statements_whose_quarters_eps_add_up_to_the_trailing_eps_in_the_price_currency_are_in_it(trailing_eps, stated):
    """HCLTECH.NS is labelled USD, but its statements are in rupees: the currency line said "Statements are in USD" and its
    cash of 8,212 crore rupees read 82.12 billion dollars."""
    stm = {**_stmts(), "income_quarterly": pd.DataFrame(HCLTECH_QUARTERS)}
    info = {**INFO, "currency": "INR", "financialCurrency": "USD", "trailingEps": trailing_eps}
    details = fundamentals.analyze("HCLTECH.NS", info, stm).details
    assert details["statement_currency"] == details["metrics"]["cash"]["currency"] == stated


def test_provider_ev_multiples_stay_refused_when_only_the_statements_are_in_the_price_currency():
    """Yahoo divides HCLTECH's rupee EV by its own dollar EBITDA, 1,331x, whatever currency the statements are in."""
    stm = {**_stmts(), "income_quarterly": pd.DataFrame(HCLTECH_QUARTERS)}
    info = {**INFO, "currency": "INR", "financialCurrency": "USD", "trailingEps": 67.59, "enterpriseToEbitda": 15.0}
    assert "mixes USD" in fundamentals.analyze("HCLTECH.NS", info, stm).details["metrics"]["ev_to_ebitda"]["note"]


def test_growth_across_a_missing_year_is_not_called_year_on_year():
    stm = _stmts(periods=[pd.Timestamp("2026-03-31"), pd.Timestamp("2024-03-31")])
    r = fundamentals.analyze("X.NS", INFO, stm)
    assert "revenue_growth" in r.details["unavailable"]
    assert not any("year on year" in e.claim for e in r.evidence)


def test_negative_equity_with_no_debt_is_still_a_red_flag():
    r = fundamentals.analyze("X.NS", INFO, _stmts(equity=[-100, 50], debt=[0, 0]))
    assert r.details["metrics"]["debt_to_equity"]["value"] is None
    assert any("equity is negative" in e.claim and e.direction == -1 for e in r.evidence)


def test_implausible_or_mixed_currency_provider_multiples_are_refused():
    info = {**INFO, "currency": "INR", "financialCurrency": "USD", "enterpriseToEbitda": 903.2,
            "priceToBook": 4.4}
    m = fundamentals.analyze("INFY.NS", info, _stmts()).details["metrics"]
    assert m["ev_to_ebitda"]["value"] is None and "mixes USD" in m["ev_to_ebitda"]["note"]
    assert m["price_to_book"]["value"] == pytest.approx(4.4)
    m = fundamentals.analyze("X.NS", {**INFO, "priceToBook": 950.0}, _stmts()).details["metrics"]
    assert "implausible" in m["price_to_book"]["note"]


def test_xbrl_insider_disclosure_is_parsed_into_a_market_purchase():
    from datetime import datetime
    from backend.database.sources.insider import parse_xbrl
    xml = ('<x><in-bse-co:TypeOfInstrument contextRef="Disclosure1">Equity</in-bse-co:TypeOfInstrument>'
           '<in-bse-co:CategoryOfPerson contextRef="Disclosure1">Promoter Group</in-bse-co:CategoryOfPerson>'
           '<in-bse-co:SecuritiesAcquiredOrDisposedNumberOfSecurity contextRef="Disclosure1">167734</in-bse-co:SecuritiesAcquiredOrDisposedNumberOfSecurity>'
           '<in-bse-co:SecuritiesAcquiredOrDisposedValueOfSecurity contextRef="Disclosure1">17818369</in-bse-co:SecuritiesAcquiredOrDisposedValueOfSecurity>'
           '<in-bse-co:SecuritiesHeldPriorToAcquisitionOrDisposalPercentageOfShareholding contextRef="Disclosure1">0.6349</in-bse-co:SecuritiesHeldPriorToAcquisitionOrDisposalPercentageOfShareholding>'
           '<in-bse-co:ModeOfAcquisitionOrDisposal contextRef="Disclosure1">Market Purchase</in-bse-co:ModeOfAcquisitionOrDisposal>'
           '<in-bse-co:CategoryOfPerson contextRef="Disclosure2">Employees</in-bse-co:CategoryOfPerson></x>')
    rows = parse_xbrl(xml, "ALEMBICLTD", datetime(2026, 9, 25, 20, 4), "f1")
    assert len(rows) == 1
    r = rows[0]
    assert r["direction"] == 1 and r["value"] == 17818369 and r["pct_before"] == pytest.approx(63.49)
