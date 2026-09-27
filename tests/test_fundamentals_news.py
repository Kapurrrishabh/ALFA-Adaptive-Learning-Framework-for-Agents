import pandas as pd
import pytest

from stockintel.analysis import fundamentals, news
from stockintel.analysis.sentiment import LexiconScorer

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
    assert m["pe"]["value"] == pytest.approx(20.0)            # 300 / 15, recomputed not copied
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
    from stockintel.data.insider import parse_xbrl
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
