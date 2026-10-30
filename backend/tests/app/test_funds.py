import json
import os
import re
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.advisory.calculators import funds
from backend.database.sources import amfi
from backend.database.sources.provider import DataUnavailable

HEADER = "Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date"
# Laid out as AMFI's NAVAll.txt is: CRLF lines, blank lines holding a space, category then fund-house headings
NAV_TEXT = "\r\n".join([
    HEADER, " ",
    "Open Ended Schemes(Equity Scheme - Mid Cap Fund)", " ", "HDFC Mutual Fund", " ",
    "105759;INF179K01CS0;INF179K01CT8;HDFC Mid Cap Fund;Regular Plan;IDCW Option;60.3000;29-Oct-2026",
    "118990;INF179K01XR8;INF179K01XS6;HDFC Mid Cap Fund;Direct Plan;IDCW Option;70.2000;29-Oct-2026",
    "105758;INF179K01CR2;-;HDFC Mid Cap Fund;Regular Plan;Growth Option;190.1000;29-Oct-2026",
    "118989;INF179K01XQ0;-;HDFC Mid Cap Fund;Direct Plan;Growth Option;210.5120;29-Oct-2026", " ",
    "Open Ended Schemes(Other Scheme - Index Funds)", " ", "HDFC Mutual Fund", " ",
    "150001;INF179KC1BA3;-;HDFC NIFTY Midcap 150 Index Fund;Direct Plan;Growth Option;18.2000;29-Oct-2026", " ",
    "Open Ended Schemes(Equity Scheme - Large Cap Fund)", " ", "ICICI Prudential Mutual Fund", " ",
    "120586;INF109K016L0;-;ICICI Prudential Large Cap Fund (erstwhile Bluechip Fund);Direct Plan;Growth;113.1700;29-Oct-2026",
    " ", "Open Ended Schemes(Fund of Funds Scheme (Overseas) - Fund of Funds Scheme (Overseas))", " ",
    "ICICI Prudential Mutual Fund", " ",
    "120186;INF109K01Z48;-;ICICI Prudential US Bluechip Equity Fund;Direct Plan;Growth;80.0000;27-Oct-2026", " ",
    "Open Ended Schemes(Equity Scheme - Sectoral/ Thematic)", " ", "Aditya Birla Sun Life Mutual Fund", " ",
    "147844;INF209KB1V70;-;Aditya Birla Sun Life PSU Equity Fund;Direct Plan;GROWTH;37.7800;29-Oct-2026", " ",
    "Open Ended Schemes(Debt Scheme - Corporate Bond Fund)", " ", "Nippon India Mutual Fund", " ",
    "109472;-;INF204K01EG7;Nippon India Corporate Bond Fund;Regular Plan;DAILY IDCW Option;16.9800;29-Oct-2026",
    "109999;INF204K01ZZ9;-;Nippon India Corporate Bond Fund;Regular Plan;Growth Option;N.A.;29-Oct-2026", " ",
    "Open Ended Schemes(Debt Scheme - Low Duration Fund)", " ", "Franklin Templeton Mutual Fund", " ",
    "147990;INF090I01TY5;-;Franklin India Low Duration Fund (No. of Segregated Portfolios-2);;;0.0000;20-Jul-2020", " ",
    "Open Ended Schemes(Debt Scheme - Medium Duration Fund)",
    "118554;INF090I01JI9;-;Franklin India Income Opportunities Fund;;;26.5759;12-Dec-2021", " ",
])
PARSED = amfi.parse(NAV_TEXT)
CURRENT = amfi.current(PARSED)


def by_code(code, schemes=PARSED):
    return next(s for s in schemes if s.code == code)


def growing(years, rate=0.10, end="2026-10-29"):
    """NAV on business days, compounding at `rate` a year of calendar time."""
    idx = pd.bdate_range(end=end, periods=int(years * 261))
    return pd.Series(100 * (1 + rate) ** ((idx - idx[0]).days / 365.25), index=idx)


def mfapi_json(code, nav):
    return json.dumps({"meta": {"scheme_code": code, "scheme_name": "x"}, "status": "SUCCESS",
                       "data": [{"date": d.strftime("%d-%m-%Y"), "nav": f"{v:.5f}"} for d, v in nav[::-1].items()]})


# --- parser ------------------------------------------------------------------------------
def test_each_row_takes_its_category_and_fund_house_from_the_headings_above_it():
    s = by_code(118989)
    assert (s.fund, s.plan, s.option, s.nav, s.nav_date) == ("HDFC Mid Cap Fund", "Direct Plan", "Growth Option",
                                                             210.512, date(2026, 10, 29))
    assert (s.scheme_type, s.category, s.amc) == ("Open Ended Schemes", "Equity Scheme - Mid Cap Fund", "HDFC Mutual Fund")
    assert by_code(120186).category == "Fund of Funds Scheme (Overseas) - Fund of Funds Scheme (Overseas)"


def test_a_category_line_with_no_fund_house_line_keeps_the_house_above_it():
    s = by_code(118554)
    assert (s.amc, s.category) == ("Franklin Templeton Mutual Fund", "Debt Scheme - Medium Duration Fund")
    assert s.plan is None and s.option is None


def test_a_dash_isin_is_missing_and_the_reinvestment_isin_is_kept():
    s = by_code(109472)
    assert (s.isin_growth, s.isin_reinvestment) == (None, "INF204K01EG7")


def test_a_row_with_no_nav_is_left_out():
    assert 109999 not in {s.code for s in PARSED}


def test_only_schemes_priced_in_the_last_week_of_the_file_are_current():
    codes = {s.code for s in CURRENT}
    assert 120186 in codes                         # two days before the newest NAV
    assert not codes & {147990, 118554}            # last priced in 2020 and 2021


def test_columns_are_found_by_name_not_position():
    swapped = NAV_TEXT.replace(HEADER, "Scheme Code;Scheme Name;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;"
                                       "Plan;Option;Net Asset Value;Date")
    swapped = swapped.replace("118989;INF179K01XQ0;-;HDFC Mid Cap Fund;", "118989;HDFC Mid Cap Fund;INF179K01XQ0;-;")
    s = by_code(118989, amfi.parse(swapped))
    assert (s.fund, s.isin_growth) == ("HDFC Mid Cap Fund", "INF179K01XQ0")


def test_a_header_without_plan_and_option_columns_says_which_are_missing():
    old = "Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date"
    with pytest.raises(DataUnavailable, match=r"\['Plan', 'Option'\]"):
        amfi.parse(NAV_TEXT.replace(HEADER, old))


def test_an_unreadable_nav_is_a_data_error_naming_its_line_not_a_bad_request():
    with pytest.raises(DataUnavailable, match="line 9 .*'abc'"):
        amfi.parse(NAV_TEXT.replace("190.1000", "abc"))


def test_an_error_page_is_not_mistaken_for_an_empty_file():
    with pytest.raises(DataUnavailable, match="no \\['Scheme Code'"):
        amfi.parse("<html><body>Service Unavailable</body></html>")


# --- search ------------------------------------------------------------------------------
def test_search_groups_a_funds_plans_with_direct_and_growth_first():
    hit = amfi.search(CURRENT, "hdfc mid cap")[0]
    assert hit["fund"] == "HDFC Mid Cap Fund" and hit["amc"] == "HDFC Mutual Fund"
    assert [p["code"] for p in hit["plans"]] == [118989, 118990, 105758, 105759]


def test_whole_word_matches_rank_before_longer_names():
    assert [f["fund"] for f in amfi.search(CURRENT, "hdfc midcap")] == ["HDFC Mid Cap Fund", "HDFC NIFTY Midcap 150 Index Fund"]


def test_a_note_in_brackets_does_not_push_a_fund_below_a_shorter_name():
    assert amfi.search(CURRENT, "icici pru bluechip")[0]["fund"] == "ICICI Prudential Large Cap Fund (erstwhile Bluechip Fund)"


def test_every_query_word_must_start_a_word_of_the_name():
    assert amfi.search(CURRENT, "hdfc small") == []


def test_a_fund_house_abbreviation_is_read_as_its_name():
    assert amfi.search(CURRENT, "absl psu")[0]["fund"] == "Aditya Birla Sun Life PSU Equity Fund"


def test_an_isin_held_only_in_the_reinvestment_column_finds_its_scheme():
    hit = amfi.search(CURRENT, "INF204K01EG7")
    assert [(f["fund"], [p["code"] for p in f["plans"]]) for f in hit] == [("Nippon India Corporate Bond Fund", [109472])]


@pytest.mark.parametrize("query", ["", " ", "a", "--"])
def test_a_query_too_short_to_search_is_refused(query):
    with pytest.raises(ValueError, match="at least 2 letters"):
        amfi.search(CURRENT, query)


def test_search_returns_at_most_the_limit():
    assert len(amfi.search(CURRENT, "fund", limit=3)) == 3


# --- daily copy --------------------------------------------------------------------------
@pytest.fixture
def nav_cache(tmp_path, monkeypatch):
    calls, texts = [], []

    def download(url):
        calls.append(url)
        return texts.pop(0) if texts else NAV_TEXT
    monkeypatch.setattr(amfi, "NAV_FILE", tmp_path / "navall.txt")
    monkeypatch.setattr(amfi, "_download", download)
    return calls, texts


def test_the_nav_file_is_downloaded_once_a_day(nav_cache):
    calls, _ = nav_cache
    amfi.schemes()
    amfi.schemes()
    assert len(calls) == 1
    yesterday = time.time() - 86400
    os.utime(amfi.NAV_FILE, (yesterday, yesterday))
    assert {s.code for s in amfi.schemes()} == {s.code for s in CURRENT} and len(calls) == 2


def test_a_download_that_does_not_parse_is_not_kept(nav_cache):
    _, texts = nav_cache
    texts.append("<html>maintenance</html>")
    with pytest.raises(DataUnavailable):
        amfi.schemes()
    assert not amfi.NAV_FILE.exists()


def test_saved_app_state_never_carries_a_fund_download(tmp_path):
    """A restored copy gets today's file time, so it would pass for today's download."""
    from backend.database import hfstate
    from backend.database.sources.panel import CACHE_DIR
    for path in (amfi.NAV_FILE, amfi.HISTORY_DIR / "118989.json"):
        copy = tmp_path / path.relative_to(CACHE_DIR)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text("x")
    assert hfstate.cache_files(tmp_path) == []


def test_history_from_mfapi_is_read_oldest_first():
    nav = growing(1)
    got = amfi.parse_history(mfapi_json(118989, nav), 118989)
    assert got.index[0] == nav.index[0] and got.iloc[-1] == pytest.approx(nav.iloc[-1])


def test_mfapi_answering_an_unknown_code_with_an_empty_scheme_is_no_history():
    with pytest.raises(DataUnavailable, match="no NAV history for scheme 118989"):
        amfi.parse_history(json.dumps({"meta": {"scheme_code": 0}, "data": [], "status": "SUCCESS"}), 118989)


# --- analytics ---------------------------------------------------------------------------
def test_trailing_returns_on_a_steady_10_percent_fund():
    r = funds.trailing_returns(growing(6))
    assert r["3Y"]["pct"] == pytest.approx(10.0) and r["5Y"]["pct"] == pytest.approx(10.0)
    assert r["3Y"]["annualised"] and not r["1Y"]["annualised"]
    assert r["1Y"]["pct"] == pytest.approx(10.0, abs=0.01)
    assert r["6M"]["pct"] == pytest.approx(100 * (1.1 ** (183 / 365.25) - 1), abs=0.01)


def test_a_return_starts_from_the_last_nav_on_or_before_the_calendar_date():
    nav = growing(2).drop(pd.Timestamp("2025-10-29"))          # no NAV published a year before the last
    assert funds.trailing_returns(nav)["1Y"]["from"] == "2025-10-28"


def test_a_period_longer_than_the_history_has_no_return():
    r = funds.trailing_returns(growing(4))
    assert r["3Y"] is not None and r["5Y"] is None


def test_a_fund_moving_1_percent_a_day_has_its_volatility_annualised():
    idx = pd.bdate_range(end="2026-10-29", periods=600)
    nav = pd.Series(100 * np.cumprod([1.01 if i % 2 else 0.99 for i in range(600)]), index=idx)
    assert funds.analytics(nav)["volatility_1y_pct"] == pytest.approx(np.sqrt(252), abs=0.05)


def test_the_worst_fall_runs_from_the_peak_to_the_trough():
    dip = pd.Series([100.0, 120.0, 90.0, 130.0], index=pd.bdate_range(end="2026-10-29", periods=4))
    dd = funds.analytics(dip)["max_drawdown"]
    assert (dd["max_drawdown_pct"], dd["peak_date"], dd["trough_date"]) == (-25.0, str(dip.index[1].date()),
                                                                            str(dip.index[2].date()))


def test_the_summary_quotes_only_figures_the_analytics_hold():
    nav = growing(6)
    nav.iloc[-300:-250] *= 0.8
    a = funds.analytics(nav)
    text = funds.summary(a)
    assert text.startswith("Over the last 5 years its NAV grew") and "worst fall" in text
    held = json.dumps(a)
    assert all(n in held for n in re.findall(r"\d{4}-\d{2}-\d{2}|\d+\.\d+", text))


def test_a_fund_too_new_for_any_return_says_so():
    a = funds.analytics(growing(0.05))
    assert a["returns"]["1M"] is None and a["volatility_1y_pct"] is None
    assert "too soon for a 1-month return" in funds.summary(a)
