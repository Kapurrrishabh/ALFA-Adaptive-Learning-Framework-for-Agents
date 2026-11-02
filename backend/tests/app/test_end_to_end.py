import re
from dataclasses import replace
from datetime import time
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from backend.advisory import report as R
from backend.user.portfolio.alerts import evaluate
from backend.api.accounts import owner_of
from backend.api.app import HSTS, PUBLIC_PATHS, SECURITY_HEADERS, create_app
from backend.config import DEFAULT_PORTFOLIO, FORECAST_DAYS
from backend.api.orchestrator import Orchestrator
from backend.advisory.calculators.screener import screen
from backend.advisory.service import AnalysisService
from backend.advisory.statistics.validation import event_study
from backend.database.sources import synthetic

from .conftest import EMAIL, END, NOW, PASSWORD, PUBLIC_URL, sign_in
from .test_funds import NAV_TEXT, growing, mfapi_json
from .test_nlu_llm import _Picker

ORACLE = Path(__file__).resolve().parents[3] / "deploy" / "oracle"
SERVE_NEEDS = ["STOCKINTEL_AUTH_SECRET", "STOCKINTEL_PUBLIC_URL"]
OTHER = "ravi@example.com"
UNAUTHORIZED = (401, {"detail": "Unauthorized"})    # fastapi-users' own refusal, not a 401 of ours
FUND_BUY = {"scheme_code": 118989, "side": "BUY", "units": 10, "nav": 200.0, "trade_date": "2025-01-02"}


@pytest.fixture
def service(local_provider):
    return AnalysisService(local_provider, use_rss_news=False, now=lambda: NOW.replace(tzinfo=None))


@pytest.fixture
def orch(local_provider, store, monkeypatch):
    o = Orchestrator(local_provider, store, use_rss_news=False, charges_model_calls=True)
    o.known.add("TEST")
    o.universe = pd.DataFrame({"symbol": ["TEST", "PEER"], "sector": ["Industrials", "Energy"]})
    return o


def signed_in_client(orch, accounts, email=EMAIL, app=None):
    client = TestClient(app or create_app(orch, accounts), base_url=PUBLIC_URL)
    sign_in(client, accounts, email)
    return client


def owner(accounts, email=EMAIL):
    return owner_of(accounts.user(email))


def test_full_analysis_produces_every_domain_with_provenance(service):
    a = service.analyze("TEST")
    assert set(a.results) == {"technical", "candlestick", "pattern", "risk", "regime", "forecast",
                              "historical", "fundamental", "news_sentiment"}
    assert a.results["fundamental"].available and a.results["news_sentiment"].available
    for r in a.results.values():
        for e in r.evidence:
            assert e.provenance.source and e.provenance.as_of
    assert a.decision.decision_support_label in ("BUY", "WATCH", "HOLD", "SELL", "AVOID")
    assert a.decision.data_timestamp == "2026-09-25"


def test_rendered_analysis_states_timestamps_and_labels_model_outputs(service):
    text = R.render_analysis(service.analyze("TEST"))
    assert "Analysis timestamp" in text and "Price data as of 2026-09-25" in text
    assert "MODEL OUTPUT" in text and "[FACT]" in text
    assert "### 12. Final evidence-based summary" in text


def test_rendered_forecast_range_states_the_interval_it_was_built_for(service):
    a = service.analyze("TEST")
    for e in a.results["forecast"].details["horizons"].values():
        e["expected_range"]["probability"] = 0.9
    assert "90% range" in R.render_analysis(a)


def test_a_drill_down_explains_the_jargon_its_own_lines_use(service):
    risk = R.render_domain(service.analyze("TEST"), "risk")
    assert "Drawdown: a fall from a peak." in risk and "EBIT:" not in risk


# every return on a flat price is zero, so this is the verdict pattern and candle checks give a null result
NO_EDGE = event_study(pd.Series(100.0, index=range(300)), pd.Series(True, index=range(300)), 10, 1)["verdict"]


@pytest.mark.parametrize("domain, verdict, called_chance", [
    ("historical", "not a statistically significant difference", True),
    ("historical", "a statistically significant difference", False),
    ("pattern", NO_EDGE, True)])
def test_a_drill_down_calls_a_result_chance_only_when_it_is_not_significant(service, domain, verdict, called_chance):
    a = service.analyze("TEST")
    r = a.results[domain]
    r.evidence = [replace(e, claim=f"{e.claim.split(' — ')[0]} — {verdict}.") for e in r.evidence]
    assert ("could easily be chance" in R.render_domain(a, domain)) == called_chance


def test_the_analysis_and_its_drill_down_give_operating_margin_one_meaning(service):
    a = service.analyze("TEST")
    meaning = re.search(r"Operating margin \((.+?)\):", R.render_analysis(a)).group(1)
    assert f"Operating margin: {meaning}." in R.render_domain(a, "fundamental")


def test_every_reply_that_explains_volatility_reads_the_one_meaning(orch, monkeypatch):
    monkeypatch.setitem(R.TERM_MEANINGS, "Volatility", "a meaning only this test writes")
    monkeypatch.setattr(type(orch.hub), "writer", lambda self: _Picker("NONE"))
    replies = [orch.handle(q, DEFAULT_PORTFOLIO, "terms").text for q in
               ("Analyze TEST", "how risky is TEST?", "Compare TEST with PEER", "Is the market in a downtrend?")]
    assert all("a meaning only this test writes" in r for r in replies)


@pytest.mark.parametrize("question, most", [("help", 8), ("Is the market in a downtrend?", 8),
                                            ("Analyze TEST", 9), ("how risky is TEST?", 9)])
def test_chat_replies_read_at_plain_language_grade(orch, monkeypatch, question, most):
    from backend.models.external.eval_user_questions import grade, plain
    monkeypatch.setattr(type(orch.hub), "writer", lambda self: _Picker("NONE"))
    assert grade(plain(orch.handle(question, DEFAULT_PORTFOLIO, "plain").text)) <= most


def test_the_market_reply_quotes_the_backtested_volatility_target(orch, monkeypatch):
    import backend.advisory.signals.strategies as strategies
    monkeypatch.setattr(strategies, "VOL_TARGET", 0.25)
    assert "To aim for 25% volatility" in orch.handle("Is the market in a downtrend?", DEFAULT_PORTFOLIO, "vol").text


def test_full_report_has_all_18_sections(service):
    body = R.render_report(service.analyze("TEST"))
    for n in range(1, 19):
        assert f"## {n}." in body


def test_missing_fundamentals_are_unavailable_not_invented(local_provider, service):
    local_provider.statements.pop("TEST.NS")
    local_provider.infos["TEST.NS"] = {"longName": "Test Industries Limited"}
    service._cache.clear()
    a = service.analyze("TEST")
    assert not a.results["fundamental"].available
    assert "Data unavailable" in R.render_analysis(a)


def test_multi_turn_conversation_keeps_context(orch):
    assert orch.handle("Analyze TEST", DEFAULT_PORTFOLIO, "s").intent == "analyze"
    r = orch.handle("Why is the technical state like that?", DEFAULT_PORTFOLIO, "s")
    assert r.intent == "domain" and "· Technicals" in r.text
    r = orch.handle("What about fundamentals?", DEFAULT_PORTFOLIO, "s")
    assert "· Fundamentals" in r.text and "Revenue changed" in r.text
    r = orch.handle("Compare it with PEER", DEFAULT_PORTFOLIO, "s")
    assert r.intent == "compare" and "TEST" in r.text and "PEER" in r.text
    r = orch.handle("Why?", DEFAULT_PORTFOLIO, "s")
    assert r.intent == "explain_decision" and r.text.startswith("Why TEST")


def test_why_reply_rounds_each_area_score_and_confidence_to_two_decimals(orch):
    orch.handle("Analyze TEST", DEFAULT_PORTFOLIO, "places")
    votes = orch.handle("why?", DEFAULT_PORTFOLIO, "places").text.split("How each area voted:")[1].split("Main ")[0]
    assert not re.search(r"\d\.\d{3}", votes)


def test_why_a_negative_score_heads_its_evidence_as_negatives(orch, local_provider):
    local_provider.frames["FALL.NS"] = synthetic.ohlcv(900, seed=3, drift=-0.003, end=END)
    orch.known.add("FALL")
    orch.handle("Analyze FALL", DEFAULT_PORTFOLIO, "fall")
    reply = orch.handle("why?", DEFAULT_PORTFOLIO, "fall")
    assert reply.data["payload"]["decision"]["score"] < 0
    assert "Main negatives:" in reply.text and "+ [FACT]" not in reply.text


def test_budget_question_uses_the_session_stock(orch):
    orch.handle("Analyze TEST", DEFAULT_PORTFOLIO, "b")
    r = orch.handle("I have ₹1 lakh. How does my budget affect the analysis?", DEFAULT_PORTFOLIO, "b")
    assert r.intent == "budget_analysis" and "position limit" in r.text


def test_portfolio_questions_are_computed_from_stored_transactions(orch):
    orch.save_portfolio(DEFAULT_PORTFOLIO, 5000.0, "moderate", "long", [
        {"symbol": "TEST", "side": "BUY", "quantity": 10, "price": 100, "trade_date": "2025-01-02", "sector": "Industrials"},
        {"symbol": "PEER", "side": "BUY", "quantity": 5, "price": 100, "trade_date": "2025-01-02", "sector": "Energy"}])
    r = orch.handle("How is my portfolio performing?", DEFAULT_PORTFOLIO, "p")
    assert r.intent == "portfolio_summary" and "Total value" in r.text and "Risk contribution" in r.text
    r = orch.handle("If I add ₹20,000 to PEER how does my portfolio risk change?", DEFAULT_PORTFOLIO, "p")
    assert r.intent == "portfolio_what_if" and "all in PEER" in r.text


def test_portfolio_question_without_a_portfolio_says_so(orch):
    r = orch.handle("How is my portfolio performing?", DEFAULT_PORTFOLIO, "p")
    assert "No portfolio" in r.text


@pytest.fixture
def signal_inputs(local_provider, monkeypatch, tmp_path):
    """The signal plan's panel, index list, insider file and track record, from the test frames."""
    import backend.advisory.calculators.performance as perf
    import backend.database.sources.panel as panel_mod
    from backend.database.sources import insider
    from backend.database.sources.provider import DataUnavailable

    def no_insider():
        raise DataUnavailable("no insider file in tests")
    frames = {s.split(".")[0]: local_provider.frames[s] for s in ("TEST.NS", "PEER.NS")}
    monkeypatch.setattr(panel_mod, "universe_panel", lambda *a, **k: panel_mod.panel_from_frames(frames))
    monkeypatch.setattr(panel_mod, "index_constituents", lambda *a, **k: pd.DataFrame({"symbol": ["PEER"]}))
    monkeypatch.setattr(insider, "load", no_insider)
    monkeypatch.setattr(perf, "PATH", tmp_path / "missing.json")
    return local_provider.frames["TEST.NS"]["close"]


def _days_ago(n):
    from datetime import date, timedelta
    return (date.today() - timedelta(days=n)).isoformat()


def test_tax_note_dates_a_rebought_holding_from_its_open_lot(orch, signal_inputs):
    close = signal_inputs
    px = {d: float(close.loc[:d].iloc[-1]) for d in (_days_ago(1000), _days_ago(700), _days_ago(70))}
    orch.save_portfolio(DEFAULT_PORTFOLIO, 0.0, "moderate", "long", [
        {"symbol": "TEST", "side": side, "quantity": 10, "price": px[d], "trade_date": d}
        for side, d in (("BUY", _days_ago(1000)), ("SELL", _days_ago(700)), ("BUY", _days_ago(70)))])
    r = orch.handle("Rebalance my portfolio", DEFAULT_PORTFOLIO, "t")
    test = next(a for a in r.data["payload"]["actions"] if a["symbol"] == "TEST")
    assert test["tax_note"].startswith("short-term")


def _split_history(close, split_day=None):
    """Provider.splits for TEST: `close` in today's shares, and a 2-for-1 split on `split_day`."""
    ratio = pd.Series(1.0, index=close.index)
    if split_day is not None:
        ratio[split_day] = 2.0
    return pd.DataFrame({"close": close, "ratio": ratio})


def _store_test_buy(orch, cash, price, day, portfolio=DEFAULT_PORTFOLIO):
    orch.save_portfolio(portfolio, cash, "moderate", "long", [
        {"symbol": "TEST", "side": "BUY", "quantity": 10, "price": price, "trade_date": day}])


def test_signal_plan_refuses_a_buy_stored_in_pre_split_shares(orch, local_provider, signal_inputs):
    close, d = signal_inputs, _days_ago(200)
    local_provider.split_histories["TEST.NS"] = _split_history(close, close.index[-50])
    _store_test_buy(orch, 100_000.0, 2 * float(close.loc[:d].iloc[-1]), d)
    r = orch.handle("Rebalance my portfolio", DEFAULT_PORTFOLIO, "t")
    assert r.text.startswith(f"I couldn't complete that: TEST: since your buy on {d}, each share has become 2")
    assert "Re-enter it as 2× the shares" in r.text


def test_signal_plan_asks_for_a_check_when_the_split_record_and_prices_disagree(orch, local_provider, signal_inputs):
    close, d = signal_inputs, _days_ago(200)
    local_provider.split_histories["TEST.NS"] = _split_history(close, close.index[-50])
    _store_test_buy(orch, 100_000.0, 10 * float(close.loc[:d].iloc[-1]), d)
    r = orch.handle("Rebalance my portfolio", DEFAULT_PORTFOLIO, "t")
    assert "The two do not agree" in r.text and "Re-enter it as" not in r.text


def test_cli_signals_checks_for_splits_before_planning(orch, local_provider, signal_inputs, monkeypatch):
    import argparse
    from backend import cli
    close, d = signal_inputs, _days_ago(200)
    local_provider.split_histories["TEST.NS"] = _split_history(close, close.index[-50])
    _store_test_buy(orch, 100_000.0, 2 * float(close.loc[:d].iloc[-1]), d)
    monkeypatch.setattr(cli, "_orchestrator", lambda *a, **k: orch)
    monkeypatch.setattr(cli, "_signal_inputs", lambda *a, **k: pytest.fail("planned without the split check"))
    with pytest.raises(ValueError, match="Re-enter it as 2× the shares"):
        cli.cmd_signals(argparse.Namespace(capital=None))


def test_cli_stops_flags_a_split_instead_of_a_stop_hit(orch, local_provider, signal_inputs, monkeypatch, capsys):
    import argparse
    from backend import cli
    close, d = signal_inputs, _days_ago(200)
    local_provider.split_histories["TEST.NS"] = _split_history(close, close.index[-50])
    _store_test_buy(orch, 0.0, 2 * float(close.loc[:d].iloc[-1]), d)
    monkeypatch.setattr(cli, "_orchestrator", lambda *a, **k: orch)
    assert cli.cmd_stops(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "through splits or bonus issues" in out and "STOP HIT" not in out


def test_signal_plan_keeps_a_buy_already_in_post_split_shares(orch, local_provider, signal_inputs):
    close, d = signal_inputs, _days_ago(200)
    local_provider.split_histories["TEST.NS"] = _split_history(close, close.index[-50])
    _store_test_buy(orch, 100_000.0, float(close.loc[:d].iloc[-1]), d)
    p = orch.handle("Rebalance my portfolio", DEFAULT_PORTFOLIO, "t").data["payload"]
    assert "TEST" in [a["symbol"] for a in p["actions"]]


def test_signal_plan_does_not_mistake_dividends_for_a_split(orch, local_provider, signal_inputs):
    # Dividends lower the adjusted close but not the split-only one, so a high-yield stock's price
    # paid drifts to 2x the adjusted close (COALINDIA 2017) with no split at all.
    close, d = signal_inputs, _days_ago(200)
    local_provider.split_histories["TEST.NS"] = _split_history(2 * close)
    _store_test_buy(orch, 100_000.0, 2 * float(close.loc[:d].iloc[-1]), d)
    p = orch.handle("Rebalance my portfolio", DEFAULT_PORTFOLIO, "t").data["payload"]
    assert "TEST" in [a["symbol"] for a in p["actions"]] and not any("split" in w for w in p["warnings"])


def test_signal_plan_says_when_it_could_not_check_for_splits(orch, signal_inputs):
    d = _days_ago(200)
    _store_test_buy(orch, 100_000.0, float(signal_inputs.loc[:d].iloc[-1]), d)
    p = orch.handle("Rebalance my portfolio", DEFAULT_PORTFOLIO, "t").data["payload"]
    assert any(w.startswith("Could not check TEST (no local split history") for w in p["warnings"])


def test_stop_check_flags_a_split_instead_of_a_stop_hit(orch, local_provider, signal_inputs):
    close, d = signal_inputs, _days_ago(200)
    local_provider.split_histories["TEST.NS"] = _split_history(close, close.index[-50])
    _store_test_buy(orch, 0.0, 2 * float(close.loc[:d].iloc[-1]), d)
    r = orch.handle("Check my stops", DEFAULT_PORTFOLIO, "t")
    assert "through splits or bonus issues" in r.text and "STOP HIT" not in r.text


def test_knowledge_answer_never_contains_live_data(orch):
    r = orch.handle("What is RSI?", DEFAULT_PORTFOLIO, "k")
    assert r.intent == "knowledge" and "Static knowledge" in r.text and "Relative Strength" in r.text


def test_screen_excludes_unaffordable_stocks_and_explains_candidates(service):
    uni = pd.DataFrame({"symbol": ["TEST", "PEER"], "sector": ["Industrials", "Energy"]})
    res = screen(service, 50_000, universe=uni, with_fundamentals=False)
    for c in res["candidates"]:
        assert c["reasons"] and c["max_shares"] >= 1
        assert c["price"] <= 50_000 * 0.20
    tiny = screen(service, 10, universe=uni, with_fundamentals=False)
    assert not tiny["candidates"] and all("affordability" in e["reason"] or "liquidity" in e["reason"]
                                          or "volatility" in e["reason"] for e in tiny["excluded"])


def test_screen_respects_sector_exclusions(service):
    uni = pd.DataFrame({"symbol": ["TEST", "PEER"], "sector": ["Industrials", "Energy"]})
    res = screen(service, 1e7, universe=uni, exclude_sectors=["Energy"], with_fundamentals=False)
    assert any(e["symbol"] == "PEER" and "excluded by you" in e["reason"] for e in res["excluded"])


def test_alert_explains_why_it_fired(service):
    a = service.analyze("TEST")
    previous = {"decision": {**a.decision.to_dict(), "decision_support_label": "INSUFFICIENT_EVIDENCE"},
                "results": {}}
    alert = evaluate(a, previous, now=NOW)
    assert alert is not None and any("classification changed" in r for r in alert.reasons)


@pytest.mark.parametrize("missing", SERVE_NEEDS)
def test_serve_will_not_start_without_its_sign_in_settings(monkeypatch, tmp_path, missing):
    import argparse
    import uvicorn
    from backend import cli
    from .test_accounts import GOOD_ENV, set_env
    set_env(monkeypatch, {**GOOD_ENV, missing: None})
    for name in ("ALFA_MODELS_REPO", "STOCKINTEL_STATE_REPO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("served without sign-in"))
    with pytest.raises(RuntimeError, match=f"^{missing}"):
        cli.cmd_serve(argparse.Namespace(db=str(tmp_path / "s.db"), demo=True, llm="off", host="127.0.0.1", port=0))


def test_api_refuses_an_orchestrator_that_does_not_charge_model_calls(local_provider, store, accounts):
    with pytest.raises(RuntimeError, match="charges_model_calls=True"):
        create_app(Orchestrator(local_provider, store, use_rss_news=False, charges_model_calls=False), accounts)


AUTH_ROUTES = sorted([("/auth/cookie/login", "POST"), ("/auth/cookie/logout", "POST"), ("/auth/forgot-password", "POST"),
                      ("/auth/google/authorize", "GET"), ("/auth/google/callback", "GET"), ("/auth/methods", "GET"),
                      ("/auth/register", "POST"), ("/auth/request-verify-token", "POST"),
                      ("/auth/reset-password", "POST"), ("/auth/verify", "POST")])


def routes_of(app):
    """Every route, the included routers' too, as FastAPI matches them (with their prefixes)."""
    from fastapi.routing import iter_route_contexts
    return list(iter_route_contexts(app.routes))


def anonymous_answers(app, headers):
    """What every route outside PUBLIC_PATHS and /auth/ answers an anonymous caller, by method and path."""
    from fastapi.routing import APIRoute
    client, answers = TestClient(app, base_url=PUBLIC_URL, headers=headers), {}
    for route in routes_of(app):
        if isinstance(route.original_route, APIRoute) and route.path not in PUBLIC_PATHS and not route.path.startswith("/auth/"):
            path = route.path_format.format(**{p.name: "1" if p.field_info.annotation is int else "TEST"
                                               for p in route.original_route.dependant.path_params})
            for method in route.methods:
                r = client.request(method, path)
                answers[(method, path)] = (r.status_code, r.json())
    return answers


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "k" * 24}], ids=["nothing", "the old shared key"])
def test_every_private_route_refuses_anonymous_callers(orch, accounts, headers):
    answers = anonymous_answers(create_app(orch, accounts), headers)
    assert len(answers) > 50 and {k: v for k, v in answers.items() if v != UNAUTHORIZED} == {}


def test_every_route_is_one_the_anonymous_walk_checks(orch, accounts):
    """The walk above reads only APIRoutes, so a plain route or a websocket added later would escape it unnoticed."""
    from fastapi.routing import APIRoute
    from starlette.routing import Mount
    assert [r.path for r in routes_of(create_app(orch, accounts))
            if not isinstance(r.original_route, APIRoute) and not (isinstance(r.original_route, Mount) and r.path == "/static")] == []


def test_every_public_path_is_a_route(orch, accounts):
    assert set(PUBLIC_PATHS) - {r.path for r in routes_of(create_app(orch, accounts))} == set()


def test_static_is_the_only_mount(orch, accounts):
    from starlette.routing import Mount
    assert [r.path for r in routes_of(create_app(orch, accounts)) if isinstance(r.original_route, Mount)] == ["/static"]


def test_the_sign_in_routes_are_the_expected_ones(orch, accounts):
    """A router mounted under /auth/ skips the route walk above, so a new one must be added here on purpose."""
    routes = routes_of(create_app(orch, accounts))
    assert sorted((r.path, m) for r in routes if r.path.startswith("/auth/") for m in r.methods) == AUTH_ROUTES


def test_a_forged_cookie_gets_401(orch, accounts):
    client = signed_in_client(orch, accounts)
    client.cookies.set("__Host-stockintel", "forged")
    r = client.get("/portfolio")
    assert (r.status_code, r.json()) == UNAUTHORIZED


def test_the_old_cookie_gets_401_after_logout(orch, accounts):
    client = signed_in_client(orch, accounts)
    cookie = client.cookies["__Host-stockintel"]
    client.post("/auth/cookie/logout")
    client.cookies.set("__Host-stockintel", cookie)
    r = client.get("/portfolio")
    assert (r.status_code, r.json()) == UNAUTHORIZED


@pytest.mark.parametrize("headers", [{"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"},
                                     {"Origin": "https://evil.example"}], ids=["cross-site", "same-site", "origin"])
def test_a_signed_in_request_started_by_another_site_gets_403(orch, accounts, headers):
    assert signed_in_client(orch, accounts).get("/portfolio", headers=headers).status_code == 403


@pytest.mark.parametrize("headers", [{}, {"Sec-Fetch-Site": "same-origin", "Origin": PUBLIC_URL},
                                     {"Sec-Fetch-Site": "none"}], ids=["no headers", "same origin", "typed address"])
def test_a_signed_in_request_from_our_own_site_is_served(orch, accounts, headers):
    assert signed_in_client(orch, accounts).get("/ui/usage", headers=headers).status_code == 200


def test_a_login_from_another_site_gets_403(orch, accounts):
    client = TestClient(create_app(orch, accounts), base_url=PUBLIC_URL)
    accounts.create_user(EMAIL, PASSWORD, verified=True)
    r = client.post("/auth/cookie/login", data={"username": EMAIL, "password": PASSWORD},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and "__Host-stockintel" not in client.cookies


def test_the_11th_login_from_one_address_in_a_minute_gets_429(orch, accounts):
    client = TestClient(create_app(orch, accounts), base_url=PUBLIC_URL)
    codes = [client.post("/auth/cookie/login", data={"username": f"u{n}@example.com", "password": PASSWORD}).status_code
             for n in range(11)]
    assert codes == [400] * 10 + [429]


@pytest.mark.parametrize("path", ["/auth/cookie/login", "/auth/forgot-password", "/auth/request-verify-token",
                                  "/auth/register"])
def test_the_per_email_limit_counts_an_address_however_it_is_typed(orch, accounts, monkeypatch, path):
    import backend.api.app as api_mod
    monkeypatch.setattr(api_mod, "EMAIL_TRIES", 2)
    client = TestClient(create_app(orch, accounts), base_url=PUBLIC_URL)
    accounts.create_user(EMAIL, PASSWORD, verified=True)
    login = path.endswith("login")
    shouted = {"Content-Type": "Application/X-WWW-Form-Urlencoded" if login else "Application/JSON"}
    typed = [(email, {}) for email in (EMAIL, EMAIL.upper(), EMAIL.title(), f" {EMAIL}", f"{EMAIL} ",
                                       f"Asha <{EMAIL}>", f"Ravi <{EMAIL}>")] + [(EMAIL, shouted)]
    tries = [client.post(path, data={"username": email, "password": PASSWORD}, headers=headers) if login
             else client.post(path, json={"email": email, "password": PASSWORD}, headers=headers)
             for email, headers in typed]
    assert [(r.status_code, "this email address" in r.text) for r in tries[2:]] == [(429, True)] * 6


def test_signing_out_with_a_json_header_and_no_body_works(orch, accounts):
    """The web app sends Content-Type: application/json on every call, with or without a body."""
    client = signed_in_client(orch, accounts)
    assert client.post("/auth/cookie/logout", headers={"Content-Type": "application/json"}).status_code == 204


def test_the_route_map_is_not_served_to_anonymous_callers(orch, accounts):
    client = TestClient(create_app(orch, accounts), base_url=PUBLIC_URL)
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(path).status_code == 404, path


@pytest.mark.parametrize("path, page", [("/privacy", "privacy.html"), ("/login/google", "index.html")])
def test_the_privacy_page_and_googles_return_page_open_signed_out(orch, accounts, path, page):
    from backend.paths import FRONTEND
    r = TestClient(create_app(orch, accounts), base_url=PUBLIC_URL).get(path)
    assert (r.status_code, r.text) == (200, (FRONTEND / page).read_text())


@pytest.mark.parametrize("path", [*PUBLIC_PATHS, "/static/app.js", "/static/app.css"])
def test_every_public_path_and_the_app_files_open_signed_out(orch, accounts, path):
    assert TestClient(create_app(orch, accounts), base_url=PUBLIC_URL).get(path).status_code == 200


@pytest.mark.parametrize("signed_in, path, status", [(False, "/", 200), (False, "/health", 200), (True, "/ui/usage", 200),
                                                     (False, "/portfolio", 401), (False, "/static/app.js", 200)])
def test_every_kind_of_answer_carries_the_security_headers(orch, accounts, signed_in, path, status):
    client = signed_in_client(orch, accounts) if signed_in else TestClient(create_app(orch, accounts), base_url=PUBLIC_URL)
    r = client.get(path)
    assert (r.status_code, {k: r.headers.get(k) for k in SECURITY_HEADERS}) == (status, SECURITY_HEADERS)


@pytest.mark.parametrize("public_url, hsts", [(PUBLIC_URL, HSTS), ("http://127.0.0.1:8931", None)], ids=["https", "http"])
def test_browsers_are_told_to_keep_to_https_only_when_the_address_is_https(orch, db_path, public_url, hsts):
    from backend.api.accounts import Accounts
    from .conftest import SECRET
    client = TestClient(create_app(orch, Accounts(db_path, public_url, SECRET, None, None)), base_url=public_url)
    assert client.get("/health").headers.get("strict-transport-security") == hsts


@pytest.mark.parametrize("signed_in, path", [(True, "/ui/usage"), (False, "/portfolio")], ids=["answer", "refusal"])
def test_json_answers_are_never_stored_by_a_cache(orch, accounts, signed_in, path):
    client = signed_in_client(orch, accounts) if signed_in else TestClient(create_app(orch, accounts), base_url=PUBLIC_URL)
    assert client.get(path).headers.get("cache-control") == "no-store"


@pytest.mark.parametrize("method, path", [("GET", "/health"), ("GET", "/ui/usage"), ("OPTIONS", "/ui/usage")])
def test_no_answer_lets_another_site_read_it(orch, accounts, method, path):
    r = signed_in_client(orch, accounts).request(method, path, headers={"Origin": "https://evil.example",
                                                                        "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers


def test_the_install_screen_takes_its_colours_from_the_light_theme(orch, accounts):
    from backend.paths import FRONTEND
    light = (FRONTEND / "app.css").read_text().split(':root:not([data-theme="dark"]) {')[1].split("}")[0]
    m = TestClient(create_app(orch, accounts), base_url=PUBLIC_URL).get("/manifest.webmanifest").json()
    assert (f"--bg: {m['theme_color']};" in light, f"--bg-2: {m['background_color']};" in light) == (True, True)


@pytest.mark.parametrize("on", [True, False])
def test_auth_methods_says_which_ways_to_sign_in_are_on(orch, db_path, google, outbox, on):
    from backend.api.accounts import Accounts
    from .conftest import SECRET
    accounts = Accounts(db_path, PUBLIC_URL, SECRET, google if on else None, outbox if on else None)
    assert TestClient(create_app(orch, accounts), base_url=PUBLIC_URL).get("/auth/methods").json() == \
        {"google": on, "signup": on}


def test_users_me_says_who_is_signed_in(orch, accounts):
    assert signed_in_client(orch, accounts).get("/users/me").json() == {"email": EMAIL, "is_verified": True,
                                                                        "google": False}


def test_rate_limit_buckets_that_have_refilled_are_dropped_past_max_keys(monkeypatch):
    import time as _t
    import backend.api.app as api_mod
    monkeypatch.setattr(api_mod, "MAX_KEYS", 1)
    bucket = api_mod.TokenBucket(1, 0.01)
    bucket.allow("a"), bucket.allow("b")
    _t.sleep(0.05)
    bucket.allow("c")
    assert list(bucket.state) == ["c"]


@pytest.fixture
def asha_and_ravi(orch, accounts):
    """Two users signed in to one app, each with a portfolio holding TEST."""
    app = create_app(orch, accounts)
    clients = signed_in_client(orch, accounts, EMAIL, app), signed_in_client(orch, accounts, OTHER, app)
    for client in clients:
        client.post("/portfolio", json={"cash": 500, "transactions": [
            {"symbol": "TEST", "side": "BUY", "quantity": 5, "price": 100, "trade_date": "2025-06-02"}]})
    return clients


def test_one_users_portfolio_is_not_anothers(orch, accounts):
    asha, ravi = signed_in_client(orch, accounts), signed_in_client(orch, accounts, OTHER)
    asha.post("/portfolio/cash", json={"cash": 500})
    assert ravi.get("/portfolio").status_code == 404


def test_one_users_reports_are_not_listed_for_another(asha_and_ravi):
    asha, ravi = asha_and_ravi
    made = asha.post("/reports/TEST").json()["id"]
    assert ([r["id"] for r in asha.get("/reports").json()], ravi.get("/reports").json()) == ([made], [])


def test_one_users_signal_plan_is_kept_for_that_user_only(orch, accounts, asha_and_ravi, signal_inputs):
    asha_and_ravi[0].get("/signals?capital=100000")
    assert [orch.store.latest_plan(owner(accounts, email)) is None for email in (EMAIL, OTHER)] == [False, True]


def test_one_users_acknowledgement_does_not_hide_anothers_alert(orch, asha_and_ravi):
    asha, ravi = asha_and_ravi
    alert = orch.store.save_alert("TEST", "price_move", "high", "TEST moved 6%", ["moved 6%"])
    asha.post(f"/alerts/{alert}/ack")
    assert [[a["id"] for a in client.get("/alerts").json()] for client in (asha, ravi)] == [[], [alert]]


def test_one_users_chat_is_not_anothers_with_the_same_session_id(orch, accounts, monkeypatch):
    monkeypatch.setattr(type(orch.hub), "writer", lambda self: _Picker("NONE"))
    asha, ravi = signed_in_client(orch, accounts), signed_in_client(orch, accounts, OTHER)
    asha.post("/chat", json={"message": "Analyze TEST", "session_id": "s1"})
    assert ravi.post("/chat", json={"message": "What about fundamentals?", "session_id": "s1"}).json()["intent"] == "help"


def test_one_users_agent_conversation_is_not_anothers_with_the_same_session_id(orch, accounts, monkeypatch):
    from backend.models.agent.client import SelfAgentClient
    sessions = []
    monkeypatch.setattr(SelfAgentClient, "ask", lambda self, question, subject="", *, session: sessions.append(session) or {})
    app = create_app(orch, accounts)
    for email in (EMAIL, OTHER):
        signed_in_client(orch, accounts, email, app).post("/ui/agent/ask", json={"message": "Why?", "session_id": "s1"})
    assert sessions == [f"{owner(accounts, email)}:s1" for email in (EMAIL, OTHER)]


def count_rows(store, db_path):
    """Rows per table that a user owns, in the portfolio store and the accounts file; usage leaves out the all-users row."""
    import sqlite3
    from contextlib import closing
    owned = ("portfolios", "transactions", "fund_transactions", "portfolio_snapshots", "reports", "signal_plans",
             "alert_acks")
    counts = {t: store.conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in owned}
    counts["usage"] = store.conn.execute("SELECT count(*) FROM usage WHERE who != '*'").fetchone()[0]
    with closing(sqlite3.connect(db_path)) as conn:
        counts.update({t: conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in ("user", "accesstoken")})
    return counts


@pytest.fixture
def asha_deleted(orch, accounts, db_path, asha_and_ravi, monkeypatch, funds_served):
    """Both users made one row of each kind, then asha deleted her account; returns the counts before and after."""
    from backend.models.agent.client import SelfAgentClient
    monkeypatch.setattr(SelfAgentClient, "ask", lambda self, question, subject="", *, session: {})
    alert = orch.store.save_alert("TEST", "price_move", "high", "TEST moved 6%", ["moved 6%"])
    for client, email in zip(asha_and_ravi, (EMAIL, OTHER)):
        for method, path, body in (("post", "/reports/TEST", None), ("post", f"/alerts/{alert}/ack", None),
                                   ("post", "/ui/agent/ask", {"message": "Why?"}),
                                   ("post", "/portfolio/funds/transactions", FUND_BUY),
                                   ("post", "/chat", {"message": "How is my portfolio performing?"})):
            assert getattr(client, method)(path, json=body).status_code == 200, path
        orch.store.save_plan(owner(accounts, email), {"as_of": END, "strategy": "momentum", "actions": []})
    before = count_rows(orch.store, db_path)
    deleted = asha_and_ravi[0].delete("/users/me")
    return deleted, before, count_rows(orch.store, db_path)


def test_deleting_an_account_removes_every_row_it_owned(asha_deleted):
    deleted, before, after = asha_deleted
    assert (deleted.status_code, set(before.values()), after) == (204, {2}, {t: 1 for t in before})


def test_a_deleted_accounts_cookie_gets_401(asha_deleted, asha_and_ravi):
    r = asha_and_ravi[0].get("/portfolio")
    assert (r.status_code, r.json()) == UNAUTHORIZED


def test_deleting_an_account_leaves_another_users_portfolio(asha_deleted, asha_and_ravi):
    assert len(asha_and_ravi[1].get("/portfolio").json()["transactions"]) == 1


def test_deleting_an_account_drops_its_chat_sessions(orch, accounts, asha_and_ravi):
    gone = owner(accounts)
    asha_and_ravi[0].post("/chat", json={"message": "Analyze TEST", "session_id": "s1"})
    asha_and_ravi[0].delete("/users/me")
    assert [key for key in orch.sessions if key[0] == gone] == []


def test_a_save_from_another_tab_while_the_account_is_deleted_is_removed_too(orch, accounts, asha_and_ravi,
                                                                            monkeypatch):
    from backend.api.accounts import UserManager
    real_delete, gone = UserManager.delete, owner(accounts)

    async def delete_then_save(self, user, *args):
        await real_delete(self, user, *args)
        orch.save_portfolio(gone, 500, "moderate", "medium", [])     # that tab signed in before the account went
    monkeypatch.setattr(UserManager, "delete", delete_then_save)
    asha_and_ravi[0].delete("/users/me")
    assert orch.store.portfolio(gone) is None


def test_other_requests_are_served_while_an_account_is_erased(orch, accounts, monkeypatch):
    import threading
    started, served, waits, real_forget = threading.Event(), threading.Event(), [], orch.forget

    def slow_forget(portfolio):
        started.set()
        waits.append(served.wait(20))       # times out if the erase holds the event loop that serves /health
        real_forget(portfolio)
    monkeypatch.setattr(orch, "forget", slow_forget)
    with signed_in_client(orch, accounts) as client:       # one event loop for both requests
        deleting = threading.Thread(target=client.delete, args=("/users/me",))
        deleting.start()
        started.wait(20)
        client.get("/health")
        served.set()
        deleting.join()
    assert waits == [True, True]


def test_the_state_backup_holds_no_sign_in_while_the_live_database_keeps_it(orch, accounts, db_path, tmp_path):
    import sqlite3
    from contextlib import closing
    from backend.database import hfstate
    token = signed_in_client(orch, accounts).cookies["__Host-stockintel"]
    hfstate.snapshot(Path(db_path), tmp_path / "backup", tmp_path / "no-cache")
    with closing(sqlite3.connect(db_path)) as live:
        kept = live.execute("SELECT count(*) FROM accesstoken WHERE token = ?", (token,)).fetchone()[0]
    assert (token.encode() in (tmp_path / "backup" / hfstate.DB_NAME).read_bytes(), kept) == (False, 1)


def accounts_command(db_path, action, email=EMAIL):
    import argparse
    from backend import cli
    return cli.cmd_accounts(argparse.Namespace(db=db_path, action=action, email=email))


@pytest.fixture
def shared_portfolio(monkeypatch, db_path):
    """The old shared key's portfolio in a database file, and the server settings the accounts command reads."""
    from backend.database.storage import Store
    from .test_accounts import GOOD_ENV, set_env
    set_env(monkeypatch, GOOD_ENV)
    Store(db_path).save_portfolio(DEFAULT_PORTFOLIO, 500.0, "moderate", "medium", [])


def portfolio_names(db_path):
    from backend.database.storage import Store
    return [r[0] for r in Store(db_path).conn.execute("SELECT name FROM portfolios")]


def test_claim_then_release_moves_the_shared_portfolio_to_an_account_and_back(accounts, db_path, shared_portfolio):
    accounts.create_user(EMAIL, PASSWORD, verified=True)
    names = []
    for action in ("claim", "release"):
        accounts_command(db_path, action)
        names.append(portfolio_names(db_path))
    assert names == [[owner(accounts)], [DEFAULT_PORTFOLIO]]


def test_claim_refuses_an_account_whose_email_is_not_confirmed(accounts, db_path, shared_portfolio):
    accounts.create_user(EMAIL, PASSWORD, verified=False)
    with pytest.raises(SystemExit, match="confirmed"):
        accounts_command(db_path, "claim")


def test_claim_refuses_an_account_that_already_has_a_portfolio(accounts, db_path, shared_portfolio):
    from backend.database.storage import Store
    Store(db_path).save_portfolio(owner_of(accounts.create_user(EMAIL, PASSWORD, verified=True)), 0.0, "moderate",
                                  "medium", [])
    with pytest.raises(SystemExit, match="already exists"):
        accounts_command(db_path, "claim")


def test_the_local_app_refuses_to_listen_beyond_this_computer(tmp_path):
    import argparse
    from backend import cli
    with pytest.raises(SystemExit, match="stockintel serve"):
        cli.cmd_app(argparse.Namespace(host="0.0.0.0", port=8931, db=str(tmp_path / "s.db")))


def test_the_oracle_host_opens_only_https_to_the_internet():
    compose = yaml.safe_load((ORACLE / "compose.yaml").read_text())
    assert [p for s in compose["services"].values() for p in s.get("ports", [])] == ["443:443"]
    assert re.findall(r"^for port in (.*); do$", (ORACLE / "setup.sh").read_text(), re.M) == ["443"]


def oracle_services(file="compose.yaml"):
    return yaml.safe_load((ORACLE / file).read_text())["services"]


def test_the_app_trusts_a_forwarded_caller_address_only_from_caddy():
    services = oracle_services()
    assert services["app"]["environment"]["FORWARDED_ALLOW_IPS"] == services["caddy"]["networks"]["edge"]["ipv4_address"]


def test_caddy_cannot_reach_the_agent():
    services = oracle_services()
    assert set(services["caddy"]["networks"]) & set(services["agent"]["networks"]) == set()


def test_only_the_app_gets_the_sign_in_secrets():
    assert [name for name, s in oracle_services().items() if "auth.env" in s.get("env_file", [])] == ["app"]


def test_setup_no_longer_makes_or_prints_the_shared_key():
    setup = (ORACLE / "setup.sh").read_text()
    assert ("#key=" in setup, "STOCKINTEL_API_KEY" in setup) == (False, False)


def test_caddy_caps_the_request_body_size():
    assert re.search(r"request_body \{\s*max_size \d+MB\s*\}", (ORACLE / "Caddyfile").read_text())


def test_the_gate_check_boot_neither_restores_nor_saves_the_real_state():
    assert oracle_services("gatecheck.yaml")["app"]["environment"]["STOCKINTEL_STATE_REPO"] == ""


def test_the_gate_check_boot_opens_a_database_other_than_the_real_one():
    real = re.search(r"STOCKINTEL_DB=(\S+)", (ORACLE.parents[1] / "Dockerfile").read_text())[1]
    assert oracle_services("gatecheck.yaml")["app"]["environment"]["STOCKINTEL_DB"] not in (real, "")


def test_render_asks_for_every_setting_serve_will_not_start_without():
    render = yaml.safe_load((ORACLE.parents[1] / "render.yaml").read_text())
    assert set(SERVE_NEEDS) - {v["key"] for v in render["services"][0]["envVars"]} == set()


def test_api_analysis_and_domain_endpoints(orch, accounts):
    client = signed_in_client(orch, accounts)
    r = client.get("/stocks/TEST/analysis")
    assert r.status_code == 200 and r.json()["decision"]["decision_support_label"]
    assert client.get("/stocks/TEST/technical").json()["result"]["domain"] == "technical"
    assert client.get("/stocks/TEST/fundamentals").json()["result"]["domain"] == "fundamental"
    assert client.get("/stocks/TEST/astrology").status_code == 404
    chart = client.get("/stocks/TEST/chart?bars=100").json()
    assert len(chart["dates"]) == 100 and len(chart["sma50"]) == 100


def test_api_validates_inputs(orch, accounts):
    client = signed_in_client(orch, accounts)
    assert client.get("/stocks/DROP;TABLE/analysis").status_code in (400, 404)
    assert client.post("/chat", json={"message": ""}).status_code == 422
    bad = {"cash": 0, "transactions": [{"symbol": "TEST", "side": "SELL", "quantity": 5, "price": 10,
                                        "trade_date": "2025-01-01"}]}
    assert client.post("/portfolio", json=bad).status_code == 400   # oversell rejected
    assert client.get("/screener?budget=-5").status_code == 422


def test_api_missing_data_is_a_clean_404(orch, accounts):
    client = signed_in_client(orch, accounts)
    r = client.get("/stocks/NOSUCH/analysis")
    assert r.status_code == 404 and r.json()["error"] == "data_unavailable"


def test_api_rate_limit(orch, monkeypatch, accounts):
    import backend.api.app as api_mod
    monkeypatch.setattr(api_mod, "RATE_PER_MIN", 3)
    client = signed_in_client(orch, accounts)
    codes = [client.get("/search?q=TE").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]


def test_one_users_requests_do_not_use_up_another_users_limit(orch, monkeypatch, accounts):
    import backend.api.app as api_mod
    monkeypatch.setattr(api_mod, "RATE_PER_MIN", 3)
    app = create_app(orch, accounts)
    first, second = signed_in_client(orch, accounts, app=app), signed_in_client(orch, accounts, "other@example.com", app=app)
    assert 429 in [first.get("/search?q=TE").status_code for _ in range(5)]
    assert second.get("/search?q=TE").status_code == 200


def test_api_chat_round_trip(orch, accounts):
    client = signed_in_client(orch, accounts)
    r = client.post("/chat", json={"message": "What is a Sharpe ratio?", "session_id": "x1"})
    assert r.status_code == 200 and r.json()["intent"] == "knowledge"


def test_api_json_is_strict(orch, accounts):
    client = signed_in_client(orch, accounts)
    raw = client.get("/stocks/TEST/analysis").text
    assert "NaN" not in raw and "Infinity" not in raw


def test_rejected_portfolio_post_leaves_the_stored_portfolio_intact(orch, accounts):
    client = signed_in_client(orch, accounts)
    good = {"cash": 100, "transactions": [{"symbol": "TEST", "side": "BUY", "quantity": 5, "price": 10,
                                           "trade_date": "2025-01-01"}]}
    assert client.post("/portfolio", json=good).status_code == 200
    bad = {"cash": 0, "transactions": [{"symbol": "TEST", "side": "SELL", "quantity": 5, "price": 10,
                                        "trade_date": "2025-01-01"}]}
    assert client.post("/portfolio", json=bad).status_code == 400
    stored = orch.store.portfolio(owner(accounts))
    assert stored["cash"] == 100 and len(stored["transactions"]) == 1


def test_held_status_does_not_depend_on_how_the_ticker_was_typed(orch):
    orch.save_portfolio(DEFAULT_PORTFOLIO, 0.0, "moderate", "medium", [
        {"symbol": "TEST.NS", "side": "BUY", "quantity": 5, "price": 10, "trade_date": "2025-01-01"}])
    assert orch.analysis(orch.session(DEFAULT_PORTFOLIO, "h"), "TEST").held
    assert orch.store.portfolio(DEFAULT_PORTFOLIO)["transactions"][0]["symbol"] == "TEST"


def test_one_analysis_serves_two_portfolios_with_the_same_held_flag(orch, monkeypatch):
    svc = orch.service()
    analyze, calls = svc.analyze, []
    monkeypatch.setattr(svc, "analyze", lambda *a, **k: calls.append(a) or analyze(*a, **k))
    for portfolio in ("alice", "bob"):
        orch.analysis(orch.session(portfolio, "shared"), "TEST")
    assert len(calls) == 1


def test_old_sessions_are_dropped_past_the_cap(orch, monkeypatch):
    import backend.api.orchestrator as orchestrator
    monkeypatch.setattr(orchestrator, "SESSIONS_KEPT", 2)
    monkeypatch.setattr(type(orch.hub), "writer", lambda self: _Picker("NONE"))
    for session_id in ("a", "b", "c"):
        orch.handle("Analyze TEST", DEFAULT_PORTFOLIO, session_id)
    follow_ups = [orch.handle("What about fundamentals?", DEFAULT_PORTFOLIO, session_id).intent for session_id in ("a", "c")]
    assert follow_ups == ["help", "domain"]


def _weak_ref_to_analysis(orch, s, symbol):
    """A weak reference to `symbol`'s analysis, so a test can see when the orchestrator lets go of it."""
    import weakref
    return weakref.ref(orch.analysis(s, symbol))


def test_expired_analyses_are_evicted(orch, monkeypatch):
    import gc
    import backend.api.orchestrator as orchestrator
    monkeypatch.setattr(orchestrator, "ANALYSIS_REUSE_S", 0)
    s = orch.session(DEFAULT_PORTFOLIO, "evict")
    expired = _weak_ref_to_analysis(orch, s, "TEST")
    orch.analysis(s, "PEER")
    gc.collect()
    assert expired() is None


def test_analyses_past_the_cap_drop_the_oldest(orch, monkeypatch):
    import gc
    import backend.api.orchestrator as orchestrator
    monkeypatch.setattr(orchestrator, "ANALYSES_KEPT", 1)
    s = orch.session(DEFAULT_PORTFOLIO, "cap")
    oldest = _weak_ref_to_analysis(orch, s, "TEST")
    orch.analysis(s, "PEER")
    gc.collect()
    assert oldest() is None


def test_api_market_endpoint(orch, local_provider, accounts):
    local_provider.frames["^NSEI"] = local_provider.frames["^NSEI"]
    client = signed_in_client(orch, accounts)
    r = client.get("/market")
    assert r.status_code == 200 and r.json()["trend"] in ("up", "down")


@pytest.fixture
def app_client(orch, local_provider, accounts):
    import time as _t
    from backend.database.sources.panel import panel_from_frames
    app = create_app(orch, accounts)
    hub = app.state.hub
    frames = {s.split(".")[0]: local_provider.frames[s] for s in ("TEST.NS", "PEER.NS")}
    hub._panel, hub._panel_at = panel_from_frames(frames), _t.time()
    hub._names = pd.DataFrame({"symbol": ["TEST", "PEER"], "name": ["Test Industries Ltd.", "Peer Power Ltd."],
                               "sector": ["Industrials", "Energy"], "nifty200": [True, True]})
    return signed_in_client(orch, accounts, app=app)


def test_web_app_shell_and_assets_are_served(app_client):
    assert "StockIntel" in app_client.get("/").text
    assert app_client.get("/static/app.js").status_code == 200
    assert app_client.get("/static/app.css").status_code == 200


def test_ui_search_matches_symbol_and_company_name(app_client):
    assert app_client.get("/ui/search?q=te").json()[0]["symbol"] == "TEST"
    assert app_client.get("/ui/search?q=power").json()[0]["symbol"] == "PEER"


def test_ui_stock_overview_and_chart(app_client):
    o = app_client.get("/ui/stock/TEST").json()
    assert o["name"] == "Test Industries Limited" and o["low_52w"] <= o["price"] <= o["high_52w"]
    c = app_client.get("/ui/stock/TEST/chart?range=1M").json()
    assert len(c["close"]) == 22
    assert app_client.get("/ui/stock/TEST/chart?range=7Y").status_code == 400


def test_ui_verdict_explains_both_sides(app_client):
    v = app_client.get("/ui/stock/TEST/verdict").json()
    assert v["verdict"] in ("BUY", "BUY — HALF SIZE", "WAIT", "DON'T BUY", "HOLD / ADD", "HOLD", "SELL AT REBALANCE")
    assert v["momentum_rank"] in (1, 2) and v["universe_size"] == 2
    assert v["why_buy"] or v["why_not"]
    assert v["evidence_note"].startswith("The buy/sell rule is NSE's momentum method.")


def test_verdict_bands_say_which_portfolio_size_they_assume(app_client):
    """plan() sizes the portfolio from the capital, so a ₹2 lakh plan keeps only the top 20; bands set for 20 stocks
    gave the opposite advice on the same holding without saying why."""
    from backend.advisory.signals.advisor import N_DEFAULT
    v = app_client.get("/ui/stock/TEST/verdict").json()
    assert (v["buy_rank"], v["keep_rank"]) == (N_DEFAULT, 2 * N_DEFAULT)
    assert f"These bands are for a {N_DEFAULT}-stock portfolio" in v["headline"]


def test_a_half_size_verdict_sizes_the_position_at_half(app_client, monkeypatch):
    import backend.api.hub as hubmod
    from backend.advisory.signals.advisor import CRASH_EXPOSURE, N_DEFAULT
    state = hubmod.market_state
    monkeypatch.setattr(hubmod, "market_state", lambda bench, asof: {**state(bench, asof), "momentum_crash_risk": True})
    v = app_client.get("/ui/stock/TEST/verdict").json()
    assert v["verdict"] == "BUY — HALF SIZE"
    assert f"(≈{100 * CRASH_EXPOSURE / N_DEFAULT:.1f}% with {N_DEFAULT} stocks)" in v["plan"]["position_size"]


def test_verdict_promoter_totals_count_every_filing_in_the_window(app_client, monkeypatch):
    """The display list keeps the 20 newest filings; a promoter buy behind 25 newer director sales still counts."""
    from backend.database.sources import insider
    today = pd.Timestamp.today().normalize()
    sales = [{"broadcast": today - pd.Timedelta(days=1 + i % 20), "symbol": "TEST", "category": "Director",
              "direction": -1, "value": 1e6, "shares": 100, "pid": i} for i in range(25)]
    buy = {"broadcast": today - pd.Timedelta(days=60), "symbol": "TEST", "category": "Promoters",
           "direction": 1, "value": 5e7, "shares": 10_000, "pid": 99}
    monkeypatch.setattr(insider, "load", lambda: pd.DataFrame(sales + [buy]))
    v = app_client.get("/ui/stock/TEST/verdict").json()
    assert "Promoters bought ₹5.00 cr in the last 90 days (weak signal on its own)." in v["why_buy"]


@pytest.mark.parametrize("trend, listed_against", [("deteriorating", True), ("stable/mixed", False)])
def test_neutral_fundamentals_are_not_a_reason_to_buy(app_client, orch, monkeypatch, trend, listed_against):
    from backend.advisory.fusion.evidence import DomainResult
    analysis = orch.analysis

    def neutral(s, sym):
        a = analysis(s, sym)
        a.results["fundamental"] = DomainResult("fundamental", "mixed", -0.05, details={"trend": trend})
        return a
    monkeypatch.setattr(orch, "analysis", neutral)
    v = app_client.get("/ui/stock/TEST/verdict").json()
    line = f"Fundamentals look mixed (trend: {trend})."
    assert line not in v["why_buy"] and (line in v["why_not"]) == listed_against


def test_ui_verdict_that_gives_a_stop_says_the_track_record_left_it_out(app_client):
    v = app_client.get("/ui/stock/TEST/verdict").json()
    assert v["plan"]["stop_loss"] and "measured without it" in v["evidence_note"]


def test_portfolio_single_transaction_add_and_delete(app_client, orch, accounts):
    tx = {"symbol": "TEST", "side": "BUY", "quantity": 3, "price": 100, "trade_date": "2025-01-02"}
    assert app_client.post("/portfolio/transactions", json=tx).status_code == 200
    assert app_client.post("/portfolio/transactions", json={**tx, "side": "SELL", "quantity": 9}).status_code == 400
    assert len(orch.store.portfolio(owner(accounts))["transactions"]) == 1
    assert app_client.delete("/portfolio/transactions/0").status_code == 200
    assert orch.store.portfolio(owner(accounts))["transactions"] == []


def test_two_transactions_added_at_once_are_both_kept(app_client, orch, monkeypatch, accounts):
    """Each add reads the stored list and writes it back whole, so two at once could keep only the later one."""
    import threading
    read, both_reading = orch.store.portfolio, threading.Barrier(2, timeout=1)

    def read_together(name):
        try:            # under the lock the second add cannot read until the first has saved, so this times out
            both_reading.wait()
        except threading.BrokenBarrierError:
            pass
        return read(name)
    monkeypatch.setattr(orch.store, "portfolio", read_together)
    tx = {"symbol": "TEST", "side": "BUY", "quantity": 3, "price": 100, "trade_date": "2025-01-02"}
    adds = [threading.Thread(target=app_client.post, args=("/portfolio/transactions",),
                             kwargs={"json": tx}) for _ in range(2)]
    for add in adds:
        add.start()
    for add in adds:
        add.join()
    assert len(read(owner(accounts))["transactions"]) == 2


def test_ui_technical_chart_carries_geometry_markers_cone_and_records(app_client):
    t = app_client.get("/ui/stock/TEST/technical?bars=120").json()
    assert len(t["dates"]) == 120 and len(t["open"]) == len(t["close"]) == 120
    assert len(t["cone"]) == FORECAST_DAYS and all(c["lo80"] <= c["lo50"] <= c["mid"] <= c["hi50"] <= c["hi80"] for c in t["cone"])
    for mk in t["markers"]:
        assert mk["date"] in t["dates"] and "summary" in mk["record"]
    for p in t["patterns"]:
        assert "geometry" in p and "summary" in p["record"]
    assert "no model we tested predicted direction" in t["cone_note"]


def test_pattern_geometry_points_are_the_swing_points_used():
    import numpy as np
    from backend.advisory.statistics import patterns
    from .conftest import bars
    closes = np.concatenate([np.linspace(95, 100, 60), np.linspace(100, 130, 30), np.linspace(130, 115, 12)[1:],
                             np.linspace(115, 129.5, 12)[1:], np.linspace(129.5, 110, 14)[1:]])
    top = next(p for p in patterns.detect(bars(closes, spread=0.004)) if p["pattern"] == "double_top")
    labels = [q[2] for q in top["geometry"]["points"]]
    assert labels == ["Start", "Top 1", "Neckline", "Top 2"]
    neck = top["geometry"]["lines"][0]
    assert neck["label"] == "Neckline" and neck["from"][1] == pytest.approx(top["breakout_level"], abs=0.01)


def test_portfolio_history_rebuilds_value_from_transactions(app_client, orch, accounts):
    orch.save_portfolio(owner(accounts), 0.0, "moderate", "medium", [
        {"symbol": "TEST", "side": "BUY", "quantity": 10, "price": 100, "trade_date": "2025-06-02"}])
    h = app_client.get("/ui/portfolio/history").json()
    assert h["dates"][0] >= "2025-06-02" and h["invested"][-1] == pytest.approx(1000)
    assert all(v > 0 for v in h["value"])


@pytest.mark.parametrize("sees_alices_buy", [
    lambda orch, s: orch.hub.verdict("TEST", s)["held"],
    lambda orch, s: orch.hub.portfolio_history(s.portfolio)["invested"][-1] == 1000,
    lambda orch, s: "TEST" in [h["symbol"] for h in orch.portfolio_report(s)[0].holdings]],
    ids=["verdict", "history", "report"])
def test_two_portfolios_do_not_see_each_others_holdings(app_client, orch, sees_alices_buy):
    for portfolio, symbol, quantity in (("alice", "TEST", 10), ("bob", "PEER", 5)):
        orch.save_portfolio(portfolio, 0.0, "moderate", "medium", [
            {"symbol": symbol, "side": "BUY", "quantity": quantity, "price": 100, "trade_date": "2025-06-02"}])
    assert sees_alices_buy(orch, orch.session("alice", "web")) and not sees_alices_buy(orch, orch.session("bob", "web"))


@pytest.fixture
def two_portfolios_reports(app_client, orch, accounts):
    """Report ids made by alice in chat and by the signed-in user through the API."""
    for portfolio in ("alice", owner(accounts)):
        _store_test_buy(orch, 0.0, 100, "2025-06-02", portfolio)
    alices = orch.handle("Generate a report on TEST", "alice", "r").data["report_id"]
    return {"alice": alices, "mine": app_client.post("/reports/TEST").json()["id"]}


def test_the_report_list_leaves_out_another_portfolios_reports(app_client, two_portfolios_reports):
    listed = app_client.get("/reports").json()
    assert [r["id"] for r in listed] == [two_portfolios_reports["mine"]]


def test_another_portfolios_report_cannot_be_opened(app_client, two_portfolios_reports):
    opened = {who: app_client.get(f"/reports/{rid}").status_code
              for who, rid in two_portfolios_reports.items()}
    assert opened == {"alice": 404, "mine": 200}


def test_a_report_before_the_portfolio_is_set_up_is_a_plain_400(app_client):
    r = app_client.post("/reports/TEST")
    assert r.status_code == 400 and r.json()["detail"].startswith("Set up your portfolio first")


def test_a_portfolio_keeps_only_its_own_newest_reports(orch):
    from backend.database.storage import REPORTS_KEPT
    for portfolio in ("alice", DEFAULT_PORTFOLIO):
        _store_test_buy(orch, 0.0, 100, "2025-06-02", portfolio)
    orch.store.save_report("alice", "TEST", "alice's report")
    for n in range(REPORTS_KEPT + 1):
        orch.store.save_report(DEFAULT_PORTFOLIO, "TEST", f"report {n}")
    kept = {p: [r["body"] for r in orch.store.reports(p, limit=REPORTS_KEPT + 1)] for p in ("alice", DEFAULT_PORTFOLIO)}
    assert kept == {"alice": ["alice's report"], DEFAULT_PORTFOLIO: [f"report {n}" for n in range(REPORTS_KEPT, 0, -1)]}


def test_a_portfolio_keeps_only_its_own_newest_plans(orch, monkeypatch):
    """Every /signals call keeps a plan, so without a cap one user could grow the database without end."""
    import backend.database.storage as storage
    monkeypatch.setattr(storage, "PLANS_KEPT", 2)
    for portfolio in ("alice", DEFAULT_PORTFOLIO):
        _store_test_buy(orch, 0.0, 100, "2025-06-02", portfolio)
    for portfolio in ("alice", DEFAULT_PORTFOLIO, DEFAULT_PORTFOLIO, DEFAULT_PORTFOLIO):
        orch.store.save_plan(portfolio, {"as_of": END, "strategy": "momentum", "actions": []})
    kept = orch.store.conn.execute("SELECT p.name, count(*) FROM signal_plans s JOIN portfolios p ON p.id = s.portfolio_id"
                                   " GROUP BY p.name").fetchall()
    assert dict(map(tuple, kept)) == {"alice": 1, DEFAULT_PORTFOLIO: 2}


def test_one_portfolios_plan_does_not_move_anothers_stops(orch):
    from backend.advisory.signals.advisor import STOP_PCT
    for portfolio in ("alice", "bob"):
        _store_test_buy(orch, 0.0, 100, "2025-06-02", portfolio)
    orch.store.save_plan("bob", {"as_of": END, "strategy": "momentum", "actions": [{"symbol": "TEST", "stop_level": 55.5}]})
    stops = [re.search(r"stop ₹([\d,.]+)", orch.handle("Check my stops", p, "s").text).group(1) for p in ("alice", "bob")]
    assert stops == [f"{100 * (1 - STOP_PCT):,.2f}", "55.50"]


def test_a_plan_for_new_money_needs_no_portfolio_set_up(app_client, signal_inputs):
    """A plan is kept with its portfolio, and a new user has none yet; the plan itself must still come back."""
    assert app_client.get("/signals?capital=100000").status_code == 200


def test_acknowledging_an_alert_hides_it_only_from_that_portfolio(app_client, orch, accounts):
    for portfolio in ("alice", owner(accounts)):
        _store_test_buy(orch, 0.0, 100, "2025-06-02", portfolio)
    alert = orch.store.save_alert("TEST", "price_move", "high", "TEST moved 6%", ["moved 6%"])
    app_client.post(f"/alerts/{alert}/ack")
    assert [len(app_client.get("/alerts").json()), len(orch.alerts("alice"))] == [0, 1]


def test_alerts_list_only_the_stocks_the_portfolio_holds(app_client, orch, accounts):
    """Anyone's analysis writes an alert, so an unfiltered list would show which stocks other users looked up."""
    _store_test_buy(orch, 0.0, 100, "2025-06-02", owner(accounts))
    for symbol in ("TEST", "PEER"):
        orch.store.save_alert(symbol, "price_move", "high", f"{symbol} moved 6%", ["moved 6%"])
    assert [a["symbol"] for a in app_client.get("/alerts").json()] == ["TEST"]


def test_an_alert_from_analysing_the_suffixed_symbol_reaches_the_holder(orch):
    """Holdings drop the ".NS" suffix, and an alert kept as typed ("TEST.NS") was hidden from TEST's holder."""
    _store_test_buy(orch, 0.0, 100, "2025-06-02")
    orch.store.save_analysis("TEST.NS", {"decision": {"analysis_timestamp": "2026-01-02", "data_timestamp": "2026-01-02",
                                                      "decision_support_label": "INSUFFICIENT_EVIDENCE", "score": 0.0,
                                                      "confidence": 0.0}, "results": {}})
    orch.analysis(orch.session(DEFAULT_PORTFOLIO, "a"), "TEST.NS")
    assert [a["symbol"] for a in orch.alerts(DEFAULT_PORTFOLIO)] == ["TEST"]


OLD_OWNERLESS_TABLES = """
CREATE TABLE portfolios (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, cash REAL NOT NULL DEFAULT 0,
    risk_profile TEXT NOT NULL DEFAULT 'moderate', horizon TEXT NOT NULL DEFAULT 'medium', created_at TEXT NOT NULL);
CREATE TABLE reports (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, created_at TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE signal_plans (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, as_of TEXT NOT NULL,
    strategy TEXT NOT NULL, payload TEXT NOT NULL);
INSERT INTO reports (symbol, created_at, body) VALUES ('TEST', '2026-01-02', 'an old report');
INSERT INTO signal_plans (created_at, as_of, strategy, payload) VALUES ('2026-01-02', '2026-01-02', 'momentum', '{}');
"""


@pytest.mark.parametrize("had_default, visible", [(True, True), (False, False)])
def test_an_older_database_gives_its_reports_and_plans_to_the_default_portfolio_only(tmp_path, had_default, visible):
    """With no default portfolio the old rows have no known owner; one set up later, or a restart, must not claim them."""
    import sqlite3
    from backend.database.storage import Store
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.executescript(OLD_OWNERLESS_TABLES)
    if had_default:
        old.execute("INSERT INTO portfolios (name, created_at) VALUES (?, '2026-01-01')", (DEFAULT_PORTFOLIO,))
    old.commit()
    old.close()
    Store(path).save_portfolio(DEFAULT_PORTFOLIO, 0.0, "moderate", "medium", [])
    reopened = Store(path)
    seen = [bool(reopened.reports(DEFAULT_PORTFOLIO)), reopened.latest_plan(DEFAULT_PORTFOLIO) is not None]
    assert seen == [visible, visible]


def test_performance_endpoint_reports_missing_build_cleanly(app_client, monkeypatch, tmp_path):
    import backend.advisory.calculators.performance as perf
    monkeypatch.setattr(perf, "PATH", tmp_path / "none.json")
    r = app_client.get("/ui/performance")
    assert r.status_code == 404 and "build-performance" in r.json()["detail"]


def test_builder_spends_within_capital_and_apply_records_the_buys(app_client, orch, accounts):
    from datetime import date as _date
    b = app_client.get("/ui/builder?capital=100000&n=3&risk=full").json()
    assert b["holdings"] and b["cash_left"] >= 0
    assert sum(x["value"] for x in b["holdings"]) + b["cash_left"] == pytest.approx(100000, abs=0.01)
    assert len(b["backcast"]["dates"]) == len(b["backcast"]["basket"]) == len(b["backcast"]["nifty"])
    body = {"holdings": [{"symbol": x["symbol"], "shares": x["shares"], "price": x["price"]} for x in b["holdings"]]}
    assert app_client.post("/ui/builder/apply", json=body).status_code == 200
    txs = orch.store.portfolio(owner(accounts))["transactions"]
    assert [t["symbol"] for t in txs] == [x["symbol"] for x in b["holdings"]]
    assert all(t["side"] == "BUY" and t["trade_date"] == _date.today().isoformat() for t in txs)


def test_builder_leaves_out_excluded_sectors_and_rejects_bad_input(app_client):
    b = app_client.get("/ui/builder?capital=100000&n=3&risk=full&exclude=Energy").json()
    assert "PEER" not in [x["symbol"] for x in b["holdings"]]
    assert app_client.get("/ui/builder?capital=500").status_code == 422
    assert app_client.get("/ui/builder?capital=100000&n=2").status_code == 422


def test_ai_forecast_says_why_when_models_are_switched_off(app_client, monkeypatch):
    monkeypatch.setenv("STOCKINTEL_ML", "0")
    f = app_client.get("/ui/features").json()
    assert f["ml"] is False and f["kronos"] is False      # ALFA's NumPy model is not behind this switch
    r = app_client.get("/ui/stock/TEST/ai?model=chronos")
    assert r.status_code == 404 and "STOCKINTEL_ML" in r.json()["detail"]


def _closed_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_agent_ask_is_503_not_500_when_the_agent_is_down(orch, monkeypatch, tmp_path, accounts):
    import backend.models.agent.client as sc
    monkeypatch.setattr(sc, "CREDENTIALS", tmp_path / "agent.json")
    monkeypatch.setenv("SELFAGENT_URL", f"http://127.0.0.1:{_closed_port()}")
    client = signed_in_client(orch, accounts)
    assert client.get("/ui/agent/status").json()["available"] is False
    r = client.post("/ui/agent/ask", json={"message": "Should I buy ITC?"})
    assert r.status_code == 503 and "not reachable" in r.json()["detail"]


def test_agent_ask_carries_the_chat_session_to_the_agent(orch, monkeypatch, accounts):
    """Every question opened a new agent conversation, so "Analyze RELIANCE" then "Why?" lost the instrument."""
    from backend.models.agent.client import SelfAgentClient
    sessions = []
    monkeypatch.setattr(SelfAgentClient, "ask", lambda self, question, subject="", *, session: sessions.append(session))
    client = signed_in_client(orch, accounts)
    client.post("/ui/agent/ask", json={"message": "Why?", "session_id": "web-1"})
    assert sessions == [f"{owner(accounts)}:web-1"]


def _answering_agent(monkeypatch, orch, accounts):
    from backend.models.agent.client import SelfAgentClient
    monkeypatch.setattr(SelfAgentClient, "ask", lambda self, question, subject="", *, session: {"stage": "turn"})
    return signed_in_client(orch, accounts)


def test_an_agent_question_over_the_daily_limit_is_a_429_saying_when_more_come(orch, monkeypatch, accounts):
    from datetime import datetime
    from backend.api import quota
    monkeypatch.setitem(quota.USER_LIMITS, "lm", 1)
    client = _answering_agent(monkeypatch, orch, accounts)
    asks = [client.post("/ui/agent/ask", json={"message": "Why?"}) for _ in range(2)]
    stopped = asks[1].json()
    assert ([a.status_code for a in asks], stopped["limit"], datetime.fromisoformat(stopped["resets_at"]).time()) == \
        ([200, 429], 1, time(0))


def test_the_usage_route_gives_the_owners_count_from_the_store(orch, monkeypatch, accounts):
    from datetime import datetime, timezone
    from backend.api import quota
    client = _answering_agent(monkeypatch, orch, accounts)
    client.post("/ui/agent/ask", json={"message": "Why?"})
    usage = client.get("/ui/usage").json()
    stored = orch.store.usage(owner(accounts), quota.day_ist(datetime.now(timezone.utc)))
    assert (usage["used"], usage["limit"]) == (stored["lm"], quota.USER_LIMITS["lm"])


def test_a_question_while_the_agent_is_busy_is_a_503_that_costs_nothing(orch, monkeypatch, accounts):
    """Questions queued for the one agent for up to a minute each, and enough of them used up the server's threads."""
    import threading
    import websockets.sync.client as ws
    import backend.models.agent.client as sc
    asking, release = threading.Event(), threading.Event()

    def answering(*a, **k):
        asking.set()
        release.wait(5)
        raise OSError("the test closed the agent")
    monkeypatch.setattr(sc, "BUSY_WAIT_S", 0.1)
    monkeypatch.setattr(sc.SelfAgentClient, "_login", lambda self: "token")
    monkeypatch.setattr(ws, "connect", answering)
    client = signed_in_client(orch, accounts)
    first = threading.Thread(target=client.post, args=("/ui/agent/ask",), kwargs={"json": {"message": "Why?"}})
    first.start()
    asking.wait(5)
    busy = client.post("/ui/agent/ask", json={"message": "Why?"})
    used = orch.usage(owner(accounts))["used"]
    release.set()
    first.join()
    assert (busy.status_code, "answering someone else" in busy.json()["detail"], used) == (503, True, 1)


@pytest.mark.parametrize("reply", [None, 503], ids=["port closed", "error reply"])
def test_the_agent_status_does_not_give_out_the_agents_address(orch, monkeypatch, reply, accounts):
    import httpx
    port = _closed_port()
    monkeypatch.setenv("SELFAGENT_URL", f"http://127.0.0.1:{port}")
    if reply:
        monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(reply, request=httpx.Request("GET", url)))
    client = signed_in_client(orch, accounts)
    assert str(port) not in client.get("/ui/agent/status").text


def test_a_question_to_an_agent_that_is_down_does_not_give_out_its_address(orch, monkeypatch, tmp_path, accounts):
    import backend.models.agent.client as sc
    port = _closed_port()
    monkeypatch.setattr(sc, "CREDENTIALS", tmp_path / "agent.json")
    monkeypatch.setenv("SELFAGENT_URL", f"http://127.0.0.1:{port}")
    client = signed_in_client(orch, accounts)
    assert str(port) not in client.post("/ui/agent/ask", json={"message": "Why?"}).text


def test_phone_install_manifest_and_service_worker_are_served(app_client):
    m = app_client.get("/manifest.webmanifest").json()
    assert m["start_url"] == "/#/home" and m["display"] == "standalone"
    sw = app_client.get("/sw.js")
    assert sw.status_code == 200 and "javascript" in sw.headers["content-type"]


def test_ticker_lists_indices_then_stocks_with_a_day_change(app_client):
    t = app_client.get("/ui/ticker").json()
    assert t[0] == {**t[0], "symbol": "NIFTY 50", "index": True}
    assert {x["symbol"] for x in t if not x["index"]} == {"TEST", "PEER"}
    assert all(isinstance(x["change_pct"], float) for x in t)


def _record(tmp_path):
    import json as _json
    rec = {"bonferroni_z": 3.25,
           "patterns": [{"z": 1.0}, {"z": -3.5}, {"z": None}],
           "forecast": {"horizons": [{"horizon": 5, "accuracy": 0.52, "always_up_accuracy": 0.53}],
                        "coverage": [{"symbol": "A", "horizon": FORECAST_DAYS, "coverage": 0.70}, {"symbol": "B", "horizon": FORECAST_DAYS, "coverage": 0.90},
                                     {"symbol": "C", "horizon": FORECAST_DAYS, "coverage": 0.80}, {"symbol": "A", "horizon": 5, "coverage": 0.10}]},
           "momentum": {"table": {"momentum_semiannual": {"max_dd_pct": -35.3}, "momentum_trend_vol_overlay": {"max_dd_pct": "-21.4"}},
                        "live_check": {"window": "2022-08-26 to 2026-09-25",
                                       "bars": [{"cagr": 12.2}, {"cagr": 10.3}],
                                       "yearly": [{"year": 2023, "etf": 39.9, "nifty500": 25.2}, {"year": 2025, "etf": -5.5, "nifty500": 4.6}]}}}
    path = tmp_path / "performance.json"
    path.write_text(_json.dumps(rec))
    return path


def test_evidence_note_reads_every_figure_off_the_track_record(tmp_path, monkeypatch):
    import backend.advisory.calculators.performance as perf
    monkeypatch.setattr(perf, "PATH", _record(tmp_path))
    note = perf.evidence_note(perf.track_record())
    assert "+1.9 points a year (2022-08-26 to 2026-09-25)" in note
    assert "+15 points in 2023, -10 in 2025" in note
    assert "fell up to 35%" in note
    assert "1 of 3 candle and chart patterns" in note and "at 0 of 1 horizons" in note
    assert "not built yet" in perf.evidence_note(None)
    assert perf.momentum_evidence(perf.track_record())[1] == (
        "Backtest on today's index list (survivorship-inflated): worst fall -35%, -21% with the trend + volatility overlay.")


def test_the_live_edge_says_its_benchmark_leaves_out_dividends(app_client, tmp_path, monkeypatch):
    """The ETF's NAV keeps its dividends and ^CRSLDX is a price index, so the gap overstates a like-for-like edge."""
    import backend.advisory.calculators.performance as perf
    monkeypatch.setattr(perf, "PATH", _record(tmp_path))
    rec = perf.track_record()
    for claim in (perf.evidence_note(rec), perf.momentum_evidence(rec)[0]):
        assert "beat the Nifty 500 price index by +1.9" in claim and perf.NO_DIVIDENDS_NOTE in claim
    live = app_client.get("/ui/performance").json()["momentum"]["live_check"]
    assert live["benchmark_note"] == perf.NO_DIVIDENDS_NOTE
    # the home page and the performance page each quote the gap, so each shows the caveat
    assert app_client.get("/static/app.js").text.count("${live.benchmark_note}") == 2


def test_technical_range_claim_and_stop_come_from_data(app_client, tmp_path, monkeypatch):
    import backend.advisory.calculators.performance as perf
    from backend.advisory.signals.advisor import STOP_PCT
    monkeypatch.setattr(perf, "PATH", _record(tmp_path))
    t = app_client.get("/ui/stock/TEST/technical?bars=60").json()
    assert "held 80% of outcomes in the 80% band (median of 3 stocks)" in t["cone_note"]
    assert t["stop"] == pytest.approx(t["close"][-1] * (1 - STOP_PCT), abs=0.01) and t["stop_pct"] == STOP_PCT * 100
    monkeypatch.setattr(perf, "PATH", tmp_path / "missing.json")
    t = app_client.get("/ui/stock/TEST/technical?bars=60").json()
    assert "not built yet" in t["cone_note"]


def test_model_verdicts_call_small_accuracy_gaps_noise():
    from backend.advisory.calculators.performance import model_verdicts
    v = model_verdicts({"chronos": {"horizons": [{"horizon": FORECAST_DAYS, "n": 660, "pinball_skill": 0.0,
                                                  "chronos_direction_acc": 0.574, "base_rate_acc": 0.565,
                                                  "direction_edge_lo": -0.01, "direction_edge_hi": 0.03}]},
                        "kronos": {"n": 528, "mae_pct": 5.2, "no_change_mae_pct": 2.2, "direction_acc": 0.492, "base_rate_acc": 0.536}})
    assert v["chronos"] == "its range was about as good as plain volatility, and its direction was right about as often as the base rate"
    assert v["kronos"].startswith("its price error was larger")
    assert v["kronos"].endswith("less often than the base rate (not grouped by date)")    # an old record says how it was judged


class _FakeSpace:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def predict(self, *args, api_name):
        self.calls.append((api_name, args))
        if self.error:
            raise self.error
        return self.reply


def test_ai_forecast_uses_the_model_space_when_one_is_set(app_client, monkeypatch):
    import json as _json
    import backend.models.external.remote_models as rm
    from backend.models.external.tsfm import CONTEXT
    fan = {f"q{q}": [100.0 + q] * FORECAST_DAYS for q in (10, 25, 50, 75, 90)}
    fake = _FakeSpace(_json.dumps(fan))
    monkeypatch.setenv("STOCKINTEL_MODEL_SPACE", "you/stockintel-models")
    monkeypatch.setattr(rm, "_client", lambda: fake)
    r = app_client.get("/ui/stock/TEST/ai?model=chronos").json()
    assert r["q50"] == fan["q50"] and len(r["dates"]) == FORECAST_DAYS
    api_name, (closes, horizon) = fake.calls[0]
    assert api_name == "/chronos_fan" and horizon == FORECAST_DAYS and 0 < len(_json.loads(closes)) <= CONTEXT


def test_a_host_that_runs_chronos_itself_does_not_send_it_to_the_space(app_client, monkeypatch):
    """An Oracle VM names only the writer: its CPU runs the forecasts, and the GPU quota is kept for writing."""
    import backend.models.external.remote_models as rm
    import backend.models.external.tsfm as ts
    monkeypatch.setenv("STOCKINTEL_MODEL_SPACE", "you/stockintel-models")
    monkeypatch.setenv("STOCKINTEL_SPACE_MODELS", "writer")
    monkeypatch.setattr(rm, "_client", lambda: pytest.fail("chronos was sent to the Space"))
    monkeypatch.setattr(ts, "fan", lambda close, horizon: {f"q{q}": [100.0] * horizon for q in (10, 25, 50, 75, 90)})
    r = app_client.get("/ui/stock/TEST/ai?model=chronos")
    assert r.status_code == 200 and len(r.json()["q50"]) == FORECAST_DAYS


def test_a_misspelt_space_model_fails_loudly_instead_of_running_everything_locally(monkeypatch):
    import backend.models.external.remote_models as rm
    monkeypatch.setenv("STOCKINTEL_MODEL_SPACE", "you/stockintel-models")
    monkeypatch.setenv("STOCKINTEL_SPACE_MODELS", "writer,chronoss")
    with pytest.raises(ValueError, match="chronoss"):
        rm.on_space("writer")


def test_a_failing_model_space_is_a_clean_404_naming_the_space(app_client, monkeypatch):
    import backend.models.external.remote_models as rm
    monkeypatch.setenv("STOCKINTEL_MODEL_SPACE", "you/stockintel-models")
    monkeypatch.setattr(rm, "_client", lambda: _FakeSpace(error=RuntimeError("queue full")))
    r = app_client.get("/ui/stock/PEER/ai?model=kronos")
    assert r.status_code == 404 and "you/stockintel-models" in r.json()["detail"] and "queue full" in r.json()["detail"]


@pytest.mark.parametrize("error, used", [(None, 1), (RuntimeError("queue full"), 0)], ids=["answered", "failed"])
def test_a_model_space_forecast_is_charged_once_and_only_when_it_answers(app_client, orch, monkeypatch, error, used,
                                                                       accounts):
    import json as _json
    import backend.models.external.remote_models as rm
    fan = {f"q{q}": [100.0 + q] * FORECAST_DAYS for q in (10, 25, 50, 75, 90)}
    monkeypatch.setenv("STOCKINTEL_MODEL_SPACE", "you/stockintel-models")
    monkeypatch.setattr(rm, "_client", lambda: _FakeSpace(_json.dumps(fan), error))
    for _ in range(2):          # the second is a cache hit when the first answered
        app_client.get("/ui/stock/TEST/ai?model=chronos")
    assert orch.usage(owner(accounts))["used"] == used


def test_a_forecast_asked_while_the_same_one_runs_on_the_space_runs_and_pays_once(app_client, orch, monkeypatch,
                                                                                   accounts):
    """The stock page's ALFA layer and its analyst note ask for one fan seconds apart, and the Space takes ~35 s."""
    import json as _json
    import threading
    import time as _t
    import backend.models.external.remote_models as rm
    asking, release = threading.Event(), threading.Event()

    class SlowSpace(_FakeSpace):
        def predict(self, *args, api_name):
            asking.set()
            release.wait(5)
            return super().predict(*args, api_name=api_name)
    space = SlowSpace(_json.dumps({f"q{q}": [100.0 + q] * FORECAST_DAYS for q in (10, 25, 50, 75, 90)}))
    monkeypatch.setenv("STOCKINTEL_MODEL_SPACE", "you/stockintel-models")
    monkeypatch.setattr(rm, "_client", lambda: space)
    ask = lambda: app_client.get("/ui/stock/TEST/ai?model=chronos")
    first, second = threading.Thread(target=ask), threading.Thread(target=ask)
    first.start()
    asking.wait(5)
    second.start()
    _t.sleep(0.3)               # time for the second request to reach the lock the first one holds
    release.set()
    first.join(), second.join()
    assert (len(space.calls), orch.usage(owner(accounts))["used"]) == (1, 1)


def test_state_snapshot_copies_the_database_and_caches(tmp_path):
    import sqlite3
    from backend.database import hfstate
    db, cache, out = tmp_path / "s.db", tmp_path / "cache", tmp_path / "out"
    cache.mkdir()
    (cache / "performance.json").write_text("{}")
    (cache / "panel_x.pkl").write_bytes(b"p")
    (cache / "ignored.tmp").write_text("x")
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("create table t (v)")
    con.execute("insert into t values (42)")
    con.commit()                        # still in the WAL file, not the main file
    hfstate.snapshot(db, out, cache)
    assert sqlite3.connect(out / "stockintel.db").execute("select v from t").fetchall() == [(42,)]
    assert sorted(p.name for p in (out / "cache").iterdir()) == ["panel_x.pkl", "performance.json"]


def test_state_pull_keeps_local_files_unless_told_to_overwrite(tmp_path, monkeypatch):
    import huggingface_hub
    from backend.database import hfstate
    saved = tmp_path / "saved"
    (saved / "cache").mkdir(parents=True)
    (saved / "stockintel.db").write_text("remote db")
    (saved / "cache" / "performance.json").write_text("remote perf")
    monkeypatch.setenv("STOCKINTEL_STATE_REPO", "you/stockintel-state")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(saved))
    db, cache = tmp_path / "local.db", tmp_path / "cache"
    db.write_text("local db")
    assert hfstate.pull(db, cache) == ["performance.json"]
    assert db.read_text() == "local db" and (cache / "performance.json").read_text() == "remote perf"
    assert hfstate.pull(db, cache, overwrite=True) == ["local.db", "performance.json"] and db.read_text() == "remote db"


def test_state_autosave_pushes_only_when_something_changed(tmp_path, monkeypatch):
    from backend.database import hfstate
    pushes = []
    monkeypatch.setattr(hfstate, "push", lambda db, cache: pushes.append(db))
    db = tmp_path / "s.db"
    db.write_text("v1")
    saver = hfstate.Autosave(db, minutes=60, cache_dir=tmp_path)
    assert saver.save_if_changed() is False
    db.write_text("v2 longer")
    assert saver.save_if_changed() is True and saver.save_if_changed() is False
    assert pushes == [db]


def test_state_needs_a_token_and_says_where_to_get_one(monkeypatch):
    from backend.database import hfstate
    monkeypatch.setenv("STOCKINTEL_STATE_REPO", "you/stockintel-state")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="huggingface.co/settings/tokens"):
        hfstate._token()


def test_alfa_forecast_quotes_its_own_saved_record(app_client, monkeypatch):
    import backend.models.serving.alfa_fan as am
    drawer = {"name": "generative", "test_from": "2023-01-02",
              "measured": {"vs_garch": -0.0323, "vs_garch_error": 0.0032, "scored": 80576}}
    monkeypatch.setattr(am, "available", lambda: True)
    monkeypatch.setattr(am, "fan", lambda close: {**{f"q{q}": [100.0] * am.STEPS for q in (10, 25, 50, 75, 90)},
                                                   "drawer": drawer, "direction_note": "No direction."})
    r = app_client.get("/ui/stock/TEST/ai?model=alfa").json()
    assert len(r["dates"]) == am.STEPS and r["q50"] == [100.0] * am.STEPS
    assert "beat GARCH(1,1)-t by 0.032 ± 0.003 nats per return over 80,576 returns" in r["note"] and r["note"].endswith("No direction.")


@pytest.mark.parametrize("days,said", [(None, "80% range held 77.0% of outcomes (95% range 72.2%–81.8%) against 78.5% for "
                                               "the volatility band, and was narrower"),
                                       (5, "No backtest has scored the 10-day fan drawn here")])
def test_alfa_forecast_quotes_the_backtest_of_the_fan_it_draws_and_no_other(app_client, monkeypatch, tmp_path, days, said):
    import json as _json
    import backend.models.serving.alfa_fan as am
    tested = tmp_path / "alfa_fan_backtest.json"
    tested.write_text(_json.dumps({"days": days or am.STEPS, "rows": 396, "dates": 33, "from": "2024-01-01",
                                   "alfa": {"cover80": 0.7702, "cover80_lo_hi": [0.7222, 0.8182], "mean_log_width": 0.1045},
                                   "band": {"cover80": 0.7854, "cover80_lo_hi": [0.7398, 0.8308], "mean_log_width": 0.112}}))
    drawer = {"name": "generative", "test_from": "2023-01-02",
              "measured": {"vs_garch": -0.0323, "vs_garch_error": 0.0032, "scored": 80576}}
    monkeypatch.setattr(am, "BACKTEST", tested)
    monkeypatch.setattr(am, "available", lambda: True)
    monkeypatch.setattr(am, "fan", lambda close: {**{f"q{q}": [100.0] * am.STEPS for q in (10, 25, 50, 75, 90)},
                                                   "drawer": drawer, "direction_note": "No direction."})
    assert said in app_client.get("/ui/stock/TEST/ai?model=alfa").json()["note"]


def test_alfa_runs_on_the_model_space_when_one_is_set_even_with_local_weights(app_client, monkeypatch):
    import json as _json
    import backend.models.external.remote_models as rm
    import backend.models.serving.alfa_fan as am
    fan = {**{f"q{q}": [100.0] * am.STEPS for q in (10, 25, 50, 75, 90)},
           "drawer": {"name": "generative", "test_from": "2023-01-02", "measured": {"vs_garch": -0.03, "vs_garch_error": 0.003, "scored": 9}},
           "direction_note": "No direction."}
    fake = _FakeSpace(_json.dumps(fan))
    monkeypatch.setattr(am, "available", lambda: True)
    monkeypatch.setattr(am, "fan", lambda *a: pytest.fail("ran on this server's CPU"))
    monkeypatch.setenv("STOCKINTEL_MODEL_SPACE", "you/stockintel-models")
    monkeypatch.setattr(rm, "_client", lambda: fake)
    app_client.get("/ui/stock/TEST/ai?model=alfa")
    assert fake.calls[0][0] == "/return_paths"


def test_published_weights_land_where_the_server_reads_them_and_a_local_copy_is_kept(monkeypatch, tmp_path):
    """The VM recalibrates the GRU every day; pulling the published copy over it on restart would undo that."""
    from pathlib import Path
    import huggingface_hub
    from backend.models.serving import gru_line, weights
    got = []
    monkeypatch.setenv("ALFA_MODELS_REPO", "you/alfa-weights")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, local_dir: got.append((repo, Path(local_dir) / name)))
    missing, held = tmp_path / "path_gru.npz", tmp_path / "returns.npz"
    held.write_text("recalibrated here")
    weights.pull([missing, held])
    assert got == [("you/alfa-weights", missing)] and gru_line.WEIGHTS in weights.PUBLISHED


def test_fetching_the_agent_keeps_the_feedback_this_deployment_has_learned(monkeypatch, tmp_path):
    import huggingface_hub
    from backend.models.serving import weights
    held = tmp_path / "repo"
    (held / "artifacts").mkdir(parents=True)
    (held / "data" / "prices").mkdir(parents=True)
    (held / "artifacts" / "served.sqlite").write_text("as published")
    (held / "artifacts" / "generator.npz").write_text("weights")
    (held / "data" / "prices" / "INFY.NS.csv").write_text("bars")
    artifacts, data = tmp_path / "artifacts", tmp_path / "data"
    artifacts.mkdir()
    (artifacts / "served.sqlite").write_text("learned since")
    monkeypatch.setattr(weights, "ARTIFACTS", artifacts)
    monkeypatch.setattr(weights, "DATA", data)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, local_dir: str(held))
    assert weights.pull_agent("you/alfa-agent") == ["generator.npz", "prices/INFY.NS.csv"]
    assert (artifacts / "served.sqlite").read_text() == "learned since" and (data / "prices" / "INFY.NS.csv").exists()


def test_the_app_reads_company_news_from_the_newswire_not_google(monkeypatch, tmp_path):
    """Google News's robots.txt does not allow the search the app used to make; the newswire archive is read
    with no network call, by the company's name, not its first word ("Tata" would be every Tata company)."""
    import json as _json
    from backend.advisory.service import company_terms, newswire_items
    from backend.database.live import news
    archive = tmp_path / "wire.jsonl"
    rows = [{"link": "https://n.test/1", "day": "2026-10-06", "source": "Mint", "title": "Tata Steel gets aid", "summary": ""},
            {"link": "https://n.test/2", "day": "2026-10-06", "source": "Mint", "title": "Tata Motors launches a car", "summary": ""}]
    archive.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(news, "ARCHIVE", archive)
    monkeypatch.setattr(news.read, "__defaults__", (archive,))
    items = newswire_items("TATASTEEL.NS", company_terms("TATASTEEL.NS", "Tata Steel Limited"))
    assert [i.headline for i in items] == ["Tata Steel gets aid"] and items[0].url == "https://n.test/1"


def test_the_news_view_lists_linked_headlines_not_evidence_lines(orch):
    from backend.advisory import report as R
    a = orch.analysis(orch.session(DEFAULT_PORTFOLIO, "news"), "TEST")
    shown = R.render_domain(a, "news_sentiment")
    assert shown.startswith("### ") and "· News & sentiment" in shown and "source=" not in shown


def test_alfa_says_how_to_reach_it_when_it_cannot(app_client, monkeypatch, tmp_path):
    import backend.models.serving.alfa_fan as am
    monkeypatch.setattr(am, "WEIGHTS", tmp_path / "missing.npz")
    r = app_client.get("/ui/stock/TEST/ai?model=alfa")
    assert r.status_code == 404 and "STOCKINTEL_MODEL_SPACE" in r.json()["detail"]


def test_fine_tuned_weights_bring_their_own_record_into_the_note(app_client, monkeypatch):
    import json as _json
    import backend.models.external.remote_models as rm
    record = {"model": "chronos-2-nse",
              "test": {"horizons": [{"horizon": FORECAST_DAYS, "n": 5472, "pinball_skill": -0.0445, "chronos_direction_acc": 0.4889,
                                     "base_rate_acc": 0.5168, "chronos_cover80": 0.8087, "ewma_cover80": 0.8282}]},
              "precision_calls": [{"horizon": FORECAST_DAYS, "call_rate": 0.1142, "precision": 0.5168, "precision_lo": 0.4254,
                                   "precision_hi": 0.6077, "base_rate": 0.5203}]}
    fan = {**{f"q{q}": [100.0] * FORECAST_DAYS for q in (10, 25, 50, 75, 90)}, "served_model": "you/chronos-2-nse", "served_record": record}
    monkeypatch.setenv("STOCKINTEL_MODEL_SPACE", "you/stockintel-models")
    monkeypatch.setattr(rm, "_client", lambda: _FakeSpace(_json.dumps(fan)))
    note = app_client.get("/ui/stock/TEST/ai?model=chronos").json()["note"]
    assert note.startswith("Chronos-2 fine-tuned on NSE prices by us.") and f"Tested on 5472 past {FORECAST_DAYS}-day forecasts" in note
    assert "its range was worse than plain volatility" in note
    assert "most confident 11% of 'up' calls were right 52% (95% range 43%–61%) vs 52% for always 'up'" in note


def test_the_gru_shows_the_simulated_paths_ending_at_the_10th_50th_and_90th_percentile():
    """Three random draws were three unlabelled lines meaning the same thing; each shown path is picked by its own
    measure, so the low one ends below 9 in 10 of the simulations and the high one above 9 in 10."""
    import numpy as np
    from backend.models.serving.gru_line import labelled_paths
    ends = np.linspace(80.0, 120.0, 101)
    simulated = np.stack([np.linspace(100.0, end, FORECAST_DAYS) for end in np.random.default_rng(0).permutation(ends)])
    shown = labelled_paths(simulated)
    assert [p["name"] for p in shown] == ["low", "middle", "high"] and [p["percentile"] for p in shown] == [10, 50, 90]
    assert [p["close"][-1] for p in shown] == [84.0, 100.0, 116.0]


def _gru_record(nll_gain_lo=0.019, edge=(-0.0079, 0.0199)):
    """A GRU record shaped like the one train_path_gru.py and calibrate_path_gru.py save."""
    return {"test_from": "2024-01-01", "test": {"n": 5954, "nll_gain_lo": nll_gain_lo, "nll_gain_hi": 0.058, "days": [
        {"day": FORECAST_DAYS, "mae_pct": 6.479, "no_change_mae_pct": 6.54, "direction_acc": 0.5391, "base_rate_acc": 0.5334,
         "direction_edge_lo": edge[0], "direction_edge_hi": edge[1]}]},
        "calibration": {"fitted_through": "2026-10-09", "test": [{"day": FORECAST_DAYS, "range": "rolling 12 months",
                        "cover80": 0.7914, "cover80_lo_hi": [0.76, 0.821], "dates": 20}]}}


def _fake_gru_line(monkeypatch, record):
    """Swap the GRU for one that returns `record`; the list returned fills with the last close each call was given."""
    import backend.models.serving.gru_line as gl
    starts = []
    monkeypatch.setattr(gl, "line", lambda c, h, l, m: starts.append(float(c[-1])) or
                        {"close": [float(c[-1])] * FORECAST_DAYS, "paths": [], "record": record})
    return starts


def test_gru_line_note_quotes_its_test_record(app_client, monkeypatch):
    import backend.models.serving.gru_line as gl
    record = _gru_record()
    paths = [{"name": n, "percentile": q, "meaning": "", "close": [100.0 + q] * FORECAST_DAYS}
             for n, q in (("low", 10), ("middle", 50), ("high", 90))]
    monkeypatch.setattr(gl, "line", lambda c, h, l, m: {"close": [101.0] * FORECAST_DAYS, "paths": paths, "record": record})
    r = app_client.get("/ui/stock/TEST/ai?model=gru").json()
    assert len(r["dates"]) == FORECAST_DAYS and r["close"] == [101.0] * FORECAST_DAYS
    assert r["paths"] == paths and "three of 1,000 simulated paths, picked by where they end" in r["note"]
    assert "beat the volatility band (likelihood better by 0.019 to 0.058 per day" in r["note"]
    assert ("recalibrated on the year to 2026-10-09, held 79.1% of outcomes over 20 test dates "
            "(95% range 76.0%–82.1% when whole dates are resampled)") in r["note"] and "Kupiec" not in r["note"]
    assert "error was 6.479% against 6.54% for assuming no change" in r["note"] and r["note"].endswith("within noise.")


@pytest.mark.parametrize("nll_gain_lo, edge, moves, direction", [
    (-0.01, (-0.07, -0.01), "did not beat the volatility band", "worse than the base rate."),
    (0.019, (0.01, 0.05), "beat the volatility band", "a small edge.")])
def test_gru_note_judges_its_record_instead_of_assuming_it_won(app_client, monkeypatch, nll_gain_lo, edge, moves, direction):
    _fake_gru_line(monkeypatch, _gru_record(nll_gain_lo, edge))
    note = app_client.get("/ui/stock/TEST/ai?model=gru").json()["note"]
    assert f"daily move sizes {moves}" in note and note.endswith(direction)


def _at_exchange_time(orch, monkeypatch, at):
    """Set the service's clock to `at`, exchange time, on the day of TEST's last bar; returns TEST's history."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    svc = orch.service()
    history = svc.history("TEST")
    moment = datetime.combine(history.index[-1].date(), at, ZoneInfo(svc.market.timezone))
    monkeypatch.setattr(svc, "now", lambda: moment.astimezone().replace(tzinfo=None))     # local and naive, like datetime.now
    return history


@pytest.mark.parametrize("at, origin", [(time(11, 0), -2), (time(15, 30), -1)])
def test_forecasts_start_from_the_last_closed_bar(app_client, orch, monkeypatch, at, origin):
    """In market hours the last bar is the moving price, which no model trained on, so the GRU starts from the bar
    before it and its first forecast day is today."""
    starts = _fake_gru_line(monkeypatch, _gru_record())
    history = _at_exchange_time(orch, monkeypatch, at)
    r = app_client.get("/ui/stock/TEST/ai?model=gru").json()
    assert starts == [pytest.approx(float(history["close"].iloc[origin]))] and r["as_of"] == str(history.index[origin].date())
    assert r["dates"][0] == str(pd.bdate_range(history.index[origin], periods=2)[1].date())


@pytest.mark.parametrize("at, runs", [(time(11, 0), 1), (time(15, 45), 2)])
def test_a_forecast_reruns_only_when_its_closed_bar_moves(app_client, orch, monkeypatch, at, runs):
    """Keyed on the moving close, every 15-minute price refresh reran the models under the lock every forecast
    shares. Once the session has closed, a cached mid-day price can still give way to the final close."""
    starts = _fake_gru_line(monkeypatch, _gru_record())
    _at_exchange_time(orch, monkeypatch, at)
    app_client.get("/ui/stock/TEST/ai?model=gru")
    svc = orch.service()
    history, moved = svc.history, svc.history("TEST").copy()
    moved.loc[moved.index[-1], "close"] *= 1.04
    monkeypatch.setattr(svc, "history", lambda sym: moved if sym == "TEST" else history(sym))
    app_client.get("/ui/stock/TEST/ai?model=gru")
    assert len(starts) == runs


def test_the_analyst_note_quotes_the_closed_bar_its_forecasts_start_from(app_client, orch, monkeypatch):
    import backend.advisory.analyst as an
    import backend.api.hub as hubmod
    from tests.app.test_nlu_llm import _analyst_facts
    monkeypatch.setenv("STOCKINTEL_ML", "0")
    monkeypatch.setattr(hubmod.Hub, "ai_forecast", lambda self, sym, model, owner: {})
    f, quoted = _analyst_facts(), []
    monkeypatch.setattr(an, "facts", lambda close, as_of, **rest: quoted.append((close, as_of)) or f)
    history = _at_exchange_time(orch, monkeypatch, time(11, 0))
    app_client.get("/ui/stock/TEST/analyst")
    assert quoted == [(round(float(history["close"].iloc[-2]), 2), str(history.index[-2].date()))]


def test_every_stop_on_the_stock_page_is_the_one_below_the_last_close(app_client, orch, monkeypatch):
    """In market hours the chart and verdict took the stop off the moving price and the analyst note off the last
    close, so one page showed two stop-losses."""
    import backend.advisory.analyst as an
    import backend.api.hub as hubmod
    from backend.advisory.signals.advisor import STOP_PCT
    from tests.app.test_nlu_llm import _analyst_facts
    monkeypatch.setenv("STOCKINTEL_ML", "0")
    monkeypatch.setattr(hubmod.Hub, "ai_forecast", lambda self, sym, model, owner: {})
    f, noted = _analyst_facts(), []
    monkeypatch.setattr(an, "facts", lambda verdict, **rest: noted.append(verdict["stop"]) or f)
    history = _at_exchange_time(orch, monkeypatch, time(11, 0))
    minute = history.iloc[-1:].set_axis(history.index[-1:] + pd.Timedelta(hours=11))
    monkeypatch.setattr(orch.provider, "intraday", lambda sym, period, interval: minute)
    app_client.get("/ui/stock/TEST/analyst")
    shown = [app_client.get("/ui/stock/TEST/technical").json()["stop"],
             app_client.get("/ui/stock/TEST/verdict").json()["plan"]["stop_loss"],
             app_client.get("/ui/stock/TEST/intraday?range=1D").json()["levels"]["stop_10pct"], *noted]
    assert shown == [round(float(history["close"].iloc[-2]) * (1 - STOP_PCT), 2)] * 4


def test_the_gru_refuses_a_newest_bar_it_cannot_read(app_client, orch, monkeypatch):
    """dropna dropped a newest bar with no high, so the line started from the day before while its dates counted
    from the newest bar."""
    _fake_gru_line(monkeypatch, _gru_record())
    svc = orch.service()
    history, gap = svc.history, svc.history("TEST").copy()
    gap.loc[gap.index[-1], "high"] = float("nan")
    monkeypatch.setattr(svc, "history", lambda sym: gap if sym == "TEST" else history(sym))
    r = app_client.get("/ui/stock/TEST/ai?model=gru")
    assert r.status_code == 404 and f"{gap.index[-1].date()} bar has no high" in r.json()["detail"]


def test_the_gru_starts_from_the_stocks_last_close_when_the_nifty_lags_a_bar(app_client, orch, monkeypatch):
    """The Nifty is cached on its own clock and can miss the stock's newest bar; an inner join dropped that bar,
    so the line started from the day before while being drawn from today."""
    starts = _fake_gru_line(monkeypatch, _gru_record())
    nifty = orch.hub.bench()
    monkeypatch.setattr(orch.hub, "bench", lambda: nifty.iloc[:-1])
    app_client.get("/ui/stock/TEST/ai?model=gru")
    assert starts == [pytest.approx(float(orch.service().history("TEST")["close"].iloc[-1]))]


def test_analyst_note_falls_back_to_the_computed_note_when_models_are_off(app_client, monkeypatch):
    import backend.api.hub as hubmod
    from tests.app.test_nlu_llm import _analyst_facts
    monkeypatch.setenv("STOCKINTEL_ML", "0")
    f = _analyst_facts()
    monkeypatch.setattr(hubmod.Hub, "ai_forecast", lambda self, sym, model, owner: {})
    import backend.advisory.analyst as an
    monkeypatch.setattr(an, "facts", lambda *a, **k: f)
    r = app_client.get("/ui/stock/TEST/analyst").json()
    assert r["written_by"] is None and r["text"] == an.computed_note(f) and "switched off" in r["draft_rejected"][0]


@pytest.fixture
def note_writer(app_client, orch, monkeypatch):
    """The fake language model behind the analyst note; "holder" holds TEST and "non-holder" does not."""
    import backend.advisory.analyst as an
    import backend.api.hub as hubmod
    from tests.app.test_nlu_llm import _analyst_facts
    f, writer = _analyst_facts(), _Picker("A draft.")
    monkeypatch.setattr(hubmod.Hub, "ml_enabled", lambda self: True)
    monkeypatch.setattr(hubmod.Hub, "writer", lambda self: writer)
    monkeypatch.setattr(hubmod.Hub, "ai_forecast", lambda self, sym, model, owner: {})
    monkeypatch.setattr(an, "facts", lambda **inputs: f)
    orch.save_portfolio("holder", 0.0, "moderate", "medium", [
        {"symbol": "TEST", "side": "BUY", "quantity": 10, "price": 100, "trade_date": "2025-06-02"}])
    return writer


def _note(orch, portfolio, symbol="TEST"):
    return orch.hub.analyst_note(symbol, orch.session(portfolio, "note"))


def test_the_analyst_note_is_not_shared_between_a_holder_and_a_non_holder(orch, note_writer):
    """The note quotes the verdict, which says HOLD / ADD to someone who holds the stock and BUY to anyone else."""
    for portfolio in ("holder", "non-holder"):
        _note(orch, portfolio)
    assert len(note_writer.prompts) == 2


def test_an_analyst_note_over_the_daily_limit_is_the_computed_note_saying_why(orch, note_writer, monkeypatch):
    from backend.api import quota
    monkeypatch.setitem(quota.USER_LIMITS, "lm", 1)
    _note(orch, "holder", "TEST")
    over = _note(orch, "holder", "PEER")
    assert (len(note_writer.prompts), over["written_by"], over["draft_rejected"][0].startswith("You have used your")) == \
        (1, None, True)


def test_a_cached_analyst_note_is_not_charged_again(orch, note_writer):
    for _ in range(2):
        _note(orch, "holder")
    assert orch.usage("holder")["used"] == 1


def test_the_all_users_cap_stops_a_second_portfolios_note(orch, note_writer, monkeypatch):
    from backend.api import quota
    monkeypatch.setitem(quota.EVERYONE_LIMITS, "lm", 1)
    notes = [_note(orch, portfolio) for portfolio in ("holder", "non-holder")]
    assert (len(note_writer.prompts), notes[1]["draft_rejected"][0].startswith("All users together")) == (1, True)


def test_a_note_the_writer_could_not_write_is_not_charged(orch, note_writer, monkeypatch):
    from backend.models.external.llm import LLMUnavailable

    def asleep(system, messages):
        raise LLMUnavailable("the Space is asleep")
    monkeypatch.setattr(note_writer, "create", asleep)
    note = _note(orch, "holder")
    assert (note["draft_rejected"], orch.usage("holder")["used"]) == (["the Space is asleep"], 0)


def test_a_note_asked_while_the_writer_is_busy_is_the_computed_note_saying_so(orch, note_writer, monkeypatch):
    """Requests queued for the one writer for up to a minute each, and enough of them used up the server's threads."""
    import threading
    import backend.api.hub as hubmod
    writing, release = threading.Event(), threading.Event()
    draft = note_writer.create

    def slow(system, messages):
        writing.set()
        release.wait(5)
        return draft(system, messages)
    monkeypatch.setattr(hubmod, "BUSY_WAIT_S", 0.1)
    monkeypatch.setattr(note_writer, "create", slow)
    first = threading.Thread(target=_note, args=(orch, "holder"))
    first.start()
    writing.wait(5)
    busy = _note(orch, "non-holder")
    release.set()
    first.join()
    assert busy["draft_rejected"][0].startswith("The writer is busy")


@pytest.mark.parametrize("mode", ["narrate", "agent"])
def test_a_claude_reply_over_the_daily_limit_is_the_computed_answer_saying_why(orch, monkeypatch, mode):
    from backend.api import quota
    from .test_nlu_llm import FakeClient, text_resp
    monkeypatch.setitem(quota.USER_LIMITS, "lm", 0)
    orch.llm_mode, orch.llm_client = mode, FakeClient([text_resp("unused")])
    reply = orch.handle("Analyze TEST", DEFAULT_PORTFOLIO, "claude")
    assert (orch.llm_client.calls, "AI answers for today" in reply.text) == ([], True)


def test_a_question_routed_over_the_daily_limit_keeps_the_reply_and_says_why(orch, monkeypatch):
    from backend.api import quota
    writer = _Picker("market_regime")
    monkeypatch.setitem(quota.USER_LIMITS, "lm", 0)
    monkeypatch.setattr(type(orch.hub), "writer", lambda self: writer)
    reply = orch.handle("what is going on out there?", DEFAULT_PORTFOLIO, "route")
    assert (writer.prompts, reply.data["no_answer"], "AI answers for today" in reply.text) == ([], True, True)


def test_the_cli_still_routes_questions_past_the_daily_limit(monkeypatch):
    """eval_user_questions.py asks the CLI's orchestrator dozens of questions; each one refused past the
    cap was scored as a wrong handler."""
    from argparse import Namespace
    from backend import cli
    from backend.api import quota
    orch, writer = cli._orchestrator(Namespace(db=":memory:", demo=True)), _Picker("market_regime")
    monkeypatch.setitem(quota.USER_LIMITS, "lm", 0)
    monkeypatch.setattr(type(orch.hub), "writer", lambda self: writer)
    orch.handle("what is going on out there?", DEFAULT_PORTFOLIO, "route")
    assert len(writer.prompts) == 1


def test_an_unrecognised_question_is_routed_by_the_language_model(orch, monkeypatch):
    writer = _Picker("market_regime")
    monkeypatch.setattr(type(orch.hub), "writer", lambda self: writer)
    reply = orch.handle("what is going on out there?", DEFAULT_PORTFOLIO, "route")
    assert reply.intent == "market_regime" and "Nifty market state" in reply.text
    assert "market_regime:" in writer.prompts[0]


def test_a_question_the_language_model_cannot_place_says_so_instead_of_guessing(orch, monkeypatch):
    monkeypatch.setattr(type(orch.hub), "writer", lambda self: _Picker("NONE"))
    reply = orch.handle("what is the weather in Mumbai?", DEFAULT_PORTFOLIO, "route2")
    assert reply.data["no_answer"] and "don't have a lesson" in reply.text


def test_chat_works_without_a_language_model(orch, monkeypatch):
    from backend.models.external.llm import LLMUnavailable
    def no_writer(self):
        raise LLMUnavailable("no torch and no Space")
    monkeypatch.setattr(type(orch.hub), "writer", no_writer)
    assert orch.handle("what is going on out there?", DEFAULT_PORTFOLIO, "route3").data["no_answer"]


def test_a_vague_question_is_not_answered_with_an_unrelated_lesson(orch, monkeypatch):
    monkeypatch.setattr(type(orch.hub), "writer", lambda self: _Picker("NONE"))
    for question in ("is it a good day", "anything new", "what is going on out there?"):
        assert orch.handle(question, DEFAULT_PORTFOLIO, "vague").data["no_answer"], question


@pytest.fixture
def funds_served(tmp_path, monkeypatch):
    from backend.database.sources import amfi
    from backend.database.sources.provider import DataUnavailable
    on = lambda nav_on_29_oct: growing(6) * (nav_on_29_oct / growing(6).iloc[-1])  # noqa: E731
    pages = {amfi.NAV_URL: NAV_TEXT, amfi.HISTORY_URL.format(code=118989): mfapi_json(118989, on(210.512)),
             amfi.HISTORY_URL.format(code=118990): mfapi_json(118990, on(70.2)),
             amfi.HISTORY_URL.format(code=105758): mfapi_json(105758, on(150.0)),
             amfi.HISTORY_URL.format(code=148304): mfapi_json(148304, growing(6) * 0)}

    def download(url):
        if url not in pages:
            raise DataUnavailable(f"could not download {url}: 404")
        return pages[url]
    monkeypatch.setattr(amfi, "NAV_FILE", tmp_path / "navall.txt")
    monkeypatch.setattr(amfi, "HISTORY_DIR", tmp_path / "mfapi")
    monkeypatch.setattr(amfi, "_download", download)
    return pages


def test_fund_search_lists_each_fund_once_with_its_plans(app_client, funds_served):
    r = app_client.get("/ui/funds/search?q=hdfc mid").json()
    assert [f["fund"] for f in r["funds"]] == ["HDFC Mid Cap Fund", "HDFC NIFTY Midcap 150 Index Fund"]
    assert r["funds"][0]["plans"][0] == {"code": 118989, "plan": "Direct Plan", "option": "Growth Option",
                                         "nav": 210.512, "nav_date": "2026-10-29", "is_ambiguous": False}


@pytest.mark.parametrize("q", ["", "h"])
def test_fund_search_refuses_an_empty_or_one_letter_query_even_while_amfi_is_down(app_client, funds_served, q):
    from backend.database.sources import amfi
    del funds_served[amfi.NAV_URL]
    r = app_client.get(f"/ui/funds/search?q={q}")
    assert r.status_code == 400 and "at least 2 letters" in r.json()["detail"]


def test_fund_page_gives_amfi_facts_and_labels_the_history_unofficial(app_client, funds_served):
    r = app_client.get("/ui/funds/118989").json()
    assert (r["scheme"]["fund"], r["scheme"]["nav"], r["scheme"]["nav_date"]) == ("HDFC Mid Cap Fund", 210.512, "2026-10-29")
    assert r["analytics"]["returns"]["5Y"]["pct"] == pytest.approx(10.0)
    assert "unofficial" in r["source"]["history"] and r["summary"].startswith("Over the last 5 years")
    assert len(r["history"]["dates"]) == len(r["history"]["nav"]) == len(growing(6))


def test_fund_page_without_history_still_gives_the_official_nav_and_says_why(app_client, funds_served):
    r = app_client.get("/ui/funds/105759").json()
    assert r["scheme"]["nav"] == 60.3 and r["analytics"] is None and "mf/105759" in r["history_error"]


def test_a_history_that_disagrees_with_amfis_nav_is_not_charted(app_client, funds_served):
    r = app_client.get("/ui/funds/105758").json()
    assert r["history"] is None and "150.0" in r["history_error"] and "190.1" in r["history_error"]


def test_a_history_that_matches_amfis_nav_says_so(app_client, funds_served):
    r = app_client.get("/ui/funds/118989").json()
    assert r["history_check"] == "matches AMFI's NAV on 2026-10-29"


def test_an_idcw_plans_figures_say_its_payouts_are_left_out(app_client, funds_served):
    r = app_client.get("/ui/funds/118990").json()
    assert r["analytics"]["is_total_return"] is False and "not list this as a Growth option" in r["summary"]


def test_fund_page_for_a_portfolio_carried_at_zero_gives_amfis_facts_and_no_return(app_client, funds_served):
    r = app_client.get("/ui/funds/148304")
    assert r.status_code == 200 and r.json()["scheme"]["nav"] == 0.0 and r.json()["analytics"] is None
    assert "not above zero" in r.json()["history_error"]


def test_an_unknown_fund_code_is_a_404_that_says_how_to_find_one(app_client, funds_served):
    r = app_client.get("/ui/funds/999999")
    assert r.status_code == 404 and "999999" in r.json()["detail"] and "/ui/funds/search" in r.json()["detail"]



def test_fund_page_lists_every_plan_of_its_fund(app_client, funds_served):
    r = app_client.get("/ui/funds/105759").json()
    assert [p["code"] for p in r["plans"]] == [118989, 118990, 105758, 105759]


def test_fund_page_carries_the_fund_lessons(app_client, funds_served):
    r = app_client.get("/ui/funds/118989").json()
    assert "Direct and regular plans" in [lesson["title"] for lesson in r["lessons"]]


def test_a_fund_code_that_is_not_a_number_is_refused_before_any_download(app_client, funds_served, monkeypatch):
    from backend.database.sources import amfi
    asked = []
    monkeypatch.setattr(amfi, "_download", asked.append)
    assert (app_client.get("/ui/funds/abc").status_code, asked) == (422, [])


def test_learn_titles_the_fund_lessons(app_client):
    assert "Mutual funds and ETFs" in [t["topic"] for t in app_client.get("/ui/learn").json()]


def test_a_fund_added_to_the_portfolio_is_valued_at_amfis_nav(app_client, funds_served):
    assert app_client.post("/portfolio/funds/transactions", json=FUND_BUY).status_code == 200
    r = app_client.get("/portfolio/funds").json()
    assert (r["invested"], r["value"], r["holdings"][0]["fund"]) == (2000.0, 2105.12, "HDFC Mid Cap Fund")
    assert "AMFI" in r["source"]


def test_selling_more_fund_units_than_held_gets_400_and_saves_nothing(app_client, funds_served):
    r = app_client.post("/portfolio/funds/transactions", json={**FUND_BUY, "side": "SELL"})
    assert (r.status_code, app_client.get("/portfolio/funds").json()["transactions"]) == (400, [])


def test_a_fund_code_amfi_does_not_list_gets_400(app_client, funds_served):
    r = app_client.post("/portfolio/funds/transactions", json={**FUND_BUY, "scheme_code": 999999})
    assert r.status_code == 400 and "999999" in r.json()["detail"]


def test_deleting_a_fund_buy_that_a_later_sale_needs_gets_400(app_client, funds_served):
    app_client.post("/portfolio/funds/transactions", json=FUND_BUY)
    app_client.post("/portfolio/funds/transactions", json={**FUND_BUY, "side": "SELL", "trade_date": "2025-03-03"})
    buy = app_client.get("/portfolio/funds").json()["transactions"][0]
    assert app_client.delete(f"/portfolio/funds/transactions/{buy['id']}").status_code == 400
    assert len(app_client.get("/portfolio/funds").json()["transactions"]) == 2


def test_a_deleted_fund_transaction_is_gone(app_client, funds_served):
    app_client.post("/portfolio/funds/transactions", json=FUND_BUY)
    tx = app_client.get("/portfolio/funds").json()["transactions"][0]
    assert app_client.delete(f"/portfolio/funds/transactions/{tx['id']}").status_code == 200
    assert app_client.delete(f"/portfolio/funds/transactions/{tx['id']}").status_code == 404


def test_a_portfolio_with_no_funds_is_served_while_amfi_is_down(app_client, funds_served):
    from backend.database.sources import amfi
    del funds_served[amfi.NAV_URL]
    assert app_client.get("/portfolio/funds").json()["holdings"] == []


def hold_funds(orch, accounts, transactions):
    """Written straight to the store: the route would ask AMFI, and refuses a code AMFI does not list now."""
    orch.save_portfolio(owner(accounts), 0.0, "moderate", "medium", [])
    orch.store.save_fund_transactions(owner(accounts), transactions)


def test_held_funds_are_listed_with_the_reason_while_amfi_is_down(app_client, orch, accounts, funds_served):
    from backend.database.sources import amfi
    del funds_served[amfi.NAV_URL]
    hold_funds(orch, accounts, [FUND_BUY])
    r = app_client.get("/portfolio/funds")
    assert (r.status_code, len(r.json()["transactions"]), "could not download" in r.json()["valuation_error"]) == (200, 1, True)


def test_a_held_fund_amfi_no_longer_lists_is_named_in_the_error_and_can_be_deleted(app_client, orch, accounts, funds_served):
    hold_funds(orch, accounts, [{**FUND_BUY, "scheme_code": 999999}])
    r = app_client.get("/portfolio/funds").json()
    assert "999999" in r["valuation_error"]
    assert app_client.delete(f"/portfolio/funds/transactions/{r['transactions'][0]['id']}").status_code == 200


def test_a_fund_transaction_carries_its_schemes_name(app_client, funds_served):
    app_client.post("/portfolio/funds/transactions", json=FUND_BUY)
    tx = app_client.get("/portfolio/funds").json()["transactions"][0]
    assert (tx["scheme"]["fund"], tx["scheme"]["plan"]) == ("HDFC Mid Cap Fund", "Direct Plan")


def test_the_stock_report_is_unchanged_by_funds(app_client, funds_served):
    app_client.post("/portfolio/transactions", json={"symbol": "TEST", "side": "BUY", "quantity": 3, "price": 100,
                                                     "trade_date": "2025-01-02"})
    before = app_client.get("/portfolio/analysis").json()
    app_client.post("/portfolio/funds/transactions", json=FUND_BUY)
    assert app_client.get("/portfolio/analysis").json() == before


def test_one_users_funds_are_not_anothers(asha_and_ravi, funds_served):
    asha, ravi = asha_and_ravi
    asha.post("/portfolio/funds/transactions", json=FUND_BUY)
    assert ravi.get("/portfolio/funds").json()["transactions"] == []
