import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockintel import report as R
from stockintel.alerts import evaluate
from stockintel.api import create_app
from stockintel.orchestrator import Orchestrator
from stockintel.screener import screen
from stockintel.service import AnalysisService

from .conftest import NOW

KEY = "k" * 24


@pytest.fixture
def service(local_provider):
    return AnalysisService(local_provider, use_rss_news=False, now=lambda: NOW.replace(tzinfo=None))


@pytest.fixture
def orch(local_provider, store, monkeypatch):
    o = Orchestrator(local_provider, store, use_rss_news=False)
    o.known.add("TEST")
    o.universe = pd.DataFrame({"symbol": ["TEST", "PEER"], "sector": ["Industrials", "Energy"]})
    return o


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
    assert "12. FINAL EVIDENCE-BASED SUMMARY" in text


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
    assert orch.handle("Analyze TEST", "s").intent == "analyze"
    r = orch.handle("Why is the technical state like that?", "s")
    assert r.intent == "domain" and "technical evidence" in r.text
    r = orch.handle("What about fundamentals?", "s")
    assert "fundamental evidence" in r.text and "Revenue changed" in r.text
    r = orch.handle("Compare it with PEER", "s")
    assert r.intent == "compare" and "TEST" in r.text and "PEER" in r.text
    r = orch.handle("Why?", "s")
    assert r.intent == "explain_decision" and r.text.startswith("Why TEST")


def test_budget_question_uses_the_session_stock(orch):
    orch.handle("Analyze TEST", "b")
    r = orch.handle("I have ₹1 lakh. How does my budget affect the analysis?", "b")
    assert r.intent == "budget_analysis" and "position limit" in r.text


def test_portfolio_questions_are_computed_from_stored_transactions(orch):
    orch.save_portfolio(5000.0, "moderate", "long", [
        {"symbol": "TEST", "side": "BUY", "quantity": 10, "price": 100, "trade_date": "2025-01-02", "sector": "Industrials"},
        {"symbol": "PEER", "side": "BUY", "quantity": 5, "price": 100, "trade_date": "2025-01-02", "sector": "Energy"}])
    r = orch.handle("How is my portfolio performing?", "p")
    assert r.intent == "portfolio_summary" and "Total value" in r.text and "Risk contribution" in r.text
    r = orch.handle("If I add ₹20,000 to PEER how does my portfolio risk change?", "p")
    assert r.intent == "portfolio_what_if" and "all in PEER" in r.text


def test_portfolio_question_without_a_portfolio_says_so(orch):
    r = orch.handle("How is my portfolio performing?", "p")
    assert "No portfolio" in r.text


def test_knowledge_answer_never_contains_live_data(orch):
    r = orch.handle("What is RSI?", "k")
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


def test_api_requires_a_key_at_startup(orch):
    with pytest.raises(RuntimeError, match="STOCKINTEL_API_KEY"):
        create_app(orch, api_key="")


def test_api_rejects_missing_or_wrong_keys(orch):
    client = TestClient(create_app(orch, api_key=KEY))
    assert client.get("/health").status_code == 200
    assert client.get("/stocks/TEST/analysis").status_code == 401
    assert client.get("/stocks/TEST/analysis", headers={"X-API-Key": "wrong" * 5}).status_code == 401


def test_api_analysis_and_domain_endpoints(orch):
    client = TestClient(create_app(orch, api_key=KEY))
    h = {"X-API-Key": KEY}
    r = client.get("/stocks/TEST/analysis", headers=h)
    assert r.status_code == 200 and r.json()["decision"]["decision_support_label"]
    assert client.get("/stocks/TEST/technical", headers=h).json()["result"]["domain"] == "technical"
    assert client.get("/stocks/TEST/fundamentals", headers=h).json()["result"]["domain"] == "fundamental"
    assert client.get("/stocks/TEST/astrology", headers=h).status_code == 404
    chart = client.get("/stocks/TEST/chart?bars=100", headers=h).json()
    assert len(chart["dates"]) == 100 and len(chart["sma50"]) == 100


def test_api_validates_inputs(orch):
    client = TestClient(create_app(orch, api_key=KEY))
    h = {"X-API-Key": KEY}
    assert client.get("/stocks/DROP;TABLE/analysis", headers=h).status_code in (400, 404)
    assert client.post("/chat", json={"message": ""}, headers=h).status_code == 422
    bad = {"cash": 0, "transactions": [{"symbol": "TEST", "side": "SELL", "quantity": 5, "price": 10,
                                        "trade_date": "2025-01-01"}]}
    assert client.post("/portfolio", json=bad, headers=h).status_code == 400   # oversell rejected
    assert client.get("/screener?budget=-5", headers=h).status_code == 422


def test_api_missing_data_is_a_clean_404(orch):
    client = TestClient(create_app(orch, api_key=KEY))
    r = client.get("/stocks/NOSUCH/analysis", headers={"X-API-Key": KEY})
    assert r.status_code == 404 and r.json()["error"] == "data_unavailable"


def test_api_rate_limit(orch, monkeypatch):
    import stockintel.api as api_mod
    monkeypatch.setattr(api_mod, "RATE_PER_MIN", 3)
    client = TestClient(create_app(orch, api_key=KEY))
    codes = [client.get("/search?q=TE", headers={"X-API-Key": KEY}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]


def test_api_chat_round_trip(orch):
    client = TestClient(create_app(orch, api_key=KEY))
    r = client.post("/chat", json={"message": "What is a Sharpe ratio?", "session_id": "x1"},
                    headers={"X-API-Key": KEY})
    assert r.status_code == 200 and r.json()["intent"] == "knowledge"


def test_api_json_is_strict(orch):
    client = TestClient(create_app(orch, api_key=KEY))
    raw = client.get("/stocks/TEST/analysis", headers={"X-API-Key": KEY}).text
    assert "NaN" not in raw and "Infinity" not in raw


def test_rejected_portfolio_post_leaves_the_stored_portfolio_intact(orch):
    client = TestClient(create_app(orch, api_key=KEY))
    h = {"X-API-Key": KEY}
    good = {"cash": 100, "transactions": [{"symbol": "TEST", "side": "BUY", "quantity": 5, "price": 10,
                                           "trade_date": "2025-01-01"}]}
    assert client.post("/portfolio", json=good, headers=h).status_code == 200
    bad = {"cash": 0, "transactions": [{"symbol": "TEST", "side": "SELL", "quantity": 5, "price": 10,
                                        "trade_date": "2025-01-01"}]}
    assert client.post("/portfolio", json=bad, headers=h).status_code == 400
    stored = orch.store.portfolio("default")
    assert stored["cash"] == 100 and len(stored["transactions"]) == 1


def test_held_status_does_not_depend_on_how_the_ticker_was_typed(orch):
    orch.save_portfolio(0.0, "moderate", "medium", [
        {"symbol": "TEST.NS", "side": "BUY", "quantity": 5, "price": 10, "trade_date": "2025-01-01"}])
    assert orch.analysis(orch.session("h"), "TEST").held
    assert orch.store.portfolio("default")["transactions"][0]["symbol"] == "TEST"


def test_api_market_endpoint(orch, local_provider):
    local_provider.frames["^NSEI"] = local_provider.frames["^NSEI"]
    client = TestClient(create_app(orch, api_key=KEY))
    r = client.get("/market", headers={"X-API-Key": KEY})
    assert r.status_code == 200 and r.json()["trend"] in ("up", "down")


@pytest.fixture
def app_client(orch, local_provider):
    import time as _t
    from stockintel.data.panel import panel_from_frames
    app = create_app(orch, api_key=KEY)
    hub = app.state.hub
    frames = {s.split(".")[0]: local_provider.frames[s] for s in ("TEST.NS", "PEER.NS")}
    hub._panel, hub._panel_at = panel_from_frames(frames), _t.time()
    hub._names = pd.DataFrame({"symbol": ["TEST", "PEER"], "name": ["Test Industries Ltd.", "Peer Power Ltd."],
                               "sector": ["Industrials", "Energy"], "nifty200": [True, True]})
    return TestClient(app)


def test_web_app_shell_and_assets_are_served(app_client):
    assert "StockIntel" in app_client.get("/").text
    assert app_client.get("/static/app.js").status_code == 200
    assert app_client.get("/static/app.css").status_code == 200


def test_ui_search_matches_symbol_and_company_name(app_client):
    h = {"X-API-Key": KEY}
    assert app_client.get("/ui/search?q=te", headers=h).json()[0]["symbol"] == "TEST"
    assert app_client.get("/ui/search?q=power", headers=h).json()[0]["symbol"] == "PEER"


def test_ui_stock_overview_and_chart(app_client):
    h = {"X-API-Key": KEY}
    o = app_client.get("/ui/stock/TEST", headers=h).json()
    assert o["name"] == "Test Industries Limited" and o["low_52w"] <= o["price"] <= o["high_52w"]
    c = app_client.get("/ui/stock/TEST/chart?range=1M", headers=h).json()
    assert len(c["close"]) == 22
    assert app_client.get("/ui/stock/TEST/chart?range=7Y", headers=h).status_code == 400


def test_ui_verdict_explains_both_sides(app_client):
    v = app_client.get("/ui/stock/TEST/verdict", headers={"X-API-Key": KEY}).json()
    assert v["verdict"] in ("BUY", "BUY — HALF SIZE", "WAIT", "DON'T BUY", "HOLD / ADD", "HOLD", "SELL AT REBALANCE")
    assert v["momentum_rank"] in (1, 2) and v["universe_size"] == 2
    assert v["why_buy"] or v["why_not"]
    assert "1.5–2.5 points" in v["evidence_note"] and "2025" in v["evidence_note"]


def test_portfolio_single_transaction_add_and_delete(app_client, orch):
    h = {"X-API-Key": KEY}
    tx = {"symbol": "TEST", "side": "BUY", "quantity": 3, "price": 100, "trade_date": "2025-01-02"}
    assert app_client.post("/portfolio/transactions", json=tx, headers=h).status_code == 200
    assert app_client.post("/portfolio/transactions", json={**tx, "side": "SELL", "quantity": 9}, headers=h).status_code == 400
    assert len(orch.store.portfolio("default")["transactions"]) == 1
    assert app_client.delete("/portfolio/transactions/0", headers=h).status_code == 200
    assert orch.store.portfolio("default")["transactions"] == []


def test_ui_technical_chart_carries_geometry_markers_cone_and_records(app_client):
    t = app_client.get("/ui/stock/TEST/technical?bars=120", headers={"X-API-Key": KEY}).json()
    assert len(t["dates"]) == 120 and len(t["open"]) == len(t["close"]) == 120
    assert len(t["cone"]) == 20 and all(c["lo80"] <= c["lo50"] <= c["mid"] <= c["hi50"] <= c["hi80"] for c in t["cone"])
    for mk in t["markers"]:
        assert mk["date"] in t["dates"] and "summary" in mk["record"]
    for p in t["patterns"]:
        assert "geometry" in p and "summary" in p["record"]
    assert "no model we tested predicted direction" in t["cone_note"]


def test_pattern_geometry_points_are_the_swing_points_used():
    import numpy as np
    from stockintel.analysis import patterns
    from .conftest import bars
    closes = np.concatenate([np.linspace(95, 100, 60), np.linspace(100, 130, 30), np.linspace(130, 115, 12)[1:],
                             np.linspace(115, 129.5, 12)[1:], np.linspace(129.5, 110, 14)[1:]])
    top = next(p for p in patterns.detect(bars(closes, spread=0.004)) if p["pattern"] == "double_top")
    labels = [q[2] for q in top["geometry"]["points"]]
    assert labels == ["Start", "Top 1", "Neckline", "Top 2"]
    neck = top["geometry"]["lines"][0]
    assert neck["label"] == "Neckline" and neck["from"][1] == pytest.approx(top["breakout_level"], abs=0.01)


def test_portfolio_history_rebuilds_value_from_transactions(app_client, orch):
    orch.save_portfolio(0.0, "moderate", "medium", [
        {"symbol": "TEST", "side": "BUY", "quantity": 10, "price": 100, "trade_date": "2025-06-02"}])
    h = app_client.get("/ui/portfolio/history", headers={"X-API-Key": KEY}).json()
    assert h["dates"][0] >= "2025-06-02" and h["invested"][-1] == pytest.approx(1000)
    assert all(v > 0 for v in h["value"])


def test_performance_endpoint_reports_missing_build_cleanly(app_client, monkeypatch, tmp_path):
    import stockintel.performance as perf
    monkeypatch.setattr(perf, "PATH", tmp_path / "none.json")
    r = app_client.get("/ui/performance", headers={"X-API-Key": KEY})
    assert r.status_code == 404 and "build-performance" in r.json()["detail"]


def test_builder_spends_within_capital_and_apply_records_the_buys(app_client, orch):
    from datetime import date as _date
    h = {"X-API-Key": KEY}
    b = app_client.get("/ui/builder?capital=100000&n=3&risk=full", headers=h).json()
    assert b["holdings"] and b["cash_left"] >= 0
    assert sum(x["value"] for x in b["holdings"]) + b["cash_left"] == pytest.approx(100000, abs=0.01)
    assert len(b["backcast"]["dates"]) == len(b["backcast"]["basket"]) == len(b["backcast"]["nifty"])
    body = {"holdings": [{"symbol": x["symbol"], "shares": x["shares"], "price": x["price"]} for x in b["holdings"]]}
    assert app_client.post("/ui/builder/apply", json=body, headers=h).status_code == 200
    txs = orch.store.portfolio("default")["transactions"]
    assert [t["symbol"] for t in txs] == [x["symbol"] for x in b["holdings"]]
    assert all(t["side"] == "BUY" and t["trade_date"] == _date.today().isoformat() for t in txs)


def test_builder_leaves_out_excluded_sectors_and_rejects_bad_input(app_client):
    h = {"X-API-Key": KEY}
    b = app_client.get("/ui/builder?capital=100000&n=3&risk=full&exclude=Energy", headers=h).json()
    assert "PEER" not in [x["symbol"] for x in b["holdings"]]
    assert app_client.get("/ui/builder?capital=500", headers=h).status_code == 422
    assert app_client.get("/ui/builder?capital=100000&n=2", headers=h).status_code == 422


def test_ai_forecast_says_why_when_models_are_switched_off(app_client, monkeypatch):
    monkeypatch.setenv("STOCKINTEL_ML", "0")
    h = {"X-API-Key": KEY}
    assert app_client.get("/ui/features", headers=h).json() == {"ml": False, "kronos": False}
    r = app_client.get("/ui/stock/TEST/ai?model=chronos", headers=h)
    assert r.status_code == 404 and "STOCKINTEL_ML" in r.json()["detail"]


def _closed_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_agent_ask_is_503_not_500_when_the_agent_is_down(orch, monkeypatch, tmp_path):
    import stockintel.selfagent_client as sc
    monkeypatch.setattr(sc, "CREDENTIALS", tmp_path / "agent.json")
    monkeypatch.setenv("SELFAGENT_URL", f"http://127.0.0.1:{_closed_port()}")
    client = TestClient(create_app(orch, api_key=KEY))
    h = {"X-API-Key": KEY}
    assert client.get("/ui/agent/status", headers=h).json()["available"] is False
    r = client.post("/ui/agent/ask", json={"message": "Should I buy ITC?"}, headers=h)
    assert r.status_code == 503 and "not reachable" in r.json()["detail"]


def test_phone_install_manifest_and_service_worker_are_served(app_client):
    m = app_client.get("/manifest.webmanifest").json()
    assert m["start_url"] == "/#/home" and m["display"] == "standalone"
    sw = app_client.get("/sw.js")
    assert sw.status_code == 200 and "javascript" in sw.headers["content-type"]
