import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.advisory import report as R
from backend.user.portfolio.alerts import evaluate
from backend.api.app import create_app
from backend.config import FORECAST_DAYS
from backend.api.orchestrator import Orchestrator
from backend.advisory.calculators.screener import screen
from backend.advisory.service import AnalysisService

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
    import backend.api.app as api_mod
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
    from backend.database.sources.panel import panel_from_frames
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
    assert v["evidence_note"].startswith("The buy/sell rule is NSE's momentum method.")


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


def test_portfolio_history_rebuilds_value_from_transactions(app_client, orch):
    orch.save_portfolio(0.0, "moderate", "medium", [
        {"symbol": "TEST", "side": "BUY", "quantity": 10, "price": 100, "trade_date": "2025-06-02"}])
    h = app_client.get("/ui/portfolio/history", headers={"X-API-Key": KEY}).json()
    assert h["dates"][0] >= "2025-06-02" and h["invested"][-1] == pytest.approx(1000)
    assert all(v > 0 for v in h["value"])


def test_performance_endpoint_reports_missing_build_cleanly(app_client, monkeypatch, tmp_path):
    import backend.advisory.calculators.performance as perf
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
    f = app_client.get("/ui/features", headers=h).json()
    assert f["ml"] is False and f["kronos"] is False      # ALFA's NumPy model is not behind this switch
    r = app_client.get("/ui/stock/TEST/ai?model=chronos", headers=h)
    assert r.status_code == 404 and "STOCKINTEL_ML" in r.json()["detail"]


def _closed_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_agent_ask_is_503_not_500_when_the_agent_is_down(orch, monkeypatch, tmp_path):
    import backend.models.agent.client as sc
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


def test_ticker_lists_indices_then_stocks_with_a_day_change(app_client):
    t = app_client.get("/ui/ticker", headers={"X-API-Key": KEY}).json()
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


def test_technical_range_claim_and_stop_come_from_data(app_client, tmp_path, monkeypatch):
    import backend.advisory.calculators.performance as perf
    from backend.advisory.signals.advisor import STOP_PCT
    monkeypatch.setattr(perf, "PATH", _record(tmp_path))
    t = app_client.get("/ui/stock/TEST/technical?bars=60", headers={"X-API-Key": KEY}).json()
    assert "held 80% of outcomes in the 80% band (median of 3 stocks)" in t["cone_note"]
    assert t["stop"] == pytest.approx(t["close"][-1] * (1 - STOP_PCT), abs=0.01) and t["stop_pct"] == STOP_PCT * 100
    monkeypatch.setattr(perf, "PATH", tmp_path / "missing.json")
    t = app_client.get("/ui/stock/TEST/technical?bars=60", headers={"X-API-Key": KEY}).json()
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
    r = app_client.get("/ui/stock/TEST/ai?model=chronos", headers={"X-API-Key": KEY}).json()
    assert r["q50"] == fan["q50"] and len(r["dates"]) == FORECAST_DAYS
    api_name, (closes, horizon) = fake.calls[0]
    assert api_name == "/chronos_fan" and horizon == FORECAST_DAYS and 0 < len(_json.loads(closes)) <= CONTEXT


def test_a_failing_model_space_is_a_clean_404_naming_the_space(app_client, monkeypatch):
    import backend.models.external.remote_models as rm
    monkeypatch.setenv("STOCKINTEL_MODEL_SPACE", "you/stockintel-models")
    monkeypatch.setattr(rm, "_client", lambda: _FakeSpace(error=RuntimeError("queue full")))
    r = app_client.get("/ui/stock/PEER/ai?model=kronos", headers={"X-API-Key": KEY})
    assert r.status_code == 404 and "you/stockintel-models" in r.json()["detail"] and "queue full" in r.json()["detail"]


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
    r = app_client.get("/ui/stock/TEST/ai?model=alfa", headers={"X-API-Key": KEY}).json()
    assert len(r["dates"]) == am.STEPS and r["q50"] == [100.0] * am.STEPS
    assert "beat GARCH(1,1)-t by 0.032 ± 0.003 nats per return over 80,576 returns" in r["note"] and r["note"].endswith("No direction.")


def test_alfa_says_how_to_reach_it_when_it_cannot(app_client, monkeypatch, tmp_path):
    import backend.models.serving.alfa_fan as am
    monkeypatch.setattr(am, "WEIGHTS", tmp_path / "missing.npz")
    r = app_client.get("/ui/stock/TEST/ai?model=alfa", headers={"X-API-Key": KEY})
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
    note = app_client.get("/ui/stock/TEST/ai?model=chronos", headers={"X-API-Key": KEY}).json()["note"]
    assert note.startswith("Chronos-2 fine-tuned on NSE prices by us.") and f"Tested on 5472 past {FORECAST_DAYS}-day forecasts" in note
    assert "its range was worse than plain volatility" in note
    assert "most confident 11% of 'up' calls were right 52% (95% range 43%–61%) vs 52% for always 'up'" in note


def test_gru_line_note_quotes_its_test_record(app_client, monkeypatch):
    import backend.models.serving.gru_line as gl
    record = {"test_from": "2024-01-01", "test": {"n": 5954, "nll_gain_lo": 0.019, "nll_gain_hi": 0.058, "days": [
        {"day": FORECAST_DAYS, "mae_pct": 6.479, "no_change_mae_pct": 6.54, "direction_acc": 0.5391, "base_rate_acc": 0.5334,
         "direction_edge_lo": -0.0079, "direction_edge_hi": 0.0199}]}}
    paths = [[100.0 + k for k in range(FORECAST_DAYS)]] * 3
    monkeypatch.setattr(gl, "line", lambda c, h, l, m: {"close": [101.0] * FORECAST_DAYS, "paths": paths, "record": record})
    r = app_client.get("/ui/stock/TEST/ai?model=gru", headers={"X-API-Key": KEY}).json()
    assert len(r["dates"]) == FORECAST_DAYS and r["close"] == [101.0] * FORECAST_DAYS
    assert r["paths"] == paths and "3 possible paths" in r["note"]
    assert "beat the volatility band (likelihood better by 0.019 to 0.058 per day" in r["note"]
    assert "error was 6.479% against 6.54% for assuming no change" in r["note"] and r["note"].endswith("within noise.")


def test_analyst_note_falls_back_to_the_computed_note_when_models_are_off(app_client, monkeypatch):
    import backend.api.hub as hubmod
    from tests.app.test_nlu_llm import _analyst_facts
    monkeypatch.setenv("STOCKINTEL_ML", "0")
    f = _analyst_facts()
    monkeypatch.setattr(hubmod.Hub, "ai_forecast", lambda self, sym, model: {})
    import backend.advisory.analyst as an
    monkeypatch.setattr(an, "facts", lambda *a, **k: f)
    r = app_client.get("/ui/stock/TEST/analyst", headers={"X-API-Key": KEY}).json()
    assert r["written_by"] is None and r["text"] == an.computed_note(f) and "switched off" in r["draft_rejected"][0]
