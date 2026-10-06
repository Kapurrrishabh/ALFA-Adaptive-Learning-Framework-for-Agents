from types import SimpleNamespace as NS

import pytest

from backend.models.external.llm import LLMUnavailable, Narrator, ToolAgent, verify_numbers
from backend.api.nlu import classify, extract_amount, extract_horizon, extract_risk, extract_sectors, extract_tickers

KNOWN = ["RELIANCE", "HDFCBANK", "TCS", "INFY", "M&M"]


@pytest.mark.parametrize("text,amount", [
    ("I have ₹50,000", 50000), ("I have Rs 50000 to invest", 50000), ("invest 1 lakh", 1e5),
    ("I have ₹1.5 lakh", 1.5e5), ("2 crore portfolio", 2e7), ("with 50k", 5e4),
    ("If I add ₹20,000", 20000), ("analyze RELIANCE over 5 days", None), ("RSI above 70", None),
])
def test_amount_parsing(text, amount):
    assert extract_amount(text) == amount


def test_ticker_extraction_uses_aliases_and_ignores_finance_acronyms():
    assert extract_tickers("Compare it with HDFC Bank.", KNOWN) == [("HDFCBANK", "NSE")]
    assert extract_tickers("Is the RSI or P/E of TCS high?", KNOWN) == [("TCS", "NSE")]
    assert extract_tickers("analyze reliance.bo", KNOWN) == [("RELIANCE", "BSE")]
    assert extract_tickers("Analyze NVIDIA and compare with AMD", KNOWN) == [("NVDA", "US"), ("AMD", "US")]
    assert extract_tickers("what is RSI?", KNOWN) == []


def test_horizon_risk_and_sector_parsing():
    assert extract_horizon("I want to invest for 2 years") == "long"
    assert extract_horizon("swing trade for a few weeks") == "short"
    assert extract_risk("I am a conservative investor") == "conservative"
    inc, exc = extract_sectors("find pharma stocks but no banks")
    assert inc == ["Healthcare"] and exc == ["Financial Services"]
    assert extract_sectors("Compare it with TCS") == ([], [])


@pytest.mark.parametrize("text,session,intent", [
    ("Analyze RELIANCE.", False, "analyze"),
    ("Why is the technical state positive?", True, "domain"),
    ("What about fundamentals?", True, "domain"),
    ("Compare it with HDFC Bank.", True, "compare"),
    ("Why?", True, "explain_decision"),
    ("I have ₹1 lakh. How does my budget affect the analysis?", True, "budget_analysis"),
    ("I have ₹50,000. Find stocks worth researching.", False, "screen"),
    ("How is my portfolio performing?", False, "portfolio_summary"),
    ("Which holdings contribute most to my portfolio risk?", False, "portfolio_risk"),
    ("Am I overexposed to a sector in my portfolio?", False, "portfolio_exposure"),
    ("If I add ₹20,000 to my portfolio how could risk change?", False, "portfolio_what_if"),
    ("What is a Sharpe ratio?", False, "knowledge"),
    ("What is the current RSI of TCS?", False, "domain"),
    ("Why did INFY fall this week?", False, "explain_move"),
    ("What changed in this company this month?", True, "what_changed"),
    ("Generate a report on TCS", False, "report"),
    ("Find fundamentally strong companies with improving earnings and positive momentum", False, "research_query"),
    ("hello", False, "help"),
])
def test_intent_classification(text, session, intent):
    phrases = ["fundamentally strong", "improving earnings", "positive momentum"]
    assert classify(text, KNOWN, session, phrases).name == intent


PAYLOAD = {"rsi_14": 68.4, "revenue_growth": {"value": 18.2}, "de": 0.31, "label": "WATCH", "vol": 0.231,
           "note": "Price is 8.4% above its 50-day average (2,450.10)."}


def test_faithful_text_passes_verification():
    text = "RSI is 68.4 and revenue grew 18.2%. Debt/equity is 0.31. Volatility is 23.1%. Price sits 8.4% above 2,450.1."
    assert verify_numbers(text, PAYLOAD) == []


def test_invented_figures_are_caught():
    assert verify_numbers("Revenue grew 22.5% and the target is ₹3,100.", PAYLOAD) == ["22.5", "3100"]


def test_rounding_to_the_written_precision_is_allowed_but_not_beyond():
    assert verify_numbers("RSI is about 68.", PAYLOAD) == []
    assert verify_numbers("RSI is 68.9.", PAYLOAD) == ["68.9"]


def test_dates_and_small_counts_are_not_treated_as_figures():
    assert verify_numbers("As of 2026-09-25, 3 domains agree.", PAYLOAD) == []


class FakeClient:
    model = "fake-claude"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, system, messages, tools=None):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    @staticmethod
    def text(resp):
        return "".join(b.text for b in resp.content if b.type == "text")


def text_resp(t):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=t)], model="fake-claude")


def test_narrator_returns_llm_text_when_faithful():
    n = Narrator(FakeClient([text_resp("Momentum is strong: RSI 68.4.")]))
    out, meta = n.narrate("why?", PAYLOAD, "RSI(14): 68.4 | trend bullish")
    assert out == "Momentum is strong: RSI 68.4." and meta["unsupported_numbers"] == []


def test_narrator_discards_unfaithful_text_and_says_so():
    n = Narrator(FakeClient([text_resp("RSI is 91.2, a screaming buy.")]))
    out, meta = n.narrate("why?", PAYLOAD, "deterministic draft")
    assert out.startswith("deterministic draft") and "91.2" in out and meta["unsupported_numbers"] == ["91.2"]


def test_agent_executes_tools_and_verifies_against_their_outputs():
    tool_call = NS(stop_reason="tool_use", model="fake-claude",
                   content=[NS(type="tool_use", id="t1", name="analyze_stock", input={"symbol": "TCS"})])
    client = FakeClient([tool_call, text_resp("TCS scores 0.42 with RSI 55.1.")])
    seen = {}

    def analyze_stock(symbol):
        seen["symbol"] = symbol
        return {"score": 0.42, "rsi": 55.1}

    agent = ToolAgent(client, {"analyze_stock": ({"description": "x", "input_schema": {"type": "object"}}, analyze_stock)})
    text, meta = agent.run("Analyze TCS", [])
    assert seen["symbol"] == "TCS" and meta["tool_calls"] == ["analyze_stock"] and meta["unsupported_numbers"] == []
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "t1"


def test_agent_reports_unknown_tools_as_errors_to_the_model():
    call = NS(stop_reason="tool_use", model="m", content=[NS(type="tool_use", id="t9", name="place_order", input={})])
    client = FakeClient([call, text_resp("I cannot place orders.")])
    ToolAgent(client, {}).run("buy it", [])
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "unknown tool" in result["content"]


def test_agent_gives_up_after_max_steps():
    loop = NS(stop_reason="tool_use", model="m", content=[NS(type="tool_use", id="t", name="x", input={})])
    agent = ToolAgent(FakeClient([loop] * 3), {"x": ({"description": "", "input_schema": {}}, lambda: {})}, max_steps=3)
    with pytest.raises(LLMUnavailable, match="did not finish"):
        agent.run("q", [])


def test_sign_flips_are_caught():
    assert verify_numbers("Revenue growth was -18.2%.", PAYLOAD) == ["-18.2"]
    assert verify_numbers("Revenue grew 18.2%.", PAYLOAD) == []


def test_rupee_prefixes_are_read_as_part_of_the_figure():
    assert verify_numbers("Target Rs.2,999 and INR9999.", PAYLOAD) == ["2999", "9999"]
    assert verify_numbers("Support near Rs. 2,450.10.", PAYLOAD) == []


def test_small_figures_with_units_are_not_exempt():
    assert verify_numbers("Roughly 10% upside and a P/E of 9.", PAYLOAD) == ["10", "9"]


def test_counts_and_list_numbers_are_exempt():
    assert verify_numbers("1. Three of 5 domains agree over a 5-day window.", PAYLOAD) == []


def test_narrator_only_accepts_figures_from_the_draft_or_evidence_claims():
    analysis = {"price": 100.0, "decision": {"score": 0.3, "confidence": 0.6, "uncertainty": 0.4},
                "results": {"technical": {"evidence": [{"claim": "RSI is 55.1."}],
                                          "details": {"hidden": 77.7}}}}
    n = Narrator(FakeClient([text_resp("RSI is 55.1 and a hidden figure is 77.7.")]))
    out, meta = n.narrate("why?", analysis, "Score +0.30")
    assert meta["unsupported_numbers"] == ["77.7"]


@pytest.mark.parametrize("text", ["Should I buy TCS?", "Is it a good time to buy INFY?", "TCS buy or not",
                                  "should I sell RELIANCE", "Is TCS worth buying?"])
def test_should_buy_questions_get_the_verdict_intent(text):
    assert classify(text, KNOWN, False, []).name == "should_buy"


def _analyst_facts(rank=161, edge=(-0.008, 0.02), gru_mae=6.479):
    from backend.advisory import analyst
    days = 20
    alfa = {"q10": [663.0] * days, "q50": [714.0] * days, "q90": [766.0] * days, "record": {"vs_garch": -0.0323}}
    gru = {"close": [712.0] * days, "record": {"test": {"days": [{"day": 20, "mae_pct": gru_mae, "no_change_mae_pct": 6.54,
                                                                    "direction_acc": 0.539, "base_rate_acc": 0.533,
                                                                    "direction_edge_lo": edge[0], "direction_edge_hi": edge[1]}]}}}
    return analyst.facts("HDFCBANK", 709.0, "2026-10-06", alfa, gru, {"lo80": 657.0, "hi80": 766.0},
                         {"stop": 638.0, "rank": rank, "universe": 194, "verdict": "DON'T BUY"})


def test_analyst_facts_carry_our_judgements_not_the_writers():
    f = _analyst_facts()
    assert "about the same as the 6.54%" in f["gru_error"] and "within noise" in f["gru_direction"]
    assert "bottom fifth" in f["momentum"]
    assert "lower than" in _analyst_facts(gru_mae=6.0)["gru_error"]          # 8% below no-change is not "the same"
    assert "real but small edge" in _analyst_facts(edge=(0.01, 0.05))["gru_direction"]


def test_analyst_check_passes_the_computed_note_and_refuses_bad_drafts():
    from backend.advisory import analyst
    f = _analyst_facts()
    assert analyst.check(analyst.computed_note(f), f) == []
    good = ("Over 20 days the range is ₹663 to ₹766. The GRU's error is similar to assuming no change, and its "
            "direction is within noise. The verdict is DON'T BUY.")
    assert any("adds terms" in r for r in analyst.check(good + " A Granger test agrees.", f))
    softened = good.replace("its direction is within noise", "its direction is slightly better than the base rate")
    assert any("leaves out within noise" in r for r in analyst.check(softened, f))
    assert analyst.check(good, f) == []
    assert any("dollars" in r for r in analyst.check(good.replace("₹", "$"), f))
    assert any("direction" in r for r in analyst.check(good + " The price will rise.", f))
    assert any("praises" in r for r in analyst.check(good + " The GRU is reliable.", f))
    assert any("leaves out ₹766" in r for r in analyst.check("The range starts at ₹663. DON'T BUY.", f))
    assert any("figure 812" in r for r in analyst.check(good + " It may reach ₹812.", f))


def test_analyst_shows_the_writers_note_only_when_it_passes():
    from backend.advisory import analyst
    from backend.models.external.language import Reply

    class Writer:
        def __init__(self, text):
            self.text_out = text

        def create(self, system, messages):
            return Reply("test-writer", self.text_out)

        @staticmethod
        def text(resp):
            return resp.content
    f = _analyst_facts()
    ok = analyst.write(f, Writer("Over 20 days the range is ₹663 to ₹766. The GRU's error is about the same as no change "
                                 "and its direction is within noise. The verdict is DON'T BUY."))
    assert ok["written_by"] == "test-writer" and ok["draft_rejected"] == []
    bad = analyst.write(f, Writer("It will rise to ₹900."))
    assert bad["written_by"] is None and bad["text"] == analyst.computed_note(f) and bad["draft_rejected"]
