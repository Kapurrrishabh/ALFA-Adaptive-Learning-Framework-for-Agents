from types import SimpleNamespace as NS

import pytest

from backend.models.external.llm import LLMUnavailable, Narrator, ToolAgent, verify_numbers
from backend.api.nlu import classify, extract_amount, extract_horizon, extract_risk, extract_sectors, extract_tickers, name_index
from backend.knowledge_base.lessons import knowledge_base

KNOWN = name_index({"RELIANCE": "Reliance Industries Ltd.", "HDFCBANK": "HDFC Bank Ltd.", "TCS": "Tata Consultancy Services Ltd.",
                    "INFY": "Infosys Ltd.", "M&M": "Mahindra & Mahindra Ltd."}, set(), set())
FORMS = knowledge_base().short_forms


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


LISTED = name_index({"NYKAA": "FSN E-Commerce Ventures Ltd.", "IDEA": "Vodafone Idea Limited", "LAURUSLABS": "Laurus Labs Ltd.",
                     "ADANIENT": "Adani Enterprises Limited", "ADANIPORTS": "Adani Ports and Special Economic Zone Limited",
                     "RPOWER": "Reliance Power Limited", "RELIANCE": "Reliance Industries Limited"}, {"idea"}, set())


@pytest.mark.parametrize("text,symbols", [
    ("how is nykaa performing", [("NYKAA", "NSE")]),            # its symbol in lowercase
    ("fsn e-commerce ventures results", [("NYKAA", "NSE")]),    # its name
    ("how is laurus doing", [("LAURUSLABS", "NSE")]),           # a first word no other company starts with
    ("how is reliance power doing", [("RPOWER", "NSE")]),       # the longer name, not the alias "reliance"
    ("what is going on with adani", []),                         # two companies start with it
])
def test_a_listed_company_is_found_by_its_symbol_or_name_in_any_case(text, symbols):
    assert extract_tickers(text, LISTED) == symbols


def test_a_symbol_that_is_an_english_word_needs_capitals_or_more_of_its_name():
    assert extract_tickers("is it a good idea to buy now?", LISTED) == []
    assert extract_tickers("Is IDEA a good buy?", LISTED) == [("IDEA", "NSE")]
    assert extract_tickers("should i buy vodafone idea", LISTED) == [("IDEA", "NSE")]


# the dictionary's words as it spells them: it has "metal" and "mind", and no plurals
LEADING = name_index({"HAPPSTMNDS": "Happiest Minds Technologies Limited", "NDTV": "New Delhi Television Ltd.",
                      "INDHOTEL": "The Indian Hotels Company Limited", "IMFA": "Indian Metals & Ferro Alloys Limited",
                      "GODREJCP": "Godrej Consumer Products Limited", "GODREJPROP": "Godrej Properties Limited",
                      "CENTRALBK": "Central Bank of India", "CDSL": "Central Depository Services (India) Ltd."},
                     {"happiest", "mind", "new", "the", "central", "bank", "of", "hotel", "metal", "consumer"},
                     {"delhi", "india", "indian"})


@pytest.mark.parametrize("text,symbols", [
    ("how is godrej consumer doing ?", [("GODREJCP", "NSE")]),  # its leading words, which no other name has
    ("what did the government say in new delhi", []),           # leading words that are all dictionary words, a place too
    ("what did the central bank say", []),
    ("how are indian metals stocks doing?", []),                # and a plural of one
    ("how has happiest minds been doing lately ?", [("HAPPSTMNDS", "NSE")]),    # all everyday words, so an alias
    ("indian hotels results", [("INDHOTEL", "NSE")]),
])
def test_a_company_is_found_by_its_leading_words_unless_they_are_everyday_words(text, symbols):
    assert extract_tickers(text, LEADING) == symbols


def test_a_word_in_capitals_that_is_only_a_symbol_is_not_a_share_in_text_all_in_capitals():
    """TAKE Ltd. has no name but its symbol, an English word: TCS conversations were handed TAKE.NS's figures."""
    take = name_index({"TAKE": "TAKE Limited", "TCS": "Tata Consultancy Services Ltd."}, {"take"}, set())
    assert (extract_tickers("SHOULD I TAKE PROFITS NOW?", take), extract_tickers("SHOULD I TAKE PROFITS IN TCS?", take)) \
        == ([], [("TCS", "NSE")])


def test_every_listed_share_is_a_value_of_the_index_even_one_whose_names_are_all_english_words():
    """"Eternal Ltd." gave no phrase, so the app read ETERNAL as unlisted and threw away the alias "zomato" for it."""
    assert set(name_index({"ETERNAL": "Eternal Ltd.", "IDEA": "Vodafone Idea Limited"}, {"eternal", "idea"}, set()).values()) \
        == {"ETERNAL", "IDEA"}


def test_how_a_stock_is_performing_is_answered_about_the_stock_not_from_the_lessons():
    i = classify("how is the nykaa stock performing and should I buy it and if yes then why", LISTED, FORMS, False)
    assert (i.name, i.symbols) == ("should_buy", [("NYKAA", "NSE")])
    assert classify("how is nykaa stock performing", LISTED, FORMS, False).name == "analyze"   # "performing" is not "forming"


def test_horizon_risk_and_sector_parsing():
    assert extract_horizon("I want to invest for 2 years") == "long"
    assert extract_horizon("swing trade for a few weeks") == "short"
    assert extract_risk("I am a conservative investor") == "conservative"
    inc, exc = extract_sectors("find pharma stocks but no banks")
    assert inc == ["Healthcare"] and exc == ["Financial Services"]
    assert extract_sectors("Compare it with TCS") == ([], [])


@pytest.mark.parametrize("text,horizon", [
    ("should i hold meta for the long term", "long"), ("i plan to hold it for 6 months", "medium"),
    ("what is the intraday high of hdfc bank today", None), ("quick question, is tcs overbought", None),
    ("how much does yes bank swing", None), ("is reliance above its 200 day average", None),
    ("how much long term debt does L&T carry", None), ("how are short term capital gains taxed in India?", None),
    ("is tcs a good long term investment", "long"), ("should i buy tcs long term?", "long"),
    ("is infy a good short term buy", "short"), ("is tcs good in the long run", "long"), ("long term view on tcs", "long"),
    ("should i buy tcs with a 3 year horizon", "long"),
])
def test_only_a_horizon_someone_holds_for_sets_one(text, horizon):
    """"intraday" in a question about a price set the chat's horizon to short, and CAMPUS's wider check read AVOID
    instead of HOLD for the rest of it."""
    assert extract_horizon(text) == horizon


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
    ("hello", False, "greeting"),
    ("hi", False, "greeting"),
    ("thanks!", False, "greeting"),
    ("why is the market going down?", False, "market_regime"),
    ("is the nifty falling?", False, "market_regime"),
    ("why has the sensex been so weak?", False, "market_regime"),    # was a full stock analysis of ^BSESN
    ("the market looks weak", False, "market_regime"),
    ("tell me a joke", False, "help"),
    ("what is a stop-loss?", False, "knowledge"),
    ("check my stops", False, "stops"),
    ("what are my stops?", False, "stops"),
    ("what is an alert?", False, "knowledge"),
    ("what is NAV?", False, "knowledge"),
    ("what is the meaning of NAV?", False, "knowledge"),
    ("what is the full form of NAV?", False, "knowledge"),
    ("can you explain NAV?", False, "knowledge"),
    ("tell me about NAV", True, "knowledge"),
    ("please explain ELSS", False, "knowledge"),
    ("what is HHI?", False, "knowledge"),
    ("tell me about TCS", False, "analyze"),
    ("what is IRCTC in the news for?", False, "domain"),
    ("how are short term capital gains taxed in India?", False, "knowledge"),
    ("how are rate cuts priced in by the market?", False, "help"),
    ("how are its margins calculated?", True, "domain"),
    ("is Infosys overbought?", False, "domain"),
    ("is TCS a better buy than Infosys?", False, "analyze"),
    ("which of TCS and Infosys is a better buy?", False, "analyze"),
    ("what is the intraday high of TCS today", False, "day_range"),
    ("how far is TCS from its 60 day high?", False, "analyze"),      # a span's high, not the day's
    ("is RELIANCE at a 50-day low", False, "analyze"),
    ("why is INFY trading low today", False, "analyze"),
    ("why did TCS fall to a 60 day low", False, "explain_move"),
    ("is TCS in a correction", False, "correction"), ("is the market in a correction?", False, "market_regime"),
])
def test_intent_classification(text, session, intent):
    phrases = ["fundamentally strong", "improving earnings", "positive momentum"]
    assert classify(text, KNOWN, FORMS, session, phrases).name == intent


@pytest.mark.parametrize("session", [False, True])
def test_a_lesson_word_that_is_one_of_our_symbols_stays_a_stock(session):
    assert classify("what is NAV?", {**KNOWN, "nav": "NAV"}, FORMS, session).symbols == [("NAV", "NSE")]


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
                                  "Is TCS worth buying?", "is RELIANCE.NS a good buy right now?",
                                  "Is TCS a buy at this price?", "should i buy or sell TCS"])
def test_should_buy_questions_get_the_verdict_intent(text):
    assert classify(text, KNOWN, FORMS, False, []).name == "should_buy"


@pytest.mark.parametrize("text", ["should I sell RELIANCE", "should i hold TCS?", "is it time to exit INFY",
                                  "Is it a good time to sell INFY?", "sell or hold RELIANCE after the results?"])
def test_a_sell_or_hold_question_gets_the_holders_verdict_intent(text):
    """"should I sell reliance" was answered "DON'T BUY", which says nothing about a share already owned."""
    assert classify(text, KNOWN, FORMS, False, []).name == "should_sell"


def _analyst_facts(rank=161, edge=(-0.008, 0.02), gru_mae=6.479, drawer="generative", nll_gain_lo=0.019,
                   test_from="2024-01-01"):
    from backend.advisory import analyst
    from backend.config import FORECAST_DAYS as days
    alfa = {"q10": [663.0] * days, "q50": [714.0] * days, "q90": [766.0] * days, "record": {"vs_garch": -0.0323},
            "drawer": {"name": drawer}}
    gru = {"close": [712.0] * days, "record": {"test_from": test_from, "test": {"nll_gain_lo": nll_gain_lo, "nll_gain_hi": 0.058, "days": [
                                                         {"day": days, "mae_pct": gru_mae, "no_change_mae_pct": 6.54,
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


def test_analyst_note_dates_the_gru_error_from_its_record_window():
    from backend.advisory import analyst
    f = _analyst_facts(test_from="2023-07-03")
    assert "since July 2023" in f["gru_error"] and "2024" not in f["gru_error"]
    assert any("2019" in r for r in analyst.check(analyst.note_to_reword(f).replace("July 2023", "June 2019"), f))
    assert any("leaves out July 2023" in r for r in analyst.check(analyst.note_to_reword(f).replace("July", "June"), f))


def test_analyst_facts_call_a_direction_reliably_below_the_base_rate_worse_and_require_it():
    f = _analyst_facts(edge=(-0.07, -0.01))
    assert "worse than the base rate" in f["gru_direction"] and "edge" not in f["gru_direction"]
    assert ("worse", "less often") in f["_required"]


def test_analyst_note_credits_garch_when_garch_drew_the_fan():
    from backend.advisory import analyst
    f = _analyst_facts(drawer="garch")
    note = analyst.computed_note(f)
    assert note.startswith(f"Over the next {f['days_ahead']} trading days the standard GARCH model puts HDFCBANK at")
    assert "beat the standard GARCH" not in note and "our return model has not beaten it" in note
    assert analyst.check(analyst.note_to_reword(f), f) == []
    credits_ours = (f"Over the next {f['days_ahead']} trading days our return model puts HDFCBANK at ₹663 to ₹766, middle "
                    "₹714. The GRU's error was about the same as assuming no change, and its direction was within noise. "
                    "Verdict: DON'T BUY, stop-loss ₹638.")
    # the required words can be met with the wrong subject, so naming our model at all is refused
    for draft in (credits_ours, credits_ours + " The standard GARCH model has not beaten it.",
                  credits_ours.replace("our return model", "our model") + " GARCH has not beaten it."):
        assert any("names our return model" in r for r in analyst.check(draft, f))


@pytest.mark.parametrize("drawer, other", [("garch", "return model"), ("generative", "GARCH")])
def test_the_writer_never_sees_how_the_two_models_compare_and_the_note_says_it_as_written(drawer, other):
    from backend.advisory import analyst
    f = _analyst_facts(drawer=drawer)
    draft = (f"Over {f['days_ahead']} days 80% of the paths end between ₹663 and ₹766. The GRU's error since January 2024 "
             "is about the same as no change and its direction is within noise against the base rate. The verdict is DON'T BUY.")
    writer = _Picker(draft)
    out = analyst.write(f, writer)
    assert other not in writer.prompts[0] and f["return_model_record"] not in writer.prompts[0]
    assert out["text"] == f"{draft} {f['return_model_record']}" and out["written_by"] == "test-lm"


def test_analyst_check_refuses_a_rewrite_that_says_the_gru_beat_the_band_when_it_did_not():
    from backend.advisory import analyst
    f = _analyst_facts(nll_gain_lo=-0.01)
    note = analyst.note_to_reword(f)
    assert analyst.check(note, f) == []
    assert any("leaves out did not beat" in r for r in analyst.check(note.replace("did not beat", "beat"), f))


def test_analyst_check_passes_the_note_it_rewords_and_refuses_bad_drafts():
    from backend.advisory import analyst
    f = _analyst_facts()
    assert analyst.check(analyst.note_to_reword(f), f) == []
    good = (f"Over {f['days_ahead']} days 80% of the paths end between ₹663 and ₹766. The GRU's error since January 2024 "
            "is similar to assuming no change, and against the base rate its direction is within noise. The verdict is DON'T BUY.")
    assert any("adds terms" in r for r in analyst.check(good + " A Granger test agrees.", f))
    softened = good.replace("its direction is within noise", "its direction is slightly better than the base rate")
    assert any("leaves out within noise" in r for r in analyst.check(softened, f))
    assert analyst.check(good, f) == []
    assert any("dollars" in r for r in analyst.check(good.replace("₹", "$"), f))
    assert any("direction" in r for r in analyst.check(good + " The price will rise.", f))
    assert any("praises" in r for r in analyst.check(good + " The GRU is reliable.", f))
    assert any("one-day limit" in r for r in analyst.check(good + " The GRU beat the band on daily swings, but only for one day ahead.", f))
    assert any("leaves out ₹766" in r for r in analyst.check("The range starts at ₹663. DON'T BUY.", f))
    assert any("figure 812" in r for r in analyst.check(good + " It may reach ₹812.", f))


@pytest.mark.parametrize("kept,reworded,dropped", [
    ("puts HDFCBANK at ₹663 to ₹766, middle ₹714 (80% of our return model's paths)",
     "expects HDFCBANK to be between ₹663 and ₹766, with a middle value of ₹714", ["80%", "paths"]),
    ("about the same as the 6.54% of assuming no change", "about the same as 6.54%", ["no change"]),
    ("against 53.3% for the base rate", "against 53.3%", ["base rate"]),
], ids=["the range's share of paths", "what the error is compared with", "what the direction is compared with"])
def test_analyst_check_refuses_a_rewrite_that_drops_what_a_range_or_comparison_is_of(kept, reworded, dropped):
    """Phi-4-mini wrote TCS "is expected to be between ₹2,251 and ₹2,566" and NYKAA's error "about the same as 4.684%",
    and both passed, as every figure in them was in the facts."""
    from backend.advisory import analyst
    f = _analyst_facts()
    note = analyst.note_to_reword(f)
    assert kept in note
    assert analyst.check(note.replace(kept, reworded), f) == [f"it leaves out {d}" for d in dropped]


def test_the_writers_prompt_names_the_qualifiers_its_check_requires():
    """The check refused every note that dropped them, and the prompt never asked for them, so more fell back."""
    from backend.advisory import analyst
    assert [q for q in ("80%", "paths", "no change", "base rate") if f'"{q}"' not in analyst.SYSTEM] == []


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
    ok = analyst.write(f, Writer(f"Over {f['days_ahead']} days 80% of the paths end between ₹663 and ₹766. The GRU's error "
                                 "since January 2024 is about the same as no change and its direction is within noise against "
                                 "the base rate. The verdict is DON'T BUY."))
    assert ok["written_by"] == "test-writer" and ok["draft_rejected"] == []
    bad = analyst.write(f, Writer("It will rise to ₹900."))
    assert bad["written_by"] is None and bad["text"] == analyst.computed_note(f) and bad["draft_rejected"]


def _writer_comparison_inputs(monkeypatch, tmp_path, analyst_inputs, n):
    """compare_writers' fact sheets for `n` of three stocks, from a hub whose analyst inputs are `analyst_inputs`."""
    import pandas as pd
    import backend.api.hub as hubmod
    import backend.cli as cli
    from backend.models.external import compare_writers
    traded = pd.DataFrame(1.0, index=pd.bdate_range(end="2026-09-25", periods=5), columns=["AAA", "BBB", "CCC"])
    hub = NS(panel=lambda: NS(close=traded, volume=traded), analyst_inputs=analyst_inputs,
             names=lambda: pd.DataFrame({"symbol": list(traded.columns), "nifty200": [True] * 3}))
    monkeypatch.setattr(cli, "_orchestrator", lambda args: NS(session=lambda portfolio, session_id: None))
    monkeypatch.setattr(hubmod, "Hub", lambda orch: hub)
    monkeypatch.setattr(compare_writers, "INPUTS", tmp_path / "analyst_inputs.json")
    return compare_writers.build_facts(n)


def test_the_writer_comparison_scores_the_inputs_the_analyst_note_is_written_from(monkeypatch, tmp_path):
    import json
    sheets = _writer_comparison_inputs(monkeypatch, tmp_path, lambda sym, s: {"symbol": sym}, 2)
    assert sheets == [{"symbol": "AAA"}, {"symbol": "BBB"}]
    assert json.loads((tmp_path / "analyst_inputs.json").read_text()) == sheets


def test_the_writer_comparison_stops_when_every_stock_is_refused(monkeypatch, tmp_path):
    """A GRU record calibrated the old way refuses every stock; skipping them all saved an empty input file."""
    from backend.database.sources.provider import DataUnavailable

    def refuse(sym, s):
        raise DataUnavailable("the GRU record carries an old calibration")
    with pytest.raises(DataUnavailable, match="only 0 of 2 stocks gave fact sheets"):
        _writer_comparison_inputs(monkeypatch, tmp_path, refuse, 2)
    assert not (tmp_path / "analyst_inputs.json").exists()


class _Picker:
    """A writer that answers with whatever label it was told to, like the Space's language model."""

    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def create(self, system, messages):
        from backend.models.external.language import Reply as LMReply
        self.prompts.append(messages[0]["content"])
        return LMReply("test-lm", self.answer)

    @staticmethod
    def text(resp):
        return resp.content


@pytest.mark.parametrize("answer,intent", [
    ("market_regime", "market_regime"),
    ("  Market_Regime.  ", "market_regime"),      # labels are read loosely, so spacing and case cannot break routing
    ("NONE", "help"),
    ("the market is falling because of global cues", None),   # prose is not a label, so it is discarded
])
def test_the_language_model_only_picks_one_of_our_handlers(answer, intent):
    from backend.api.intent_lm import choose
    from backend.api.nlu import ROUTABLE
    assert choose("what is happening out there?", ROUTABLE, _Picker(answer), declined="help") == intent


@pytest.mark.parametrize("question,title", [
    ("what is a Sharpe ratio?", "Sharpe ratio"),
    ("what is a stop-loss?", "Stop-loss"),          # a Learn-page chip, so it must have an entry
    ("what is LTCG tax", "Taxes on listed equity (verify current rules)"),   # found through a synonym and a stem
    ("how much tax is due on shares I sold", "Taxes on listed equity (verify current rules)"),  # ELSS also says "tax"
    ("is the price of a share unpredictable", "Why the system does not predict exact prices"),
    ("is debt bad for a business", "Debt/equity, interest coverage and current ratio"),
    ("is a correlated stock risky to hold", "Risk contribution"),
    ("is a candle formation a signal", "What the evidence says about candlesticks"),
    ("can a test show a result is real", "Data snooping and multiple testing"),
])
def test_the_reference_answers_the_questions_the_app_offers(question, title):
    from backend.knowledge_base.lessons import knowledge_base
    assert knowledge_base().search(question)[0][0].title == title


@pytest.mark.parametrize("question,title", [
    ("is there a fund locked for three years", "ELSS (tax-saving funds, verify current rules)"),
    ("price bands explained in simple words", "Price bands and circuit breakers"),
    ("can you help me with volatility", "Volatility"),
    ("give me the basics of moving averages", "Moving averages (SMA, EMA, WMA)"),
    ("show me how momentum works", "Momentum and rate of change"),
    ("take me through what volatility is", "Volatility"),
    ("what's the fund's p/e ratio", "P/E ratio (price to earnings)"),
    ("can the model predict tomorrow's price", "Why the system does not predict exact prices"),
    ("tax on selling stocks", "Taxes on listed equity (verify current rules)"),   # "Taxes" was cut to "taxe"
    # wordings reviewers saw lose their lesson when another lesson's prose was reworded
    ("tax on shares sold this year", "Taxes on listed equity (verify current rules)"),
    ("why can't you predict prices", "Why the system does not predict exact prices"),
    ("how do i figure out p/e ratio", "P/E ratio (price to earnings)"),
    ("volatility in both directions", "Volatility"),
    ("volatility and turbulent markets", "Volatility"),
    ("is this firm overvalued on p/e", "P/E ratio (price to earnings)"),
    ("what tax do I pay on shares I sold", "Taxes on listed equity (verify current rules)"),
    ("what is a stop loss order", "Stop-loss"),          # a hyphenated heading word is named by its parts
])
def test_the_agents_one_lesson_is_the_one_the_question_names(question, title):
    # the agent reads only the top hit (run/ask.py), so a chat word such as "help" or "simple" placed in another
    # lesson takes that hit, and the question gets the wrong lesson or none; the app's top-3 can still look right
    from backend.knowledge_base.lessons import knowledge_base
    assert [lesson.title for lesson, _ in knowledge_base().search(question, k=1)] == [title]


def test_a_lesson_whose_text_is_reworded_to_mention_another_topic_does_not_take_its_question(tmp_path):
    from backend.knowledge_base.lessons import KB_DIR, KnowledgeBase
    for path in KB_DIR.glob("*.md"):
        (tmp_path / path.name).write_text(path.read_text().replace(
            "## PEG ratio\n", "## PEG ratio\nA high p/e ratio is not a signal here.\n"))
    hits = KnowledgeBase(tmp_path).search("what is a p/e ratio", k=1)
    assert [lesson.title for lesson, _ in hits] == ["P/E ratio (price to earnings)"]


@pytest.mark.parametrize("question", ["is it a good day", "anything new", "what is going on out there?",
                                      "who are you", "tell me a joke", "what causes a price decline",
                                      "why did the price swing"])
def test_the_reference_stays_silent_rather_than_serving_a_chance_word_match(question):
    from backend.knowledge_base.lessons import knowledge_base
    assert knowledge_base().search(question) == []
