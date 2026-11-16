"""S14/S15: every figure an answer states is one the evidence states, or the answer is a refusal."""

import collections
import json
import re

import numpy as np
import pytest

from backend.models.core.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID
from backend.models.core.tokenizer.wordpiece import WordPiece
from backend.models.guardrails import guardrails
from backend.models.data import advisory, slots, stock_qa
from backend.models.training import prepare_alfa_v2, prepare_stock_qa
from backend.paths import ARTIFACTS

EVIDENCE = "ticker AAPL ; close 182.50 ; return_20d -4.8% ; volatility_20d 21.3% ; atr_14 3.05"


def snapshot(rows=80, seed=0):
    """A rising series with enough history for every indicator in a snapshot."""
    rng = np.random.default_rng(seed)
    closes = 100.0 + rng.standard_normal(rows).cumsum()
    bars = np.stack([closes, closes + 1.0, closes - 1.0, closes, np.full(rows, 1e6)], axis=1)
    return bars


def test_an_invented_figure_is_caught():
    answer = "it trades at 182.50 and earned 3.11 a share ."
    assert guardrails.unsupported_figures(answer, EVIDENCE) == ["3.11"]


def test_a_figure_inside_a_longer_number_does_not_count_as_supported():
    """The failure this would hide: 21.3% in the evidence silently vouching for a 21.35% answer."""
    assert guardrails.unsupported_figures("volatility is 21.35% .", EVIDENCE) == ["21.35%"]


@pytest.mark.parametrize("figure", ["1158.70", "1,234,567", "8.70", "-24.8%", "+115.4%"])
def test_a_figure_is_read_whole(figure):
    """A four digit price split into 115 and 8.70 failed 11% of real rows against their own evidence,
    which is the guardrail calling a faithful answer a fabrication."""
    assert guardrails.unsupported_figures(f"it is {figure} .", f"value {figure}") == []


@pytest.mark.parametrize("text", ["the range was 0.4-0.6 of the index .", "a contribution for 2017.",
                                  "it fell to 0. then it recovered ."])
def test_a_text_always_supports_itself(text):
    """The reference path serves a quote of its evidence when it cannot trust the paraphrase, so a quote
    failing its own source would turn every such answer into a refusal. Three real shapes did: a figure
    ending a sentence, and a signed one whose minus sign follows a digit."""
    assert guardrails.unsupported_figures(text, text) == []


def test_a_detached_minus_sign_still_reads_as_the_signed_figure_it_matches():
    """A real served turn, refused: the evidence stated -5.6% and the model wrote it, but its minus sign
    is a token of its own so the extractor read an unsigned 5.6%. Every test above spells the sign
    attached, which is why the suite passed while serving withheld a grounded answer."""
    answer = "it is at 1242.30 , - 5.6% over 20 days , running at 16.8% volatility ."
    evidence = "ticker RELIANCE.NS ; close 1242.30 ; return_20d -5.6% ; volatility_20d 16.8%"
    assert guardrails.unsupported_figures(answer, evidence) == []


def test_a_flipped_sign_is_still_caught():
    """The magnitude is grounded and the direction is not, which is the failure that looks fine."""
    assert guardrails.unsupported_figures("it gained +4.8% .", EVIDENCE) == ["+4.8%"]


@pytest.mark.parametrize("evidence", ["ticker AAPL ; rsi_14 73 ; sma_50 231.50",
                                      "ticker aapl ; rsi _ 14 73 ; sma _ 50 231.50"])
def test_a_window_in_a_key_name_does_not_support_a_value(evidence):
    """The 14 day window is stated by "rsi_14" and the RSI is 73, so a slipped RSI of 50 must not pass on
    "sma_50". The spaced form is the decoded evidence the measured gate reads."""
    answer = "its 14 day rsi is 50 , which is neither overbought nor oversold ."
    assert guardrails.unsupported_figures(answer, evidence) == ["50"]
    assert guardrails.unsupported_figures("its 14 day rsi is 73 , and its 50 day average 231.50 .", evidence) == []


@pytest.mark.parametrize("answer,evidence", [
    ("it returned 7% over 10 years .", "Over the past 10 years the index returned 7%."),
    ("over the past 10 years it returned 7% .", "The index returned 7% over 10 years."),
    ("it gained 5% over 3 sessions .", "It gained 5% in the past 3 sessions."),
    ("RSI (14) is 73, and the 50-DMA is 231.50.", "rsi_14 73 ; sma_50 231.50 its 14 day rsi is 73 , with a "
                                                  "50 day average of 231.50 ."),
])
def test_a_window_written_in_free_text_on_either_side_still_supports_its_figure(answer, evidence):
    """Treating "past 10" or "3 day" as windows everywhere rejected faithful passage answers and rewrites."""
    assert guardrails.unsupported_figures(answer, evidence) == []


def test_a_refusal_has_nothing_to_support():
    assert guardrails.unsupported_figures(advisory.UNSUPPORTED, "") == []


NIFTY_AND_TCS = ["ET, 27 Oct 2026: The Nifty fell 1.2% as IT stocks dropped on weak US cues.",
                 "Mint, 26 Oct 2026: TCS shares slid 2% after its results missed estimates."]


def test_a_figure_tied_to_a_name_no_passage_states_it_with_is_caught():
    """"TCS shares fell 1.2%" passed the entailment model for passages where 1.2% was the Nifty's fall."""
    assert guardrails.unbound_figures("TCS shares fell 1.2% on weak US cues.", NIFTY_AND_TCS) == [("1.2%", "TCS")]


@pytest.mark.parametrize("answer", ["The Nifty fell 1.2% as IT stocks dropped (ET, 27 Oct 2026).",
                                    "TCS shares slid 2%, and the Nifty fell 1.2%.",
                                    "Down 1.2%, the Nifty fell as IT stocks dropped."])
def test_a_figure_tied_to_the_name_its_passage_gives_it_is_bound(answer):
    assert guardrails.unbound_figures(answer, NIFTY_AND_TCS) == []


@pytest.mark.parametrize("answer", ["The Reserve Bank of India (RBI) raised the repo rate to 5.5%.",
                                    "The Reserve Bank of India raised the repo rate to 5.5%."])
def test_a_name_written_out_is_bound_by_its_short_form(answer):
    """"The Reserve Bank of India (RBI) ... 5.5%" was refused for a passage that gave 5.5% under "RBI" alone."""
    passages = ["Mint, 7 Oct 2026: RBI hikes the repo rate by 25 bps to 5.5%.",
                "NDTV Profit, 7 Oct 2026: The Reserve Bank of India raised its growth forecast."]
    assert guardrails.unbound_figures(answer, passages) == []


NESTLE = ("Stock: Nestle India Limited (NESTLEIND.NS), NSE, INR; last close 1,263.20 on 2026-11-13. ; [fundamental] "
          "Return on equity is 80.0%. (FY ending 2025-03-31). ; [fundamental] Return on invested capital is 11.2%. "
          "(FY ending 2026-03-31). ; [fundamental] Dividend yield 0.58% (snapshot retrieved 2026-11-15). ; "
          "[fundamental] Free cash flow yield is 1.7%. (snapshot retrieved 2026-11-15). ; [fundamental] PEG ratio is 1.04. "
          "(snapshot retrieved 2026-11-15).")


@pytest.mark.parametrize("answer,caught", [
    ("for nestle india limited (nestleind. ns), roic, profit on all money from owners and lenders, was 80.0% .",
     [("80.0%", "return on invested capital")]),
    ("its return on equity , profit on the owners ' money , was 80.0% , and its current ratio was 11.2% .",
     [("11.2%", "current ratio")])], ids=["ROE given as ROIC", "ROIC given as the current ratio"])
def test_a_facts_answer_giving_a_figure_as_another_metrics_is_caught(answer, caught):
    """Both were spoken past the 0.799 cut and every other guard, every figure being in the passage."""
    assert guardrails.mislabelled_figures(answer, NESTLE, stock_qa.METRICS) == caught


@pytest.mark.parametrize("answer", [
    "Nestle India (NESTLEIND.NS) had ROE of 80.0% for the FY ending 2025-03-31, and ROIC was 11.2%.",
    "its peg , the p / e divided by profit growth , was 1.04 .",
    "the 0.58% dividend yield , the yearly dividend as a share of the price , sat below a 1.7% free cash flow yield ."],
    ids=["short forms", "a meaning naming another metric", "a figure named after it"])
def test_a_facts_answer_giving_each_figure_its_own_metric_passes(answer):
    """Tied to the nearest metric, as `unbound_figures` ties names, 272 of 2,752 written passage answers were refused."""
    assert guardrails.mislabelled_figures(answer, NESTLE, stock_qa.METRICS) == []


ASIANPAINT = ("Stock: Asian Paints Limited (ASIANPAINT.NS), NSE, INR; last close 2,550.00 on 2026-11-13. ; [fundamental] "
              "Current ratio is 2.21. (FY ending 2026-03-31). ; [fundamental] Return on equity is 20.2%. (FY ending 2026-03-31).")


@pytest.mark.parametrize("answer,caught", [
    ("for the fy ending 2026 - 03 - 31, return on equity, the owners'money, was 20.2%, and return on invested capital, "
     "profit on all money from owners and lenders, was light, was 20.2%.", [("20.2%", "return on invested capital")]),
    ("roic, which was 20.2%, is high .", [("20.2%", "return on invested capital")]),
    ("our facts do not include its p / e ; the stock closed at 2550.00 .", [])],
    ids=["past a second aside", "inside the aside", "not past a semicolon"])
def test_a_facts_answer_is_read_past_its_clauses_without_a_figure_to_the_figure_its_metric_gives(answer, caught):
    """Read only to the next comma, ROE's 20.2% given as ROIC was spoken for ASIANPAINT, as were 2 more of 37 answers."""
    assert guardrails.mislabelled_figures(answer, ASIANPAINT, stock_qa.METRICS) == caught


@pytest.mark.parametrize("answer,evidence,wrong", [
    ("it sits above both its 20 and 50 day averages .", "close 344.63 ; sma_20 333.04 ; sma_50 342.08", []),
    ("it sits between its 20 and 50 day averages .", "close 344.63 ; sma_20 333.04 ; sma_50 342.08",
     [("between", "above")]),
    ("it trades at 979.40, above both its 20 and 50 day averages .", "close 979.40 ; sma_20 959.32 ; sma_50 988.22",
     [("above", "between")]),
    ("it sits below both its 20 and 50 day averages .", "close 344.63 ; rsi_14 55", [("below", None)]),
    ("it is up 3.3% over 20 days .", "close 344.63 ; rsi_14 55", [])],
    ids=["above both, averages crossed", "between said above both", "above said between", "no averages", "no trend"])
def test_trend_words_are_read_against_the_close_and_each_average_not_their_order(answer, evidence, wrong):
    """The training rule says "between" for a price above both whose 20-day average is under its 50-day one, so 47% of
    the "between" targets were false; the guard reads what the words say."""
    assert guardrails.wrong_trend(answer, evidence) == wrong


def test_a_close_quoted_without_its_thousands_separator_is_supported():
    """The tokenizer drops the comma, so the model can only write a facts passage's 1,263.20 as 1263.20."""
    assert guardrails.unsupported_figures("nestle india closed at 1263.20 .", NESTLE) == []


def test_screen_swaps_in_the_refusal_and_reports_why():
    text, unsupported = guardrails.screen("its target is 250.00 .", EVIDENCE, advisory.UNSUPPORTED)
    assert text == advisory.UNSUPPORTED
    assert unsupported == ["250.00"]


def test_screen_passes_a_grounded_answer_through_unchanged():
    answer = "it is trading at 182.50 , -4.8% over 20 days ."
    assert guardrails.screen(answer, EVIDENCE, advisory.UNSUPPORTED) == (answer, [])


def test_every_phrasing_of_every_intent_passes_its_own_guardrail():
    """The dataset's whole premise, across all 46 rows a snapshot can produce. One answer form that
    states a figure the evidence lacks is a row teaching invention, and invisible in the loss."""
    bars = snapshot()
    end = len(bars) - 1
    facts = advisory.snapshot(bars, end)
    shown = advisory.as_text(facts)
    shown.update(advisory.risk_outlook([0.21, 0.55, 0.24]))
    evidence = advisory.render_evidence("AAPL", shown)

    for intent in advisory.INTENTS:
        for phrasing in range(advisory.phrasings(intent)):
            question, answer = advisory.row(intent, "AAPL", facts, shown, phrasing)
            assert guardrails.unsupported_figures(answer, evidence) == [], f"{intent}: {question}"


def test_held_out_phrasings_exist_and_are_a_minority():
    """If training saw every phrasing there would be nothing left to measure generalisation with."""
    for intent in advisory.INTENTS:
        held = [p for p in range(advisory.phrasings(intent)) if advisory.is_held_out(intent, p)]
        assert held, intent
        assert len(held) < advisory.phrasings(intent) - 1, intent


def test_a_reserved_word_appears_in_no_trained_question_of_any_intent():
    """The leak that would invalidate the experiment in silence. Reserved words exist to ask whether
    the encoder can place a word it has never read; one appearing in any trained question, even for a
    different intent, turns the answer into recall and nothing in the loss or the probe would say so."""
    trained = " | ".join(advisory.ask(intent, "AAPL", p)
                         for intent in advisory.INTENTS
                         for p in range(advisory.phrasings(intent))
                         if not advisory.is_held_out(intent, p))
    for intent in advisory.INTENTS:
        for phrasing in range(advisory.phrasings(intent)):
            if "word" not in advisory.novelty(intent, phrasing):
                continue
            reserved = advisory.ask(intent, "AAPL", phrasing)
            for word in reserved.split():
                if word not in trained.split():
                    break
            else:
                raise AssertionError(f"{intent}: every word of {reserved!r} is already trained on")


def test_each_novelty_axis_has_phrasings_to_measure():
    """A reserved word sits in a known frame and a reserved frame uses known words, so each cell is a
    different question and an empty one is a measurement the probe reports as if it had run."""
    counted = collections.Counter(advisory.novelty(intent, p)
                                  for intent in advisory.INTENTS
                                  for p in range(advisory.phrasings(intent)))
    for axis in (advisory.TRAINED, "word", "frame", "word and frame"):
        assert counted[axis] > 0, f"nothing is novel in {axis} alone"


def test_no_question_puts_the_wrong_article_before_a_vowel():
    """Frames and words are combined rather than written out, so a mismatch like "a ugly stretch" or
    "an rough run" appears only once they meet and reads as carelessness to whoever is asking."""
    for intent in advisory.INTENTS:
        for phrasing in range(advisory.phrasings(intent)):
            asked = advisory.ask(intent, "AAPL", phrasing)
            assert not re.search(r"\ba (?=[aeiou])", asked), asked
            assert not re.search(r"\ban (?![aeiou])", asked), asked


def test_the_held_out_phrasing_reuses_an_answer_form_training_saw():
    """The unseen split has to differ from training in the question alone. If it also brought an
    answer form training never saw, a loss gap there would measure wording, not generalisation."""
    bars = snapshot()
    facts = advisory.snapshot(bars, len(bars) - 1)
    shown = advisory.as_text(facts)
    shown.update(advisory.risk_outlook([0.21, 0.55, 0.24]))
    for intent in advisory.INTENTS:
        answers = [advisory.row(intent, "AAPL", facts, shown, p)[1]
                   for p in range(advisory.phrasings(intent))]
        trained = {a for p, a in enumerate(answers) if not advisory.is_held_out(intent, p)}
        for phrasing, answer in enumerate(answers):
            if advisory.is_held_out(intent, phrasing):
                assert answer in trained, f"{intent} phrasing {phrasing} brings an unseen answer"


def test_chat_register_never_touches_the_ticker():
    """The corruption that would teach invention: the answer quotes the ticker back, so a question
    asking about a symbol with a letter dropped would train the model to copy one nobody holds. The
    ticker must be five letters here — a four letter one is too short for the misspeller to pick, so
    AAPL passes this test whether the code is right or not, and 53 tickers in the data are longer."""
    rng = np.random.default_rng(0)
    for intent in advisory.INTENTS:
        for phrasing in range(advisory.phrasings(intent)):
            for _ in range(20):
                assert "GOOGL" in advisory.ask(intent, "GOOGL", phrasing, rng)


def test_chat_register_adds_no_digit_to_a_question():
    """Every integer in a question has to be one the evidence carries. A stray digit from a typo is an
    unsupported figure in the prompt itself, and the guardrail only ever reads the answer."""
    rng = np.random.default_rng(1)
    for intent in advisory.INTENTS:
        for phrasing in range(advisory.phrasings(intent)):
            clean = set(re.findall(r"\d", advisory.ask(intent, "AAPL", phrasing)))
            for _ in range(20):
                messy = advisory.ask(intent, "AAPL", phrasing, rng)
                assert set(re.findall(r"\d", messy)) <= clean, messy


def test_chat_register_actually_rewrites_most_questions():
    """A no-op augmentation would train and measure nothing while every number still moved, which is
    the failure that looks like a result. Not all: leaving some untouched is the point of a share."""
    rng = np.random.default_rng(2)
    asked = [advisory.ask("performance", "AAPL", 0, rng) for _ in range(100)]
    clean = advisory.ask("performance", "AAPL", 0)
    assert sum(one != clean for one in asked) > 60


def test_filling_a_slot_answer_reproduces_the_figure_answer_exactly():
    """What makes the slot path safe to swap in: it is lossless. If filling ever drifted from the
    answer it replaces, the model would be trained to name facts that serving renders differently."""
    bars = snapshot()
    facts = advisory.snapshot(bars, len(bars) - 1)
    shown = advisory.as_text(facts)
    shown.update(advisory.risk_outlook([0.21, 0.55, 0.24]))
    slots = advisory.slot_names(shown)
    for intent in advisory.INTENTS:
        for phrasing in range(advisory.phrasings(intent)):
            _, wanted = advisory.row(intent, "AAPL", facts, shown, phrasing)
            _, named = advisory.row(intent, "AAPL", facts, slots, phrasing)
            assert advisory.fill(named, shown) == wanted, f"{intent} phrasing {phrasing}"


def test_filling_leaves_prose_that_merely_looks_like_a_field_name_alone():
    """The bug this pins: 'close to a coin toss' becoming '182.50 to a coin toss'. Any prose word
    that collides with an evidence key silently turns into a number in every served answer."""
    shown = {"close": "182.50", "outlook": "calm", "outlook_confidence": "44%"}
    assert advisory.fill("it is enclosed , outlook_confidence sure , at close .", shown) == (
        "it is enclosed , 44% sure , at 182.50 ."
    )


def test_a_slot_answer_states_no_figure_of_its_own():
    """The point of the slot path. Only the integers in the field names survive, and the evidence
    carries those, so nothing the model writes can be a fabricated figure."""
    bars = snapshot()
    facts = advisory.snapshot(bars, len(bars) - 1)
    shown = advisory.as_text(facts)
    shown.update(advisory.risk_outlook([0.21, 0.55, 0.24]))
    evidence = advisory.render_evidence("AAPL", shown)
    slots = advisory.slot_names(shown)
    for intent in advisory.INTENTS:
        for phrasing in range(advisory.phrasings(intent)):
            _, named = advisory.row(intent, "AAPL", facts, slots, phrasing)
            assert guardrails.unsupported_figures(named, evidence) == [], f"{intent}: {named}"


@pytest.mark.parametrize("ticker", ["AAPL", "M&M.NS"])
def test_no_answer_form_about_a_held_share_reads_as_naming_another(ticker):
    """Without the risk forms' words in SUBJECT_SLOTS, "for context AAPL has run" was refused as naming "context"."""
    facts = advisory.snapshot(snapshot(), 79)
    names = frozenset({"aapl", "m&m", "nykaa"})
    for close, sma_20, sma_50 in ((103.0, 102.0, 101.0), (101.0, 102.0, 103.0), (102.0, 101.0, 103.0)):
        for rsi_14 in (80.0, 20.0, 50.0):
            banded = {**facts, "close": close, "sma_20": sma_20, "sma_50": sma_50, "rsi_14": rsi_14}
            for outlook in advisory.RISK_NAMES:
                shown = {**advisory.as_text(banded), **advisory.risk_outlook([0.6 if name == outlook else 0.2
                                                                              for name in advisory.RISK_NAMES])}
                evidence = advisory.render_evidence(ticker, shown)
                for intent in advisory.INTENTS:
                    for phrasing in range(advisory.phrasings(intent)):
                        _, answer = advisory.row(intent, ticker, banded, shown, phrasing)
                        assert guardrails.other_shares(answer, evidence, names, advisory.SUBJECT_SLOTS) == [], answer


@pytest.mark.parametrize("confidence", [0.34, 0.5, 0.99])
def test_the_risk_answer_quotes_whatever_confidence_the_head_reports(confidence):
    """Serving hands the head's own probabilities straight through, so no value may go unsupported."""
    rest = (1.0 - confidence) / 2.0
    facts = advisory.snapshot(snapshot(), 79)
    shown = advisory.as_text(facts)
    shown.update(advisory.risk_outlook([rest, confidence, rest]))
    evidence = advisory.render_evidence("AAPL", shown)
    for phrasing in range(advisory.phrasings("risk")):
        _, answer = advisory.row("risk", "AAPL", facts, shown, phrasing)
        assert guardrails.unsupported_figures(answer, evidence) == []


@pytest.mark.parametrize("close,sma_20,sma_50,words", [
    (3, 2, 1, "above both"), (3, 1, 2, "above both"), (1, 2, 3, "below both"), (1, 3, 2, "below both"),
    (2, 1, 3, "between"), (2, 3, 1, "between")])
def test_the_performance_answer_places_the_close_against_each_average(close, sma_20, sma_50, words):
    """With the averages in the other order it said "between": false in 7,471 of the 15,904 "between" training rows."""
    shown = {"close": "3.00", "return_20d": "+1.0%", "return_5d": "+0.5%"}
    for phrasing in range(advisory.phrasings("performance")):
        _, answer = advisory.row("performance", "AAPL", dict(close=close, sma_20=sma_20, sma_50=sma_50), shown, phrasing)
        assert f" {words} its 20 and 50 day averages " in answer


def test_the_performance_answer_says_the_trend_the_servers_guard_reads(monkeypatch):
    """The gold and the guard each coded the rule, and agreed only while both were edited alike."""
    monkeypatch.setattr(guardrails, "trend_position", lambda close, sma_20, sma_50: "below")
    shown = {"close": "3.00", "return_20d": "+1.0%", "return_5d": "+0.5%"}
    _, answer = advisory.row("performance", "AAPL", dict(close=3, sma_20=2, sma_50=1), shown, 0)
    assert " below both its 20 and 50 day averages " in answer


def test_the_performance_answer_reads_the_trend_off_the_figures_its_evidence_shows():
    """From unrounded figures, a close of 100.004 beside averages of 100.001 and 100.00 said "above both" while the
    evidence showed all three as 100.00; the guard refused it, and 32 rebuilt rows were dropped for it."""
    facts = dict(close=100.004, sma_20=100.001, sma_50=100.0)
    shown = {**advisory.as_text(facts), "return_20d": "+1.0%", "return_5d": "+0.5%"}
    _, answer = advisory.row("performance", "AAPL", facts, shown, 0)
    assert guardrails.wrong_trend(answer, advisory.render_evidence("AAPL", shown)) == []


STOCK_FACTS = ["Stock: Reliance Industries Limited (RELIANCE.NS), NSE, INR; last close 1,339.70 on 2026-11-13.",
               stock_qa.DEFINITIONS,
               "[technical] RSI(14) is 70.8 — momentum strong but statistically overbought. (source computed:rsi(14); 2026-11-13).",
               "Price change to 2026-11-13: 1 month +14.83%, 1 year -10.04% (computed from yahoo daily bars)."]
STOCK_ANSWER = ("As of 2026-11-13, RSI(14) for Reliance Industries (RELIANCE) is 70.8. RSI is a 0 to 100 gauge of recent "
                "gains against recent losses, and above 70 is often called overbought. The price rose 14.83% over 1 month "
                "but fell 10.04% over 1 year, so the short rise sits inside a longer fall. A high reading is not by itself "
                "a reason to expect a drop. What to watch: whether the 1-year fall keeps easing.")
SHORT = "As of 2026-11-13, Reliance Industries (RELIANCE) has an RSI(14) of 70.8 after a 14.83% one-month rise."
OTHERS = {"TATASTEEL": ["TATASTEEL", "Tata Steel"]}


def stock_check(answer=STOCK_ANSWER, short=SHORT):
    return stock_qa.check(answer, short, "What does RSI say about RELIANCE?", STOCK_FACTS,
                          ["RELIANCE", "Reliance Industries"], OTHERS)


def test_a_stock_answer_quoting_its_facts_passes():
    assert stock_check() == []


def test_a_rounded_stock_figure_is_caught():
    """Rounding reads naturally, and is exactly what the served guard refuses: a model taught "about 71" for
    70.8 would have its answers withheld."""
    assert any("['71']" in p for p in stock_check(STOCK_ANSWER.replace("is 70.8", "is about 71")))


def test_a_stock_answer_naming_another_company_is_caught():
    """The failure measured outside the 101 trained stocks: half the answers named another company."""
    swap = lambda text: text.replace("Reliance Industries (RELIANCE)", "Tata Steel (TATASTEEL)")
    found = stock_check(swap(STOCK_ANSWER), swap(SHORT))
    assert any("never names the stock" in p for p in found) and any("TATASTEEL" in p for p in found)


def test_a_date_no_fact_states_is_caught():
    assert any("2026-11-14" in p for p in stock_check(STOCK_ANSWER.replace("As of 2026-11-13", "As of 2026-11-14")))


def test_a_row_naming_a_held_out_peer_is_a_test_row():
    """A train row that names a test stock as its peer would leak that stock's figures into training."""
    assert stock_qa.split_for(["INFY", "TCS"], "STKQ-0079", {"TCS"}, set(), set()) == "test"
    assert stock_qa.split_for(["INFY"], "STKQ-0079", {"TCS"}, {"INFY"}, set()) == "validation"
    assert stock_qa.split_for(["INFY"], "STKQ-0079", set(), set(), {"STKQ-0079"}) == "test"


def test_held_out_takes_the_same_share_of_every_group_on_every_run():
    keys = [f"S{i}" for i in range(40)]
    group = {k: "a" if i < 30 else "b" for i, k in enumerate(keys)}
    picked = stock_qa.held_out(keys, 0.1, group, "salt")
    assert picked == stock_qa.held_out(list(reversed(keys)), 0.1, group, "salt")
    assert sum(group[k] == "a" for k in picked) == 3 and sum(group[k] == "b" for k in picked) == 1


def test_a_stock_answer_with_a_double_negative_is_caught():
    """'fell -10.04%' reads to a beginner as a rise; the reviewers found it in a fifth of the first draft."""
    assert any("double negative" in p for p in stock_check(STOCK_ANSWER.replace("fell 10.04%", "fell -10.04%")))


def test_a_stock_answer_ending_in_an_order_is_caught():
    assert any("advice" in p for p in stock_check(STOCK_ANSWER + " Trim it if it rises further."))


def test_a_term_with_no_plain_meaning_is_flagged_and_an_explained_one_is_not():
    assert stock_qa.style_flags("RSI(14) is 70.8.", "q") == ["uses RSI with no plain meaning"]
    assert stock_qa.style_flags("RSI(14), a 0 to 100 gauge of recent gains against recent losses, is 70.8.", "q") == []


def analysis(price=1339.7):
    """The fields of an `AnalysisService` result that `stock_qa.facts` reads for its first lines."""
    evidence = lambda claim, source, model: {"claim": claim, "is_model_output": model,
                                            "provenance": {"source": source, "as_of": "2026-11-13"}}
    return {"symbol": "RELIANCE", "market": "NSE", "currency": "INR", "name": "Reliance Industries Limited",
            "analysis_timestamp": "2026-11-15T18:00:00", "quality": {"last_date": "2026-11-13"}, "results": {
                "technical": {"details": {"price": price, "as_of": "2026-11-13"},
                              "evidence": [evidence("RSI is 70.8 — momentum strong.", "computed:rsi(14)", False)]},
                "forecast": {"evidence": [evidence("80% expected 5d range 1,250.89–1,434.81.", "model:forecast", True)]},
                "fundamental": {"details": {"profile": {"summary": " ".join(["word"] * 30) + ". A second sentence."}}}}}


def test_the_facts_open_with_the_stock_its_full_ticker_and_its_last_close():
    """The guard reads "reliance.ns" in an answer as another share unless the passage holds it, and the close and its
    date were a separate line the window often left out."""
    lines = stock_qa.facts(analysis())
    assert lines[0] == "Stock: Reliance Industries Limited (RELIANCE.NS), NSE, INR; last close 1,339.70 on 2026-11-13."
    assert not any(line.startswith("Last close") for line in lines)


def test_a_facts_line_says_its_area_says_rsi_14_and_marks_model_output():
    lines = stock_qa.facts(analysis())
    assert "[technical] RSI(14) is 70.8 — momentum strong. (source computed:rsi(14); 2026-11-13)." in lines
    assert any(line.startswith("[forecast, model output] 80% expected") for line in lines)


def test_the_company_profile_is_its_first_sentence_cut_to_fit_the_window():
    """Two sentences of a company's profile took a third of the window and never fitted beside a figure."""
    assert "[profile] " + " ".join(["word"] * stock_qa.PROFILE_WORDS) + "..." in stock_qa.facts(analysis())


def test_a_valuation_line_carries_the_day_of_the_close_it_rests_on_not_the_day_it_was_fetched():
    """Every training row's valuation lines said "snapshot retrieved 2026-11-15", and ALFA learned to write that date."""
    fetched = "snapshot retrieved 2026-11-15"
    snap = analysis()
    fundamental = snap["results"]["fundamental"]
    fundamental["details"]["profile"]["market_cap"] = 2.0e12
    fundamental["details"]["metrics"] = {"forward_pe": {"value": 18.14, "unit": "x", "source": "yahoo", "period": fetched}}
    fundamental["evidence"] = [{"claim": "P/E is 22.4.", "provenance": {"source": "yahoo", "period": fetched}}]
    lines = stock_qa.facts(snap)
    assert {"Market value 200,000.00 crore INR (yahoo, as of 2026-11-13).",
            "[fundamental] Forward P/E 18.14x (source yahoo; 2026-11-13).",
            "[fundamental] P/E is 22.4. (source yahoo; 2026-11-13)."} <= set(lines)
    assert not any("retrieved" in line for line in lines)


@pytest.mark.parametrize("summary,profile", [
    ("Apple Inc. designs phones. It also sells services.", "Apple Inc. designs phones."),
    ("JPMorgan Chase & Co. operates as a bank. It lends.", "JPMorgan Chase & Co. operates as a bank."),
    ("Dr. Reddy's Laboratories Limited makes medicines. It sells them.", "Dr. Reddy's Laboratories Limited makes medicines."),
    ("ASML Holding N.V. makes lithography systems. It sells them.", "ASML Holding N.V. makes lithography systems.")],
    ids=["Inc", "Co", "Dr", "initials"])
def test_the_company_profile_is_not_cut_at_the_period_of_a_name(summary, profile):
    """Cut at the first period, "what does Apple do" was answered "Apple Inc." and DRREDDY's profile was "Dr."."""
    snap = analysis()
    snap["results"]["fundamental"]["details"]["profile"]["summary"] = summary
    assert "[profile] " + profile in stock_qa.facts(snap)


def test_a_profile_yahoo_does_not_give_is_left_out_not_shown_as_its_placeholder():
    snap = analysis()
    snap["results"]["fundamental"]["details"]["profile"]["summary"] = "Data unavailable"
    assert not any(line.startswith("[profile]") for line in stock_qa.facts(snap))


def test_statements_in_another_currency_than_the_price_are_said_in_plain_words():
    """INFY's free cash flow and cash read in USD beside a close in INR, and nothing said why."""
    snap = analysis()
    snap["results"]["fundamental"]["details"].update(statement_currency="USD", price_currency="INR")
    assert "[fundamental] Statements are in USD; the price is in INR." in stock_qa.facts(snap)


def test_a_p_e_over_the_last_twelve_months_says_it_is_not_of_the_financial_year_beside_it():
    """Yahoo's trailing P/E sat among FY margins as if of the same year: ESTER's 42.1 beside a loss-making year."""
    snap = analysis()
    snap["results"]["fundamental"]["evidence"] = [{"claim": "P/E is 42.1 (absolute bands; not sector-adjusted).",
                                                   "provenance": {"source": "yahoo:info.trailingPE",
                                                                  "period": "snapshot retrieved 2026-11-15"}}]
    assert ("[fundamental] P/E is 42.1 (absolute bands; not sector-adjusted). It is over the last twelve months, not "
            "the financial year the yearly figures are for. (source yahoo:info.trailingPE; 2026-11-13).") \
        in stock_qa.facts(snap)


def test_picked_facts_lead_with_the_stock_and_fit_the_window():
    lines = STOCK_FACTS
    score = lambda q: [0.0, 0.1, 0.9, 0.5]
    picked = stock_qa.pick_facts("q", lines, score, lambda t: len(t.split()), budget=30)
    assert picked[0].startswith("Stock: Reliance Industries Limited (RELIANCE.NS)")
    assert picked[1] == "[technical] RSI(14) is 70.8 — momentum strong but statistically overbought. (2026-11-13)."
    assert sum(len(p.split()) for p in picked) <= 30 and stock_qa.DEFINITIONS not in picked


def test_a_facts_passage_joins_the_picked_facts_in_one_layout():
    passage = stock_qa.facts_passage("q", STOCK_FACTS, lambda q: [0.0, 0.1, 0.9, 0.5], lambda t: len(t.split()), 30)
    assert passage.startswith("Stock: Reliance") and stock_qa.FACT_SEPARATOR + "[technical] RSI(14) is 70.8" in passage


def overlap(texts):
    """A stand-in encoder: how many words each text shares with the question."""
    words = lambda text: set(re.findall(r"[a-z]+", text.lower()))
    return lambda question: [float(len(words(text) & words(question))) for text in texts]


def test_a_headline_does_not_outrank_a_figure_by_repeating_the_stock_name():
    """Headlines repeat the company's name, and so did the questions: they won the window over the figures asked for."""
    lines = [STOCK_FACTS[0], '[news_sentiment, model output] "Reliance Industries Limited: Reliance Industries ends higher"',
             "[fundamental] P/E is 22.4."]
    headline, figure = stock_qa.facts_scorer(lines, overlap)("Is Reliance Industries (RELIANCE) cheap on its P/E?")[1:]
    assert headline == 0.0 and figure > 0.0


def test_a_line_of_the_area_the_question_names_gets_the_domain_prior():
    lines = [STOCK_FACTS[0], "[technical] a line", "[fundamental] a line", "Price change to 2026-11-13: a line"]
    score = stock_qa.facts_scorer(lines, lambda texts: lambda question: [0.0] * len(texts))
    assert score("is it expensive next to its peers?") == [0.0, 0.0, stock_qa.DOMAIN_PRIOR, 0.0]


GROUNDED = SHORT
LEFT_OUT = SHORT + " Readings above 70 are often called overbought."
HELD = {"test_stocks": ["TCS"], "validation_stocks": [], "test_templates": ["STKQ-0007"]}
NAMES = frozenset({"tata steel", "reliance industries"})


def frozen_row(short=GROUNDED, split="train", template="STKQ-0079", kind="single_turn"):
    return {"id": "STKQ-0079-RELIANCE", "split": split, "stocks": ["RELIANCE"], "template_id": template,
            "interaction_type": kind, "company": "Reliance Industries",
            "question": "What does RSI say about RELIANCE?", "facts": STOCK_FACTS, "short_answer": short,
            "answer": STOCK_ANSWER, "prior_answer": "Reliance Industries (RELIANCE) looks overbought."}


def encoder(*rows):
    texts = [text for r in rows for text in r["facts"] + [r["question"], r["short_answer"], r["answer"], r["prior_answer"]]]
    return WordPiece.train(texts * 4, 2000, log=lambda *_: None).encode


def generator_rows(*rows, written=None):
    """The definitions line scores lowest, and the window holds every line but it."""
    encode = encoder(*rows)
    scores = lambda texts: lambda q: [0.0, 0.1, 0.9, 0.5][:len(texts)]
    return prepare_stock_qa.generator_rows(rows, HELD, scores, encode, NAMES, written), encode


def test_the_generator_keeps_only_rows_whose_passage_grounds_the_short_answer():
    """Only the definitions line states the 70 above which RSI is called overbought, and it does not fit the window:
    trained on, the row would teach the generator a figure it was never shown, which the server then refuses."""
    (kept, dropped), _ = generator_rows(frozen_row(), frozen_row(LEFT_OUT))
    assert len(kept["train"]) == 1 and dropped == {("train", "a figure in a fact the window left out"): 1}


def test_the_generator_drops_a_target_the_servers_share_guard_refuses_but_not_free_prose():
    """The advisory forms' slots read "data" in "our data has no" as a share; facts answers are screened without them."""
    gap = "Our data has no ESG figures for Reliance Industries (RELIANCE.NS), so we cannot rate it."
    (kept, dropped), _ = generator_rows(frozen_row(gap), frozen_row(GROUNDED + " Tata Steel reads the same."))
    assert len(kept["train"]) == 1 and dropped == {("train", "names a share the server's guard refuses"): 1}


def test_a_follow_up_never_trains_since_its_source_lacks_the_earlier_turn():
    (kept, dropped), _ = generator_rows(frozen_row(kind="multi_turn_followup"))
    assert not kept["train"] and dropped == {("train", "a follow-up, whose earlier turn the source does not hold"): 1}


def test_the_generator_target_is_the_short_answer_not_the_full_or_prior_one():
    (kept, _), encode = generator_rows(frozen_row())
    _, _, target = kept["train"][0]
    ids = [CLS_ID] + encode(GROUNDED) + [SEP_ID]
    assert list(target[:len(ids)]) == ids and (target[len(ids):] == PAD_ID).all()


def test_with_passage_answers_the_target_is_the_answer_written_from_that_passage():
    row = frozen_row()
    passage, _ = prepare_stock_qa.passage_for(row, lambda texts: lambda q: [0.0, 0.1, 0.9, 0.5], encoder(row))
    written = "Reliance Industries (RELIANCE.NS) has an RSI(14) of 70.8 as of 2026-11-13, which reads as overbought."
    (kept, _), encode = generator_rows(row, written={row["id"]: (passage, written)})
    ids = [CLS_ID] + encode(written) + [SEP_ID]
    assert list(kept["train"][0][2][:len(ids)]) == ids


def test_a_passage_answer_written_from_another_passage_stops_the_build():
    """A passage that moved after its answer was written would train the answer against facts it never quoted."""
    with pytest.raises(ValueError, match="no longer the one its answer was written from"):
        generator_rows(frozen_row(), written={"STKQ-0079-RELIANCE": ("Stock: an older passage.", GROUNDED)})


def test_a_row_of_a_test_template_reaches_stockqa_and_never_training():
    """split.json decides, not the row: a stale row marked train would leak a held-out wording into training."""
    with pytest.raises(ValueError, match="split.json puts it in test"):
        generator_rows(frozen_row(template="STKQ-0007"))
    (kept, _), _ = generator_rows(frozen_row(split="test", template="STKQ-0007"))
    assert len(kept["stockqa"]) == 1 and not kept["train"]


PASSAGE = " ; ".join(stock_qa.lean(line) for line in (STOCK_FACTS[0], STOCK_FACTS[2]))


def test_a_passage_answer_quoting_its_passage_passes_the_check():
    answer = "As of 2026-11-13, Reliance Industries (RELIANCE.NS) has an RSI(14) of 70.8, which reads as overbought."
    assert stock_qa.check_passage_answer(answer, PASSAGE, "Reliance Industries", NAMES) == []


def test_a_passage_answer_quoting_a_fact_outside_its_passage_fails_the_check():
    """The 14.83% rise is in the stock's facts but not in this passage, which is all the server checks against."""
    answer = "Reliance Industries (RELIANCE.NS) has an RSI(14) of 70.8 after a 14.83% one-month rise."
    assert any("['14.83%']" in p for p in stock_qa.check_passage_answer(answer, PASSAGE, "Reliance Industries", NAMES))


def test_a_passage_answer_naming_a_share_its_passage_does_not_fails_the_check():
    answer = "Reliance Industries (RELIANCE.NS) has an RSI(14) of 70.8, about where Tata Steel stands."
    assert any("tata steel" in p for p in stock_qa.check_passage_answer(answer, PASSAGE, "Reliance Industries", NAMES))


def test_a_passage_answer_giving_a_figure_as_another_metrics_fails_the_check():
    """The server refuses it, so a target written this way would train an answer it never serves."""
    answer = "As of 2026-11-13, Reliance Industries (RELIANCE.NS) has a beta of 70.8."
    assert any("another metric" in p for p in stock_qa.check_passage_answer(answer, PASSAGE, "Reliance Industries", NAMES))


@pytest.mark.parametrize("answer,right", [
    ("reliance industries (reliance. ns) had an rsi (14) of 70.8 on 2026 - 11 - 13 .", True),
    ("reliance industries had an rsi (14) of 71.2 on 2026 - 11 - 13 .", False),
    ("reliance industries had a beta of 70.8 on 2026 - 11 - 13 .", False),
    ("tata steel had an rsi (14) of 70.8 on 2026 - 11 - 13 .", False)],
    ids=["right", "a figure not in the passage", "a figure as another metric's", "another share"])
def test_the_facts_oracle_wants_every_figure_the_passages_under_its_own_metric_and_the_share_named(answer, right):
    from backend.models.training.fit_facts_cut import judge
    assert judge(answer, PASSAGE, "Reliance Industries")[0] == right


SLOT_NAMES = ("Reliance Industries Limited", "RELIANCE.NS")


def test_a_slot_answer_marks_the_name_the_ticker_and_each_figure_by_its_place_with_its_sign_left_as_text():
    answer = "As of 2026-11-13, Reliance Industries Limited (RELIANCE.NS) closed at 1,339.70 with an RSI(14) of 70.8."
    assert slots.to_slots(answer, PASSAGE, *SLOT_NAMES) == "As of κ 2-κ 3-κ 4, ι (χ) closed at κ 1 with an RSI(κ 5) of κ 6."


def test_a_slot_answer_is_dropped_when_it_states_a_figure_its_passage_lacks():
    assert slots.to_slots("Reliance Industries Limited has an RSI(14) of 71.2.", PASSAGE, *SLOT_NAMES) is None


def test_filling_refuses_a_figure_marker_past_the_passages_figures():
    with pytest.raises(ValueError, match="'κ 10' names no figure of the 9"):
        slots.from_slots("ι has an RSI of κ 10.", PASSAGE, *SLOT_NAMES)


def test_slot_markers_are_single_pieces_that_decode_in_the_form_filling_reads():
    tokenizer = WordPiece.load(ARTIFACTS / "tokenizer.json")
    assert all(tokenizer.encode(m) == [tokenizer.ids[m]] for m in (slots.NAME, slots.TICKER, slots.FIGURE))
    slotted = "ι (χ) closed at κ 1 with an rsi (κ 5) of κ 6."
    assert tokenizer.decode(tokenizer.encode(slotted)) == slotted


def test_every_grounded_passage_answer_reads_back_from_a_slot_form_that_leaves_no_figure_to_write():
    """Matching signed figures exactly dropped 561 of these, since an answer says "rose 14.1%" for "+14.1%"."""
    written = prepare_stock_qa.passage_answers(prepare_stock_qa.OUT / "passage_answers")
    dropped = []
    for key, (passage, answer) in written.items():
        if guardrails.unsupported_figures(answer, passage):
            continue
        name, ticker = stock_qa.STOCK_LINE.match(passage).groups()
        slotted = slots.to_slots(answer, passage, name, ticker)
        if slotted is None:
            dropped.append(key)
            continue
        assert slots.from_slots(slotted, passage, name, ticker) == answer, key
        assert not re.search(r"\d", re.sub(rf"{slots.FIGURE} \d+", "", slotted)), key
    assert written and not dropped, dropped[:5]


def test_passage_batches_keep_a_stock_whole_and_close_once_full():
    rows = [{"symbol": s} for s in "AAABBCC"]
    assert [[r["symbol"] for r in b] for b in prepare_stock_qa.batches(rows, 4)] == [list("AAABB"), list("CC")]


V3_STOCKS = [{"symbol": symbol, "market": market, "name": name} for symbol, market, name in (
    ("TCS", "NSE", "Tata Consultancy Services"), ("INFY", "NSE", "Infosys"), ("AAPL", "US", "Apple"), ("MSFT", "US", "Microsoft"))]
V3_PRICES = {s["symbol"]: f"ticker {s['symbol']} ; close 1.00" for s in V3_STOCKS}


def v3_rows(question, label="performance", split="dev", prices=V3_PRICES):
    template = {"id": "USR-001", "category": "performance", "split": split, "question": question, "label": label}
    return prepare_stock_qa.v3_rows([template], V3_STOCKS, prices, lambda row: f"facts of {row['symbol']}", HELD)


def test_a_v3_train_template_filled_with_a_frozen_test_stock_is_a_test_row():
    assert {r["symbol"]: r["split"] for r in v3_rows("how has {t} done?")} == \
        {"TCS": "test", "INFY": "train", "AAPL": "train", "MSFT": "train"}


def test_a_v3_question_naming_a_frozen_test_stock_as_its_second_share_is_a_test_row():
    """INFY's only other NSE share is TCS, so its comparison names TCS and must not train."""
    assert {r["symbol"]: r["split"] for r in v3_rows("which has done better, {t} or {t}?", "unsupported")} == \
        {"TCS": "test", "INFY": "test", "AAPL": "train", "MSFT": "train"}


def test_every_v3_row_of_a_test_template_is_a_test_row():
    assert {r["split"] for r in v3_rows("how has {t} done?", split="test")} == {"test"}


def test_a_v3_facts_row_reads_its_facts_passage_and_every_other_label_its_price_evidence():
    assert all(r["evidence"] == f"facts of {r['symbol']}" and r["evidence_kind"] == "facts"
               for r in v3_rows("what is the p/e of {t}?", "facts"))
    assert all(r["evidence"] == V3_PRICES[r["symbol"]] and r["evidence_kind"] == "advisory"
               for r in v3_rows("what is the p/e of {t}?", "unsupported"))


def test_a_v3_price_template_short_of_shares_with_price_evidence_stops_the_build():
    with pytest.raises(SystemExit, match="needs 2 NSE shares with advisory evidence; there are 1"):
        v3_rows("how has {t} done?", prices={k: v for k, v in V3_PRICES.items() if k != "TCS"})


def test_no_built_v3_train_or_validation_row_names_a_frozen_test_stock_or_uses_a_test_template():
    rows = prepare_stock_qa.read_jsonl(prepare_stock_qa.V3 / "rows.jsonl")
    held = json.loads((prepare_stock_qa.OUT / "frozen_v1" / "split.json").read_text())
    test_templates = {t["id"] for t in prepare_stock_qa.short_templates() if t["split"] == "test"}
    names = {s["symbol"]: s["name"] for s in json.loads((prepare_stock_qa.OUT / "universe.json").read_text())}
    leaked = [r["id"] for r in rows if r["split"] != "test" and (r["template_id"] in test_templates or any(
        stock_qa._names_in(symbol, text) or stock_qa._names_in(names[symbol], text)
        for symbol in held["test_stocks"] for text in (r["question"], r["evidence"])))]
    assert len(rows) == 4 * 255 and not leaked, leaked[:5]


V3_PRICE_ROW = {"id": "USR-001-RELIANCE", "company": "Reliance Industries", "evidence_kind": "advisory",
                "evidence": "ticker RELIANCE.NS ; close 1261.70 ; return_5d +5.0% ; return_20d +8.1% ; sma_20 1191.11 ; "
                            "sma_50 1234.90"}
V3_ANSWERS = ["Over the last 20 days RELIANCE.NS gained 8.1%, and it added 5.0% over the last 5. It closed at 1261.70, "
              "above its 20 day average of 1191.11 and its 50 day average of 1234.90, so the past few weeks have been a "
              "steady climb.",
              "RELIANCE.NS rose 8.1% across 20 days and 5.0% over the last 5. The close of 1261.70 sits over both the 20 "
              "day average of 1191.11 and the 50 day average of 1234.90, which says buyers have held the upper hand in "
              "recent weeks."]


def test_a_v3_answer_pair_quoting_its_price_evidence_in_two_wordings_passes():
    assert prepare_stock_qa.v3_problems(V3_PRICE_ROW, {"answers": V3_ANSWERS}, NAMES) == []


def test_a_v3_answer_written_twice_fails_as_one_wording():
    assert prepare_stock_qa.v3_problems(V3_PRICE_ROW, {"answers": V3_ANSWERS[:1] * 2}, NAMES) == [
        "the two answers share 100% of their word trigrams; reword one until they share under 50%"]


def test_a_v3_answer_stating_a_figure_its_evidence_lacks_fails():
    answer = V3_ANSWERS[0].replace("5.0%", "5.5%")
    assert any("['5.5%']" in p for p in prepare_stock_qa.v3_answer_problems("answer 1", V3_PRICE_ROW, answer, NAMES))


def test_a_v3_price_answer_naming_its_company_where_the_evidence_gives_only_the_ticker_fails():
    answer = V3_ANSWERS[0].replace("RELIANCE.NS", "Reliance Industries")
    assert "answer 1 never names the stock as its evidence does: RELIANCE.NS" in \
        prepare_stock_qa.v3_answer_problems("answer 1", V3_PRICE_ROW, answer, NAMES)


def test_a_v3_price_answer_is_read_by_the_servers_price_share_guard_not_the_facts_one():
    """The server reads a word before "has" in a price answer as a share, which the facts guard does not."""
    answer = V3_ANSWERS[0].replace("the past few weeks have", "the past month has")
    assert any(p.startswith("the server's guard reads another share in answer 1: ['month']")
               for p in prepare_stock_qa.v3_answer_problems("answer 1", V3_PRICE_ROW, answer, NAMES))


def test_a_v3_price_answer_saying_a_trend_its_averages_contradict_fails_as_the_server_refuses_it():
    """RELIANCE closed above both averages; the check passed "between", which the server's trend guard refuses."""
    answer = V3_ANSWERS[0].replace("above its 20 day average of 1191.11 and its 50 day average of 1234.90",
                                   "between its 20 and 50 day averages of 1191.11 and 1234.90")
    assert "answer 1 says the price is between its 20 and 50 day averages; its evidence says above" in \
        prepare_stock_qa.v3_answer_problems("answer 1", V3_PRICE_ROW, answer, NAMES)


V3_FACTS_ROW = {"id": "USR-059-RELIANCE", "company": "Reliance Industries", "evidence_kind": "facts", "evidence": PASSAGE}


def test_a_v3_facts_answer_giving_a_figure_as_another_metrics_fails_as_the_server_refuses_it():
    answer = ("As of 2026-11-13, Reliance Industries Limited (RELIANCE.NS) closed at 1,339.70 and had a beta of 70.8, "
              "so the share has moved far more than the market has in recent weeks and could swing hard again.")
    assert "the server's guard reads 70.8 in answer 1 as beta" in \
        prepare_stock_qa.v3_answer_problems("answer 1", V3_FACTS_ROW, answer, NAMES)


def test_a_v3_facts_answer_giving_a_term_a_meaning_of_its_own_fails_as_the_server_refuses_it():
    answer = ("As of 2026-11-13, Reliance Industries Limited (RELIANCE.NS) closed at 1,339.70 with an RSI, a gauge of "
              "how much buyers want the share, of 70.8, so recent gains have far outweighed recent losses.")
    assert "answer 1 gives rsi a meaning of its own" in \
        prepare_stock_qa.v3_answer_problems("answer 1", V3_FACTS_ROW, answer, NAMES)


def test_a_v3_facts_answer_dropping_a_thousands_separator_has_no_slot_form():
    """The figure guard compares separators away, so only the slot form catches a close the model cannot copy back."""
    row = {"id": "USR-059-RELIANCE", "company": "Reliance Industries", "evidence_kind": "facts", "evidence": PASSAGE}
    answer = ("As of 2026-11-13, Reliance Industries Limited (RELIANCE.NS) closed at 1339.70 with an RSI(14) of 70.8. "
              "That reading is high, so recent gains have far outweighed recent losses and the run looks stretched.")
    assert prepare_stock_qa.v3_answer_problems("answer 1", row, answer, NAMES) == [
        "answer 1 writes a figure otherwise than its evidence does, so it has no slot form"]


def test_facts_rows_are_repeated_to_their_share_of_scored_tokens_not_of_rows():
    """The loss is a mean over target tokens, so a facts target twice as long counts twice."""
    assert prepare_alfa_v2.oversampling(facts_tokens=100, other_tokens=920, share=0.08) == 1
    assert prepare_alfa_v2.oversampling(facts_tokens=10, other_tokens=920, share=0.08) == 8
    assert prepare_alfa_v2.scored_tokens(np.array([[CLS_ID, 7, SEP_ID, PAD_ID], [CLS_ID, SEP_ID, PAD_ID, PAD_ID]])) == 3


def encoded(tokenizer, texts, width=0):
    ids = [tokenizer.encode(text) for text in texts]
    source = np.full((len(ids), 1, width or max(map(len, ids))), PAD_ID, dtype=np.uint16)
    for row, row_ids in enumerate(ids):
        source[row, 0, :len(row_ids)] = row_ids
    return source, None, None


def test_a_wide_or_facts_train_row_naming_a_test_stock_stops_the_build_but_a_served_one_does_not():
    texts = ["how has TCS done this month ?", "how has INFY done this month ?", "how has TCSL done ?"]
    tokenizer = WordPiece.train(texts * 4, 200, log=lambda *_: None)
    served = {prepare_alfa_v2.SERVED: encoded(tokenizer, texts[:1])}
    prepare_alfa_v2.check_unseen({**served, "wide": encoded(tokenizer, texts[1:])}, ["TCS"], tokenizer)
    with pytest.raises(SystemExit, match="wide train rows name the frozen set's test stocks {'TCS': 1}"):
        prepare_alfa_v2.check_unseen({**served, "wide": encoded(tokenizer, texts)}, ["TCS"], tokenizer)


@pytest.fixture
def alfa_v2(tmp_path, monkeypatch):
    """prepare_alfa_v2 run on one-question served, wide and facts sets, with a frozen set under DATA."""
    texts = {"served": "how has TCS done ?", "wide": "how has WIPRO done ?", "facts": "what is the P/E of ITC ?"}
    tokenizer = WordPiece.train(list(texts.values()) * 4, 200, log=lambda *_: None)
    artifacts, wide, facts, frozen = (tmp_path / "artifacts", tmp_path / "wide", tmp_path / "facts",
                                      tmp_path / "data" / "stock_qa" / "frozen_v1")
    for built in (artifacts, wide, facts, frozen):
        built.mkdir(parents=True)
    for built in (artifacts, wide, facts):
        tokenizer.save(built / "tokenizer.json")
    (artifacts / "pretrained.npz").write_bytes(b"")

    def write(where, dataset, splits, text, rows):
        source, _, _ = encoded(tokenizer, [text] * rows, width=16)
        target = np.full((rows, 4), PAD_ID, dtype=np.uint16)
        target[:, :2] = [CLS_ID, SEP_ID]
        for split in splits:
            for part, array in zip(("source", "keep", "target"), (source, (source != PAD_ID).astype(np.uint8), target)):
                np.save(where / f"{dataset}_{split}_{part}.npy", array)
    write(artifacts, prepare_alfa_v2.SERVED, ("train", "validation", "unseen", "casual"), texts["served"], 3)
    write(wide, prepare_alfa_v2.SERVED, ("train", "validation", "unseen", "casual"), texts["wide"], 2)
    write(facts, prepare_stock_qa.GENERATOR_DATASET, ("train", "validation", "stockqa"), texts["facts"], 1)
    (facts / f"{prepare_stock_qa.GENERATOR_DATASET}_MANIFEST.json").write_text("{}")
    (frozen / "split.json").write_text('{"test_stocks": ["TCS"]}')
    (frozen / "MANIFEST.json").write_text("{}")
    monkeypatch.setattr(prepare_alfa_v2, "ARTIFACTS", artifacts)
    monkeypatch.setattr(prepare_alfa_v2, "DATA", tmp_path / "data")
    out = tmp_path / "out"
    prepare_alfa_v2.main(["--wide", str(wide), "--wide-stride", "34", "--wide-excluded", "X.NS.csv", "--facts", str(facts),
                          "--frozen", str(frozen), "--out", str(out)])
    return out, facts


def test_the_alfa_v2_manifest_records_the_sets_promote_reads_it_trained_on(alfa_v2):
    """The facts route turns on only when a checkpoint's record holds finance.STOCK_QA, and promote reads it here."""
    from backend.models.agent import finance
    manifest = json.loads((alfa_v2[0] / "MANIFEST.json").read_text())
    assert manifest["trained_on"] == [prepare_alfa_v2.SERVED, "wide", finance.STOCK_QA]


def test_alfa_v2_validation_leads_with_the_facts_rows(alfa_v2):
    """train_generator scores a split's first rows only; facts rows after the served ones were never scored."""
    out, facts = alfa_v2
    validation, facts_rows = np.load(out / "alfa_v2_validation_source.npy"), np.load(facts / "stock_qa_validation_source.npy")
    assert len(validation) == 4 and (validation[:1] == facts_rows).all() and not (validation[1:] == facts_rows[0]).all()


def test_a_facts_row_is_laid_out_as_the_server_lays_out_a_facts_turn():
    """The server picks a facts passage within finance's question budget and lays it out with context.assemble. Forty
    facts and a long question make the budget bind, so a row built within any other budget is laid out otherwise."""
    from backend.models.agent import finance
    from backend.models.agent.context import assemble, evidence_room
    from backend.models.core.config import ModelConfig
    lines = [STOCK_FACTS[0]] + [f"[fundamental] Measure {k} is {k}.5 (FY ending 2026-03-31)." for k in range(40)]
    question = ("Read me every measure you hold for Reliance Industries (RELIANCE) one at a time, with the period each "
                "one describes, and say which changed the most over the last year and why that matters to a holder.")
    row = {**frozen_row("Our data has no peer figures for Reliance Industries (RELIANCE.NS)."), "facts": lines,
           "question": question}
    tokenizer = WordPiece.train((lines + [row["question"], row["short_answer"]]) * 4, 2000, log=lambda *_: None)
    similarity = lambda texts: lambda question: [0.0] * len(texts)
    kept, _ = prepare_stock_qa.generator_rows([row], HELD, similarity, tokenizer.encode, NAMES)
    length = ModelConfig.max_text_length
    passage = finance.facts_passage(row["question"], lines, stock_qa.facts_scorer(lines, similarity),
                                    lambda text: len(tokenizer.encode(text)), evidence_room(length, finance.FACTS_QUESTION_TOKENS))
    served = assemble(tokenizer, row["question"], [passage], length, finance.FACTS_QUESTION_TOKENS)
    assert (kept["train"][0][0] == np.asarray(served.source[0])).all()
