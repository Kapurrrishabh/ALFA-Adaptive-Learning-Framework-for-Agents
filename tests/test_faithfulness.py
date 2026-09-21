"""S14/S15: every figure an answer states is one the evidence states, or the answer is a refusal."""

import collections
import re

import numpy as np
import pytest

from selfagent.agent import guardrails
from selfagent.data import advisory

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


def test_a_refusal_has_nothing_to_support():
    assert guardrails.unsupported_figures(advisory.UNSUPPORTED, "") == []


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
