"""C1: one typed question in, one answer or one named refusal out, with no network anywhere.

The serving layer's whole value is what it refuses. A wrong-intent answer passes the figure guard —
every digit in it is supported — so the tests that matter here are the five stages that stop a turn,
and the one rule that the row a served question is built into is the row training was built from.
"""

import csv
from datetime import date, timedelta

import numpy as np
import pytest

from backend.agent import Router, assemble, finance, lexical
from backend.agent.core import Agent, answered
from selfagent.data import advisory
from selfagent.data.encode import build_sources
from selfagent.learn.abstain import Abstainer
from selfagent.tokenizer.wordpiece import pretokenize

LENGTH = 128
FIRST_BAR = date(2020, 1, 1)


def price_file(directory, name, bars=advisory.BARS_NEEDED):
    """A rising series with exactly as much history as a snapshot needs, unless asked for less."""
    path = directory / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "high", "low", "close", "adj_close", "volume"])
        for day in range(bars):
            close = 100.0 + day
            writer.writerow([(FIRST_BAR + timedelta(days=day)).isoformat(),
                             close, close + 1.0, close - 1.0, close, close, 1e6])
    return path


@pytest.fixture
def market(tmp_path):
    price_file(tmp_path, "AAPL", bars=advisory.BARS_NEEDED + 5)
    price_file(tmp_path, "RELIANCE.NS")
    return finance.Market(tmp_path)


@pytest.fixture
def router():
    known = finance.examples()
    return Router(known, lexical([text for _, _, text in known], pretokenize))


class FakeTokenizer:
    """Ids in first-seen order, and a decode that returns whatever the fake model was told to write."""

    def __init__(self, written=""):
        self.written = written
        self.ids = {}

    def encode(self, text):
        return [self.ids.setdefault(word, len(self.ids) + 5) for word in pretokenize(text)]

    def decode(self, ids):
        return self.written


class FakeModel:
    """Generates one fixed answer at one fixed confidence, so the core's branches are what is tested."""

    def __init__(self, confidence=0.999):
        self.value = confidence

    def generate(self, source, keep, temperature, top_p, rng):
        return [[1, 2, 3]]

    def confidence(self, source, produced, keep):
        return np.array([self.value])


class ExplodingModel:
    """Any call means the core ran the decoder on a turn it should have stopped before."""

    def generate(self, *_):
        raise AssertionError("the decoder ran on a turn that should have stopped before it")

    def confidence(self, *_):
        raise AssertionError("the decoder ran on a turn that should have stopped before it")


class Config:
    max_text_length = LENGTH


def agent(market, router, written="", confidence=0.999, cut=0.9, gate_cut=0.01, model=None):
    return Agent(finance, market, router, Abstainer(gate_cut), FakeTokenizer(written),
                 model or FakeModel(confidence), Config, Abstainer(cut))


# --- routing ----------------------------------------------------------------

def test_a_question_sharing_no_words_with_anything_known_reports_no_margin(router):
    """Zero everywhere means the argmax is whichever example came first, which is not a decision.
    Without a zero margin here the gate has nothing to refuse on and the agent answers at random."""
    assert router.route("zzzz qqqq wwww").margin == 0.0


def test_the_margin_is_against_a_different_intent_not_the_next_example(router):
    # Every intent has several trained phrasings, so a margin against the runner-up example would be
    # near zero on exactly the questions the router is most sure about.
    route = router.route("is overbought right now ?")
    assert route.label == "overbought" and route.margin > 0.0


def test_a_router_without_examples_fails_at_construction():
    with pytest.raises(ValueError, match="labelled examples"):
        Router([], lambda text: [])


def test_the_ticker_gap_is_not_itself_the_mismatch(market, router):
    """Substituting the ticker out leaves a double space. If the query and the known phrasings were
    squeezed differently, the router would score its own training text as unfamiliar."""
    intent, phrasing, known = finance.examples()[0]
    asked = finance.ask(intent, phrasing, "AAPL")
    assert finance.without_subject(asked, "AAPL") == known
    assert router.route(finance.without_subject(asked, "AAPL")).label == intent


# --- the instrument ---------------------------------------------------------

def test_only_instruments_on_disk_resolve(market):
    assert market.resolve("how is AAPL doing ?") == "AAPL"
    assert market.resolve("how is aapl doing ?") == "AAPL"
    assert market.resolve("how is TSLA doing ?") is None


def test_an_indian_ticker_resolves_without_its_suffix(market):
    # Nobody types the exchange suffix, and inventing one for a name we hold no file for would serve
    # a snapshot of the wrong instrument.
    assert market.resolve("what about RELIANCE ?") == "RELIANCE.NS"


def test_a_snapshot_reads_no_bar_after_the_one_asked_for(market):
    """The leakage rule the training snapshots obey. A snapshot that read one bar past its as-of date
    would quote a price nobody could have known, and every backtest built on it would be worthless."""
    _, latest, last = market.snapshot("AAPL")
    wanted = (FIRST_BAR + timedelta(days=advisory.BARS_NEEDED - 1)).isoformat()
    _, earlier, taken_at = market.snapshot("AAPL", as_of=wanted)
    assert taken_at == wanted < last
    assert earlier["close"] != latest["close"]


def test_an_as_of_before_the_first_bar_is_an_error(market):
    with pytest.raises(ValueError, match="earliest on file"):
        market.snapshot("AAPL", as_of="1999-01-01")


def test_too_little_history_is_an_error_not_a_shorter_window(tmp_path):
    """Computing a 60 day drawdown off 30 bars produces a figure, and the guard would pass it."""
    price_file(tmp_path, "SHORT", bars=advisory.BARS_NEEDED - 1)
    with pytest.raises(ValueError, match="snapshot needs"):
        finance.Market(tmp_path).snapshot("SHORT")


# --- the row the decoder reads ---------------------------------------------

def test_a_served_row_is_laid_out_exactly_as_a_training_row(market):
    """The only reason this module exists. A served row built with a different cap or separator is a
    different input distribution, and the model would answer worse for a reason no metric names."""
    tokenizer = FakeTokenizer()
    evidence, _, _ = market.snapshot("AAPL")
    asked = finance.ask("performance", advisory.trained_phrasings("performance")[0], "AAPL")
    context = assemble(tokenizer, asked, [evidence], LENGTH, finance.QUESTION_TOKENS)
    source, keep = build_sources(
        tokenizer.encode(asked)[: finance.QUESTION_TOKENS], [tokenizer.encode(evidence)], 1, LENGTH)
    assert np.array_equal(context.source[0], source) and np.array_equal(context.keep[0], keep)


def test_evidence_too_long_for_the_window_is_refused_not_truncated():
    # A cut passage drops a figure the answer quotes, and then the guard refuses every answer for a
    # reason that looks like the model's fault.
    with pytest.raises(ValueError, match="exceeds"):
        assemble(FakeTokenizer(), "how is it doing ?", [" ".join(["word"] * LENGTH)], LENGTH,
                 finance.QUESTION_TOKENS)


# --- the five ways a turn stops --------------------------------------------

def test_no_instrument_named_stops_before_the_decoder(market, router):
    turn = agent(market, router, model=ExplodingModel()).answer("how is TSLA doing ?")
    assert turn.because == "no subject" and not turn.spoke
    assert turn.served == finance.NO_SUBJECT


def test_a_thin_routing_margin_stops_before_the_decoder(market, router):
    turn = agent(market, router, gate_cut=1e9, model=ExplodingModel()).answer("how is AAPL doing ?")
    assert turn.because == "unclear question" and turn.served == finance.UNKNOWN_QUESTION


def test_an_intent_whose_figures_are_missing_refuses_rather_than_inventing(market, router):
    """The week-ahead call comes from the price head, which is not loaded here. Absent, the evidence
    does not carry it, and the one intent that quotes it must refuse instead of writing a number."""
    turn = agent(market, router, model=ExplodingModel()).answer(
        "how risky is AAPL over the next week ?")
    assert turn.intent == "risk" and turn.because.startswith("missing outlook")
    assert turn.served == finance.REFUSAL


def test_an_unsupported_figure_refuses_even_at_full_confidence(market, router):
    turn = agent(market, router, written="it trades at 4242.42 .", confidence=1.0).answer(
        "how is AAPL doing ?")
    assert turn.unsupported == ["4242.42"] and turn.because == "unsupported figure"
    assert turn.served == finance.REFUSAL and turn.answer == "it trades at 4242.42 ."


def test_a_confidence_under_the_cut_refuses_a_supported_answer(market, router):
    turn = agent(market, router, written="it is doing fine .", confidence=0.5, cut=0.9).answer(
        "how is AAPL doing ?")
    assert turn.because == "low confidence" and turn.served == finance.REFUSAL
    assert turn.answer == "it is doing fine ."


def test_a_supported_confident_answer_is_served(market, router):
    turn = agent(market, router, written="it is doing fine .", confidence=0.99, cut=0.9).answer(
        "how is AAPL doing ?")
    assert turn.spoke and turn.because == "" and turn.served == "it is doing fine ."
    assert answered([turn]) == (1, 1)


def test_the_decoder_is_given_the_trained_wording_not_the_users(market, router):
    """The measured payoff: 90.5% exact on wording the decoder trained on against 28.5% on wording it
    did not. The user's words choose the question; the model only ever reads words it has seen."""
    typed = "hows aapl been doin lately"
    turn = agent(market, router, written="it is doing fine .").answer(typed)
    assert turn.question == typed and turn.asked != typed
    assert finance.without_subject(turn.asked, turn.ticker) in [
        text for _, _, text in finance.examples()]


def test_a_stated_chance_is_absent_until_a_calibrator_is_fitted(market, router):
    turn = agent(market, router, written="it is doing fine .").answer("how is AAPL doing ?")
    assert turn.stated is None
