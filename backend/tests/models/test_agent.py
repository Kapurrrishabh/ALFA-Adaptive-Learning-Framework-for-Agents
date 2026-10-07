"""C1: one typed question in, one answer or one named refusal out, with no network anywhere.

The serving layer's whole value is what it refuses. A wrong-intent answer passes the figure guard —
every digit in it is supported — so the tests that matter here are the five stages that stop a turn,
and the one rule that the row a served question is built into is the row training was built from.
"""

import csv
import json
from datetime import date, timedelta

import numpy as np
import pytest

from backend.models import serving as models
from backend.models.agent import PARAPHRASED, QUOTED, Reference, Router, assemble, combined, finance, lexical
from backend.models.agent.core import Agent, answered
from backend.knowledge_base.retrieval import Chunk, Hybrid
from backend.models.core import pretrained
from backend.models.core.config import ModelConfig
from backend.models.data import advisory, prices, returns
from backend.models.data.encode import build_sources
from backend.models.data.qa_pairs import PASSAGES, QUESTION_TOKENS
from backend.models.networks.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P
from backend.models.learning import Calibrator
from backend.models.learning.abstain import Abstainer
from backend.models.networks import PriceWindowClassifier, ReturnGenerator
from backend.models.core.tokenizer.wordpiece import pretokenize

LENGTH = 128
FIRST_BAR = date(2020, 1, 1)

# Small enough to build in a test, and 6 channels because that is what the feature builder produces:
# five indicators plus the window's own logged scale.
OUTLOOK = ModelConfig(price_channels=6, price_window=32, dim=32, num_heads=2, ffn_dim=64, price_layers=1)

# Distinct rows, so which bucket was read is visible in what comes back.
TABLE = ((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7))

# What an artifact records, for the advisors these tests build by hand instead of loading.
MEASURED = {"horizon": 5, "edges": [0.011, 0.019]}


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
def registered(tmp_path, market):
    """The same instruments, with SEC's table saying which of them files there.

    AAPL is a registrant and RELIANCE.NS is not, which is the real split -- 50 of our 101 can be asked for
    by number and the other 51 cannot be asked for at all -- and it is what decides the order of the arms.
    """
    table = tmp_path / "company_tickers.json"
    table.write_text(json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}))
    return finance.Market(tmp_path, symbols=table)


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
        # Its own words back when the fake was told to write nothing, because a cut passage decodes ids
        # that came from real text rather than from the model.
        if self.written:
            return self.written
        words = {value: word for word, value in self.ids.items()}
        return " ".join(words[one] for one in ids)


class FakeModel:
    """Generates one fixed answer at one fixed confidence, so the core's branches are what is tested."""

    def __init__(self, confidence=0.999):
        self.value = confidence

    def generate(self, source, keep, temperature, top_p, rng):
        return [[1, 2, 3]]

    def confidence(self, source, produced, keep):
        return np.array([self.value])


class FlatEncoder:
    """Reads every text as the same vector, so the cosine half of `combined` carries no information.

    What is left is the term-coverage half alone, which is the part being tested.
    """

    def text(self, ids, keep):
        return type("States", (), {"data": np.ones((len(ids), ids.shape[1], 4))})()


class ExplodingModel:
    """Any call means the core ran the decoder on a turn it should have stopped before."""

    def generate(self, *_):
        raise AssertionError("the decoder ran on a turn that should have stopped before it")

    def confidence(self, *_):
        raise AssertionError("the decoder ran on a turn that should have stopped before it")


class FakeLive:
    """Fixed passages in place of a search, so no test here needs a network to reach the reader."""

    def __init__(self, fetched):
        self.fetched = fetched

    def passages(self, query, terms, as_of=None, subject=None):
        return self.fetched


class ExplodingLive:
    """Any call means a question the stored passages answered went to the network anyway."""

    def passages(self, *_, **__):
        raise AssertionError("a live search ran for a question the stored corpus answered")


class RefusedLive:
    """A search that could not be run at all, which is what every source being blocked looks like.

    Raised as the plain builtin the searcher's own errors derive from, so this test needs no import from
    `backend/live/` -- the agent must not depend on the package that needs `requests`.
    """

    def passages(self, *_, **__):
        raise RuntimeError("sec_edgar: 403 from efts.sec.gov")


class Config:
    max_text_length = LENGTH


def agent(market, router, written="", confidence=0.999, cut=0.9, gate_cut=0.01, model=None,
          reference=None):
    return Agent(finance, market, router, Abstainer(gate_cut), FakeTokenizer(written),
                 model or FakeModel(confidence), Config, Abstainer(cut), reference=reference)


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


def test_unknown_query_words_cost_the_combined_scorer_its_margin():
    """The defect `combined` divides out: raw BM25 charges nothing for words it does not know.

    That is why its margin serves a poem as a performance question. Dividing by the query's own total
    term weight makes an unrecognised word cost the score, which is the only reason the summed scorer
    can share a gate with the cosine.
    """
    known = finance.examples()
    texts = [text for _, _, text in known]
    clean = "how volatile is it at the moment ?"
    padded = f"{clean} zzzz qqqq wwww vvvv yyyy xxxx"

    raw = Router(known, lexical(texts, pretokenize))
    assert raw.route(padded).margin == raw.route(clean).margin

    scored = Router(known, combined(texts, FlatEncoder(), FakeTokenizer(), LENGTH))
    assert scored.route(padded).margin < 0.5 * scored.route(clean).margin
    # Nothing recognised at all still has to report no margin, or the argmax is whatever came first.
    assert scored.route("zzzz qqqq wwww").margin == 0.0


def test_every_router_example_rewrites_into_a_phrasing_the_decoder_trained_on():
    """The invariant the whole rewrite rests on, and it fails silently: a paraphrase keyed to a
    held-out phrasing would hand the decoder wording it never saw, and exact match would fall from
    90.5% to 28.5% with nothing in the serving path saying so."""
    for intent, phrasing, _ in finance.examples():
        assert not advisory.is_held_out(intent, phrasing), f"{intent} example keyed to {phrasing}"


def test_the_shapes_the_router_is_measured_on_are_not_in_its_pool():
    """`shape` measures a family the pool does not carry, so a paraphrase leaking into both would turn
    that measurement into a lookup of itself."""
    pool = {text for _, _, text in finance.examples()}
    for intent in advisory.INTENTS:
        for frame in advisory.held_out_paraphrases(intent):
            assert finance.without_subject(frame.format(t="AAPL"), "AAPL") not in pool


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


def test_a_cut_passage_reports_the_part_the_model_read_as_its_evidence():
    """What `cut` is for, and the failure it avoids. The reference index is chunked at the window that
    finds answers, which is wider than this decoder's slot, so its passages have to be cut somewhere.
    `build_sources` would cut them anyway and leave the evidence line whole -- and a figure checked
    against text the model never read is not checked."""
    tokenizer = FakeTokenizer()
    room = LENGTH - finance.QUESTION_TOKENS - 3
    words = [f"{first}{second}" for first in "abcdefghij" for second in "abcdefghijklm"]
    context = assemble(tokenizer, "how is it doing ?", [" ".join(words)], LENGTH,
                       finance.QUESTION_TOKENS, cut=True)
    assert context.evidence == " ".join(words[:room])


# --- the five ways a turn stops --------------------------------------------

def test_no_instrument_named_stops_before_the_decoder(market, router):
    turn = agent(market, router, model=ExplodingModel()).answer("how is TSLA doing ?")
    assert turn.because == "no subject" and not turn.spoke
    assert turn.served == finance.NO_SUBJECT
    # Empty, not None: a caller storing this turn has columns that refuse a null.
    assert (turn.ticker, turn.intent, turn.asked) == ("", "", "")


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


# --- reading an answer out of the documents ---------------------------------

PASSAGE = "A stop loss order sells a holding automatically once its price falls to a level you set."


def reference(written, confidence=0.5, paraphrase=False, model=None, live=None):
    """The retrieval path over a five-chunk index, with the decoder writing one fixed answer.

    Its own tokenizer, not the advisory one: what the two decoders write is what these tests vary, and a
    shared fake would make one of the two answers stand for both. One chunk names AAPL and none names
    RELIANCE.NS, which is what makes the gate on the subject observable in either direction.
    """
    held = [Chunk(PASSAGE, [], "d0", "2021-06-01"),
            Chunk("The committee voted to hold the target range at this meeting.", [], "d1", "2021-06-02"),
            Chunk("Dividends are paid out of retained earnings after tax.", [], "d2", "2021-06-03"),
            Chunk("A limit order buys only at the price you name or better.", [], "d3", "2021-06-04"),
            Chunk("AAPL holders often set one below the last close.", [], "d4", "2021-06-05")]
    return Reference(Hybrid(held, pretokenize), FakeTokenizer(written),
                     model or FakeModel(confidence), Config, QUESTION_TOKENS, PASSAGES,
                     ANSWER_TEMPERATURE, ANSWER_TOP_P, paraphrase=paraphrase, live=live)


def test_a_question_naming_no_instrument_is_answered_from_the_documents(market, router):
    """The point of the path, and what it serves by default: the source's own sentence, cited. "no
    subject" means the price snapshot cannot answer it, not that we hold nothing on it. An exploding
    decoder is what pins the other half of the claim -- quoting does not run the model at all, so there
    is no confidence to report."""
    turn = agent(market, router, reference=reference("never written", model=ExplodingModel())).answer(
        "what is a stop loss order ?")
    assert turn.spoke and turn.served == PASSAGE and turn.intent == f"reference {QUOTED}"
    assert turn.asked == turn.question and turn.ticker == ""
    assert PASSAGE in turn.evidence and "published 2021-06-01" in turn.evidence
    assert turn.confidence != turn.confidence and turn.stated is None


def test_paraphrasing_serves_the_models_words_rather_than_the_passage(market, router):
    """The branch the --paraphrase flag exists for. The generator's behaviour has to stay demonstrable
    next to the decision to stop serving it, so asking for it has to actually change what is served."""
    written = "a stop loss sells your holding once the price falls to a level you set ."
    turn = agent(market, router, reference=reference(written, paraphrase=True)).answer(
        "what is a stop loss order ?")
    assert turn.spoke and turn.served == written and turn.intent == f"reference {PARAPHRASED}"


def test_a_paraphrase_stating_a_figure_the_passages_lack_is_replaced_by_the_passage(market, router):
    """Why paraphrasing is not served by default, and what still guards it when it is asked for: 57.1% of
    the figures that generator writes are unsupported, and a quote cannot state a figure its source does
    not."""
    turn = agent(market, router, reference=reference("it triggers at 4242.42 .", paraphrase=True)).answer(
        "what is a stop loss order ?")
    assert turn.spoke and turn.served == PASSAGE and turn.intent == f"reference {QUOTED}"
    assert turn.unsupported == ["4242.42"] and turn.answer == "it triggers at 4242.42 ."


def test_a_reference_turn_states_no_chance_of_being_right(market, router):
    """A confidence is reported but never calibrated here. The calibrator maps one to a correctness rate
    measured on the advisory generator, so running it on these weights would quote a fitted likelihood."""
    turn = agent(market, router,
                 reference=reference("a stop loss sells your holding .", paraphrase=True)).answer(
        "what is a stop loss order ?")
    assert turn.stated is None and turn.confidence == 0.5


def test_a_question_the_router_cannot_place_keeps_its_ticker_and_margin(market, router):
    """The second of the two places a reference answer is tried. The turn still records what routing
    found, because a thin margin is why this answer came from the documents rather than the figures."""
    turn = agent(market, router, gate_cut=1e9,
                 reference=reference("never written", model=ExplodingModel())).answer(
        "should i set a stop loss on AAPL ?")
    assert turn.spoke and turn.intent == f"reference {QUOTED}"
    assert turn.ticker == "AAPL" and 0.0 <= turn.margin < 1e9


def test_passages_that_never_name_the_subject_are_not_read_as_an_answer_about_it(market, router):
    """The gate, and the failure it was added for: a question about one instrument was answered out of a
    passage about something else, because BM25 ranks and cannot abstain. Nothing here mentions RELIANCE.NS,
    so the refusal routing already had is what comes back rather than the best of four unrelated passages."""
    turn = agent(market, router, gate_cut=1e9, model=ExplodingModel(),
                 reference=reference("never written", model=ExplodingModel())).answer(
        "should i set a stop loss on RELIANCE.NS ?")
    assert not turn.spoke and turn.because == "unclear question"
    assert turn.served == finance.UNKNOWN_QUESTION and turn.ticker == "RELIANCE.NS"


def test_the_subject_carried_from_the_conversation_is_what_it_is_looked_up_under(market, router):
    """"How is it doing" is a whole question to a person and no question at all to a router. The carried
    subject is resolved against the price files like any other, so it can name nothing a user could not."""
    turn = agent(market, router, written="it is doing fine .").answer("hows it doing ?", subject="AAPL")
    assert turn.ticker == "AAPL" and turn.spoke


def test_a_fetched_passage_is_read_by_the_same_reader_and_the_same_figure_guard(market, router):
    """What the live search is for, and the rule that keeps it honest: a fetched passage goes through the
    reader the stored ones go through, so the quote rule and the figure guard apply to it unchanged. The URL
    it came from travels in the evidence, because a quote the user cannot follow is worth less than none.

    It is served without passing the subject gate, and that is the point of the two arms: this passage says
    "Reliance Industries" and RELIANCE.NS has no registered name here, so a word check would refuse the one
    source that can answer for it. What stands in for the check is that the source was asked for this
    instrument by identity rather than found by overlap."""
    fetched = Chunk("Reliance Industries said it had commissioned the unit.", [],
                    "https://en.wikipedia.org/wiki/Reliance_Industries", "2026-09-01")
    turn = agent(market, router, gate_cut=1e9,
                 reference=reference("it commissioned 4242.42 of them .", paraphrase=True,
                                     live=FakeLive([fetched]))).answer(
        "what has RELIANCE.NS announced ?")
    assert turn.spoke and turn.served == fetched.text and turn.intent == f"reference {QUOTED}"
    assert turn.unsupported == ["4242.42"] and fetched.document in turn.evidence


def test_the_passage_that_is_quoted_is_one_that_names_the_subject_not_the_best_ranked(market, router):
    """Why the gate filters rather than just tests. "stop loss" ranks the definition first and only the last
    chunk says AAPL, so passing the set through whole would quote the definition and cite its document --
    an answer attributed to a passage that is not about the instrument asked about."""
    turn = agent(market, router, gate_cut=1e9,
                 reference=reference("never written", model=ExplodingModel())).answer(
        "should i set a stop loss on AAPL ?")
    assert turn.served == "AAPL holders often set one below the last close."
    assert "document d4" in turn.evidence and PASSAGE not in turn.evidence


def test_a_company_with_an_sec_number_is_searched_for_before_the_corpus_is_read(registered, router):
    """Measured, and the reason the order is this way round: on five such questions every stored passage
    named the ticker, none was about the company -- Stack Exchange threads that merely say it -- and all five
    answers were wrong, while asking EDGAR by number returned the matching filing five times out of five.
    The corpus holds no news, so for a question about a company it is the weaker source even when it answers.
    Chunk d4 says AAPL and would have been served; that it is not is the whole observable change."""
    fetched = Chunk("Apple Inc. said fourth-quarter revenue rose.", [],
                    "https://www.sec.gov/Archives/edgar/data/320193/ex99.htm", "2026-07-29")
    turn = agent(registered, router, gate_cut=1e9,
                 reference=reference("never written", model=ExplodingModel(),
                                     live=FakeLive([fetched]))).answer(
        "should i set a stop loss on AAPL ?")
    assert turn.spoke and turn.served == fetched.text
    assert fetched.document in turn.evidence and "document d4" not in turn.evidence


def test_a_search_by_a_guessed_name_must_come_back_with_something_that_says_the_guess(registered, router):
    """The asymmetry between the arms is provenance, and a guess has none. RELIANCE.NS is in no SEC table,
    so the only search term is the symbol's own stem -- and measured, "infy" returned the article on the
    Royal Canadian Regiment, which this path served against a question about Infosys. A weak check, and the
    right strength: 2 of those 3 searches were right and are kept."""
    wrong = Chunk("The regiment was raised in London, Ontario in 1883.", [],
                  "https://en.wikipedia.org/wiki/The_Royal_Canadian_Regiment", "2026-09-01")
    turn = agent(registered, router, gate_cut=1e9, model=ExplodingModel(),
                 reference=reference("never written", model=ExplodingModel(),
                                     live=FakeLive([wrong]))).answer(
        "should i set a stop loss on RELIANCE.NS ?")
    assert not turn.spoke and turn.because == "unclear question"
    assert turn.served == finance.UNKNOWN_QUESTION


def test_a_search_that_could_not_be_run_refuses_in_its_own_words(market, router):
    """A blocked search is not an empty corpus. The trained refusal would blame the question for a network
    that was down, so this gets its own reason and the error that caused it travels in `because`."""
    turn = agent(market, router, gate_cut=1e9, model=ExplodingModel(),
                 reference=reference("never written", model=ExplodingModel(), live=RefusedLive())).answer(
        "what has RELIANCE.NS announced ?")
    assert not turn.spoke and turn.served == finance.UNREACHABLE
    assert turn.because == "search failed: sec_edgar: 403 from efts.sec.gov"


def test_a_live_search_is_only_reached_once_the_stored_passages_have_failed_the_gate(market, router):
    """Additive, and in one direction only. A question the stored corpus can answer must not go to the
    network, because that would make every measured answer depend on something outside the repository."""
    turn = agent(market, router, reference=reference("never written", model=ExplodingModel(),
                                                    live=ExplodingLive())).answer(
        "what is a stop loss order ?")
    assert turn.spoke and turn.served == PASSAGE


def test_retrieving_nothing_leaves_the_refusal_exactly_as_it_was(market, router):
    """The path is additive or it is not safe to add. A question the corpus cannot match has to come back
    as the same named refusal the agent gave before this existed, not as a fifth kind of silence."""
    turn = agent(market, router, model=ExplodingModel(),
                 reference=reference("never written")).answer("zzzz qqqq wwww ?")
    assert turn.because == "no subject" and not turn.spoke
    assert (turn.served, turn.ticker, turn.intent, turn.asked) == (finance.NO_SUBJECT, "", "", "")


# --- the week-ahead outlook -------------------------------------------------

def head(config=OUTLOOK, tower="gru"):
    """An untrained head. Every test here compares two of its outputs, never their value."""
    return models.PriceHead(PriceWindowClassifier(config, recurrent=True), config, tower, MEASURED)


def rule(window=20):
    return models.Persistence((0.01, 0.02), TABLE, MEASURED, window)


def alternating_bars(count, step=0.01):
    """Closes whose log returns are exactly +/- `step`, so the trailing volatility is known rather than
    measured: the standard deviation of that over 20 returns is step * sqrt(20 / 19)."""
    closes = 100.0 * np.exp(step * (np.arange(count) % 2))
    return np.stack([closes, closes, closes, closes, np.full(count, 1e6)], axis=-1)


def artifact(tmp_path, accuracy, persistence=0.47, evaluated=28676, tower="gru", drop=()):
    """A served file, written the way scripts/train_price_head.py writes one."""
    recorded = {"tower": tower, "target": "volatility", "horizon": 5, "edges": [0.011, 0.019],
                "accuracy": accuracy, "evaluated": evaluated, "persistence": persistence,
                "trailing_edges": [0.01, 0.02], "table": [list(row) for row in TABLE]}
    for name in drop:
        del recorded[name]
    path = tmp_path / f"{tower}.npz"
    pretrained.save(path, PriceWindowClassifier(OUTLOOK, recurrent=True), OUTLOOK, metadata=recorded)
    return path


def test_a_served_window_matches_one_rebuilt_from_bars_up_to_that_date():
    """C3's gate, and the reason the window standardises inside itself. Standardising over the whole
    series, or taking the scale from bars[-1], both produce plausible numbers that no longer match what
    a backtest at that date could have computed."""
    rng = np.random.default_rng(0)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 300)))
    bars = np.stack([closes, closes + 1.0, closes - 1.0, closes, np.full(300, 1e6)], axis=-1)
    truncated = bars[: 201]
    assert np.array_equal(prices.window_at(bars, 200, 128),
                          prices.window_at(truncated, len(truncated) - 1, 128))


class RecordingAdvisor:
    """Keeps the window it was asked to score. The evidence line rounds an outlook to a class name and a
    whole percent, so comparing rendered text would pass on windows that are not the same array."""

    bars_needed = OUTLOOK.price_window + 2

    def __init__(self):
        self.windows = []

    def probabilities(self, bars, end):
        self.windows.append(prices.window_at(bars, end, OUTLOOK.price_window))
        return np.asarray(TABLE[0])

    @staticmethod
    def confidence(probabilities):
        return float(max(probabilities))


def test_the_window_a_snapshot_scores_is_the_one_a_truncated_file_produces(tmp_path):
    """The same rule through the served path, where the as-of date also has to resolve to the same bar."""
    price_file(tmp_path, "AAPL", bars=80)
    shorter = tmp_path / "shorter"
    shorter.mkdir()
    price_file(shorter, "AAPL", bars=61)
    watcher = RecordingAdvisor()
    _, _, taken_at = finance.Market(tmp_path, watcher).snapshot(
        "AAPL", as_of=(FIRST_BAR + timedelta(days=60)).isoformat())
    _, _, rebuilt_at = finance.Market(shorter, watcher).snapshot("AAPL")
    served, rebuilt = watcher.windows
    assert taken_at == rebuilt_at and np.array_equal(served, rebuilt)


def test_persistence_quotes_the_measured_hit_rate_of_the_bucket_it_lands_in():
    # 1% alternating returns put the 20 day volatility at 1.03%, between the 1% and 2% edges.
    assert np.array_equal(rule().probabilities(alternating_bars(40), 39), np.asarray(TABLE[1]))


def test_the_top_band_is_left_open_and_the_others_widen_with_the_horizon():
    """The forecast panel draws these. The top class has no upper edge, so closing its band would put a
    ceiling on screen that no measurement supports, and a band that ignored the horizon would understate
    a week by the square root of five."""
    day, week = models.price_bands(100.0, (0.01, 0.02), 1), models.price_bands(100.0, (0.01, 0.02), 5)

    assert [band["open"] for band in week] == [False, False, True]
    assert week[2]["sigma"] == week[1]["sigma"]  # A floor, not a range: the open band starts at the edge.
    assert week[0]["sigma"] == pytest.approx(day[0]["sigma"] * 5 ** 0.5)
    assert week[0]["high"] > day[0]["high"] > 100.0 > day[0]["low"] > week[0]["low"]


def test_calibrating_the_confidence_leaves_the_class_the_answer_names_alone():
    """The fit says how often the top class comes true, and that can fall below a runner-up. Folding it
    back into the vector and rescaling the rest is what an earlier version did, and it turned a calm week
    turbulent: 45% calibrated to 4% while the 35% class was scaled up to 61%."""
    advisor, raw = head(), np.array([0.2, 0.45, 0.35])
    advisor.calibrator = Calibrator(weight=1.0, bias=-3.0)  # Pushes every confidence down hard.

    assert advisor.confidence(raw) < 0.35 < float(max(raw))
    assert advisory.risk_outlook(raw, confidence=advisor.confidence(raw))["outlook"] == "normal"


def test_a_table_that_does_not_cover_every_bucket_is_refused_at_construction():
    # Three edges make four buckets, and the missing row would only be missed on the values that reach it.
    with pytest.raises(ValueError, match="expected"):
        models.Persistence((0.01, 0.02, 0.03), TABLE, MEASURED)


def test_too_little_history_for_the_advisor_leaves_the_outlook_out_rather_than_scoring_it(tmp_path):
    """A head needs more bars than the indicators do. Scoring a short window returns a class, and the
    answer would then quote a figure computed off half the history it claims."""
    price_file(tmp_path, "AAPL")
    _, shown, _ = finance.Market(tmp_path, rule(window=200)).snapshot("AAPL")
    assert "outlook" not in shown


def test_the_risk_intent_answers_once_an_advisor_is_loaded(tmp_path, router):
    """The other half of the refusal above: loading one advisor turns this intent on everywhere, with
    no second switch to forget."""
    price_file(tmp_path, "AAPL")
    market = finance.Market(tmp_path, rule())
    turn = agent(market, router, written="it looks calm .", confidence=0.99).answer(
        "how risky is AAPL over the next week ?")
    assert turn.intent == "risk" and turn.spoke and "outlook calm" in turn.evidence


def test_a_head_that_beat_the_rule_on_held_out_data_is_what_serving_loads(tmp_path):
    advisor = models.load(artifact(tmp_path, accuracy=0.52))
    assert isinstance(advisor, models.PriceHead) and advisor.version.startswith("gru@")


def test_a_head_that_wins_by_less_than_its_own_measurement_noise_is_not_served(tmp_path):
    """The case the first comparison was decided on: 0.1 points, where one standard error on 28,676
    windows is 0.3. A head that has not cleared one line of arithmetic must not be what answers."""
    assert isinstance(models.load(artifact(tmp_path, accuracy=0.471)), models.Persistence)


def test_the_same_gap_becomes_a_win_once_enough_windows_have_measured_it(tmp_path):
    # The gap alone cannot decide it, which is why the count is in the file: 0.1 points is noise on 28,676
    # windows and a real edge on 2.5 million.
    advisor = models.load(artifact(tmp_path, accuracy=0.471, evaluated=2_500_000))
    assert isinstance(advisor, models.PriceHead)


def test_an_artifact_missing_what_serving_needs_names_the_script_that_writes_it(tmp_path):
    with pytest.raises(ValueError, match="train_price_head.py"):
        models.load(artifact(tmp_path, accuracy=0.52, drop=("table", "trailing_edges")))


def test_a_tower_this_loader_cannot_rebuild_is_an_error(tmp_path):
    # Refused rather than rebuilt as the default tower, which would serve weights of a different shape.
    with pytest.raises(ValueError, match="cannot"):
        models.load(artifact(tmp_path, accuracy=0.52, tower="lstm"))


def test_a_window_the_head_was_not_trained_on_is_refused_not_scored():
    """Adding a feature channel changes the input the tower was fitted to. Without this the first layer
    still multiplies, because a weight matrix does not know which channel is which."""
    with pytest.raises(ValueError, match="retrain or rebuild"):
        head(ModelConfig(price_channels=5, price_window=32, dim=32, num_heads=2, ffn_dim=64,
                         price_layers=1)).probabilities(alternating_bars(60), 59)


RETURNS_SHAPE = ModelConfig(vocab_size=returns.BINS, price_window=16, dim=16, num_heads=2, num_layers=1,
                            ffn_dim=32, dropout=0.0)
GARCH_FIT = {"alpha": 0.08, "beta": 0.9, "nu": 4.0}


def return_artifact(tmp_path, gap, error=0.002, drop=()):
    """A return-model file, written the way scripts/train_return_generator.py writes one."""
    recorded = {"garch": GARCH_FIT, "cutoff": "2021-01-01", "test_from": "2023-01-01",
                "test": {"vs_garch": gap, "vs_garch_error": error}}
    for name in drop:
        del recorded[name]
    path = tmp_path / "returns.npz"
    pretrained.save(path, ReturnGenerator(RETURNS_SHAPE), RETURNS_SHAPE, metadata=recorded)
    return path


def test_a_return_model_that_beat_garch_by_more_than_its_noise_draws_the_paths(tmp_path):
    assert isinstance(models.scenarios.load(return_artifact(tmp_path, gap=-0.01)), models.scenarios.Generative)


def test_a_return_model_that_did_not_clear_garch_leaves_garch_drawing_the_paths(tmp_path):
    """Lower is better on the log score, so the sign is the easy thing to get backwards: a model 0.001
    nats better than GARCH with a standard error of 0.002 has not beaten it, and one worse certainly has
    not."""
    for gap in (-0.001, 0.01):
        assert isinstance(models.scenarios.load(return_artifact(tmp_path, gap=gap)),
                          models.scenarios.GarchPaths)


def test_a_return_artifact_missing_what_serving_needs_names_the_script_that_writes_it(tmp_path):
    with pytest.raises(ValueError, match="train_return_generator.py"):
        models.scenarios.load(return_artifact(tmp_path, gap=-0.01, drop=("garch",)))


def _closes_file(directory, name, closes):
    path = directory / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "high", "low", "close", "adj_close", "volume"])
        for day, close in enumerate(closes):
            writer.writerow([(FIRST_BAR + timedelta(days=day)).isoformat(), close, close, close, close,
                             close, 1e6])


@pytest.mark.parametrize("drawer", ["garch", "generative"])
def test_paths_drawn_as_of_a_date_ignore_every_bar_after_it(tmp_path, drawer):
    """Look-ahead, pinned for both drawers. Two files agree up to bar 299 and then one triples: paths drawn
    as of that bar must be the same array, or the picture a user sees has read a day that had not happened."""
    wiggle = 100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 400)))
    later = wiggle.copy()
    later[300:] *= 3.0
    served = models.scenarios.load(return_artifact(tmp_path, gap=-0.01 if drawer == "generative" else 0.01))
    drawn = []
    for name, closes in (("same", wiggle), ("shocked", later)):
        folder = tmp_path / name
        folder.mkdir()
        _closes_file(folder, "AAPL", closes)
        market = finance.Market(folder, scenarios=served)
        as_of = (FIRST_BAR + timedelta(days=299)).isoformat()
        drawn.append(market.draw_paths("AAPL", 10, 20, np.random.default_rng(1), as_of))
    assert drawn[0][0] == drawn[1][0] and drawn[0][1] == drawn[1][1]
    np.testing.assert_array_equal(drawn[0][2], drawn[1][2])


def test_the_fan_is_ordered_at_every_step():
    """Quantiles computed per step in the wrong axis still come out as five lists of prices; ordered at
    every step is what says each list is a quantile across paths and not a path."""
    drawn = np.random.default_rng(2).normal(0, 0.02, (500, 15))
    fan = models.scenarios.fan(100.0, drawn)
    low, mid, high = (np.array(fan[q]) for q in ("0.05", "0.5", "0.95"))
    assert len(mid) == 15 and (low < mid).all() and (mid < high).all()


# --- the language model as the last step: rewording, never deciding ---------

class FakeWriter:
    """Answers every request with one fixed text, the way the Space's language model replies."""

    def __init__(self, reply="", error=None):
        self.reply, self.error, self.asked = reply, error, []

    def create(self, system, messages):
        from backend.models.external.language import Reply
        self.asked.append(messages[0]["content"])
        if self.error:
            raise self.error
        return Reply("test-lm", self.reply)

    @staticmethod
    def text(resp):
        return resp.content


class ExplodingWriter:
    def create(self, *_):
        raise AssertionError("the language model was asked to reword something it must not touch")


def phrased_agent(market, router, written, writer, **kw):
    built = agent(market, router, written=written, **kw)
    built.writer = writer
    return built


DRAFT = "it is up 13.8% over 20 days and closed at 165.00 ."
ASKED = "how is AAPL doing ?"


def test_a_rewrite_that_keeps_every_figure_is_served_and_names_its_writer(market, router):
    """The rewrite is what is read; ALFA's draft stays in the turn as what the model wrote."""
    plain = "AAPL has risen 13.8% over the last 20 days and closed at 165.00."
    turn = phrased_agent(market, router, DRAFT, FakeWriter(plain)).answer(ASKED)
    assert turn.spoke and turn.served == plain and turn.answer == DRAFT
    assert (turn.phrased_by, turn.unphrased_because) == ("test-lm", "")


@pytest.mark.parametrize("reply,reason", [
    ("AAPL has been rising lately.", "leaves out"),                                            # drops the figures
    ("Up 13.8% in 20 days, closed at 165.00, and 9.99% this week.", "figure 9.99%"),            # adds one
    ("Up 13.8% in 20 days because of rates, closed at 165.00.", "gives a reason"),             # adds a cause
    ("Up 13.8% in 20 days, closed at 165.00, and it will rise further.", "claims a direction"),  # adds a call
    ("Up 13.8% in 20 days, closed at ₹165.00.", "writes ₹"),                                   # swaps currency
    ("Up:\n- 13.8% in 20 days\n- closed at 165.00", "breaks into lines"),                     # a list
])
def test_a_rewrite_that_changes_the_facts_serves_alfas_words_and_says_why(market, router, reply, reason):
    turn = phrased_agent(market, router, DRAFT, FakeWriter(reply)).answer(ASKED)
    assert turn.spoke and turn.served == DRAFT and turn.phrased_by == ""
    assert reason in turn.unphrased_because


def test_a_writer_out_of_quota_serves_alfas_words_with_the_reason(market, router):
    from backend.database.sources.provider import DataUnavailable
    writer = FakeWriter(error=DataUnavailable("model Space failed on /write: GPU quota exceeded"))
    turn = phrased_agent(market, router, DRAFT, writer).answer(ASKED)
    assert turn.spoke and turn.served == DRAFT and "GPU quota exceeded" in turn.unphrased_because


def test_whether_to_speak_is_still_decided_on_alfas_own_draft(market, router):
    """The cut and the faithfulness record were measured on the generator's draft, so a refusal is never
    handed to the language model to dress up."""
    turn = phrased_agent(market, router, DRAFT, ExplodingWriter(), confidence=0.5, cut=0.9).answer(ASKED)
    assert not turn.spoke and turn.served == finance.REFUSAL


NEWS = Chunk("Mint, 6 Oct 2026: Reliance Industries commissioned its new refinery unit.", [],
             "https://n.test/reliance", "2026-10-06")


def news_reference(writer, news=(NEWS,), live=None):
    built = reference("never written", model=ExplodingModel(), live=live or ExplodingLive())
    built.writer, built.news = writer, FakeLive(list(news))
    return built


def test_a_question_about_lately_is_answered_from_the_newswire_alone(market, router):
    """"Any news on tata steel" ranked over Wikipedia and the newswire together was answered from 2007."""
    plain = "Reliance Industries commissioned a new refinery unit (Mint, 6 Oct 2026)."
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(FakeWriter(plain))).answer(
        "any news on RELIANCE.NS ?")
    assert turn.spoke and turn.served.endswith(plain) and turn.phrased_by == "test-lm"
    assert turn.served.startswith("RELIANCE.NS closed at ₹") and NEWS.document in turn.evidence


def test_a_lately_question_with_no_news_gives_the_price_move_and_says_nothing_explains_it(market, router):
    """"Why is infosys falling" read from Wikipedia became "infosys is falling because it is a technology
    company". With no headline about it, the price files still say how far it moved, and nothing says why."""
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(ExplodingWriter(), news=())).answer(
        "why is RELIANCE.NS falling ?")
    assert turn.spoke and turn.served.startswith("RELIANCE.NS closed at ₹") and "found no news" in turn.served
    assert "over 5 days" in turn.served and "prices " in turn.evidence


def test_a_lately_question_with_neither_prices_nor_news_says_so(market, router):
    turn = agent(market, router, reference=news_reference(ExplodingWriter(), news=())).answer(
        "what is the stock market doing today ?")
    assert not turn.spoke and turn.served == finance.NOT_IN_SOURCES


def test_the_language_model_reads_the_news_and_our_own_sentence_gives_the_prices(market, router):
    """Given the prices too, it wrote "down 0.5% from the previous close of ₹704.80" for HDFC Bank: a 5-day
    change pinned to a quote saying the shares were up. So it never sees them, and a figure from them is refused."""
    from backend.models.agent import phrase
    moved = market.snapshot("RELIANCE.NS")[1]["return_5d"]
    writer = FakeWriter("Reliance Industries commissioned a refinery unit (Mint, 6 Oct 2026).")
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(writer)).answer("any news on RELIANCE.NS ?")
    assert moved not in writer.asked[0] and "PRICES" not in writer.asked[0]
    assert turn.served == f"{phrase.price_move(finance.Market.subject(market, 'RELIANCE.NS'), market.snapshot('RELIANCE.NS'))} " \
                          "Reliance Industries commissioned a refinery unit (Mint, 6 Oct 2026)."
    quoting = news_reference(FakeWriter(f"RELIANCE.NS is {moved} over 5 days after the refinery unit (Mint, 6 Oct 2026)."))
    turn = agent(market, router, gate_cut=1e9, reference=quoting).answer("any news on RELIANCE.NS ?")
    assert turn.phrased_by == "" and f"figure {moved}" in turn.unphrased_because
    assert turn.served == phrase.recent_plain(finance.Market.subject(market, "RELIANCE.NS"),
                                              market.snapshot("RELIANCE.NS"), [NEWS])


def test_a_writer_that_finds_no_answer_serves_the_plain_answer_not_its_half_refusal(market, router):
    """Half an answer and half a refusal ("...founded in 1981. The passage does not provide a reason. None.")
    is a refusal: only that half was true. The prices and the newest report are what is left to say."""
    writer = FakeWriter("It was founded in 1981. The passages do not provide a reason. None.")
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(writer)).answer("why is RELIANCE.NS falling ?")
    assert turn.spoke and "founded" not in turn.served and NEWS.text in turn.served


def test_forum_and_encyclopedia_passages_are_quoted_not_reworded(market, router):
    """The language model misattributed long prose ("HDFCBANK.NS is a subsidiary of HDFC Bank"), as ALFA's
    own generator did, so only dated headlines are reworded."""
    built = reference("never written", model=ExplodingModel())
    built.writer = ExplodingWriter()
    turn = agent(market, router, reference=built).answer("what is a stop loss order ?")
    assert turn.spoke and turn.served == PASSAGE


def test_a_question_not_about_money_is_refused_even_when_a_passage_matches(market, router):
    turn = agent(market, router, reference=reference("never written", model=ExplodingModel())).answer(
        "what did the committee vote on at the meeting ?")
    assert not turn.spoke and turn.served == finance.OFF_TOPIC


def test_an_exchange_listed_name_resolves_to_its_ticker_as_whole_words(tmp_path):
    price_file(tmp_path, "INFY.NS")
    price_file(tmp_path, "HDFCBANK.NS")
    named = finance.Market(tmp_path, listed={"INFY.NS": "Infosys", "HDFCBANK.NS": "HDFC Bank", "TCS.NS": "TCS"})
    assert named.resolve("why is infosys falling ?") == "INFY.NS"
    assert named.resolve("how is hdfc bank doing") == "HDFCBANK.NS"
    assert named.resolve("an infosystem question") is None and "TCS.NS" not in named.listed
    assert named.subject("INFY.NS").mentions("Infosys shares slip")


def test_a_concept_question_naming_no_company_is_answered_from_the_lessons_first(market, router):
    """The corpus is mostly US forums: "how does a SIP work" read from it was Wikipedia's sales incentive plan."""
    built = reference("never written", model=ExplodingModel())
    built.lessons = lambda question: [("SIP (systematic investment plan)", "A SIP invests a fixed amount every month.")]
    turn = agent(market, router, reference=built).answer("how does a SIP work ?")
    assert turn.spoke and turn.served == "A SIP invests a fixed amount every month." and turn.intent == "reference lesson"
    built.lessons = lambda question: []
    assert agent(market, router, reference=built).answer("what is a stop loss order ?").served == PASSAGE


def test_a_question_that_says_lately_reads_the_dated_sources_before_the_advisory_path(market, router):
    """"how is AAPL doing lately" was refused at ALFA's confidence cut while the price files held its answer."""
    turn = agent(market, router, model=ExplodingModel(),
                 reference=news_reference(FakeWriter("Reliance commissioned a refinery unit (Mint, 6 Oct 2026)."))).answer(
        "how has RELIANCE.NS been doing lately ?")
    assert turn.spoke and turn.intent == "reference recent"


def test_a_rewrite_that_misspells_the_ticker_serves_alfas_words(market, router):
    turn = phrased_agent(market, router, DRAFT, FakeWriter("AAAPL.NS is up 13.8% over 20 days, closing at 165.00.")).answer(ASKED)
    assert turn.served == DRAFT and "names AAAPL.NS" in turn.unphrased_because


def test_what_a_reader_sees_is_titled_figured_and_sourced(market, router):
    from backend.models.agent.present import present
    turn = agent(market, router, written="its 14 day rsi is 100 , which is overbought .").answer("is AAPL overbought ?")
    shown = present(turn, {}, {})
    assert shown.startswith("### AAPL · Momentum and RSI") and "Its 14 day RSI is 100, which is overbought." in shown
    assert "**Key figures** · Last close $165.00" in shown and "RSI (14) 100" in shown
    recent = agent(market, router, gate_cut=1e9, reference=news_reference(
        FakeWriter("Reliance Industries commissioned a refinery unit (Mint, 6 Oct 2026)."))).answer("any news on RELIANCE.NS ?")
    shown = present(recent, {"RELIANCE.NS": "Reliance Industries"}, {})
    assert shown.startswith("### Reliance Industries (RELIANCE.NS) · What happened lately")
    assert "**Sources**\n- [n.test, 6 Oct 2026](https://n.test/reliance)" in shown
    quiet = agent(market, router, written="it is fine .", confidence=0.1).answer("is AAPL overbought ?")
    assert present(quiet, {}, {}) == quiet.served
