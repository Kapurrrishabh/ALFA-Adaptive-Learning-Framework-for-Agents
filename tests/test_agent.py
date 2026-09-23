"""C1: one typed question in, one answer or one named refusal out, with no network anywhere.

The serving layer's whole value is what it refuses. A wrong-intent answer passes the figure guard —
every digit in it is supported — so the tests that matter here are the five stages that stop a turn,
and the one rule that the row a served question is built into is the row training was built from.
"""

import csv
from datetime import date, timedelta

import numpy as np
import pytest

from backend import models
from backend.agent import Router, assemble, combined, finance, lexical
from backend.agent.core import Agent, answered
from selfagent import pretrained
from selfagent.config import ModelConfig
from selfagent.data import advisory, prices
from selfagent.data.encode import build_sources
from selfagent.learn.abstain import Abstainer
from selfagent.models import PriceWindowClassifier
from selfagent.tokenizer.wordpiece import pretokenize

LENGTH = 128
FIRST_BAR = date(2020, 1, 1)

# Small enough to build in a test, and 6 channels because that is what the feature builder produces:
# five indicators plus the window's own logged scale.
OUTLOOK = ModelConfig(price_channels=6, price_window=32, dim=32, num_heads=2, ffn_dim=64, price_layers=1)

# Distinct rows, so which bucket was read is visible in what comes back.
TABLE = ((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7))


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


# --- the week-ahead outlook -------------------------------------------------

def head(config=OUTLOOK, tower="gru"):
    """An untrained head. Every test here compares two of its outputs, never their value."""
    return models.PriceHead(PriceWindowClassifier(config, recurrent=True), config, tower)


def rule(window=20):
    return models.Persistence((0.01, 0.02), TABLE, window)


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


def test_a_table_that_does_not_cover_every_bucket_is_refused_at_construction():
    # Three edges make four buckets, and the missing row would only be missed on the values that reach it.
    with pytest.raises(ValueError, match="expected"):
        models.Persistence((0.01, 0.02, 0.03), TABLE)


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
