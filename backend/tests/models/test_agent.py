"""C1: one typed question in, one answer or one named refusal out, with no network anywhere.

The serving layer's whole value is what it refuses. A wrong-intent answer passes the figure guard —
every digit in it is supported — so the tests that matter here are the five stages that stop a turn,
and the one rule that the row a served question is built into is the row training was built from.
"""

import csv
import json
import os
import re
import threading
import time
from collections import Counter
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from backend.models import serving as models
from backend.models.agent import PARAPHRASED, QUOTED, Reference, Router, assemble, combined, finance, lexical, phrase
from backend.models.agent import core, facts_reply
from backend.models.agent.context import evidence_room
from backend.models.agent.core import Agent, answered
from backend.models.agent.present import KEY_FIGURES
from backend.models.agent.reference import LESSON_FLOOR, UNANSWERED, Looked, SearchFailed
from backend.knowledge_base.retrieval import Chunk, Hybrid
from backend.database.sources.provider import DataUnavailable
from backend.models.core import pretrained
from backend.models.core.config import ModelConfig
from backend.models.data import advisory, prices, returns, stock_qa
from backend.models.data.encode import build_sources
from backend.models.data.qa_pairs import PASSAGES, QUESTION_TOKENS
from backend.models.networks.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P
from backend.models.learning import Calibrator
from backend.models.learning.abstain import Abstainer
from backend.models.networks import PriceWindowClassifier, ReturnGenerator
from backend.models.core.tokenizer.wordpiece import pretokenize
from backend.models.guardrails import guardrails

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


@pytest.mark.parametrize("question,named", [
    ("What is the transaction cost of buying shares in India?", None), ("write me a poem about a cat", None),
    ("what is a V-shaped recovery", None), ("is the price above the 50-day MA", None),
    ("what happens after 3 pm", None), ("what happens after 3 PM", None),
    ("how is COST doing ?", "COST"), ("is V a buy ?", "V"), ("is aapl above its 50-day MA ?", "AAPL"),
    ("how is reliance doing ?", "RELIANCE.NS"), ("Q3 COST results", "COST"), ("FY24 KO", "KO"),
    ("what did PM Modi say about the budget ?", None), ("give me a pep talk", None),
    ("Is GE Aerospace a buy ?", "GE"), ("is BA Boeing a buy ?", "BA"), ("what did the PM say about the budget ?", None),
])
def test_a_symbol_that_is_an_everyday_word_names_its_instrument_only_when_typed_as_one(tmp_path, question, named):
    """"What is the transaction cost of buying shares in India?" was answered about Costco."""
    for name in ("COST", "CAT", "PM", "MA", "V", "KO", "PEP", "GE", "BA", "AAPL", "RELIANCE.NS"):
        price_file(tmp_path, name)
    assert finance.Market(tmp_path).resolve(question) == named


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


# --- a listed share with no file, fetched when asked --------------------------

SHARES = {"NYKAA.NS": "FSN E-Commerce Ventures", "TAKE.NS": "Take Solutions", "DOLLAR.NS": "Dollar Industries",
          "M&M.NS": "Mahindra & Mahindra", "360ONE.NS": "360 ONE WAM"}


class Yahoo:
    """Yahoo in place of the network: `bars` days for any share asked for, or no rows when `bars` is 0."""

    def __init__(self, bars=advisory.BARS_NEEDED):
        self.bars, self.asked = bars, []

    def __call__(self, ticker, path):
        self.asked.append(ticker)
        if not self.bars:
            raise RuntimeError(f"yahoo returned no daily prices for {ticker}")
        path.parent.mkdir(exist_ok=True)
        price_file(path.parent, ticker, self.bars)


def fetching(tmp_path, yahoo):
    """A market holding one collector file, RELIANCE.NS, that may fetch any of SHARES."""
    price_file(tmp_path, "RELIANCE.NS")
    return finance.Market(tmp_path, shares=SHARES, fetch=yahoo)


def test_a_listed_share_with_no_file_is_fetched_when_asked_and_not_again_the_same_day(tmp_path, router):
    yahoo = Yahoo()
    asking = agent(fetching(tmp_path, yahoo), router, written="it is doing fine .")
    turns = [asking.answer("how is nykaa doing ?", subject="NYKAA.NS") for _ in range(2)]
    assert [(turn.ticker, turn.spoke) for turn in turns] == [("NYKAA.NS", True)] * 2
    assert yahoo.asked == ["NYKAA.NS"]


@pytest.mark.parametrize("question", ["should i take profits now ?", "what is the dollar doing", "how is ZZZ.NS doing ?",
                                      "how is take solutions doing ?"])
def test_a_listed_share_is_never_read_from_a_word_or_its_name_and_an_unlisted_symbol_is_never_fetched(
        tmp_path, router, question):
    """180 NSE symbols are dictionary words: "should i take profits now ?" would have fetched TAKE.NS."""
    yahoo = Yahoo()
    turn = agent(fetching(tmp_path, yahoo), router, model=ExplodingModel()).answer(question)
    assert (turn.because, yahoo.asked) == ("no subject", [])


@pytest.mark.parametrize("text,named", [("M&M.NS", "M&M.NS"), ("360ONE.NS", "360ONE.NS"),
                                        ("is NYKAA.NS overbought ?", "NYKAA.NS"), ("RELIANCE.NS", "RELIANCE.NS")])
def test_a_listed_share_resolves_by_its_symbol_typed_or_handed_over_whole(tmp_path, text, named):
    """The word pattern cannot read "M&M.NS" or "360ONE.NS", which is how the app hands a share over."""
    assert fetching(tmp_path, Yahoo()).resolve(text) == named


def test_an_empty_reply_from_yahoo_is_a_named_refusal_not_an_error(tmp_path, router, monkeypatch):
    """yfinance answers an unreachable host and an unknown symbol alike, with an empty frame and no exception."""
    import pandas as pd
    from backend.database.live import prices as yahoo_prices
    monkeypatch.setattr(yahoo_prices.yfinance, "download", lambda *_, **__: pd.DataFrame())
    price_file(tmp_path, "RELIANCE.NS")
    market = finance.Market(tmp_path, shares=SHARES, fetch=lambda ticker, path: yahoo_prices.fetch(ticker, path, 1))
    turn = agent(market, router, model=ExplodingModel()).answer("how is nykaa doing ?", subject="NYKAA.NS")
    assert (turn.spoke, turn.served, turn.ticker) == (False, finance.NO_PRICES, "")
    assert turn.because.startswith("no prices: yahoo returned no daily prices for NYKAA.NS")


def test_a_share_listed_too_recently_for_a_snapshot_is_refused_by_name_not_raised(tmp_path, router):
    """A snapshot of 60 days raised out of `answer`, and out of the chat socket with it."""
    short = advisory.BARS_NEEDED - 1
    turn = agent(fetching(tmp_path, Yahoo(bars=short)), router, model=ExplodingModel()).answer(
        "how is nykaa doing ?", subject="NYKAA.NS")
    assert (turn.spoke, turn.served) == (False, finance.NO_PRICES)
    assert turn.because == f"no prices: NYKAA.NS has {short} days of prices, and its figures need {advisory.BARS_NEEDED}"


def test_a_fetched_file_from_before_its_markets_close_is_fetched_again_and_refused_rather_than_served_when_that_fails(
        tmp_path, router):
    yahoo = Yahoo()
    asking = agent(fetching(tmp_path, yahoo), router, written="it is doing fine .")
    asking.answer("how is nykaa doing ?", subject="NYKAA.NS")
    old = finance.last_close(finance.market_of("NYKAA.NS"), time.time()) - 60
    os.utime(tmp_path / "fetched" / "NYKAA.NS.csv", (old, old))
    yahoo.bars = 0
    turn = asking.answer("how is nykaa doing ?", subject="NYKAA.NS")
    assert (yahoo.asked, turn.spoke, turn.served) == (["NYKAA.NS", "NYKAA.NS"], False, finance.NO_PRICES)


def test_a_file_the_collector_keeps_is_fetched_again_once_its_market_has_closed_since_and_is_never_written(tmp_path,
                                                                                                           router):
    """The collector runs on a day's timer, so a price answer read TCS at 2,505.00 from 2 Nov beside facts of 16 Nov."""
    yahoo = Yahoo()
    market = fetching(tmp_path, yahoo)
    held = tmp_path / "RELIANCE.NS.csv"
    before = finance.last_close(finance.market_of("RELIANCE.NS"), time.time()) - 60
    os.utime(held, (before, before))
    kept = held.read_bytes()
    turn = agent(market, router, written="it is doing fine .").answer("how is reliance doing ?")
    assert (turn.spoke, yahoo.asked, market.paths["RELIANCE.NS"]) == \
        (True, ["RELIANCE.NS"], tmp_path / "fetched" / "RELIANCE.NS.csv")
    assert (held.read_bytes(), held.stat().st_mtime) == (kept, pytest.approx(before))


def test_a_file_the_collector_keeps_from_before_its_markets_close_is_refused_when_it_cannot_be_fetched_again(tmp_path,
                                                                                                             router):
    yahoo = Yahoo(bars=0)
    market = fetching(tmp_path, yahoo)
    os.utime(tmp_path / "RELIANCE.NS.csv", (0, 0))
    turn = agent(market, router, model=ExplodingModel()).answer("how is reliance doing ?")
    assert (turn.spoke, turn.served, yahoo.asked) == (False, finance.NO_PRICES, ["RELIANCE.NS"])


@pytest.mark.parametrize("live,mtime", [(True, None), (False, 0)], ids=["written since the close", "offline"])
def test_a_file_the_collector_keeps_is_read_as_it_is_when_written_since_its_markets_close_or_offline(tmp_path, router,
                                                                                                     live, mtime):
    yahoo = Yahoo()
    price_file(tmp_path, "RELIANCE.NS")
    if mtime is not None:
        os.utime(tmp_path / "RELIANCE.NS.csv", (mtime, mtime))
    market = finance.Market(tmp_path, shares=SHARES, fetch=yahoo) if live else finance.Market(tmp_path)
    turn = agent(market, router, written="it is doing fine .").answer("how is reliance doing ?")
    assert (turn.ticker, turn.spoke, yahoo.asked) == ("RELIANCE.NS", True, [])


def test_a_share_or_an_index_closes_with_its_own_market():
    """Read as a US symbol, the Nifty's file counted as written after the close for eleven hours after NSE closed, and a
    .BO share was priced in dollars."""
    assert [finance.market_of(ticker).code for ticker in ("^NSEI", "^BSESN", "^GSPC", "TCS.NS", "RELIANCE.BO", "AAPL")] \
        == ["NSE", "BSE", "US", "NSE", "BSE", "US"]


@pytest.mark.parametrize("given", [{"shares": SHARES}, {"fetch": Yahoo()}])
def test_a_share_list_and_a_fetch_come_together_or_not_at_all(tmp_path, given):
    price_file(tmp_path, "AAPL")
    with pytest.raises(ValueError, match="together"):
        finance.Market(tmp_path, **given)


@pytest.mark.parametrize("symbol", ["../AAPL.NS", "A/B.NS", "nykaa.NS", "NYKAA"])
def test_a_share_symbol_that_could_reach_a_url_or_path_unchecked_is_refused_at_construction(tmp_path, symbol):
    price_file(tmp_path, "AAPL")
    with pytest.raises(ValueError, match=re.escape(symbol)):
        finance.Market(tmp_path, shares={symbol: "a company"}, fetch=Yahoo())


def test_a_fetched_table_is_written_whole_in_the_collectors_format_and_reads_back(tmp_path):
    import pandas as pd
    from backend.database.live import prices as yahoo_prices
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    frame = pd.DataFrame([[10, 11, 9, 10.5, 10.4, 1000], [float("nan")] * 6, [11, 12, 10, 11.5, 11.4, 2000]],
                         index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                         columns=pd.MultiIndex.from_product([["NYKAA.NS"], fields]))
    path = tmp_path / "fetched" / "NYKAA.NS.csv"
    yahoo_prices.write(yahoo_prices.table_for(frame, "NYKAA.NS"), path)
    lines = path.read_text().splitlines()
    assert lines[0] == "date,open,high,low,close,adj_close,volume" and lines[1].startswith("2020-01-01,10")
    assert [written.name for written in path.parent.iterdir()] == ["NYKAA.NS.csv"]
    assert prices.load_bars(path)[0] == ["2020-01-01", "2020-01-03"]


def test_a_fetch_that_fails_stops_every_fetch_for_a_while_so_a_hung_yahoo_costs_one_question(tmp_path, monkeypatch):
    """yfinance's cookie and crumb calls ignore the 10 s timeout, so a hung Yahoo cost every question about 40 s."""
    clock = [time.time()]
    monkeypatch.setattr(finance, "time", SimpleNamespace(time=lambda: clock[0]))
    yahoo = Yahoo(bars=0)
    market = fetching(tmp_path, yahoo)
    for ticker, later in (("NYKAA.NS", 0), ("TAKE.NS", finance.YAHOO_SLOW_S + 1),
                          ("TAKE.NS", finance.YAHOO_BACK_OFF_S)):
        clock[0] += later
        with pytest.raises(finance.NoPrices):
            market.hold(ticker)
    assert yahoo.asked == ["NYKAA.NS", "TAKE.NS"]


def test_a_fetch_over_the_time_limit_is_given_up_on_and_its_late_file_is_still_used(tmp_path, monkeypatch):
    """The back-off limited how often a hung Yahoo stalled, not how long: one stall held the generating lock 40 s."""
    monkeypatch.setattr(finance, "YAHOO_SLOW_S", 0.05)
    yahoo, released, written = Yahoo(), threading.Event(), threading.Event()

    def hung(ticker, path):
        released.wait(5)
        yahoo(ticker, path)
        written.set()
    market = fetching(tmp_path, hung)
    for ticker in ("NYKAA.NS", "TAKE.NS"):
        with pytest.raises(finance.NoPrices):
            market.hold(ticker)
    released.set()
    assert written.wait(5)
    market.hold("NYKAA.NS")
    assert yahoo.asked == ["NYKAA.NS"]


def test_a_fetched_share_is_known_by_its_listed_name(tmp_path):
    market = fetching(tmp_path, Yahoo())
    market.hold("NYKAA.NS")
    assert market.subject("NYKAA.NS").names == ("fsn e-commerce ventures",)


# --- a share's dated facts, for a question the router places there ----------

FORWARD_PE = {"value": 18.14, "unit": "x", "source": "yahoo", "period": "snapshot retrieved 2026-11-15"}


def analysis(**metrics):
    """The app's analysis of RELIANCE as `AnalysisService.analyze(...).to_dict()` gives it, cut to what `facts` reads."""
    return {"symbol": "RELIANCE", "name": "Reliance Industries Limited", "market": "NSE", "currency": "INR",
            "analysis_timestamp": "2026-11-15T18:13:22+00:00", "quality": {"last_date": "2026-11-13"},
            "results": {"fundamental": {"details": {"metrics": metrics or {"forward_pe": FORWARD_PE}}}}}


class Analyses:
    """The app's analysis in place of Yahoo: `analysed` for any share asked about, or Yahoo's failure when it is None."""

    def __init__(self, analysed=None, fails=False):
        self.analysed, self.fails, self.asked = analysed or analysis(), fails, []

    def __call__(self, ticker):
        self.asked.append(ticker)
        if self.fails:
            raise RuntimeError(f"yahoo gave no analysis of {ticker}: no data")
        return self.analysed


ASSESSED = "give me a simple fundamental and technical assessment of {} as of the latest completed trading session ."
ASSESSMENT = ASSESSED.format("RELIANCE.NS")
WRITTEN_FROM_FACTS = "reliance industries limited trades at 18.14 times the profit it is expected to make ."


def facts_router(*asked):
    known = finance.examples() + [(finance.FACTS, "csv:integrated_stock_read", finance._squeeze(question.format("")))
                                  for question in (ASSESSED, *asked)]
    return Router(known, lexical([text for _, _, text in known], pretokenize))


def shared_words(texts):
    """A similarity by the long words each text shares with the question, so which facts are picked depends on it."""
    def words(text):
        return {word for word in pretokenize(text) if len(word) > 4}
    return lambda question: [len(words(question) & words(text)) for text in texts]


def facts_agent(tmp_path, analyse, written=WRITTEN_FROM_FACTS, model=None, facts_cut=0.9, asked=()):
    price_file(tmp_path, "RELIANCE.NS")
    return Agent(finance, finance.Market(tmp_path, analyse=analyse), facts_router(*asked), Abstainer(0.01),
                 FakeTokenizer(written), model or FakeModel(), Config, Abstainer(0.9), facts=shared_words,
                 facts_abstainer=Abstainer(facts_cut) if core.GENERATOR_WRITES_FACTS else None)


@pytest.fixture
def generator_writes_facts(monkeypatch):
    """The facts route with ALFA's generator writing the answers, as it served before our code wrote them."""
    monkeypatch.setattr(core, "GENERATOR_WRITES_FACTS", True)


@pytest.mark.usefixtures("generator_writes_facts")
def test_a_facts_question_is_answered_from_the_shares_dated_analysis(tmp_path):
    analyses = Analyses()
    turn = facts_agent(tmp_path, analyses).answer(ASSESSMENT)
    assert (turn.intent, turn.spoke, turn.served, turn.as_of) == (finance.FACTS, True, WRITTEN_FROM_FACTS, "2026-11-13")
    assert turn.evidence == ("Stock: Reliance Industries Limited (RELIANCE.NS), NSE, INR. ; "
                             "[fundamental] Forward P/E 18.14x (2026-11-13).")
    assert (turn.asked, analyses.asked) == (ASSESSMENT, ["RELIANCE.NS"])


@pytest.mark.usefixtures("generator_writes_facts")
def test_a_facts_answer_stating_a_figure_its_passage_lacks_is_refused(tmp_path):
    turn = facts_agent(tmp_path, Analyses(), written="reliance industries limited trades at 21.40 times profit .").answer(
        ASSESSMENT)
    assert (turn.spoke, turn.because, turn.unsupported) == (False, "unsupported figure", ["21.40"])


@pytest.mark.usefixtures("generator_writes_facts")
def test_a_facts_answer_giving_a_figure_as_another_metrics_is_refused(tmp_path):
    """Every figure is in the passage, so only the label guard stands between this answer and the user."""
    roa = {"value": 11.0, "unit": "%", "source": "yahoo", "period": "FY ending 2026-03-31"}
    turn = facts_agent(tmp_path, Analyses(analysis(forward_pe=FORWARD_PE, roa=roa)),
                       written="reliance industries limited had a forward p / e of 11.00% .").answer(ASSESSMENT)
    assert (turn.spoke, turn.because) == (False, "mislabelled figure: 11.00% as forward p / e")


@pytest.mark.usefixtures("generator_writes_facts")
@pytest.mark.parametrize("facts_cut,spoke", [(0.4, True), (0.6, False)])
def test_a_facts_answer_is_spoken_by_the_facts_cut_not_the_advisory_one(tmp_path, facts_cut, spoke):
    """Facts answers' confidence averaged 0.45, and the advisory cut, fitted near 0.8, refused nearly all of them."""
    turn = facts_agent(tmp_path, Analyses(), model=FakeModel(0.5), facts_cut=facts_cut).answer(ASSESSMENT)
    assert (turn.spoke, turn.because) == (spoke, "" if spoke else "low confidence")


@pytest.mark.usefixtures("generator_writes_facts")
@pytest.mark.parametrize("written", ["our data has no esg figures for reliance industries limited .",
                                     "for reliance industries limited , momentum is rising ."])
def test_a_facts_answer_is_not_read_for_a_share_where_the_advisory_answer_forms_put_one(tmp_path, written):
    """Those slots refused 56-71% of the frozen set's grounded short answers: "names another share: data"."""
    turn = facts_agent(tmp_path, Analyses(), written=written).answer(ASSESSMENT)
    assert (turn.spoke, turn.because) == (True, "")


@pytest.mark.usefixtures("generator_writes_facts")
def test_a_facts_answer_naming_its_own_share_by_its_full_ticker_is_served(tmp_path):
    """The Stock line gave the bare symbol, so "reliance.ns", as questions and advisory answers name it, was another share."""
    turn = facts_agent(tmp_path, Analyses(), written="reliance.ns is valued against the profit it is expected to make .") \
        .answer(ASSESSMENT)
    assert (turn.spoke, turn.because) == (True, "")


def test_the_facts_passage_fits_the_window_with_the_separators_between_its_lines(tmp_path):
    """Its lines fill the window, and with their separators left uncounted the passage was built over it and the turn
    raised out of `assemble`."""
    many = {key: {"value": 10.0 + at, "unit": "%", "source": "yahoo", "period": "FY ending 2026-03-31"}
            for at, key in enumerate(("gross_margin", "roa", "revenue_cagr", "latest_quarter_revenue_yoy", "forward_pe"))}
    turn = facts_agent(tmp_path, Analyses(analysis(**many)), written="its gross margin is 10.00% .").answer(ASSESSMENT)
    room = evidence_room(LENGTH, finance.FACTS_QUESTION_TOKENS)
    assert turn.spoke and len(FakeTokenizer().encode(turn.evidence)) <= room


ROA = {"value": 11.0, "unit": "%", "source": "yahoo", "period": "FY ending 2026-03-31"}
PROFITABLE = "how profitable is {} ?"
DIVIDEND_ASKED = "what is the dividend yield of {} ?"


def test_a_facts_question_is_answered_in_our_words_from_its_passage_and_never_by_the_generator(tmp_path):
    """Read by hand, 54 of the 68 facts answers step36000 would speak were wrong in their words, every figure right."""
    turn = facts_agent(tmp_path, Analyses(analysis(roa=ROA)), model=ExplodingModel(), asked=(PROFITABLE,)).answer(
        PROFITABLE.format("RELIANCE.NS"))
    assert (turn.intent, turn.spoke, turn.because, turn.served) == (finance.FACTS, True, "", turn.answer)
    assert "Reliance Industries Limited (RELIANCE.NS)" in turn.served
    assert turn.served.endswith(" Return on assets is 11.00% (FY ending 2026-03-31).")


def test_a_facts_line_the_passage_left_out_is_answered_and_shown_among_the_facts_used(tmp_path):
    """59 of 323 replies said a figure was not held when the share's facts held it: the picker fits the passage to the
    generator's window, and the reply was given nothing else."""
    many = {key: {"value": 10.0 + at, "unit": "%", "source": "yahoo", "period": "FY ending 2026-03-31"}
            for at, key in enumerate(stock_qa.FUNDAMENTAL_LABELS)}
    price_file(tmp_path, "RELIANCE.NS")
    picked_last = lambda texts: lambda question: [-float("Dividend" in text) for text in texts]
    turn = Agent(finance, finance.Market(tmp_path, analyse=Analyses(analysis(**many))), facts_router(DIVIDEND_ASKED),
                 Abstainer(0.01), FakeTokenizer(), ExplodingModel(), Config, Abstainer(0.9), facts=picked_last).answer(
        DIVIDEND_ASKED.format("RELIANCE.NS"))
    assert turn.served.endswith(" Dividend yield is 19.00% (FY ending 2026-03-31).")
    assert turn.evidence.endswith(f"{stock_qa.FACT_SEPARATOR}[fundamental] Dividend yield 19.00% (FY ending 2026-03-31).")


TCS_PASSAGE = stock_qa.FACT_SEPARATOR.join((
    "Stock: Tata Consultancy Services Limited (TCS.NS), NSE, INR; last close 2,407.50 on 2026-11-13.",
    "[profile] Tata Consultancy Services Limited provides information technology (IT) and IT enabled services.",
    "Sector Technology, industry Information Technology Services (yahoo profile).",
    "[fundamental] P/E is 22.5 (absolute bands; not sector-adjusted). (2026-11-13).",
    "[fundamental] Debt/equity is 0.10. (FY ending 2026-03-31).",
    "[fundamental] Revenue changed +5.4% year on year. (FY ending 2026-03-31 vs FY ending 2025-03-31).",
    "[fundamental] Return on equity is 45.9%. (FY ending 2026-03-31).",
    "[fundamental] Dividend yield 1.14% (2026-11-13).",
    "[forecast, model output] 80% expected 5d range 2,350.10–2,466.90 (EWMA volatility; historical coverage 78%). "
    "(2026-11-13).", "[fundamental] Not available in our data: peg."))
# the share's facts when its passage holds every one
TCS_LINES = TCS_PASSAGE.split(stock_qa.FACT_SEPARATOR)


def lacks_any(told, name="Tata Consultancy Services Limited (TCS.NS)"):
    return any(frame.split("{what}")[0].format(name=name) in told for frame in facts_reply.GAPS)


@pytest.mark.parametrize("question,answering", [
    ("what does TCS do ?", "Tata Consultancy Services Limited provides information technology (IT) and IT enabled "
                           "services. Yahoo lists its sector as Technology and its industry as Information Technology Services."),
    ("is TCS expensive ?", "P/E is 22.5 (absolute bands; not sector-adjusted) (2026-11-13)."),
    ("how much debt does TCS have ?", "Debt/equity is 0.10 (FY ending 2026-03-31)."),
    ("how fast is TCS growing ?", "Revenue changed +5.4% year on year (FY ending 2026-03-31 vs FY ending 2025-03-31)."),
    ("is TCS making money ?", "Return on equity is 45.9% (FY ending 2026-03-31)."),
    ("what dividend does TCS pay ?", "Dividend yield is 1.14% (2026-11-13)."),
    ("how fast are TCS sales rising ?", "Revenue changed +5.4% year on year (FY ending 2026-03-31 vs FY ending 2025-03-31)."),
    ("is TCS risky ?", "Our model's reading: 80% expected 5d range 2,350.10–2,466.90 (EWMA volatility; historical "
                       "coverage 78%) (2026-11-13).")],
    ids=["profile", "valuation", "debt", "growth", "profit", "dividend", "growth with a price word", "model output"])
def test_a_facts_reply_gives_the_lines_of_the_area_its_question_asks_about_after_one_lead_in(question, answering):
    """Every figure and date as the passage gives them, and none of the other areas' lines."""
    told = facts_reply.reply(question, TCS_PASSAGE, TCS_LINES).text
    assert told.endswith(f". {answering}") and "Tata Consultancy Services Limited (TCS.NS)" in told
    assert guardrails.figures(told[:-len(answering)]) == []


@pytest.mark.parametrize("question,lacking,closest", [
    ("what is the forward P/E of TCS ?", "its forward P/E", "P/E is 22.5 (absolute bands; not sector-adjusted) (2026-11-13)."),
    ("what is the earnings per share of TCS ?", "its EPS", "Not available in our data: peg."),
    ("who are the main competitors of TCS ?", "a list of its competitors", "Not available in our data: peg.")],
    ids=["a metric", "a metric spelt out", "an area"])
def test_a_facts_reply_says_plainly_what_its_passage_lacks_then_gives_the_closest_facts(question, lacking, closest):
    told = facts_reply.reply(question, TCS_PASSAGE, TCS_LINES).text
    gaps = [frame.format(name="Tata Consultancy Services Limited (TCS.NS)", what=lacking) for frame in facts_reply.GAPS]
    assert any(told.startswith(gap) for gap in gaps) and told.endswith(f". {closest}")


def test_a_facts_question_naming_several_areas_is_answered_from_every_fact_its_passage_holds():
    """Taken by its first area, a scorecard asking for growth, profit and debt was answered on debt alone."""
    told = facts_reply.reply("score TCS on its growth , its profit and its debt .", TCS_PASSAGE, TCS_LINES).text
    assert all(facts_reply.stated(line) in told for line in TCS_LINES[1:])


def test_a_facts_reply_varies_its_lead_in_with_the_question_and_picks_the_same_one_in_every_process():
    asked = ["is TCS expensive ?", "is TCS expensive now ?", "is TCS too expensive ?", "is TCS expensive today ?"]
    told = [facts_reply.reply(question, TCS_PASSAGE, TCS_LINES).text for question in asked]
    assert len({text.split(" P/E is")[0] for text in told}) > 1
    # a hash salted per process, as hash() is, would pick another lead-in on another run
    assert told[0].startswith("Our facts give this on the valuation of Tata Consultancy Services Limited (TCS.NS).")


@pytest.mark.parametrize("question,at", [("what is the price to earnings ratio for TCS.NS ?", 3),
                                         ("is TCS making money ?", 6)], ids=["a metric", "an area"])
def test_a_facts_reply_answers_from_a_line_of_the_share_its_passage_left_out_and_lists_it_with_the_facts(question, at):
    """59 of 323 replies said our facts lacked a figure the share's facts held: 57 a P/E the picker left out."""
    # the share's line with its source, as `stock_qa.facts` writes it; the passage's lines are as the picker leans them
    lines = [*TCS_LINES[:at], TCS_LINES[at].replace(". (", ". (source yahoo; "), *TCS_LINES[at + 1:]]
    passage = stock_qa.FACT_SEPARATOR.join(TCS_LINES[:at] + TCS_LINES[at + 1:])
    told = facts_reply.reply(question, passage, lines)
    assert told.text.endswith(f". {facts_reply.stated(TCS_LINES[at])}") and not lacks_any(told.text)
    assert told.evidence == stock_qa.FACT_SEPARATOR.join((passage, TCS_LINES[at]))


@pytest.mark.parametrize("peak", [
    "[risk] Price is 35.9% below its peak in the loaded history. (2021-11-16 to 2026-11-13).",
    "[risk] The price is 35.87% below its highest close in the loaded history (as of 2026-11-13)."])
def test_a_price_question_is_answered_by_how_far_the_price_is_below_its_peak_and_never_told_it_is_not_held(peak):
    """DMART was told nothing in our facts gave readings of its price, then shown how far it was below its peak."""
    passage = stock_qa.FACT_SEPARATOR.join((TCS_LINES[0], "[fundamental] Price to book (P/B) 8.64x (2026-11-13).", peak))
    told = facts_reply.reply("how much has the price of TCS moved against the market ?", passage,
                             passage.split(stock_qa.FACT_SEPARATOR)).text
    assert told.endswith(f". {facts_reply.stated(peak)}") and "P/B" not in told and not lacks_any(told)


PRICE_LINES = ("Price change to 2026-11-13: 1 day +0.50%, 1 week -1.20%, 1 month +3.10% (computed from yahoo daily bars).",
               "[technical] Price is above the 50-day average (2,350.00) and the average is rising. (2026-11-13).",
               "[risk] The price is 4.20% below its highest close in the loaded history (as of 2026-11-13).")


@pytest.mark.parametrize("question", ["why did TCS rise even though the market fell ?",
                                      "is the rise in TCS getting stronger or weaker ?",
                                      "is selling volume increasing in TCS ?",
                                      "what does today's trading volume say about TCS ?",
                                      "compare the relative strength of TCS with its sector index ."])
def test_a_question_on_how_its_price_moved_is_answered_by_its_price_lines_not_its_profile_or_valuation(question):
    """Naming no area, "is selling volume increasing in Titan" was answered with its P/E and EV/sales."""
    told = facts_reply.reply(question, TCS_PASSAGE, [*TCS_LINES, *PRICE_LINES]).text
    assert told.endswith(" ".join(map(facts_reply.stated, PRICE_LINES))) and "P/E" not in told


@pytest.mark.parametrize("question", ["is the business of TCS getting stronger ?",
                                      "what does TCS do at the global level ?"])
def test_a_question_on_its_business_with_a_price_move_word_is_answered_by_its_profile_not_its_price_lines(question):
    """Both were answered with TCS's price change and averages, for their "stronger" and "level"."""
    told = facts_reply.reply(question, TCS_PASSAGE, [*TCS_LINES, *PRICE_LINES]).text
    assert told.endswith(" ".join(map(facts_reply.stated, TCS_LINES[1:3]))) and "Price change" not in told


def test_a_facts_reply_answering_one_metric_of_two_ends_saying_which_it_lacks():
    told = facts_reply.reply("what are the P/E and forward P/E of TCS ?", TCS_PASSAGE, TCS_LINES).text
    gaps = [frame.format(name="Tata Consultancy Services Limited (TCS.NS)", what="its forward P/E")
            for frame in facts_reply.GAPS]
    assert "P/E is 22.5" in told and any(told.endswith(f". {gap}") for gap in gaps)


@pytest.mark.parametrize("question", ["who are the main competitors of TCS ?", "what is the beta of TCS ?"])
def test_a_facts_reply_never_gives_the_definitions_line_as_a_fact(question):
    """`stock_qa.facts` writes it second for every share, and its "a beta of 1 means" names a beta TCS's facts lack."""
    lines = [TCS_LINES[0], stock_qa.DEFINITIONS, *TCS_LINES[1:]]
    told = facts_reply.reply(question, stock_qa.FACT_SEPARATOR.join(lines), lines).text
    assert "Definitions" not in told and lacks_any(told)


BETA_BIONICS = "Stock: Beta Bionics, Inc. (BBNX), US, USD; last close 21.30 on 2026-11-13."
BETA = "[risk] Beta to the benchmark is 1.21 (correlation 0.35). (2025-11-13 to 2026-11-13)."


@pytest.mark.parametrize("stock,question,answering,other", [
    ("Stock: Persistent Systems Limited (PERSISTENT.NS), NSE, INR; last close 5,012.00 on 2026-11-13.",
     "is Persistent Systems expensive ?", "[fundamental] P/E is 61.4 (absolute bands; not sector-adjusted). (2026-11-13).",
     "[technical] RSI(14) is 55.0 — neutral momentum zone. (2026-11-13)."),
    (BETA_BIONICS, "what is the beta of Beta Bionics ?", BETA, "[profile] Beta Bionics, Inc. develops a bionic pancreas."),
    (BETA_BIONICS, "is Beta Bionics risky ?", BETA, "[profile] Beta Bionics, Inc. develops a bionic pancreas.")],
    ids=["its question", "a line naming a metric", "a line of the area"])
def test_a_facts_reply_reads_its_question_and_its_lines_without_the_shares_names(stock, question, answering, other):
    """Read with them, "Persistent" asks for an RSI and "Beta Bionics" gives a beta."""
    passage = stock_qa.FACT_SEPARATOR.join((stock, other, answering))
    told = facts_reply.reply(question, passage, passage.split(stock_qa.FACT_SEPARATOR)).text
    assert told.endswith(f". {facts_reply.stated(answering)}") and facts_reply.stated(other) not in told


def test_a_facts_reply_our_code_wrote_is_labelled_as_ours(tmp_path):
    """Labelled "", the browser showed it as ALFA's answer from its sources, or "in its own words" past a writer."""
    turn = facts_agent(tmp_path, Analyses(analysis(roa=ROA)), model=ExplodingModel(), asked=(PROFITABLE,)).answer(
        PROFITABLE.format("RELIANCE.NS"))
    assert (turn.spoke, turn.phrased_by, core.alfa_spoke(turn)) == (True, core.BUILT, False)


def test_a_facts_reply_is_shown_with_term_meanings_entry_for_the_figure_it_gives(tmp_path):
    from backend.advisory.report import TERM_MEANINGS
    from backend.models.agent.present import present
    dividend = {"value": 0.4, "unit": "%", "source": "yahoo", "period": "FY ending 2026-03-31"}
    turn = facts_agent(tmp_path, Analyses(analysis(dividend_yield=dividend)), asked=(DIVIDEND_ASKED,)).answer(
        DIVIDEND_ASKED.format("RELIANCE.NS"))
    assert f"- Dividend yield: {TERM_MEANINGS['Dividend yield']}." in present(turn, {}, {})


def test_alfas_figures_say_they_read_closes_not_adjusted_for_dividends_and_do(tmp_path):
    """INFY's 20-day move read +4.9% in ALFA's figures and +7.3% in our analysis on one screen, with nothing to say why:
    ALFA trained on closes as traded, and our analysis reads closes adjusted for a dividend inside the window."""
    from backend.models.agent.present import figures_line
    path = price_file(tmp_path, "INFY.NS")
    rows = list(csv.reader(open(path)))
    for row in rows[1:-1]:          # a dividend went ex on the last day
        row[5] = f"{float(row[4]) * 0.98:.2f}"
    csv.writer(open(path, "w", newline="")).writerows(rows)
    evidence, shown, _ = finance.Market(tmp_path).snapshot("INFY.NS")
    raw = [float(row[4]) for row in rows[1:]]
    assert shown["return_20d"] == f"{raw[-1] / raw[-21] - 1:+.1%}"
    assert figures_line(evidence, "₹").startswith("**Key figures**, on closes not adjusted for dividends · ")


def test_a_facts_reply_our_code_wrote_keeps_its_own_capitals():
    """"Apple Inc. designs, manufactures and markets…" was shown as "Apple Inc. Designs"."""
    from backend.models.agent.present import present
    said = "For Apple Inc. (AAPL), our facts on its business are these. Apple Inc. designs, manufactures and markets phones."
    turn = core.Turn("what does apple do ?", "", "AAPL", finance.FACTS, said, said, "[profile] Apple Inc. designs phones.",
                     float("nan"), None, True, 0.5, [], "", "2026-11-13", core.BUILT)
    assert said in present(turn, {}, {})


@pytest.mark.usefixtures("generator_writes_facts")
def test_a_facts_answer_the_generator_gives_a_meaning_of_its_own_is_refused(tmp_path):
    dividend = {"value": 0.4, "unit": "%", "source": "yahoo", "period": "FY ending 2026-03-31"}
    written = "for reliance industries limited , its dividend yield , spare cash as a share of market value , was 0.40% ."
    turn = facts_agent(tmp_path, Analyses(analysis(dividend_yield=dividend)), written=written,
                       asked=(DIVIDEND_ASKED,)).answer(DIVIDEND_ASKED.format("RELIANCE.NS"))
    assert (turn.spoke, turn.because) == (False, "its own meaning of dividend yield")


@pytest.mark.parametrize("analyses", [None, Analyses(fails=True)], ids=["offline", "yahoo failed"])
def test_a_facts_question_with_no_analysis_to_read_is_refused_by_name(tmp_path, analyses):
    turn = facts_agent(tmp_path, analyses, model=ExplodingModel()).answer(ASSESSMENT)
    assert (turn.spoke, turn.served, turn.ticker, turn.intent) == (False, finance.NO_FACTS, "RELIANCE.NS", finance.FACTS)
    assert turn.because.startswith("no facts: ")


def test_an_index_is_refused_facts_without_an_analysis_being_read(tmp_path):
    """Analysed as a US share, the Nifty's beta was quoted against the S&P 500, and no facts row covers an index."""
    analyses = Analyses()
    price_file(tmp_path, "^NSEI")
    turn = facts_agent(tmp_path, analyses, model=ExplodingModel()).answer(ASSESSED.format("the nifty"), subject="^NSEI")
    assert (turn.served, turn.ticker, turn.because) == (finance.NO_FACTS, "^NSEI",
                                                        "no facts: ^NSEI is an index, and the facts route trained on shares only")
    assert analyses.asked == []


def test_an_analysis_newer_than_the_date_asked_for_is_refused_not_read(tmp_path):
    turn = facts_agent(tmp_path, Analyses(), model=ExplodingModel()).answer(ASSESSMENT, as_of="2026-10-01")
    assert turn.because == "no facts: the analysis of RELIANCE.NS reads prices to 2026-11-13, after 2026-10-01"


def test_an_analysis_is_kept_until_its_markets_next_close_then_fetched_again(tmp_path):
    """Kept a flat day, one fetched at 11:00 quoted that morning's price as the day's close until the next morning."""
    analyses = Analyses()
    price_file(tmp_path, "RELIANCE.NS")
    market = finance.Market(tmp_path, analyse=analyses)
    assert market.facts("RELIANCE.NS") == market.facts("RELIANCE.NS") and analyses.asked == ["RELIANCE.NS"]
    kept = tmp_path / "facts" / "RELIANCE.NS.json"
    closed = finance.last_close(finance.market_of("RELIANCE.NS"), time.time())
    os.utime(kept, (closed + 1, closed + 1))
    market.facts("RELIANCE.NS")
    assert analyses.asked == ["RELIANCE.NS"]
    os.utime(kept, (closed - 60, closed - 60))
    market.facts("RELIANCE.NS")
    assert analyses.asked == ["RELIANCE.NS"] * 2


@pytest.mark.parametrize("now,closed", [("2026-11-16T10:00+05:30", "2026-11-15T15:30+05:30"),
                                        ("2026-11-16T15:30+05:30", "2026-11-16T15:30+05:30"),
                                        ("2026-11-16T03:00+00:00", "2026-11-15T16:00-05:00")], ids=["NSE open", "NSE close",
                                                                                                    "US overnight"])
def test_a_markets_last_close_is_the_latest_one_at_or_before_now(now, closed):
    from datetime import datetime
    market = finance.market_of("RELIANCE.NS" if "+05:30" in now else "AAPL")
    assert finance.last_close(market, datetime.fromisoformat(now).timestamp()) == datetime.fromisoformat(closed).timestamp()


def test_an_analysis_fetched_again_is_written_whole_so_a_failed_write_leaves_the_last_one_readable(tmp_path, monkeypatch):
    """Written in place, a fetch that failed halfway, or one still writing after it was given up on, left half a file."""
    from pathlib import Path
    first = analysis()
    price_file(tmp_path, "RELIANCE.NS")
    market = finance.Market(tmp_path, analyse=Analyses(first))
    market.facts("RELIANCE.NS")
    kept = tmp_path / "facts" / "RELIANCE.NS.json"
    os.utime(kept, (0, 0))
    market.analyse, write = Analyses(analysis(forward_pe={**FORWARD_PE, "value": 21.4})), Path.write_text

    def half_then_full_disk(path, text, **options):
        write(path, text[: len(text) // 2], **options)
        raise OSError("no space left on device")
    monkeypatch.setattr(Path, "write_text", half_then_full_disk)
    with pytest.raises(OSError, match="no space"):
        market.facts("RELIANCE.NS")
    assert json.loads(kept.read_text()) == first


def test_a_slow_analysis_is_given_up_on_and_backs_off_price_fetches_too(tmp_path, monkeypatch):
    """One Yahoo, one worker and one back-off for prices and analyses alike, so a hung Yahoo stalls one question."""
    monkeypatch.setattr(finance, "YAHOO_SLOW_S", 0.05)
    released, yahoo = threading.Event(), Yahoo()

    def hung(ticker):
        released.wait(5)
        return analysis()
    price_file(tmp_path, "RELIANCE.NS")
    market = finance.Market(tmp_path, shares=SHARES, fetch=yahoo, analyse=hung)
    with pytest.raises(finance.NoFacts, match="in 0.05 s"):
        market.facts("RELIANCE.NS")
    with pytest.raises(finance.NoPrices, match="not fetched until then"):
        market.hold("NYKAA.NS")
    released.set()
    assert yahoo.asked == []


@pytest.mark.parametrize("places_facts,scorer,cut,generated", [
    (True, None, None, False), (False, shared_words, None, False), (True, shared_words, Abstainer(0.5), False),
    (True, shared_words, None, True), (False, None, Abstainer(0.5), True)],
    ids=["router without scorer", "scorer without router", "a facts cut our code's answers never read",
         "no facts cut for the generator's answers", "facts cut alone"])
def test_a_facts_scorer_a_facts_cut_and_a_router_placing_facts_questions_come_together_or_not_at_all(
        monkeypatch, market, router, places_facts, scorer, cut, generated):
    monkeypatch.setattr(core, "GENERATOR_WRITES_FACTS", generated)
    with pytest.raises(ValueError, match="facts scorer exactly when"):
        Agent(finance, market, facts_router() if places_facts else router, Abstainer(0.01), FakeTokenizer(),
              FakeModel(), Config, Abstainer(0.9), facts=scorer, facts_abstainer=cut)


def test_the_facts_route_keeps_the_old_intents_examples_and_adds_its_frames_and_only_the_dev_short_templates():
    """A test template among the examples would be placed by matching itself, and its score reported as held out."""
    known, old = finance.examples(facts=True), finance.examples()
    added = {text for _, _, text in known[len(old):]}

    def asked(split):
        return {finance._squeeze(question.format(t="")) for _, question in finance.short_templates(split)}
    assert known[:len(old)] == old
    assert asked("dev") <= added and not asked("test") & added


def test_the_short_templates_test_split_is_a_fifth_of_each_category_the_same_on_every_run():
    held = json.loads(finance.SHORT_TEMPLATES.read_text(encoding="utf-8"))
    category = {row["id"]: row["category"] for row in held["templates"]}
    assert {row["id"] for row in held["templates"] if row["split"] == "test"} == \
        stock_qa.held_out(sorted(category), 0.2, category, "price_short_v1")


def test_a_short_template_labelled_with_no_route_is_refused_by_name(tmp_path, monkeypatch):
    held = json.loads(finance.SHORT_TEMPLATES.read_text(encoding="utf-8"))
    held["categories"]["volume"]["label"] = "volume"
    monkeypatch.setattr(finance, "SHORT_TEMPLATES", tmp_path / "templates.json")
    finance.SHORT_TEMPLATES.write_text(json.dumps(held), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\['volume'\], which no route answers"):
        finance.short_templates("dev")


def test_the_short_templates_ship_with_the_package():
    """Read from the installed package on the VM, where only package-data patterns are copied."""
    import tomllib
    from fnmatch import fnmatch
    from backend.paths import PACKAGE
    patterns = tomllib.loads((PACKAGE.parent / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    assert any(fnmatch(finance.SHORT_TEMPLATES.relative_to(PACKAGE).as_posix(), pattern)
               for pattern in patterns["backend"])


@pytest.mark.parametrize("text,facts", [
    ("is expensive ?", True), ("how much debt does have ?", True), ("what does do ?", True),
    ("what does the rsi say about ?", False), ("what does your model say about risk in ?", False),
    ("give me a summary of how is trading .", False), ("give me a summary of how is tradinng", False),
    ("is likely to be eventful from here ?", False), ("should i buy for its dividend ?", False),
    ("should i hold through earnings", False), ("what does next week look like for", False),
    ("what does the price action say about", False), ("is this a cheap entry on", False),
    ("how volatile is 's revenue", True), ("how diversified is 's business", True),
    ("what does the dip in mean", False), ("what does 's slump mean", False), ("what does tata steel make", True),
    ("what does sell ?", True), ("is 's debt high", True), ("what is 's business model", True),
    ("what is 's long term debt", True), ("what is the p/e of right now", True),
    ("what is 's one year revenue growth", True), ("is it time to exit before earnings", False),
    ("is it time to sell now that the dividend is paid", False), ("should i book profits after its earnings", False),
    ("what does the dip do to", False), ("what does the selloff do to", False),
    ("did its price rise on the profit news", False), ("is 's price sliding on weak revenue", False),
    ("did tank on its debt news", False), ("did rise on its earnings", False), ("when does report earnings", False),
    ("when is 's next earnings date ?", False)])
def test_a_question_goes_to_facts_by_its_words_only_when_it_names_nothing_a_price_intent_answers(text, facts):
    """The price intents' words are their own questions', so "hold" and "next week" were not missed again. "what does"
    alone or "what does the dip do to" named a profile, a move word beside "profit" went unseen, "when does it report
    earnings" got its profit figures for a date, and everyday words such as "high" kept facts questions off."""
    assert finance.asks_facts(text) == facts


def test_no_facts_frame_and_no_trained_unsupported_phrasing_holds_a_price_word():
    """"high" was a price word, so "is tcs's debt high" and the facts frame "is {t} priced too high ?" were kept off."""
    asked = [text for label, _, text in finance.examples() if label == finance.UNANSWERABLE] + \
        [frame.format(t="") for frame in finance.FACTS_PARAPHRASES]
    assert [text for text in asked if finance._PRICE_WORD.search(f"{text} ")] == []


def test_a_question_the_word_rule_names_gets_the_facts_bonus_and_its_margin_still_faces_the_routing_cut():
    """At a margin of 1.0 the word rule put "should i book profit in tcs now", a buy question, past every cut."""
    known = finance.examples(facts=True)
    near = 0.8 - finance.FACTS_BONUS / 2
    scores = np.array([0.8 if label == "buy" else near if label == finance.FACTS else 0.0 for label, _, _ in known])
    router = finance.FactsFirst(known, lambda text: scores)
    placed = router.route("is expensive ?")
    assert (placed.label, placed.margin) == (finance.FACTS, pytest.approx(finance.FACTS_BONUS / 2))
    assert router.route("should i buy for its dividend ?").label == "buy"


def test_a_question_the_word_rule_names_placed_on_the_unsupported_intent_is_answered_from_the_facts():
    """Seven of the 60 facts_dev questions, "how much debt does tcs have" among them, matched "the debt load" there."""
    known = finance.examples(facts=True)
    scores = np.array([0.9 if label == finance.UNANSWERABLE else 0.0 for label, _, _ in known])
    router = finance.FactsFirst(known, lambda text: scores)
    assert [router.route(text).label for text in ("how much debt does have ?", "who is the chief executive of ?")] == \
        [finance.FACTS, finance.UNANSWERABLE]


def test_a_question_the_word_rule_names_keeps_the_old_examples_wider_unsupported_margin_as_facts():
    """The added facts frames took five control questions under the routing cut that the old examples alone place on
    the unsupported intent past it."""
    known, old = finance.examples(facts=True), set(finance.examples())
    scores = np.array([0.5 if example in old and example[0] == finance.UNANSWERABLE else
                       0.52 if example[0] == finance.FACTS else 0.0 for example in known])
    router = finance.FactsFirst(known, lambda text: scores)
    assert [(placed.label, placed.margin) for placed in map(router.route, ("is debt free ?", "who runs ?"))] == \
        [(finance.FACTS, pytest.approx(0.5)), (finance.FACTS, pytest.approx(0.02))]


def test_a_question_placed_on_facts_that_names_an_indicator_no_evidence_holds_is_unsupported():
    """The MACD and RSI templates put "what does the 20-day EMA say about tcs" on facts past the routing cut."""
    known = finance.examples(facts=True)
    scores = np.array([0.9 if label == finance.FACTS else 0.0 for label, _, _ in known])
    router = finance.FactsFirst(known, lambda text: scores)
    assert [router.route(text).label for text in ("what does the 20-day EMA say about ?", "what does the MACD say ?",
                                                  "is emami expensive ?")] == \
        [finance.UNANSWERABLE, finance.FACTS, finance.FACTS]


@pytest.mark.parametrize("question,intent", [("is going down ?", "drawdown"), ("is dropping", "drawdown"),
                                             ("how is trending", "performance"), ("is strong", "performance"),
                                             ("has entered a correction", "drawdown")])
def test_a_short_direction_question_stays_on_its_intent_beside_the_dev_templates_asking_why_it_moved(monkeypatch,
                                                                                                    question, intent):
    """"why is {t} going down?" is a dev facts template one word from "tcs going down?", which it took off a sure
    drawdown answer. The one on a "normal correction" sent "is tcs in a correction" to facts and its current ratio."""
    from backend.models.agent.run.ask import router_from
    from backend.models.external import sentence
    monkeypatch.setattr(sentence, "similarity", overlap)
    assert router_from("pretrained", finance.examples(facts=True), None, None, None, facts=True).route(
        question).label == intent


def test_a_trading_question_is_not_placed_on_facts_by_a_facts_frame_one_word_from_it(monkeypatch):
    """With "is {t} profitable ?" a facts frame, "is tcs profitable to trade" was a sure facts answer."""
    from backend.models.agent.run.ask import router_from
    from backend.models.external import sentence
    monkeypatch.setattr(sentence, "similarity", overlap)
    assert router_from("pretrained", finance.examples(facts=True), None, None, None, facts=True).route(
        "is profitable to trade").label != finance.FACTS


def test_a_dev_template_takes_its_label_from_its_categorys_exceptions():
    """USR-005 is a performance template and USR-170 a buy one, but only the facts lines answer either."""
    labels = {text: label for label, _, text in finance.examples(facts=True)[len(finance.examples()):]}
    assert [labels["is doing better this year?"], labels["is near a possible buying level?"]] == [finance.FACTS] * 2


def test_a_question_placed_on_a_dev_buy_template_is_asked_in_the_buy_intents_trained_wording(market):
    """With the template's own words as its key, a question placed on "is this a good time to buy {t}?" raised."""
    known, old = finance.examples(facts=True), len(finance.examples())
    scores = np.zeros(len(known))
    scores[next(at for at, (label, _, _) in enumerate(known) if at >= old and label == "buy")] = 1.0
    turn = Agent(finance, market, finance.FactsFirst(known, lambda text: scores), Abstainer(0.01),
                 FakeTokenizer("it is doing fine ."), FakeModel(), Config, Abstainer(0.9),
                 facts=shared_words).answer("is this a good time to buy AAPL ?")
    assert (turn.intent, turn.asked) == ("buy", advisory.ask("buy", "AAPL", advisory.canonical_phrasing("buy")))


def overlap(texts):
    """A cosine of word counts, the same for a text whatever else is known, as the pretrained encoder's is."""
    def bag(text):
        counts = Counter(pretokenize(text))
        norm = sum(n * n for n in counts.values()) ** 0.5 or 1.0
        return {word: n / norm for word, n in counts.items()}
    known = [bag(text) for text in texts]
    return lambda question: np.array([sum(n * words.get(word, 0.0) for word, n in bag(question).items())
                                      for words in known])


@pytest.mark.parametrize("question", ["how risky is ?", "is overbought right now ?", "should i buy ?",
                                      "what does the rsi say about ?", "how far has fallen from its high ?"])
def test_the_served_router_leaves_an_old_intents_question_where_the_facts_off_router_puts_it_and_no_less_sure(
        monkeypatch, question):
    """The added examples took "how risky is infosys" under the routing cut, and "how risky is {t} ?" as a facts
    frame would have placed it on facts."""
    from backend.models.agent.run.ask import router_from
    from backend.models.external import sentence
    monkeypatch.setattr(sentence, "similarity", overlap)
    on = router_from("pretrained", finance.examples(facts=True), None, None, None, facts=True)
    off = router_from("pretrained", finance.examples(), None, None, None, facts=False)
    placed, before = on.route(question), off.route(question)
    assert placed.label == before.label and placed.margin >= before.margin > 0


def test_the_facts_gate_is_fitted_counting_an_old_unsupported_question_placed_on_facts_as_placed_right():
    """Counted wrong, the gate refused at a cut set by the very questions the route was made to answer."""
    from backend.models.agent.router import Route
    from backend.models.external.route_encoders import ANSWERED_BY_FACTS
    from backend.models.training import route
    on_facts = SimpleNamespace(route=lambda text: Route(finance.FACTS, "", text, 1.0, 0.5))
    asked = [(finance.UNANSWERABLE, "word", "what is its free cash flow ?"), ("volatility", "word", "how much does it swing ?")]
    assert route.measure(on_facts, asked, ANSWERED_BY_FACTS) == [("word", True, 0.5), ("word", False, 0.5)]


def test_the_gate_fitter_measures_the_facts_route_as_it_is_served(monkeypatch):
    """Without the facts route's words in front, its cut and the facts label were read off a router never served."""
    import sys
    from backend.models.external import route_encoders, sentence
    from backend.models.training import route
    routers, measure = [], route.measure
    monkeypatch.setattr(route, "scorers", lambda *args: [])
    monkeypatch.setattr(route, "measure", lambda router, *args: routers.append(router) or measure(router, *args))
    monkeypatch.setattr(route_encoders, "facts_label", lambda router, gate, asked: routers.append(router))
    monkeypatch.setattr(sentence, "similarity", lambda texts: lexical(texts, pretokenize))
    monkeypatch.setattr(sys, "argv", ["route_encoders", "--facts", "--weights"])
    route_encoders.main()
    assert [type(router) for router in routers] == [finance.FactsFirst] * 2


def test_the_gate_fitter_reports_both_halves_of_the_held_out_routing_set(capsys):
    """No set the gate was fitted or checked on had the word rule name a price question, so its mistakes went unseen."""
    from backend.models.agent.router import Route
    from backend.models.external import route_encoders
    on_facts = SimpleNamespace(route=lambda text: Route(finance.FACTS, finance.FACTS, text, 1.0, 1.0))
    route_encoders.held_out(on_facts, Abstainer(0.5), lambda question: None)
    assert [line.split(";")[0] for line in capsys.readouterr().out.splitlines()] == [
        "  held-out dev: 46/130 placed right past the cut, 84 wrong past it",
        "  held-out test: 41/116 placed right past the cut, 75 wrong past it"]


@pytest.mark.parametrize("copied", ["an example", "a dev question"])
def test_no_question_of_the_held_out_test_half_is_routed_as_the_text_of(copied):
    """Eight were the round's new examples once the share was taken out, and were placed by matching themselves: three
    of the 18 gains reported as held out. The dev half is what the router was tuned on."""
    from backend.models.external import route_encoders
    held, resolve = json.loads(route_encoders.HELD_OUT.read_text(encoding="utf-8")), route_encoders.served_resolve()

    def bare(text):
        return " ".join(text.lower().replace("?", " ").split())

    def routed(question):
        ticker = resolve(question)
        return bare(finance.without_subject(question, ticker) if ticker else question)
    against = ({bare(text) for _, _, text in finance.examples(facts=True)} if copied == "an example"
               else {routed(row["question"]) for row in held["dev"]})
    assert [row["question"] for row in held["test"] if routed(row["question"]) in against] == []


def promoted_generator(tmp_path, **entry):
    """A generator file and a registry promoting it with `entry`'s fields, as a promotion writes them."""
    from backend.models.serving import registry
    artifact = tmp_path / "generator.npz"
    artifact.write_bytes(b"weights")
    (tmp_path / "registry.json").write_text(json.dumps({registry.GENERATOR: {"digest": registry.digest(artifact),
                                                                             **entry}}))
    return artifact


@pytest.mark.parametrize("entry,live,on", [({}, True, False), ({"trained_on": ["advisory_casual"]}, True, False),
                                           ({"trained_on": ["advisory_casual", finance.STOCK_QA]}, False, False),
                                           ({"trained_on": ["advisory_casual", finance.STOCK_QA]}, True, True)],
                         ids=["record before trained_on", "advisory rows only", "facts rows offline", "facts rows live"])
def test_the_facts_route_is_on_only_live_for_a_generator_promoted_as_trained_on_facts_rows_and_keeps_the_gate(tmp_path,
                                                                                                             entry, live, on):
    """Offline no analysis is read, so a facts question routed there could only be refused."""
    from backend.models.agent.run.ask import routing
    known, gate, facts = routing("pretrained", promoted_generator(tmp_path, **entry), tmp_path / "registry.json", live=live)
    assert (facts, gate) == (on, "route_gate_bge.json")
    assert any(label == finance.FACTS for label, _, _ in known) == on


def test_a_candidate_the_registry_does_not_promote_never_turns_the_facts_route_on(tmp_path):
    from backend.models.agent.run.ask import routing
    promoted_generator(tmp_path, trained_on=[finance.STOCK_QA])
    candidate = tmp_path / "candidate.npz"
    candidate.write_bytes(b"other weights")
    assert routing("pretrained", candidate, tmp_path / "registry.json", live=True)[1:] == ("route_gate_bge.json", False)


def test_a_generator_trained_on_facts_rows_is_refused_a_router_whose_gate_was_not_measured_with_them(tmp_path):
    from backend.models.agent.run.ask import routing
    with pytest.raises(ValueError, match="--router pretrained"):
        routing("alfa", promoted_generator(tmp_path, trained_on=[finance.STOCK_QA]), tmp_path / "registry.json", live=True)


def test_the_facts_cut_is_the_one_fitted_for_this_generator_and_its_absence_names_the_script(tmp_path):
    from backend.models.agent.run.ask import FACTS_CUT, facts_cut_from
    from backend.models.serving import registry
    served = promoted_generator(tmp_path)
    with pytest.raises(FileNotFoundError, match="fit_facts_cut"):
        facts_cut_from(tmp_path, served)
    Abstainer(0.45).save(tmp_path / FACTS_CUT.format(registry.digest(served)))
    assert facts_cut_from(tmp_path, served).cut == 0.45


def test_a_generator_trained_on_facts_rows_is_served_live_with_no_facts_cut_while_our_code_writes_facts_answers(
        tmp_path, monkeypatch):
    """No facts cut is shipped to the VM, and requiring one there restarted the agent again and again."""
    from backend.models.agent.run import ask
    from backend.models.external import sentence
    from backend.models.learning import AGENT
    from backend.models.serving import registry
    served = promoted_generator(tmp_path)
    monkeypatch.setattr(registry, "trained_on", lambda role, artifact, path=None: (finance.STOCK_QA,))
    monkeypatch.setattr(ask, "load_model", lambda artifacts, dataset, checkpoint: (FakeTokenizer(), FakeModel(),
                                                                                    SimpleNamespace(max_text_length=LENGTH, seed=0)))
    monkeypatch.setattr(sentence, "similarity", overlap)
    monkeypatch.setattr(ask, "nse_shares", lambda: SHARES)
    monkeypatch.setattr(ask, "prices_from", Yahoo)
    monkeypatch.setattr(ask, "analyses_from", Analyses)
    (tmp_path / "prices").mkdir()
    price_file(tmp_path / "prices", "RELIANCE.NS")
    Abstainer(0.04).save(tmp_path / "gate.json")
    agent = ask.build(tmp_path, served.name, str(tmp_path / "prices"), str(tmp_path / "gate.json"),
                      str(tmp_path / "log.sqlite"), 0.6, 10, "", router="pretrained", live=True, labeller=AGENT)[0]
    assert finance.FACTS in agent.router.labels and agent.facts_abstainer is None


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


class Recording(FakeModel):
    """Keeps the source it was asked to answer from, which is the row the decoder read."""

    def generate(self, source, keep, *sampling):
        self.source = source
        return super().generate(source, keep, *sampling)


@pytest.mark.usefixtures("generator_writes_facts")
def test_a_served_facts_row_is_laid_out_exactly_as_a_training_row(tmp_path):
    """The facts window was computed twice: serving cut a facts question at the advisory 24 tokens and every test passed."""
    from backend.models.training import prepare_stock_qa
    question = ASSESSMENT.replace(" .", ", and tell me what its gross margin and return on assets say about how well "
                                        "the business turns sales into profit for its owners .")
    many = {key: {"value": 10.0 + at, "unit": "%", "source": "yahoo", "period": "FY ending 2026-03-31"}
            for at, key in enumerate(("gross_margin", "roa", "revenue_cagr", "latest_quarter_revenue_yoy", "forward_pe"))}
    analysed, tokenizer, model = analysis(**many), FakeTokenizer("its gross margin is 10.00% ."), Recording()
    # shares a word with the question only by the company's name, which the scorer takes out of both
    analysed["results"]["fundamental"]["details"]["profile"] = {"summary": "Reliance Industries Limited is a Reliance company."}
    price_file(tmp_path, "RELIANCE.NS")
    market = finance.Market(tmp_path, analyse=Analyses(analysed))
    turn = Agent(finance, market, facts_router(), Abstainer(0.01), tokenizer, model, Config, Abstainer(0.9),
                 facts=shared_words, facts_abstainer=Abstainer(0.9)).answer(question)
    row = {"id": "STKQ-0001-RELIANCE", "split": "train", "stocks": ["RELIANCE"], "template_id": "csv:integrated_stock_read",
           "interaction_type": "single_turn", "question": question, "facts": stock_qa.facts(analysed),
           "short_answer": "its gross margin is 10.00% ."}
    kept, dropped = prepare_stock_qa.generator_rows([row], {"test_stocks": [], "validation_stocks": [], "test_templates": []},
                                                    shared_words, tokenizer.encode, market.names)
    assert turn.intent == finance.FACTS and len(tokenizer.encode(question)) > finance.FACTS_QUESTION_TOKENS
    assert not dropped and np.array_equal(model.source[0], kept["train"][0][0])


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
    assert turn.served != turn.answer == "it trades at 4242.42 ."


@pytest.mark.parametrize("written,because", [
    ("aapl sits above both its 20 and 50 day averages .", ""),
    ("aapl sits between its 20 and 50 day averages .", "wrong trend: says between, the figures say above"),
    ("aapl sits below the 20 and 50 day averages .", "wrong trend: says below, the figures say above")])
def test_a_price_answer_whose_trend_words_its_averages_contradict_refuses_even_at_full_confidence(market, router, written,
                                                                                                 because):
    """No guard read words, so 54 spoken answers on outside shares put the price between its averages when it was not."""
    turn = agent(market, router, written=written, confidence=1.0).answer("how is AAPL doing ?")
    assert (turn.intent, turn.served == written, turn.because) == ("performance", not because, because)


@pytest.mark.parametrize("written,named", [
    ("over 20 days aapl is doing fine .", "aapl"),                        # a share we hold, by its symbol
    ("nykaa is doing fine .", "nykaa"), ("fsn e-commerce ventures is doing fine .", "fsn e-commerce ventures"),
    ("maruti. ns is doing fine .", "maruti.ns"),                          # a symbol no list of ours carries
    ("i cannot tell you which way ge goes .", "ge"),                      # where the share goes: GE, too short a name
    ("i do not have a directional signal for cat , and a one week call on it is a coin toss .", "cat"),  # after "for"
    ("the move is small . amacda , is below both its averages .", "amacda"),      # or no share's name at all
    ("ns is doing fine .", "ns")])                                        # or a symbol's suffix alone
def test_an_answer_naming_another_share_refuses_even_at_full_confidence(tmp_path, router, written, named):
    """Outside the 101 it trained on, the generator named another share in 349 of 700 answers, every figure supported."""
    price_file(tmp_path, "AAPL")
    turn = agent(fetching(tmp_path, Yahoo()), router, written=written, confidence=1.0).answer("how is RELIANCE.NS doing ?")
    assert (turn.served == written, turn.because) == (False, f"names another share: {named}")


@pytest.mark.parametrize("written,because", [
    ("maruti. ns is doing fine .", "names another share: maruti.ns"),
    ("aapl sits between its 20 and 50 day averages .", "wrong trend: says between, the figures say above")],
    ids=["a wrong name", "a wrong trend"])
def test_a_sure_price_answer_a_guard_refuses_is_replaced_by_our_codes_sentence_and_the_turn_says_whose(market, router,
                                                                                                       written, because):
    """256 of 540 turns on unseen shares were refused, most for a garbled name, while the price files held the answer.
    ALFA's words and confidence stay in the turn, for the feedback log to judge as the answer that did not speak."""
    turn = agent(market, router, written=written, confidence=1.0).answer("how is AAPL doing ?")
    assert (turn.spoke, turn.served, turn.because, turn.phrased_by) == (
        True, phrase.price_move(market.subject("AAPL"), market.snapshot("AAPL")), because, core.BUILT)
    assert (turn.answer, turn.confidence, turn.stated) == (written, 1.0, None)


@pytest.mark.parametrize("index", ["^NSEI", "^BSESN"])
def test_our_codes_price_sentence_gives_an_index_in_points_not_dollars(tmp_path, router, index):
    """"^NSEI closed at $23278.50" was served: every symbol not ending in .NS was priced in dollars."""
    price_file(tmp_path, index)
    market = finance.Market(tmp_path)
    turn = agent(market, router, written="maruti. ns is doing fine .", confidence=1.0).answer(
        "how is the nifty doing ?", subject=index)
    assert (turn.phrased_by, turn.served.startswith(f"{index} closed at {market.snapshot(index)[1]['close']} on ")) == (
        core.BUILT, True)


def test_a_sure_price_answer_under_the_cut_is_replaced_by_the_answer_form_our_code_fills_from_the_figures(market, router):
    """Every intent but how the price moved has its trained answer form, whose RSI words come from the same figures."""
    turn = agent(market, router, written="it is fine .", confidence=0.5, cut=0.9).answer("is AAPL overbought ?")
    _, shown, _ = market.snapshot("AAPL")
    facts = {name: float(shown[name]) for name in ("close", "sma_20", "sma_50", "rsi_14")}
    forms = {advisory.row("overbought", "AAPL", facts, shown, phrasing)[1] for phrasing in range(3)}
    assert (turn.spoke, turn.because, turn.phrased_by, turn.served in forms) == (True, "low confidence", core.BUILT, True)
    assert "overbought territory" in turn.served and guardrails.unsupported_figures(turn.served, turn.evidence) == []


@pytest.mark.parametrize("question,written", [
    ("how is RELIANCE.NS doing ?", "reliance. ns is doing fine ."),
    ("how is RELIANCE.NS doing ?", "it is doing fine, take a look ."),
    ("how is AAPL doing ?", "aapl is doing fine ."), ("how is NYKAA.NS doing ?", "nykaa is doing fine .")])
def test_an_answer_naming_its_own_share_or_a_word_a_share_is_named_by_is_served(tmp_path, router, question, written):
    """"reliance" and "take" (TAKE.NS) are English words, which an answer about any share may use. "aapl" and "nykaa"
    are names the guard reads for, and the evidence states them."""
    price_file(tmp_path, "AAPL")
    turn = agent(fetching(tmp_path, Yahoo()), router, written=written).answer(question)
    assert (turn.spoke, turn.served) == (True, written)


def test_a_confidence_under_the_cut_refuses_a_supported_answer(market, router):
    turn = agent(market, router, written="it is doing fine .", confidence=0.5, cut=0.9).answer(
        "how is AAPL doing ?")
    assert turn.because == "low confidence" and turn.served != turn.answer == "it is doing fine ."


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


def reference(written, confidence=0.5, paraphrase=False, model=None, live=None, **parts):
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
                     ANSWER_TEMPERATURE, ANSWER_TOP_P, paraphrase=paraphrase, live=live, **parts)


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


@pytest.mark.parametrize("fetched,quoted", [("Apple Inc. said fourth-quarter revenue rose.", False),
                                            ("Apple shares held their upward trend through the quarter.", True)],
                         ids=["off its topic", "on it"])
def test_a_passage_that_speaks_of_no_area_the_question_asks_about_is_not_quoted(registered, router, fetched, quoted):
    """24 of round 7's 64 quotes met a question on a share's trend, risk or debt with its infobox or its history."""
    chunk = Chunk(fetched, [], "https://www.sec.gov/Archives/edgar/data/320193/ex99.htm", "2026-07-29")
    turn = agent(registered, router, gate_cut=1e9, model=ExplodingModel(),
                 reference=reference("never written", model=ExplodingModel(), live=FakeLive([chunk]))).answer(
        "how strong is the current trend in AAPL ?")
    assert (turn.spoke, turn.served == fetched) == (quoted, quoted)


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


def entailing(pairs):
    """An entailment model that finds every sentence entailed, for tests about something else."""
    return [1.0] * len(pairs)


def doubting(claim):
    """An entailment model that finds one sentence entailed by nothing and every other entailed."""
    return lambda pairs: [0.0 if said == claim else 1.0 for _, said in pairs]


def literal(pairs):
    """An entailment model that finds a sentence entailed by a premise holding each of its words."""
    words = lambda text: set(re.findall(r"\w+", text.lower()))
    return [float(words(said) <= words(premise)) for premise, said in pairs]


def phrased_agent(market, router, written, writer, **kw):
    built = agent(market, router, written=written, **kw)
    built.writer = writer
    return built


DRAFT = "it is up 13.8% over 20 days and closed at 165.00 ."
ASKED = "how is AAPL doing ?"


def test_an_answer_of_alfas_that_passes_every_guard_is_served_in_its_own_words_and_never_reworded(market, router):
    """ALFA's answer comes first, and the language model only last: never over an answer of ALFA's that passed."""
    turn = phrased_agent(market, router, DRAFT, ExplodingWriter()).answer(ASKED)
    assert (turn.spoke, turn.served, turn.phrased_by, turn.unphrased_because) == (True, DRAFT, "", "")


def test_a_facts_reply_is_served_as_our_code_wrote_it_and_never_handed_to_the_writer(tmp_path):
    """Reworded, "HDFC Bank has no debt." for our "the facts do not include figures on its debt" passed every check the
    rewording met: none reads a negation."""
    built = facts_agent(tmp_path, Analyses(analysis(roa=ROA)), model=ExplodingModel(), asked=(PROFITABLE,))
    built.writer = ExplodingWriter()
    turn = built.answer(PROFITABLE.format("RELIANCE.NS"))
    assert turn.spoke and turn.served == turn.answer


def test_whether_to_speak_is_still_decided_on_alfas_own_draft(market, router):
    """The cut and the faithfulness record were measured on the generator's draft, so a refusal is never
    handed to the language model to dress up: our code's sentence speaks in its place."""
    turn = phrased_agent(market, router, DRAFT, ExplodingWriter(), confidence=0.5, cut=0.9).answer(ASKED)
    assert (turn.spoke, turn.because, turn.phrased_by) == (True, "low confidence", core.BUILT)


NEWS = Chunk("Mint, 6 Oct 2026: Reliance Industries commissioned its new refinery unit.", [],
             "https://n.test/reliance", "2026-10-06")


def news_reference(writer, news=(NEWS,), live=None, entails=entailing, **parts):
    return reference("never written", model=ExplodingModel(), live=live or ExplodingLive(), writer=writer,
                     news=FakeLive(list(news)), entails=entails, **parts)


def test_a_question_about_lately_is_answered_from_the_newswire_alone(market, router):
    """"Any news on tata steel" ranked over Wikipedia and the newswire together was answered from 2007."""
    plain = "Reliance Industries commissioned a new refinery unit (Mint, 6 Oct 2026)."
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(FakeWriter(plain))).answer(
        "any news on RELIANCE.NS ?")
    assert turn.spoke and turn.served.endswith(plain) and turn.phrased_by == "test-lm"
    assert turn.served.startswith("RELIANCE.NS closed at ₹") and NEWS.document in turn.evidence


def test_a_question_for_the_days_high_is_refused_plainly_not_answered_from_the_news(market, router):
    """Its "today" sent "what is the intraday high of hdfc bank today" to the newswire, which answered with a headline."""
    plain = "Reliance Industries commissioned a new refinery unit (Mint, 6 Oct 2026)."
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(FakeWriter(plain))).answer(
        "what is the intraday high of RELIANCE.NS today ?")
    assert (turn.spoke, turn.served, turn.ticker) == (False, finance.NO_DAY_RANGE, "RELIANCE.NS")


@pytest.mark.parametrize("question", ["how far is RELIANCE.NS from its 60 day high ?", "is RELIANCE.NS at a 50-day low ?",
                                      "why is RELIANCE.NS trading low today ?"])
def test_a_question_on_a_spans_high_or_a_low_price_is_not_refused_as_asking_for_the_days_high(market, router, question):
    """The day's-high rule read "60 day high" and "trading low today" as the day's bar, and ALFA refused them all."""
    assert agent(market, router, gate_cut=1e9).answer(question).served != finance.NO_DAY_RANGE


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
    # only the newswire was read, so the refusal may not say the documents were
    assert "documents" not in turn.served


def judged(lesson, headline):
    """A judge rating the lesson `lesson` and every headline `headline`."""
    return lambda question, texts: [lesson] + [headline] * (len(texts) - 1)


def test_a_move_word_question_is_taught_when_the_judge_rates_the_lesson_above_the_news(market, router):
    """"How are capital gains on shares taxed?" says "gains", and was sent to the newswire instead of the
    lesson on taxes that answers it."""
    taxes = [("Taxes on listed equity", "Gains held over a year are taxed at 12.5%.")]
    built = news_reference(ExplodingWriter(), judge=judged(LESSON_FLOOR + 0.1, LESSON_FLOOR), lessons=ranked(taxes))
    turn = agent(market, router, reference=built).answer("how are capital gains on shares taxed ?")
    assert turn.spoke and turn.intent == "reference lesson"


def test_a_lesson_below_the_floor_is_not_served_even_when_no_headline_rates_higher(market, router):
    """"How do i book profits after gains" was matched to the lesson on the P/B ratio by the word "book"."""
    booked = Chunk("Mint, 28 Oct 2026: Traders book profits after gains in bank stocks.", [], "https://n.test/book",
                   "2026-10-28")
    built = news_reference(None, news=(booked,), judge=judged(LESSON_FLOOR - 0.01, 0.0),
                           lessons=ranked([("P/B ratio (price to book)", "P/B is price over book value per share.")]))
    turn = agent(market, router, reference=built).answer("how do i book profits after gains ?")
    assert turn.intent == "reference recent" and booked.text in turn.served


def test_a_move_word_in_a_question_naming_no_company_and_no_lesson_reads_the_news(market, router):
    """"Why did the sensex crash?" read from the corpus was quoted a passage on the 2016 demonetisation crash."""
    crash = Chunk("Mint, 28 Oct 2026: The Sensex crashed 900 points as bank stocks fell.", [], "https://n.test/crash",
                  "2026-10-28")
    built = news_reference(None, news=(crash,), lessons=lambda question, k: [], judge=judged(LESSON_FLOOR, 0.0))
    turn = agent(market, router, reference=built).answer("why did the sensex crash ?")
    assert turn.intent == "reference recent" and crash.text in turn.served


def test_a_move_question_reads_the_news_when_the_judge_rates_a_headline_above_the_lesson(market, router):
    """"Why are bond yields rising" was served the lesson on free cash flow yield, because its heading says "yield"."""
    moved = Chunk("Mint, 28 Oct 2026: The Nifty slipped as bond yields kept rising.", [], "https://n.test/yields",
                  "2026-10-28")
    yields = [("Free cash flow and FCF yield", "FCF yield is free cash flow over market value.")]
    built = news_reference(None, news=(moved,), lessons=ranked(yields),
                           judge=judged(LESSON_FLOOR + 0.1, LESSON_FLOOR + 0.2))
    turn = agent(market, router, reference=built).answer("are bond yields going to rise ?")
    assert turn.intent == "reference recent" and moved.text in turn.served


TAX_LESSONS = [("ELSS (tax-saving funds)", "ELSS funds lock your money in for three years."),
               ("Taxes on listed equity", "Short-term gains on shares are taxed at 20%.")]


def ranked(lessons):
    """The knowledge base's first `k` lessons, as it gives them when its best section names one."""
    return lambda question, k: lessons[:k]


def headless(lessons):
    """Lessons as the knowledge base gives them when its best section's heading shares no word with the question."""
    return lambda question, k: lessons if k > 1 else []


def rating(*lessons):
    """A judge rating the lessons as given and every headline after them at 0."""
    return lambda question, texts: list(lessons) + [0.0] * (len(texts) - len(lessons))


def test_the_lessons_bm25_finds_are_served_in_the_rerankers_order():
    """BM25's order taught ELSS for "what is the tax on short term profit when shares rise", over the lesson on taxes."""
    from backend.models.agent.run.ask import lessons_from
    found = lessons_from()("what do candle patterns show ?", 4)
    reranked = lessons_from(lambda question, texts: list(range(len(texts))))("what do candle patterns show ?", 4)
    assert len(found) > 1 and reranked == found[::-1]


def test_on_a_move_word_question_the_rerankers_first_lesson_is_served_over_one_the_judge_rates_higher(market, router):
    """Rated by closeness, "what is concentration risk" got the lesson on risk contribution at 0.758 against 0.742."""
    rates = lambda question, texts: [LESSON_FLOOR + 0.2 if TAX_LESSONS[1][1] in text else
                                     LESSON_FLOOR + 0.01 if TAX_LESSONS[0][1] in text else 0.0 for text in texts]
    built = news_reference(None, lessons=ranked(TAX_LESSONS), judge=rates)
    turn = agent(market, router, reference=built).answer("do elss funds gain tax free ?")
    assert turn.intent == "reference lesson" and turn.served == TAX_LESSONS[0][1]


def test_a_question_whose_best_section_names_no_lesson_is_taught_one_the_judge_rates_over_the_floor(market, router):
    """"What tax applies when i sell shares at a profit after 6 months" was served a US forum post, though the judge
    rated the lesson on taxes, the only one BM25 found, at 0.748."""
    built = news_reference(None, lessons=headless(TAX_LESSONS[1:]), judge=rating(LESSON_FLOOR + 0.05))
    turn = agent(market, router, reference=built).answer("what is owed on a quick share profit ?")
    assert turn.intent == "reference lesson" and turn.served == TAX_LESSONS[1][1]


def test_a_question_whose_best_section_names_no_lesson_is_not_taught_one_below_the_floor(market, router):
    built = news_reference(None, lessons=headless(TAX_LESSONS), judge=rating(LESSON_FLOOR - 0.01, 0.9))
    turn = agent(market, router, reference=built).answer("what is a stop loss order ?")
    assert turn.served == PASSAGE


def test_with_no_judge_only_a_lesson_bm25s_best_section_names_is_served(market, router):
    built = reference("never written", model=ExplodingModel(), lessons=headless(TAX_LESSONS))
    turn = agent(market, router, reference=built).answer("what is a stop loss order ?")
    assert turn.served == PASSAGE


def test_lessons_and_news_without_a_judge_are_refused_when_the_reference_is_built():
    with pytest.raises(ValueError, match="judge"):
        news_reference(None, lessons=ranked(TAX_LESSONS))


@pytest.mark.parametrize("question", ["what does it mean if a stock fell over the last few weeks on low volume ?",
                                      "why does RSI use the last 14 days ?"])
def test_a_window_in_a_question_naming_no_company_does_not_keep_it_from_its_lesson(market, router, question):
    """Read as a time word, "over the last few weeks" sent the first question to an unrelated headline on JGB yields."""
    volume = [("Volume, OBV and accumulation/distribution", "Volume confirms price moves.")]
    built = news_reference(None, judge=judged(LESSON_FLOOR + 0.05, 0.0), lessons=ranked(volume))
    turn = agent(market, router, reference=built).answer(question)
    assert turn.intent == "reference lesson"


@pytest.mark.parametrize("question,intent", [("is AAPL overbought right now ?", "overbought"),
                                             ("how far is AAPL off its recent high ?", "drawdown"),
                                             ("is AAPL a good buy right now ?", "buy")])
def test_a_time_word_in_an_intents_own_wording_keeps_it_on_that_intent(market, router, question, intent):
    """"is INFY a good buy right now" lost the buy answer, which says there is no directional signal, to a
    price-move summary under "What happened lately"."""
    turn = agent(market, router, written="it is doing fine .", reference=news_reference(ExplodingWriter())).answer(
        question)
    assert turn.spoke and turn.intent == intent


def test_the_latest_report_is_the_newest_passage_not_the_best_ranked():
    """Passages arrive ranked by relevance with filings first, so the first is not the latest."""
    from backend.models.agent import phrase
    older = Chunk("Mint, 1 Sep 2026: Reliance Industries shares fell.", [], "https://n.test/old", "2026-09-01")
    assert phrase.recent_plain(None, None, [older, NEWS]) == f"The latest report: {NEWS.text}"


def test_a_latest_report_with_no_prices_is_dated_by_the_report_it_serves(market, router):
    """The chat shows "data as of" beside the answer, and it gave the best-ranked passage's day, not the one served."""
    older = Chunk("Mint, 1 Sep 2026: Industries shares fell.", [], "https://n.test/old", "2026-09-01")
    turn = agent(market, router, reference=news_reference(None, news=(older, NEWS))).answer(
        "why did industries shares fall today ?")
    assert turn.served == f"The latest report: {NEWS.text}" and turn.as_of == NEWS.day


def test_the_language_model_reads_the_news_and_our_own_sentence_gives_the_prices(market, router):
    """Given the prices too, it wrote "down 0.5% from the previous close of ₹704.80" for HDFC Bank: a 5-day
    change pinned to a quote saying the shares were up. So it never sees them, and a figure from them is refused."""
    from backend.models.agent import phrase
    moved = market.snapshot("RELIANCE.NS")[1]["return_5d"]
    writer = FakeWriter("Reliance Industries commissioned a refinery unit (Mint, 6 Oct 2026).")
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(writer)).answer("any news on RELIANCE.NS ?")
    assert moved not in writer.asked[0] and "PRICES" not in writer.asked[0]
    assert turn.served == f"{phrase.price_move(finance.Market.subject(market, 'RELIANCE.NS'), market.snapshot('RELIANCE.NS'))} " \
                          f"{phrase.news_dated([NEWS], market.snapshot('RELIANCE.NS'))} " \
                          "Reliance Industries commissioned a refinery unit (Mint, 6 Oct 2026)."
    quoting = news_reference(FakeWriter(f"RELIANCE.NS is {moved} over 5 days after the refinery unit (Mint, 6 Oct 2026)."))
    turn = agent(market, router, gate_cut=1e9, reference=quoting).answer("any news on RELIANCE.NS ?")
    assert turn.phrased_by == core.BUILT and f"figure {moved}" in turn.unphrased_because
    assert turn.served == phrase.recent_plain(finance.Market.subject(market, "RELIANCE.NS"),
                                              market.snapshot("RELIANCE.NS"), [NEWS])


PREVIEW = Chunk("Mint, 21 Jan 2020: Reliance Industries is expected to report a higher profit.", [],
                "https://n.test/preview", "2020-01-21")


@pytest.mark.parametrize("writer,said", [
    (FakeWriter("Reliance Industries is expected to report a higher profit (Mint, 21 Jan 2020)."),
     "The news that follows is from 21 Jan 2020, {} days before that close. Reliance Industries is expected"),
    (None, "On 21 Jan 2020, {} days before that close, Mint wrote: Reliance Industries is expected")],
    ids=["reworded", "plain"])
def test_news_beside_the_prices_says_how_long_before_their_close_it_was_written(market, router, writer, said):
    """A 7 Oct preview of results already out was told on 16 Nov in the present tense, as if it were news."""
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(writer, news=(PREVIEW,))).answer(
        "any news on RELIANCE.NS ?")
    gap = (date.fromisoformat(str(market.snapshot("RELIANCE.NS")[2])[:10]) - date(2020, 1, 21)).days
    assert turn.served.startswith("RELIANCE.NS closed at ₹") and said.format(gap) in turn.served


PLANNED = Chunk("Mint, 6 Oct 2026: Reliance Industries said its new refinery unit opens on 2026-11-05.", [],
                "https://n.test/unit", "2026-10-06")


def test_a_writer_reply_writing_a_passages_day_another_way_is_served(market, router):
    """figures() read "2026-11-05" as 2026, -11 and -05, so "5 November 2026" was refused for a figure 5."""
    writer = FakeWriter("Reliance Industries said its new refinery unit opens on 5 November 2026 (Mint, 6 Oct 2026).")
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(writer, news=(PLANNED,))).answer(
        "any news on RELIANCE.NS ?")
    assert (turn.phrased_by, turn.unphrased_because) == ("test-lm", "")


def test_a_writer_reply_giving_a_day_no_passage_gives_is_refused_in_plain_words(market, router):
    """The reason reached the user as the date's pieces: "it leaves out -03, -11, -16, -31, 2026"."""
    writer = FakeWriter("Reliance Industries said its new refinery unit opens on 5 December 2026 (Mint, 6 Oct 2026).")
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(writer, news=(PLANNED,))).answer(
        "any news on RELIANCE.NS ?")
    assert turn.unphrased_because == "it gives the date 5 Dec 2026, which the source does not"


PASSED = "Reliance Industries commissioned its new refinery unit (Mint, 6 Oct 2026)"


@pytest.mark.parametrize("writer,reason", [
    (FakeWriter(f"{PASSED}, and 9.99% this year."), "figure 9.99%"),                            # adds one
    (FakeWriter(f"{PASSED} because of rates."), "gives a reason"),                              # adds a cause
    (FakeWriter(f"{PASSED}, and its price will rise further."), "claims a direction"),          # adds a call
    (FakeWriter(f"{PASSED}, counted in $."), "writes $"),                                       # swaps currency
    (FakeWriter("Reliance Industries:\n- commissioned its new refinery unit\n- Mint, 6 Oct 2026"), "breaks into lines"),
    (FakeWriter(f"{PASSED.replace('Reliance Industries', 'RELIEANCE.NS')}."), "names RELIEANCE.NS"),  # misspelt
    (FakeWriter(error=DataUnavailable("model Space failed on /write: GPU quota exceeded")), "GPU quota exceeded")])
def test_a_writer_reply_that_changes_the_news_serves_the_plain_answer_and_says_why(market, router, writer, reason):
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(writer)).answer("any news on RELIANCE.NS ?")
    assert turn.served == phrase.recent_plain(finance.Market.subject(market, "RELIANCE.NS"),
                                              market.snapshot("RELIANCE.NS"), [NEWS]) and turn.phrased_by == core.BUILT
    assert reason in turn.unphrased_because


@pytest.mark.parametrize("writer,asked", [
    (FakeWriter(f"{PASSED}."), True), (FakeWriter(f"{PASSED} because of rates."), True),
    (FakeWriter(error=DataUnavailable("model Space failed on /write: GPU quota exceeded")), False)],
    ids=["its words served", "its words refused", "not reached"])
def test_a_turn_says_whether_its_request_reached_the_writer(market, router, writer, asked):
    """A writer the Space could not run was charged to the user as asked."""
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(writer)).answer("any news on RELIANCE.NS ?")
    assert turn.asked_writer is asked


def test_a_question_not_about_money_is_refused_before_the_writer_is_asked(market, router):
    """"how is your mom doing" found a headline saying "doing" and asked the writer before its refusal: 2 answers."""
    writer = FakeWriter("Markets are doing well (Mint, 6 Oct 2026).")
    doing = Chunk("Mint, 6 Oct 2026: Markets are doing well.", [], "https://n.test/doing", "2026-10-06")
    turn = agent(market, router, reference=news_reference(writer, news=(doing,))).answer("how is your mom doing ?")
    assert (turn.served, turn.asked_writer, writer.asked) == (finance.OFF_TOPIC, False, [])


def test_a_refusal_after_the_writer_read_the_news_says_it_asked_the_writer(market, router):
    """"what happened in the markets today?" asked the writer, which found no answer, and was refunded as asking none."""
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(FakeWriter("NONE"))).answer(
        "what happened with the refinery unit in the markets today ?")
    assert (turn.spoke, turn.served, turn.asked_writer) == (False, finance.NOT_IN_SOURCES, True)


def test_a_writer_that_finds_no_answer_serves_the_plain_answer_not_its_half_refusal(market, router):
    """Half an answer and half a refusal ("...founded in 1981. The passage does not provide a reason. None.")
    is a refusal: only that half was true. The prices and the newest report are what is left to say."""
    writer = FakeWriter("It was founded in 1981. The passages do not provide a reason. None.")
    turn = agent(market, router, gate_cut=1e9, reference=news_reference(writer)).answer("why is RELIANCE.NS falling ?")
    assert turn.spoke and "founded" not in turn.served and NEWS.text.partition(": ")[2] in turn.served


ASKED_YES_NO = "are refinery stocks falling ?"


@pytest.mark.parametrize("reply", ["NO", "Yes, Reliance Industries commissioned a refinery unit (Mint, 6 Oct 2026)."])
def test_a_writer_that_answers_yes_or_no_serves_the_plain_answer(market, router, reply):
    """"are it stocks falling" was served "NO", a verdict the prompt asks the writer not to give."""
    from backend.models.agent import phrase
    turn = agent(market, router, reference=news_reference(FakeWriter(reply))).answer(ASKED_YES_NO)
    assert turn.spoke and turn.served == phrase.recent_plain(None, None, [NEWS]) and turn.phrased_by == core.BUILT
    assert "answers yes or no" in turn.unphrased_because


def test_a_writer_sentence_no_passage_entails_serves_the_plain_answer_and_names_it(market, router):
    """"Bond yields are not falling." was served for passages saying global yields were rising, and HHEM judged it
    0.25: a verdict no passage gives, in more wordings than any word rule caught."""
    from backend.models.agent import phrase
    verdict = "Refinery shares show no sign of a fall."
    built = news_reference(FakeWriter(f"{verdict} Reliance Industries commissioned a unit (Mint, 6 Oct 2026)."),
                           entails=doubting(verdict))
    turn = agent(market, router, reference=built).answer(ASKED_YES_NO)
    assert turn.spoke and turn.served == phrase.recent_plain(None, None, [NEWS]) and turn.phrased_by == core.BUILT
    assert f"no passage says “{verdict}”" in turn.unphrased_because


@pytest.mark.parametrize("reply", ["Reliance Industries commissioned its new refinery unit. (Mint, 6 Oct 2026)",
                                   "Reliance Industries commissioned its new refinery unit (Mint, 6 Oct 2026).",
                                   "Refinery stocks moved after the new unit (Mint, 6 Oct 2026; ET, 7 Oct 2026)."])
def test_a_passages_own_source_and_date_in_brackets_is_not_read_as_a_claim(market, router, reply):
    """Read alone, "(Mint, 6 Oct 2026)" scored as low as 0.002 against the passage it names, and a sentence carrying its
    source failed against the passage sentence that said it word for word."""
    moved = Chunk("ET, 7 Oct 2026: Refinery stocks moved.", [], "https://n.test/moved", "2026-10-07")
    built = news_reference(FakeWriter(reply), news=(NEWS, moved),
                           entails=lambda pairs: [0.0 if "(" in said else 1.0 for _, said in pairs])
    turn = agent(market, router, reference=built).answer(ASKED_YES_NO)
    assert turn.served == reply and turn.phrased_by == "test-lm"


@pytest.mark.parametrize("reply", ["Reliance Industries commissioned its new refinery unit (its biggest in years).",
                                   "Reliance Industries commissioned its new refinery unit (Mint, 6 Oct 2026). "
                                   "(Two more units are planned.)"])
def test_a_claim_in_brackets_that_no_passage_entails_serves_the_plain_answer(market, router, reply):
    """"TCS shares fell 2% (their worst day in years)" was served: each bracket was left out before the gate read it."""
    from backend.models.agent import phrase
    turn = agent(market, router, reference=news_reference(FakeWriter(reply), entails=literal)).answer(ASKED_YES_NO)
    assert turn.served == phrase.recent_plain(None, None, [NEWS]) and "no passage says" in turn.unphrased_because


def test_a_passages_figure_in_brackets_after_another_subject_is_read_with_its_sentence(market, router):
    """"The Nifty fell (over 500 points) after the RBI's announcement." was served, though the 500 points were the
    Sensex's: the entailment model reads a bracket as an aside. This fake does too, and knows whose fall the figure
    was; `literal` cannot pin this, as the passages joined hold every word."""
    from backend.models.agent import phrase
    fell = Chunk("ET, 7 Oct 2026: Tata Steel shares fell 3%. Refinery stocks fell after the RBI hike.", [],
                 "https://n.test/fell", "2026-10-07")
    tied = lambda pairs: [float("3%" not in re.sub(r"\(.*?\)", "", said) or "Tata Steel" in said) for _, said in pairs]
    built = news_reference(FakeWriter("Refinery stocks fell (3%) after the RBI hike (ET, 7 Oct 2026)."), news=(fell,),
                           entails=tied)
    turn = agent(market, router, reference=built).answer(ASKED_YES_NO)
    assert turn.served == phrase.recent_plain(None, None, [fell])
    assert "ET, 7 Oct 2026 does not say “Refinery stocks fell 3% after the RBI hike.”" in turn.unphrased_because


def test_a_figure_the_reply_ties_to_another_name_than_its_passage_does_serves_the_plain_answer(market, router):
    """"Infosys and Wipro each fell about 1.2%" passed the entailment model, though 1.2% was the Nifty's fall."""
    from backend.models.agent import phrase
    nifty = Chunk("ET, 7 Oct 2026: The Nifty fell 1.2% as refinery stocks dropped.", [], "https://n.test/n", "2026-10-07")
    built = news_reference(FakeWriter("Reliance Industries shares fell 1.2% (ET, 7 Oct 2026)."), news=(NEWS, nifty))
    turn = agent(market, router, reference=built).answer(ASKED_YES_NO)
    assert turn.served == phrase.recent_plain(None, None, [NEWS, nifty])
    assert "it ties 1.2% to Reliance Industries" in turn.unphrased_because


@pytest.mark.parametrize("reply", ["Refinery stocks are not falling.", "Refinery stocks aren't falling (Mint, 6 Oct 2026)."])
def test_a_reply_that_only_denies_the_question_serves_the_plain_answer(market, router, reply):
    """"The market is not crashing." passed the entailment model, which reads negation badly, and no passage said it."""
    from backend.models.agent import phrase
    turn = agent(market, router, reference=news_reference(FakeWriter(reply))).answer(ASKED_YES_NO)
    assert turn.served == phrase.recent_plain(None, None, [NEWS]) and "gives its own verdict" in turn.unphrased_because


def test_a_denial_that_goes_on_to_report_the_passages_is_left_to_the_entailment_model(market, router):
    reply = "Refinery stocks are not falling; Reliance Industries commissioned its new refinery unit (Mint, 6 Oct 2026)."
    turn = agent(market, router, reference=news_reference(FakeWriter(reply))).answer(ASKED_YES_NO)
    assert turn.served == reply and turn.phrased_by == "test-lm"


STEEL = Chunk("ET, 7 Oct 2026: Refinery stocks fell as Tata Steel shares fell 3%.", [], "https://n.test/steel", "2026-10-07")


@pytest.mark.parametrize("reply", ["Reliance Industries commissioned its new refinery unit (ET, 7 Oct 2026).",
                                   "Reliance Industries commissioned its new refinery unit. (ET, 7 Oct 2026)"])
def test_a_sentence_citing_a_source_that_does_not_say_it_serves_the_plain_answer(market, router, reply):
    """"Wipro shares dropped 3% after a guidance cut (Economic Times, 28 Oct 2026)." passed: Mint had reported it."""
    from backend.models.agent import phrase
    built = news_reference(FakeWriter(reply), news=(NEWS, STEEL), entails=literal)
    turn = agent(market, router, reference=built).answer(ASKED_YES_NO)
    assert turn.served == phrase.recent_plain(None, None, [NEWS, STEEL])
    assert "ET, 7 Oct 2026 does not say “Reliance Industries commissioned its new refinery unit.”" in turn.unphrased_because


def test_a_sentence_citing_the_source_that_says_it_is_served(market, router):
    reply = "Reliance Industries commissioned its new refinery unit (Mint, 6 Oct 2026)."
    turn = agent(market, router, reference=news_reference(FakeWriter(reply), news=(NEWS, STEEL), entails=literal)).answer(
        ASKED_YES_NO)
    assert turn.served == reply and turn.phrased_by == "test-lm"


def test_a_reply_that_only_names_a_source_serves_the_plain_answer(market, router):
    from backend.models.agent import phrase
    turn = agent(market, router, reference=news_reference(FakeWriter("(Mint, 6 Oct 2026)"))).answer(ASKED_YES_NO)
    assert turn.served == phrase.recent_plain(None, None, [NEWS]) and "states nothing" in turn.unphrased_because


def test_a_sentence_that_joins_two_passages_is_entailed_by_them_together(market, router):
    """"The Sensex rose 410 points, while Wipro shares fell 3%" scored 0.0 against each of its two passages alone."""
    slid = Chunk("Mint, 6 Oct 2026: TCS shares slid while results were due.", [], "https://n.test/tcs", "2026-10-06")
    reply = "Reliance Industries commissioned its new refinery unit, while TCS shares slid"
    built = news_reference(FakeWriter(reply), news=(NEWS, slid), entails=literal)
    turn = agent(market, router, reference=built).answer("are refinery and tcs stocks falling ?")
    assert turn.served == reply and turn.phrased_by == "test-lm"


def test_a_writer_without_an_entailment_model_is_refused_when_the_reference_is_built():
    with pytest.raises(ValueError, match="entails"):
        news_reference(FakeWriter("never asked"), entails=None)


def test_a_writer_with_no_news_to_reword_needs_no_entailment_model(market, router):
    """Only news replies are checked by it, and `--phrase` without `--live` loaded its 568 MB for nothing."""
    built = reference("never written", model=ExplodingModel(), writer=ExplodingWriter())
    assert agent(market, router, reference=built).answer("what is a stop loss order ?").served == PASSAGE


def test_a_writer_saying_the_passages_do_not_mention_it_is_a_refusal_not_an_answer(market, router):
    """"Pharma stocks are not specifically mentioned in the passages provided." was served, and HHEM judged it 0.17."""
    writer = FakeWriter("Refinery stocks are not specifically mentioned in the passages provided.")
    turn = agent(market, router, reference=news_reference(writer)).answer(ASKED_YES_NO)
    assert not turn.spoke and turn.served == finance.NOT_IN_SOURCES


def test_a_writer_that_reports_what_happened_on_a_yes_no_question_is_served(market, router):
    reply = "Reliance Industries commissioned its new refinery unit (Mint, 6 Oct 2026)."
    turn = agent(market, router, reference=news_reference(FakeWriter(reply))).answer(ASKED_YES_NO)
    assert turn.served == reply and turn.phrased_by == "test-lm"


def test_a_report_in_the_questions_own_words_is_served_when_the_passages_entail_it(market, router):
    """"The Nifty has fallen 1.2% as IT stocks dropped on weak US cues (ET, 27 Oct 2026)." was refused for "has the
    nifty fallen", by a word rule that read any restatement of the question as a verdict."""
    said = Chunk("Mint, 6 Oct 2026: Refinery stocks fell 2% after the new unit opened.", [], "https://n.test/r",
                 "2026-10-06")
    reply = "Refinery stocks are falling, down 2% after the new unit opened (Mint, 6 Oct 2026)."
    turn = agent(market, router, reference=news_reference(FakeWriter(reply), news=(said,))).answer(ASKED_YES_NO)
    assert turn.served == reply and turn.phrased_by == "test-lm"


def test_forum_and_encyclopedia_passages_are_quoted_not_reworded(market, router):
    """The language model misattributed long prose ("HDFCBANK.NS is a subsidiary of HDFC Bank"), as ALFA's
    own generator did, so only dated headlines are reworded."""
    built = reference("never written", model=ExplodingModel(), writer=ExplodingWriter(), entails=entailing)
    turn = agent(market, router, reference=built).answer("what is a stop loss order ?")
    assert turn.spoke and turn.served == PASSAGE


def test_a_question_not_about_money_is_refused_even_when_a_passage_matches(market, router):
    turn = agent(market, router, reference=reference("never written", model=ExplodingModel())).answer(
        "what did the committee vote on at the meeting ?")
    assert not turn.spoke and turn.served == finance.OFF_TOPIC


# The two Wikipedia infobox passages served for TCS and Nykaa, as the corpus stored them
TCS_INFOBOX = ("(2026) Total equity ₹ 108,478 crore (US$11 billion) (2026) Owner Tata Sons (71.74%) [ 2 ] Number of "
               "employees 584,519 (March 2026) [ 3 ] Parent Tata Group Subsidiaries TRDDC Website tcs .com Footnotes "
               "Financials as of 31 March 2025 . [ 4 ] Tata Consultancy Services Limited [ 5 ] ( TCS ) is an Indian")
NYKAA_INFOBOX = ("₹ 72 crore (US$7.5 million) (2025) Owner Nayar family (53.5%) [ 2 ] Number of employees 2,000+ (2021) "
                 "Website www .nykaa .com Footnotes Financials as of 31 March 2025 . [ 3 ] FSN E-Commerce Ventures Ltd , "
                 "doing business as Nykaa , is an Indian retail company, headquartered in Mumbai . It sells beaut")


MOON = Chunk("Titan is the largest moon of Saturn, and it has been doing what moons do, orbiting its planet lately.",
             [], "moon", "")
ZAGGLE = Chunk("ZAGGLE.NS has been doing well lately, a company note said.", [], "zaggle", "2026-10-01")


def infoboxes(*more):
    held = [Chunk(TCS_INFOBOX, [], "188e9441c1b74e60", ""), Chunk(NYKAA_INFOBOX, [], "d74d2e8b9a944217", ""),
            Chunk(PASSAGE, [], "d0", "2021-06-01"), *more]
    return Reference(Hybrid(held, pretokenize), FakeTokenizer(), ExplodingModel(), Config, QUESTION_TOKENS, PASSAGES,
                     ANSWER_TEMPERATURE, ANSWER_TOP_P)


@pytest.mark.parametrize("question,quoted", [
    ("who owns tata consultancy services ?",
     "(2026) Total equity ₹ 108,478 crore (US$11 billion) (2026) Owner Tata Sons (71.74%) Number of employees 584,519 "
     "(March 2026) Parent Tata Group Subsidiaries TRDDC Website tcs.com Footnotes Financials as of 31 March 2025. "
     "Tata Consultancy Services Limited (TCS) is an Indian"),
    ("who owns nykaa ?",
     "₹ 72 crore (US$7.5 million) (2025) Owner Nayar family (53.5%) Number of employees 2,000+ (2021) Website "
     "www.nykaa.com Footnotes Financials as of 31 March 2025. FSN E-Commerce Ventures Ltd, doing business as Nykaa, "
     "is an Indian retail company, headquartered in Mumbai. It sells beaut")])
def test_a_quote_drops_citation_marks_and_the_spaces_the_corpus_split_off_but_keeps_every_figure(question, quoted):
    assert infoboxes().look_up(question).served == quoted


RELIANCE_INFOBOX = ("Reliance Industries Not to be confused with Reliance Group . Reliance Industries Limited Type Public "
                    "Traded as BSE : 500325 NSE : RELIANCE Founded 1958 Founder Dhirubhai Ambani Headquarters Mumbai Key")
RELIANCE_PEOPLE = "people Mukesh Ambani ( Chairman and MD ) Products Petroleum Reliance Industries Limited"


def test_a_question_on_who_runs_a_company_is_quoted_only_a_passage_naming_its_management():
    """"who runs reliance industries?" was quoted the start of the infobox, cut off before "Key people", which said nothing
    of who runs it."""
    held = infoboxes(Chunk(RELIANCE_INFOBOX, [], "95a5b0fe22e159db", ""), Chunk(RELIANCE_PEOPLE, [], "95a5b0fe22e159dc", ""))
    assert held.look_up("who runs reliance industries ?").served == "people Mukesh Ambani (Chairman and MD) Products " \
                                                                     "Petroleum Reliance Industries Limited"


@pytest.mark.parametrize("question,subject,off_topic", [("how has zaggle been doing lately ?", "ZAGGLE.NS", False),
                                                        ("how is your mom doing ?", "", True)])
def test_a_question_about_a_stock_we_hold_no_prices_for_is_looked_up_not_refused_as_off_topic(market, router, question,
                                                                                               subject, off_topic):
    """The first was refused as not about money; it and the second each find a passage that says "doing". The turn
    carries the stock it was asked under, so "should i buy it" next is not about the conversation's share before."""
    turn = agent(market, router, model=ExplodingModel(), reference=infoboxes(ZAGGLE)).answer(question, subject=subject)
    assert (turn.served == finance.OFF_TOPIC) == off_topic and turn.spoke != off_topic and turn.ticker == subject


NSE_LIST = Chunk("NSE: HAPPSTMNDS Happiest Minds Technologies Limited NSE: HARITASEAT Harita Seating Systems Limited "
                 "NSE: HARRMALAYA Harrisons Malayalam Limited", [], "c931913ec51a747f", "")


def test_a_passage_that_only_names_the_share_is_not_quoted_for_how_it_is_performing(tmp_path, router):
    """"how is happstmnds performing and should I buy it" was quoted a stored Wikipedia list of 14 NSE companies."""
    price_file(tmp_path, "HAPPSTMNDS.NS")
    held = finance.Market(tmp_path, listed={"HAPPSTMNDS.NS": "Happiest Minds Technologies"})
    turn = agent(held, router, gate_cut=1e9, model=ExplodingModel(),
                 reference=infoboxes(NSE_LIST)).answer("how is happstmnds performing and should I buy it")
    assert (turn.ticker, turn.spoke, turn.because) == ("HAPPSTMNDS.NS", False, "unclear question")


def test_a_question_on_a_share_naming_no_area_is_still_quoted_the_passage_that_answers_it(tmp_path, router):
    """"who owns TCS ?", which the router is sure our figures cannot answer, lost its infobox when a question about a
    share had to name an area."""
    price_file(tmp_path, "TCS.NS")
    held = finance.Market(tmp_path, listed={"TCS.NS": "Tata Consultancy Services"})
    turn = agent(held, router, model=ExplodingModel(), reference=infoboxes()).answer("who owns TCS ?")
    assert (turn.intent, "Owner Tata Sons (71.74%)" in turn.served) == (f"reference {QUOTED}", True)


class Unanswering:
    """Documents read and found not to answer, or a live search that could not be run when `fails`."""

    def __init__(self, fails):
        self.fails = fails

    def look_up(self, *_):
        if self.fails:
            raise SearchFailed("wikipedia: 503 from en.wikipedia.org")
        return Looked("", ZAGGLE.text, UNANSWERED, [], float("nan"), "", "2026-10-01", "", "it does not say")


@pytest.mark.parametrize("fails,refusal", [(False, finance.NOT_IN_SOURCES), (True, finance.UNREACHABLE)],
                         ids=["not in sources", "unreachable"])
def test_a_refusal_about_a_stock_we_hold_no_prices_for_carries_the_stock(market, router, fails, refusal):
    """Carried empty, the conversation kept the share before it, and "should i buy it" next was about that one."""
    turn = agent(market, router, model=ExplodingModel(), reference=Unanswering(fails)).answer(
        "how has zaggle been doing lately ?", subject="ZAGGLE.NS")
    assert (turn.spoke, turn.served, turn.ticker) == (False, refusal, "ZAGGLE.NS")


@pytest.mark.parametrize("question,subject", [("how has titan been doing lately ?", "TITAN.NS"),
                                              ("how has zaggle been doing lately ?", "ZAGGLE.NS")])
def test_a_stock_we_hold_no_prices_for_is_answered_only_from_a_passage_that_names_it(market, router, question, subject):
    """With no prices for either, "titan" was served the moon of Saturn and "zaggle" the Nykaa infobox."""
    turn = agent(market, router, model=ExplodingModel(), reference=infoboxes(MOON)).answer(question, subject=subject)
    assert (turn.spoke, turn.because) == (False, "no subject")


@pytest.mark.parametrize("question", ["what does RSI mean ?", "what is diversification ?",
                                      "what does it mean when the macd line rises above the signal line ?",
                                      "does volume rise before a breakout ?", "why are small caps rallying ?",
                                      "are fiis and diis buying ?", "how do i read a candlestick chart ?",
                                      "what is cagr ?", "what is xirr ?", "what is a hammer candle ?",
                                      "do breakouts need volume ?"])
def test_market_words_alone_put_a_question_about_money(question):
    """Each of the first five found its lesson or a headline and was refused as not about money."""
    assert finance.about_money(question)


@pytest.mark.parametrize("question", ["write me a poem about the sea", "what is the weather in Mumbai ?",
                                      "who won the cricket match yesterday ?", "tell me a joke", "asdfgh qwerty",
                                      "how many hours should i sleep ?", "recommend a good movie for tonight",
                                      "what are the top songs on the music charts ?", "show me a chart of the weather",
                                      "how do i make a pie chart in excel ?", "what is stochastic gradient descent ?",
                                      "what is the volume of a sphere ?", "how do i turn up the volume on my phone ?",
                                      "how do i treat an acne breakout ?",
                                      "how many candles go on a cake, one candle a year ?",
                                      "why do weather charts put the key below the chart ?", "how is your mom doing ?"])
def test_a_question_about_something_else_is_not_about_money(question):
    assert not finance.about_money(question)


@pytest.mark.parametrize("question,lesson", [
    ("what is a pennant ?", ("Flags and pennants", "A pennant is a small triangle after a sharp move.")),
    ("what is a hanging man ?", ("Hammer and hanging man", "Both have a small body near the top of the range.")),
])
def test_a_lesson_that_uses_every_word_of_the_question_is_served_though_no_money_word_is_in_it(market, router, question,
                                                                                                lesson):
    """A lesson is about money by construction, so naming its topic is enough: "what is a pennant" names no market word."""
    built = reference("never written", model=ExplodingModel())
    built.lessons = lambda asked, k: [lesson]
    turn = agent(market, router, reference=built).answer(question)
    assert turn.spoke and turn.intent == "reference lesson"


def test_the_words_a_lesson_is_cited_with_do_not_put_a_question_on_topic(market, router):
    """Its evidence wraps a lesson in "document lesson: ... published undated", which no lesson says."""
    built = reference("never written", model=ExplodingModel())
    built.lessons = lambda asked, k: [("Candle anatomy", "A candle shows the open, high, low and close of one period.")]
    turn = agent(market, router, reference=built).answer("what is an undated document ?")
    assert not turn.spoke and turn.served == finance.OFF_TOPIC


@pytest.mark.parametrize("question,lesson", [
    ("how do candles burn ?", ("Candle anatomy", "A candle shows the open, high, low and close of one period.")),
    ("what is a black hole ?", ("Three white soldiers and three black crows", "Three long black candles in a row.")),
    ("what is the volume of a sphere ?", ("Volume, OBV and accumulation/distribution", "Volume confirms price moves.")),
    ("how do i treat an acne breakout ?", ("Breakouts and false breakouts", "A breakout is a close beyond a range.")),
])
def test_a_lesson_sharing_one_word_with_a_question_about_something_else_is_refused(market, router, question, lesson):
    """The lessons' heading check lets one shared word through, and every lesson is about money, not the question."""
    built = reference("never written", model=ExplodingModel())
    built.lessons = lambda asked, k: [lesson]
    turn = agent(market, router, reference=built).answer(question)
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
    built.lessons = lambda question, k: [("SIP (systematic investment plan)", "A SIP invests a fixed amount every month.")]
    turn = agent(market, router, reference=built).answer("how does a SIP work ?")
    assert turn.spoke and turn.served == "A SIP invests a fixed amount every month." and turn.intent == "reference lesson"
    built.lessons = lambda question, k: []
    assert agent(market, router, reference=built).answer("what is a stop loss order ?").served == PASSAGE


@pytest.mark.parametrize("question", ["how has RELIANCE.NS been doing lately ?",
                                      "how has RELIANCE.NS been doing over the last month ?"])
def test_a_question_sure_to_ask_how_the_price_moved_with_a_look_back_word_reads_the_dated_sources_first(market, router,
                                                                                                         question):
    """"how is AAPL doing lately" and "how has TCS been doing over the last month" were refused at ALFA's confidence cut
    while the price files held their answer."""
    turn = agent(market, router, model=ExplodingModel(),
                 reference=news_reference(FakeWriter("Reliance commissioned a refinery unit (Mint, 6 Oct 2026)."))).answer(
        question)
    assert turn.spoke and turn.intent == "reference recent"


@pytest.mark.parametrize("question,recent", [("how has RELIANCE.NS done over the past month ?", True),
                                             ("RELIANCE.NS last 30 days performance", True),
                                             ("how did RELIANCE.NS do over the last six months ?", True),
                                             ("how is RELIANCE.NS performing ?", False)])
def test_a_question_the_router_is_unsure_of_reads_the_dated_sources_only_when_it_names_a_window(market, router, question,
                                                                                                recent):
    """The router read "what does TCS do" as a question on how its price moved, at a margin under the gate."""
    built = news_reference(FakeWriter("Reliance commissioned a refinery unit (Mint, 6 Oct 2026)."), live=FakeLive([]))
    turn = agent(market, router, gate_cut=1e9, model=ExplodingModel(), reference=built).answer(question)
    assert (turn.intent == "reference recent") == recent


@pytest.mark.parametrize("filings", [RefusedLive(), ExplodingLive()])
@pytest.mark.parametrize("question", ["how has AAPL done over the past month ?"])
def test_a_question_sure_to_ask_how_the_price_moved_is_answered_from_the_price_files_without_filings(registered, router,
                                                                                                    filings, question):
    """"how is AAPL doing" was refused as unreachable when EDGAR refused us, though the price files held the answer, and
    it waited on EDGAR when EDGAR did answer. A window is a span of prices, not a reason to read filings."""
    built = news_reference(ExplodingWriter(), news=(), filings=filings)
    turn = agent(registered, router, model=ExplodingModel(), reference=built).answer(question)
    assert turn.spoke and turn.served.startswith("AAPL closed at $") and turn.because == ""


@pytest.mark.parametrize("news", [False, True], ids=["offline", "with news"])
def test_a_question_sure_to_ask_how_the_price_moved_with_no_look_back_word_is_answered_by_alfa_first(market, router, news):
    """`serve` always attaches a Reference, and "how is AAPL doing" was answered by our price sentence, or the writer,
    before ALFA's generator ran at all. Our sentence is for when ALFA's answer is refused."""
    written = "aapl sits above both its 20 and 50 day averages ."
    built = news_reference(ExplodingWriter()) if news else reference("never written", model=ExplodingModel())
    turn = agent(market, router, written=written, confidence=1.0, reference=built).answer("how is AAPL doing ?")
    assert (turn.intent, turn.served, turn.phrased_by) == ("performance", written, "")


def test_a_question_about_a_fetched_share_in_no_money_words_reaches_its_price_files(tmp_path, router):
    """"how has nykaa been doing lately ?" was refused as not about money, though the chat handed over NYKAA.NS."""
    turn = agent(fetching(tmp_path, Yahoo()), router, model=ExplodingModel(),
                 reference=news_reference(ExplodingWriter(), news=())).answer(
        "how has nykaa been doing lately ?", subject="NYKAA.NS")
    assert turn.spoke and turn.intent == "reference recent" and turn.served.startswith("NYKAA.NS closed at ₹")


ASKED_TOGETHER = "how is TCS performing and should I buy it"


def together(tmp_path, router, question=ASKED_TOGETHER, gate_cut=4.0, writer=None, subject="", **kw):
    """(market, turn) for `question`. The lexical router's margins are 2.78 for ASKED_TOGETHER and 5.90 and 5.52 for its
    parts. INFY.NS is held too, so a part can name another share."""
    price_file(tmp_path, "TCS.NS")
    price_file(tmp_path, "INFY.NS")
    market = finance.Market(tmp_path)
    asking = agent(market, router, written="i do not have a directional signal for it .", gate_cut=gate_cut,
                   **{"reference": reference("never written", model=ExplodingModel()), **kw})
    asking.writer = writer
    return market, asking.answer(question, subject=subject)


class Confidences(FakeModel):
    """Reports `values` one answer at a time."""

    def __init__(self, *values):
        self.values = list(values)

    def confidence(self, source, produced, keep):
        return np.array([self.values.pop(0)])


def test_two_questions_asked_together_are_answered_in_one_turn_each_from_its_own_path(tmp_path, router):
    """It sat between buy and performance under the routing cut, so it was served the TCS infobox."""
    market, turn = together(tmp_path, router)
    assert (turn.intent, turn.ticker, turn.spoke, turn.because) == ("performance and buy", "TCS.NS", True, "")
    assert turn.served == "i do not have a directional signal for it ."
    assert turn.confidence == 0.999 and market.snapshot("TCS.NS")[0] in turn.evidence


def test_a_part_refused_by_its_own_guard_is_answered_by_our_code_in_the_turn_and_named(tmp_path, router):
    """Neither part spoke in ALFA's words, so the turn reports no confidence of ALFA's."""
    market, turn = together(tmp_path, router, confidence=0.5)
    assert turn.spoke and turn.served.endswith(" this is not a recommendation to trade .")
    assert (turn.because, turn.phrased_by) == ("performance: low confidence; buy: low confidence", core.BUILT)
    assert turn.confidence != turn.confidence and turn.stated is None


@pytest.mark.parametrize("question,intent", [(ASKED_TOGETHER, "performance and buy"),
                                             ("is TCS overbought and is it oversold ?", "overbought"),
                                             ("how volatile is TCS and is that high ?", "volatility")])
def test_a_question_the_router_is_sure_of_is_split_only_when_its_parts_are_placed_apart(tmp_path, router, question,
                                                                                         intent):
    """Sure of the whole, one route answered "what is the p/e of tcs and how volatile is it" with the P/E alone. "is that
    high" asks about the volatility, and was split off as how far TCS.NS is below its high."""
    _, turn = together(tmp_path, router, question, gate_cut=0.01)
    assert (turn.intent, turn.spoke) == (intent, True)


def test_two_facts_questions_the_router_is_sure_of_are_each_answered(tmp_path):
    """A facts reply gives the first area asked: "what is the p/e of RELIANCE.NS and how much debt does it have" was
    given the P/E alone."""
    dividend = {"value": 0.4, "unit": "%", "source": "yahoo", "period": "FY ending 2026-03-31"}
    turn = facts_agent(tmp_path, Analyses(analysis(roa=ROA, dividend_yield=dividend)), model=ExplodingModel(),
                       asked=(PROFITABLE, DIVIDEND_ASKED)).answer(
        "how profitable is RELIANCE.NS and what is the dividend yield of it ?")
    assert ("Return on assets is 11.00%" in turn.served, "Dividend yield is 0.40%" in turn.served) == (True, True)


def test_two_questions_asked_together_are_not_split_when_a_part_is_under_the_routing_cut(tmp_path, router):
    _, turn = together(tmp_path, router, gate_cut=5.7)
    assert (turn.spoke, turn.because) == (False, "unclear question")


@pytest.mark.parametrize("question,gate_cut", [
    ("how is TCS performing and should I buy INFY", 4.0),                # another share we hold
    ("how is TCS performing and what is its dividend ?", 2.5),           # a part routed as asking for what we lack
    ("how is TCS performing and should I buy apple", 4.0),               # shares resolve() cannot read were answered
    ("how is TCS performing and is the nifty overbought ?", 3.0),        # with TCS's figures
    ("how is it doing and should i buy TCS", 4.0),                       # "it" points ahead, to the share named next
    ("is it a good time to buy apple and how is TCS doing", 4.0),        # "it" points ahead, to no share it names
    ("is TCS overbought and how is apple doing this week ?", 2.5)])      # "this week" points at no share
def test_two_questions_asked_together_are_not_split_when_a_part_is_not_about_the_share_or_not_answerable(
        tmp_path, router, question, gate_cut):
    """Each cut is under every part's margin and over the whole question's, so only the part itself stops the split."""
    _, turn = together(tmp_path, router, question, gate_cut, model=ExplodingModel())
    assert (turn.spoke, turn.because) == (False, "unclear question")


def test_a_first_part_may_point_back_to_the_share_the_conversation_is_about(tmp_path, router):
    _, turn = together(tmp_path, router, "how is it doing and should i buy it", subject="TCS.NS")
    assert (turn.intent, turn.ticker, turn.spoke) == ("performance and buy", "TCS.NS", True)


@pytest.mark.parametrize("named", ["apple", "nykaa"])     # SEC's name for AAPL, an English word; a share not fetched
def test_two_questions_asked_together_are_not_split_when_a_later_part_names_another_share_by_name(tmp_path, market,
                                                                                                router, named):
    """"is it a good time to buy apple" points back, and resolve() cannot read "apple", so it had RELIANCE.NS's figures."""
    table = tmp_path / "company_tickers.json"
    table.write_text(json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}))
    held = finance.Market(tmp_path, symbols=table, shares=SHARES, fetch=Yahoo())
    turn = agent(held, router, gate_cut=4.0, model=ExplodingModel(),
                 reference=reference("never written", model=ExplodingModel())).answer(
        f"how is RELIANCE.NS performing and is it a good time to buy {named}")
    assert (turn.spoke, turn.because) == (False, "unclear question")


def test_two_questions_asked_together_with_a_look_back_word_are_split_before_the_newswire_reads_them(tmp_path, router):
    """With the newswire on, the whole question went to the price move and its buy part was dropped unsaid."""
    _, turn = together(tmp_path, router, "how has TCS done over the last month and should I buy it", 4.7,
                       reference=news_reference(ExplodingWriter(), news=()))
    assert (turn.intent, turn.spoke) == ("performance and buy", True)


@pytest.mark.parametrize("confidences", [(0.9991, 0.9995), (0.9995, 0.9991)])
def test_two_answers_given_together_report_the_least_confidence_of_the_two(tmp_path, router, confidences):
    _, turn = together(tmp_path, router, "how volatile is TCS and is it overbought ?", 3.0,
                       model=Confidences(*confidences))
    assert (turn.intent, turn.spoke, turn.confidence) == ("volatility and overbought", True, 0.9991)


def test_two_answers_given_together_are_served_in_alfas_and_our_words_and_never_reworded(tmp_path, router):
    _, turn = together(tmp_path, router, writer=ExplodingWriter())
    assert (turn.spoke, turn.phrased_by, turn.unphrased_because) == (True, "", "")


def test_two_answers_one_in_alfas_words_and_one_in_our_codes_name_both(tmp_path, router):
    """Joined as "our code", the turn read as our code's alone: shown as "Our code's sentence, where ALFA held back", in
    our colour, without the confidence of ALFA's own answer to the other part."""
    _, turn = together(tmp_path, router, model=Confidences(0.999, 0.5))
    assert (turn.spoke, turn.phrased_by, turn.confidence) == (True, f"{core.ALFA}, {core.BUILT}", 0.999)


@pytest.mark.parametrize("question,parts", [
    (ASKED_TOGETHER, ["how is TCS performing", "should I buy it"]),
    ("how volatile is TCS and is it overbought ?", ["how volatile is TCS", "is it overbought ?"]),
    ("what is the difference between stocks and bonds ?", []), ("how do RSI and MACD differ ?", []),
    ("how has TCS done ?", []),
    ("how volatile is TCS and is it overbought and should i buy it", [])])   # each part is a pass of the decoder
def test_a_question_is_split_on_and_only_when_every_part_opens_with_a_question_word(question, parts):
    assert finance.parts(question) == parts


@pytest.mark.parametrize("part,named", [
    ("should I buy it this week", "should I buy TCS.NS this week"),       # was "should I buy TCS.NS TCS.NS week"
    ("should i buy that ?", "should i buy TCS.NS ?"),
    ("should i buy this stock ?", "should i buy TCS.NS stock ?"), ("is this overbought ?", "is TCS.NS overbought ?"),
    ("how far is it below its high", "how far is TCS.NS below its high")])   # see `naming` on "its"
def test_a_part_pointing_back_names_the_share_in_place_of_a_pronoun_only(part, named):
    assert finance.naming(part, "TCS.NS") == named


def test_two_answers_given_together_are_shown_under_both_titles_with_their_figures(tmp_path, router):
    from backend.models.agent.present import present
    shown = present(together(tmp_path, router)[1], {}, {})
    assert shown.startswith("### TCS.NS · Recent performance · Should I buy?\n\nI do not have a directional signal for it.")
    assert f"{KEY_FIGURES} · Last close ₹" in shown


APPLE = Chunk("Mint, 6 Oct 2026: Apple shares rose after AAPL launched a new phone.", [], "https://n.test/aapl",
              "2026-10-06")


@pytest.mark.parametrize("news,said", [((), "could not read its SEC filings"), ((APPLE,), APPLE.text.partition(": ")[2])])
def test_filings_that_cannot_be_read_cost_their_own_passages_not_the_turn(registered, router, news, said):
    """The reply names the filings only when nothing else was found; the turn records the refusal either way."""
    built = news_reference(None, news=news, filings=RefusedLive())
    turn = agent(registered, router, model=ExplodingModel(), reference=built).answer("how has AAPL done lately ?")
    assert turn.spoke and turn.served.startswith("AAPL closed at $") and said in turn.served
    assert ("SEC filings" in turn.served) == (not news) and "403 from efts.sec.gov" in turn.because


def test_what_a_reader_sees_is_titled_figured_and_sourced(market, router):
    from backend.models.agent.present import present
    turn = agent(market, router, written="its 14 day rsi is 100 , which is overbought .").answer("is AAPL overbought ?")
    shown = present(turn, {}, {})
    assert shown.startswith("### AAPL · Momentum and RSI") and "Its 14 day RSI is 100, which is overbought." in shown
    assert f"{KEY_FIGURES} · Last close $165.00" in shown and "RSI (14) 100" in shown
    recent = agent(market, router, gate_cut=1e9, reference=news_reference(
        FakeWriter("Reliance Industries commissioned a refinery unit (Mint, 6 Oct 2026)."))).answer("any news on RELIANCE.NS ?")
    shown = present(recent, {"RELIANCE.NS": "Reliance Industries"}, {})
    assert shown.startswith("### Reliance Industries (RELIANCE.NS) · What happened lately")
    assert "**Sources**\n- [n.test, 6 Oct 2026](https://n.test/reliance)" in shown
    quiet = agent(market, router, model=ExplodingModel()).answer("how risky is AAPL over the next week ?")
    assert not quiet.spoke and present(quiet, {}, {}) == "I do not have that in the evidence for this question, so I cannot answer it."


def test_what_a_reader_sees_gives_an_index_close_in_points_not_dollars(tmp_path, router):
    """Key figures showed "Last close $25,910.05" for the nifty: every symbol not ending in .NS was priced in dollars."""
    from backend.models.agent.present import present
    price_file(tmp_path, "^NSEI")
    market = finance.Market(tmp_path)
    turn = agent(market, router, written="maruti. ns is doing fine .", confidence=1.0).answer(
        "how is the nifty doing ?", subject="^NSEI")
    assert f"{KEY_FIGURES} · Last close {market.snapshot('^NSEI')[1]['close']} · " in present(turn, {}, {})


def test_the_60_day_drawdown_is_labelled_the_biggest_fall_within_the_60_days(market, router):
    """"Off its 60-day high -12.4%" was on RELIANCE's card the day it closed at its 60-day high: the figure is the worst
    fall inside the 60 days."""
    from backend.models.agent.present import present
    turn = agent(market, router, written="its 14 day rsi is 100 , which is overbought .").answer("is AAPL overbought ?")
    assert f"Biggest fall from a peak in 60 days {market.snapshot('AAPL')[1]['drawdown_60d']}" in present(turn, {}, {})


@pytest.mark.parametrize("question,title", [("should I sell AAPL ?", "Should I sell?"),
                                            ("should I hold AAPL ?", "Should I hold?"),
                                            ("should I buy AAPL ?", "Should I buy?")])
def test_a_buy_answer_is_headed_by_the_trade_its_question_asks_about(market, router, question, title):
    """"should I sell reliance" was headed "Should I buy?"."""
    from backend.models.agent.present import present
    turn = agent(market, router, written="i do not have a directional signal for aapl .").answer(question)
    assert (turn.intent, present(turn, {}, {}).split("\n")[0]) == ("buy", f"### AAPL · {title}")


def test_alfas_lower_case_words_are_shown_as_sentences_with_the_symbol_as_written():
    """After a part ending "(Economic Times, 7 Oct 2026)", "i do not have…" stayed lower case and "tcs . ns" read "tcs. Ns"."""
    from backend.models.agent.present import present
    served = ("TCS.NS closed at ₹2405.00 (Economic Times, 7 Oct 2026)\n\n"
              "i do not have a directional signal for tcs . ns , and what i can say is little .")
    turn = SimpleNamespace(spoke=True, intent="performance and buy", ticker="TCS.NS", question=ASKED_TOGETHER,
                           served=served, evidence="", phrased_by="", because="")
    assert "\n\nI do not have a directional signal for TCS.NS, and what I can say is little." in present(turn, {}, {})


@pytest.mark.parametrize("written,confidence,why", [
    ("it trades at 4242.42 .", 1.0, "it gave a figure the data does not hold"),
    ("maruti. ns is doing fine .", 1.0, "it named another share"),
    ("aapl sits between its 20 and 50 day averages .", 1.0, "it misread where the price stands against its averages"),
    ("aapl sits above both its 20 and 50 day averages .", 0.5, "it was not sure enough of it")],
    ids=["unsupported figure", "another share", "wrong trend", "low confidence"])
def test_our_codes_sentence_in_alfas_place_says_alfas_draft_was_held_back_and_why(market, router, written, confidence,
                                                                                  why):
    """It led as ALFA's answer, "confidence 0.98 · worded by our code", for a draft refused for a figure 134444.80."""
    from backend.models.agent.present import present
    turn = agent(market, router, written=written, confidence=confidence).answer("how is AAPL doing ?")
    assert f"\n\nALFA's draft was held back: {why}. This is our code's sentence.\n\n" in present(turn, {}, {})


def test_a_part_our_code_answered_in_alfas_place_is_named_by_its_heading(tmp_path, router):
    from backend.models.agent.present import present
    _, turn = together(tmp_path, router, model=Confidences(0.999, 0.5))
    assert ("ALFA's draft for “Should I buy?” was held back: it was not sure enough of it. That part is our code's "
            "sentence.") in present(turn, {}, {})


@pytest.mark.usefixtures("generator_writes_facts")
def test_a_facts_answer_is_shown_with_what_its_figures_mean_every_fact_it_used_and_the_day_they_are_of(tmp_path):
    from backend.models.agent.present import present
    shown = present(facts_agent(tmp_path, Analyses()).answer(ASSESSMENT), {"RELIANCE.NS": "Reliance Industries"}, {})
    assert shown == ("### Reliance Industries (RELIANCE.NS) · Company facts\n\n"
                     "Reliance industries limited trades at 18.14 times the profit it is expected to make.\n\n"
                     "**What these figures mean**\n"
                     "- Forward P/E: the price against the profit per share expected over the next year.\n\n"
                     "**Facts used**\n- Stock: Reliance Industries Limited (RELIANCE.NS), NSE, INR.\n"
                     "- [fundamental] Forward P/E 18.14x (2026-11-13).\n\n"
                     "Source · Yahoo Finance via StockIntel, as of 13 Nov 2026")


def test_a_facts_reply_explains_at_most_four_terms_in_the_reports_words_and_none_from_the_stock_line(monkeypatch):
    """Written by code from TERM_MEANINGS, so a generated sentence can never define a term wrongly."""
    from backend.advisory import report
    from backend.models.agent.present import meanings
    monkeypatch.setitem(report.TERM_MEANINGS, "Current ratio", "a meaning only this test writes")
    facts = ["Stock: Beta Drugs Limited (BETA.NS), NSE, INR; last close 1,820.00 on 2026-11-13.",
             "[fundamental] Current ratio is 1.10. (FY ending 2026-03-31).", "[fundamental] Not available in our data: peg.",
             "[fundamental] Free cash flow yield is 3.8%. (snapshot retrieved 2026-11-15).",
             "[technical] RSI(14) is 70.8. (2026-11-13).", "[fundamental] Return on equity is 8.9%. (FY ending 2026-03-31).",
             "[fundamental] Gross margin 41.00% (FY ending 2026-03-31)."]
    assert meanings(facts, "") == ["- Current ratio: a meaning only this test writes.",
                                   f"- Free cash flow yield: {report.TERM_MEANINGS['Free cash flow yield']}.",
                                   f"- RSI: {report.TERM_MEANINGS['RSI']}.", f"- Equity: {report.TERM_MEANINGS['Equity']}."]


def test_a_facts_reply_explains_the_term_it_states_though_four_others_come_before_its_line():
    """The reply adds the line it took from the share's other lines last, so "what is the price to earnings ratio for
    AMZN" explained the four terms of its passage and not the P/E it stated."""
    from backend.advisory.report import TERM_MEANINGS
    from backend.models.agent.present import present
    lines = [TCS_LINES[0], "[fundamental] Current ratio is 1.05. (FY ending 2026-03-31).", TCS_LINES[6],
             "[fundamental] PEG ratio is 1.24. (2026-11-13).", "[fundamental] Free cash flow yield is 0.3%. (2026-11-13)."]
    told = facts_reply.reply("what is the price to earnings ratio for TCS ?", stock_qa.FACT_SEPARATOR.join(lines),
                             [*lines, TCS_LINES[3]])
    turn = SimpleNamespace(spoke=True, intent=finance.FACTS, ticker="TCS.NS", served=told.text, evidence=told.evidence,
                           as_of="2026-11-13", phrased_by=core.BUILT)
    assert f"- P/E: {TERM_MEANINGS['P/E']}." in present(turn, {}, {})
