"""The one domain this agent is implemented for, behind the seam the core routes through.

Every finance-specific fact lives here: which instruments exist, how a question names one, what the
evidence for a question looks like, and which facts an intent's answer is allowed to quote. The core
holds none of it, which is the whole point of the seam — a second domain would be another module of
this shape, and the plan says explicitly that we build one and leave the interface as the evidence.

Offline unless given a fetch. Instruments are the price files on disk, so the gate "answers a free-text
question end to end, no network" is met by there being no network call to make. Given `shares` and
`fetch` (ask.py passes them only with --live), a share on NSE's list is fetched from Yahoo the first time
it is named by symbol, and any share's bars again once its market has closed since they were written. A name is never
looked up, so anything else we hold no bars for is answered by saying so.
"""

import json
import os
import re
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from backend.advisory.domains import DAY_RANGE, DOMAIN_WORDS, FACTS_DOMAIN_WORDS, named_domains
from backend.config import market_of
from backend.models.agent.router import Router
from backend.models.core.backend import xp
from backend.models.data import advisory, prices, stock_qa
from backend.models.guardrails import guardrails
from backend.paths import DATA

from backend.advisory.sentiment.feeds import feed
from backend.knowledge_base import lessons

# Questions are lowercased by the tokenizer anyway, and a ticker is the one token whose case a user
# varies freely: "aapl", "Aapl", "AAPL" all name the same file.
_WORD = re.compile(r"[A-Za-z][A-Za-z.]*")
# Except a symbol that is also an everyday word: "transaction cost", "a cat", "after 3 pm" and "V-shaped" named
# Costco, Caterpillar, Philip Morris and Visa. Those need capitals, not after a number or an article ("50-day MA",
# "the PM said"; "Q3 COST" is fine) nor before a hyphen. Hand-kept: the tokenizer's vocabulary cannot tell "pep" from "aapl".
# The titles also stand before a person's name ("PM Modi", "V Anantha Nageswaran"), where "GE Aerospace" is GE.
_TITLES = frozenset({"HD", "PM", "V"})
_ALSO_WORDS = _TITLES | {"BA", "CAT", "COST", "DIS", "GE", "KO", "LIN", "MA", "PEP", "PG"}
_AFTER_NUMBER_OR_ARTICLE = re.compile(r"((?<![A-Za-z0-9])\d\S*\s*|-|\b(the|an?)\s+)$", re.I)
_BEFORE_NAME = re.compile(r"\s+[A-Z][a-z]")
# Checked before a symbol reaches a URL or a file name: no "/" or "..", and every NSE main-board symbol fits
_NSE_SYMBOL = re.compile(r"[A-Z0-9&-]{1,20}\.NS")
# PROVISIONAL, not measured: a fetch is waited for YAHOO_SLOW_S, and after one that fails or takes longer none is tried
# for YAHOO_BACK_OFF_S, so a hung Yahoo (40 s a question, measured) costs one question YAHOO_SLOW_S, not each 40 s.
# Timing fetches through a real outage and on healthy days would confirm both.
YAHOO_SLOW_S = 20
YAHOO_BACK_OFF_S = 300

REFUSAL = advisory.UNSUPPORTED

# The routed intent that means "placed, and the figures do not carry it". Re-exported for the same reason
# the refusal text is: the core acts on it and has no business knowing which dataset named it.
UNANSWERABLE = advisory.UNSUPPORTED_INTENT

# The intent that asks how the price moved, which is the one a time word ("lately") hands to the dated sources:
# the figures say how far it moved and the news may say why. The other intents are asked "right now" too.
MOVES = "performance"

# Re-exported rather than read from advisory by the core: the core knows about a domain, not about which
# dataset built it, and a second domain would cap its questions somewhere else.
QUESTION_TOKENS = advisory.QUESTION_TOKENS
# Where the answer forms put the instrument, which is where the wrong-share guard reads an answer for a share's name
SUBJECT_SLOTS = advisory.SUBJECT_SLOTS

# The intent answered from a share's dated facts, its window, and the stock Q&A set it trained on.
# The set's name is also what a generator's promotion records it trained on, which is what turns the route on.
FACTS = "facts"
FACTS_QUESTION_TOKENS = stock_qa.QUESTION_TOKENS
FACTS_SLOTS = stock_qa.FACTS_SLOTS
FACTS_METRICS = stock_qa.METRICS
facts_passage = stock_qa.facts_passage
facts_scorer = stock_qa.facts_scorer
STOCK_QA = "stock_qa/frozen_v1"
# The areas only the facts route reads, and the price intents' areas, which keep a question theirs: "is the drawdown
# worth it for the dividend" names a dividend. News is left to the router, as " event" is in "eventful", a risk word.
FACTS_AREAS = frozenset({"fundamental", "profile"})
PRICE_AREAS = frozenset({"risk", "technical"})
# Added to each facts example's score for a question `asks_facts` names, so the routing cut still decides it.
# PROVISIONAL: mid-plateau on the dev sets (the control, facts_dev, a held-out dev half, the dev templates, stock Q&A
# validation), where 0.08 to 0.15 placed the same questions wrongly. More price questions with facts words would tell.
FACTS_BONUS = 0.1
# The user's short price questions: the dev ones are router examples with the facts route on, the test ones measure it
SHORT_TEMPLATES = Path(__file__).with_name("price_short_templates.json")

NO_SUBJECT = "i need to know which stock or index you are asking about."

UNKNOWN_QUESTION = "i am not sure what you are asking. ask me about one stock or index: how its price has moved, " \
                   "if it looks overbought, how much it swings, how far it is below its high, or how risky next week looks."

OFF_TOPIC = "i can only help with money and markets. that means shares, companies and the economy."

NOT_IN_SOURCES = "i read the recent news on that, and none of it answers the question."

UNREACHABLE = "i would have to read something outside my own documents to answer that, and i could not " \
              "reach it just now. ask again in a minute."

NO_PRICES = "i could not get enough daily prices for that share. it may have listed too recently, or yahoo " \
            "may be out of reach just now."

NO_FACTS = "i could not get that company's latest figures, so i cannot answer that just now."

NO_DAY_RANGE = "i do not give a day's high or low. my answers are about how a price has moved over days and weeks."


class NoPrices(RuntimeError):
    """A share whose prices could not be had: Yahoo gave none, or too few for a snapshot.

    The core turns it into a refusal, the way it does `reference.SearchFailed`.
    """


class NoFacts(RuntimeError):
    """A share whose dated facts could not be had: no analysis is read here, or Yahoo gave none. Refused like NoPrices."""


class Subject(namedtuple("Subject", "ticker names cik")):
    """An instrument a question is about: what a passage would call it, and its SEC number if it has one.

    A domain fact, not a retrieval one, which is why it lives here and is handed to the reader. What
    counts as a mention of RELIANCE.NS is the same judgement the news tagger already makes, and there is
    one of it.
    """

    def mentions(self, text):
        """Whether `text` is about this instrument, by its symbol or by one of the company's names."""
        return self.ticker in feed.tag(text, {self.ticker: self.names})

    @property
    def named(self):
        """The one name to search an outside source by: the registered name if known, else the symbol.

        A wider net than `names` deliberately, and the two are not interchangeable. A search may be asked a
        guess and ranks the answer; the gate may not, because a guess there is what lets a passage that is
        not about this instrument be served as though it were.
        """
        return max(self.names, key=lambda name: (len(name), name)) if self.names \
            else self.ticker.split(".")[0].lower()


class Market:
    """The instruments on disk, and the as-of snapshot of any one of them.

    Bars are read per question rather than held in memory: 101 files of ten years is small, and a cache
    that went stale against a re-download would be a wrong figure served confidently.

    `advisor` is what produces the week-ahead outlook, from `backend/models/price.py`. Given here rather
    than per question so that loading one turns the risk intent on everywhere at once; left out, the
    evidence simply does not carry the figure and the core refuses the one intent that quotes it.

    `shares` ({"NYKAA.NS": "FSN E-Commerce Ventures"}) and `fetch(ticker, path)` come together or not at
    all: a listed share with no file is fetched when a question names it, by `hold`, and so is any share whose file
    was written before its market's last close. `analyse(ticker)` is the app's
    own analysis of a share as a dict (`AnalysisService.analyze(...).to_dict()`), which `facts` reads.
    """

    def __init__(self, price_dir, advisor=None, symbols=None, scenarios=None, listed=None, shares=None,
                 fetch=None, analyse=None):
        self.paths = {path.stem.upper(): path for path in sorted(price_dir.glob("*.csv"))}
        if not self.paths:
            raise ValueError(f"no price files under {price_dir}; the agent would have no evidence to read")
        # "RELIANCE" should find RELIANCE.NS, but only when no plain RELIANCE.csv exists to prefer.
        self.aliases = {name.split(".")[0]: name for name in self.paths if "." in name}
        self.registrants = feed.registrants(symbols, self.paths) if symbols else {}
        # SEC's table holds no Indian listing, so their names come from the exchange's own constituent list
        # ({"INFY.NS": "infosys"}): the name a question and a headline use, which the symbol often is not
        self.listed = {ticker: name.lower() for ticker, name in (listed or {}).items() if ticker in self.paths}
        for ticker, name in self.listed.items():
            self.registrants.setdefault(ticker, (None, (name,)))
        self.advisor = advisor
        self.scenarios = scenarios
        if (shares is None) != (fetch is None):
            raise ValueError("give Market `shares` and `fetch` together or not at all: a list alone names shares "
                             "it cannot fetch, and a fetch alone has no list to check a symbol against")
        bad = sorted(ticker for ticker in shares or () if not _NSE_SYMBOL.fullmatch(ticker))
        if bad:
            raise ValueError(f"{bad[:5]} are not NSE symbols of the form {_NSE_SYMBOL.pattern}; check the list "
                             "the shares were read from")
        self.shares = {ticker: name.lower() for ticker, name in (shares or {}).items()}
        self.fetch = fetch
        # One worker: a fetch given up on still runs until yfinance gives up, so a hung Yahoo holds one thread, not many
        self.fetching = ThreadPoolExecutor(max_workers=1)
        self.unreachable_until = 0.0
        # What the wrong-share guard reads an answer for. Not a word an answer about any share may use ("take", "idea")
        english = feed.english_words()
        self.names = frozenset(name for ticker in {*self.paths, *self.shares}
                               for name in (ticker.split(".")[0].lower(), self.shares.get(ticker, ""),
                                            *self.registrants.get(ticker, (None, ()))[1])
                               if name and feed.is_name(name, english))
        # Apart from the collector's files, so no training glob reads them and only the agent writes here
        self.fetched = price_dir / "fetched"
        self.analyse = analyse
        self.analysed = price_dir / "facts"

    def subject(self, ticker):
        """`ticker` as a thing a passage can be about. What the gate on a read-out answer is checked with.

        Registered names only, never the symbol's own stem, which was measured: the stem of RELIANCE.NS is
        the ordinary English word "reliance", and 1,082 dated corpus passages contain it -- Federal Reserve
        releases about banks' reliance on funding, none of them about the company. So an instrument SEC's
        table does not carry is known here by its full symbol and nothing else, no stored passage says that,
        and its questions are answered by searching for it by name instead. Narrow, and honest about which
        half of what we hold this file can recognise in prose at all.
        """
        cik, registered = self.registrants.get(ticker, (None, ()))
        return Subject(ticker, tuple(sorted(registered)), cik)

    def resolve(self, text):
        """The ticker a question names, or None. Matched against what is on disk, never guessed.

        By symbol first, then by an exchange-listed company name as whole words, the longest name winning
        so "hdfc bank" is not read as some shorter name inside it. A listed share we hold no file for is
        found only by its symbol: typed with ".NS", or handed over whole ("M&M.NS", which `_WORD` cannot read).
        """
        if text.strip() in self.paths or text.strip() in self.shares:
            return text.strip()
        for found in _WORD.finditer(text):
            symbol = found.group(0).upper().strip(".")
            if symbol in _ALSO_WORDS and not _typed_as_symbol(text, found):
                continue
            if symbol in self.paths:
                return symbol
            if symbol in self.aliases:
                return self.aliases[symbol]
            if symbol in self.shares:
                return symbol
        named = feed.tag(text, {ticker: (name,) for ticker, name in self.listed.items()})
        return max(named, key=lambda ticker: len(self.listed[ticker])) if named else None

    def names_another(self, text, ticker):
        """Whether `text` names a share other than `ticker` by a name known here that `resolve` does not read: SEC's
        name for one, even an English word ("apple"), or one the guard reads for, such as a share not yet fetched. A
        question has none of the answer forms' slots."""
        own = " ".join((ticker, *self.subject(ticker).names))
        return bool(feed.tag(text, {other: names for other, (_, names) in self.registrants.items() if other != ticker})
                    or guardrails.other_shares(text, own, self.names, (frozenset(),) * 3))

    def hold(self, ticker):
        """Make sure `ticker` has bars on file, and with a fetch given, bars written after its market's last close, as
        its facts are. The collector's files count too: it runs on a day's timer, not after each close, so a price
        answer read TCS at 2,505.00 from 2 Nov beside facts that read 2,405.00 on 16 Nov.

        Raises NoPrices when Yahoo gives none, or too few for a snapshot. A stale file is refused, not served.
        """
        closed = last_close(market_of(ticker), time.time())
        if ticker in self.paths and (self.fetch is None or self.paths[ticker].stat().st_mtime >= closed):
            return
        path = self.fetched / f"{ticker}.csv"
        if not path.exists() or path.stat().st_mtime < closed:
            self._from_yahoo(lambda: self.fetch(ticker, path), ticker, "prices", NoPrices)
        dates, _ = prices.load_bars(path)
        if len(dates) < advisory.BARS_NEEDED:
            raise NoPrices(f"{ticker} has {len(dates)} days of prices, and its figures need {advisory.BARS_NEEDED}")
        self.paths[ticker] = path
        if ticker in self.shares:
            self.registrants.setdefault(ticker, (None, (self.shares[ticker],)))

    def facts(self, ticker, as_of=None):
        """(the lines `stock_qa.facts` reads off the app's analysis of `ticker`, the date of its last close).

        Kept under facts/ until its market's next close, and fetched under the price files' back-off. Raises NoFacts for
        an index, when this market reads no analysis, Yahoo gives none, or `as_of` is before the analysis's last close.
        """
        if ticker.startswith("^"):
            raise NoFacts(f"{ticker} is an index, and the facts route trained on shares only")
        if self.analyse is None:
            raise NoFacts(f"this process reads no analysis of {ticker}; facts are read only with --live")
        path = self.analysed / f"{ticker}.json"
        if not path.exists() or path.stat().st_mtime < last_close(market_of(ticker), time.time()):
            self._from_yahoo(lambda: _written(self.analyse(ticker), path), ticker, "analysis", NoFacts)
        analysis = json.loads(path.read_text(encoding="utf-8"))
        day = analysis["quality"]["last_date"]
        if as_of is not None and str(as_of) < day:
            raise NoFacts(f"the analysis of {ticker} reads prices to {day}, after {as_of}")
        return stock_qa.facts(analysis), day

    def _from_yahoo(self, job, ticker, what, failed):
        """Run `job` on the one Yahoo worker for at most YAHOO_SLOW_S; when it fails, back off and raise `failed`."""
        if time.time() < self.unreachable_until:
            raise failed(f"yahoo failed or was slow under {YAHOO_BACK_OFF_S} s ago, so {ticker} is not fetched "
                         "until then; ask again in a few minutes")
        try:
            # Given up on, not cancelled: a fetch over the limit still ends its whole-file write in the background
            self.fetching.submit(job).result(timeout=YAHOO_SLOW_S)
        except TimeoutError as error:
            self.unreachable_until = time.time() + YAHOO_BACK_OFF_S
            raise failed(f"yahoo gave no {what} for {ticker} in {YAHOO_SLOW_S} s; ask again in a few minutes") \
                from error
        except RuntimeError as error:
            self.unreachable_until = time.time() + YAHOO_BACK_OFF_S
            raise failed(str(error)) from error

    def snapshot(self, ticker, as_of=None):
        """(evidence text, the figures it states, the date it was taken at) for one instrument.

        `as_of` is the last date the snapshot may read, and the default is the last bar on file. The
        filter is the same leakage rule the training snapshots obey: no bar after the one asked for.
        """
        dates, bars = prices.load_bars(self.paths[ticker])
        end = prices.index_on_or_before(dates, as_of)
        if end < advisory.BARS_NEEDED - 1:
            raise ValueError(
                f"{ticker} has {end + 1} bars up to {as_of or dates[-1]}, and a snapshot needs "
                f"{advisory.BARS_NEEDED}; the indicators would be computed from a shorter window"
            )
        shown = advisory.as_text(advisory.snapshot(bars, end))
        # A head needs more history than the indicators do, and short of it the honest move is to leave
        # the figure out and let the core name it in the refusal, not to score a partial window.
        if self.advisor is not None and end + 1 >= self.advisor.bars_needed:
            scored = self.advisor.probabilities(bars, end)
            shown.update(advisory.risk_outlook(scored, confidence=self.advisor.confidence(scored)))
        return advisory.render_evidence(ticker, shown), shown, dates[end]

    def outlook(self, ticker, as_of=None, history=120):
        """(dates, closes, class probabilities) behind the week-ahead call, on the bars `snapshot` reads.

        Here rather than in a client for the same reason the snapshot is: two paths computing "the model's
        call" is two answers to one question. Probabilities are None when no advisor is loaded or the
        instrument is short of the history one needs, which is when the evidence carries no outlook either.
        """
        dates, bars = prices.load_bars(self.paths[ticker])
        end = prices.index_on_or_before(dates, as_of)
        start = max(0, end - history + 1)
        enough = self.advisor is not None and end + 1 >= self.advisor.bars_needed
        return (dates[start : end + 1], [float(close) for close in bars[start : end + 1, 3]],
                [float(p) for p in self.advisor.probabilities(bars, end)] if enough else None)

    def draw_paths(self, ticker, steps, count, rng, as_of=None):
        """(as-of date, last close, (count, steps) sampled daily log returns) from bars up to `as_of`."""
        dates, bars = prices.load_bars(self.paths[ticker])
        end = prices.index_on_or_before(dates, as_of)
        if end + 1 < self.scenarios.bars_needed:
            raise ValueError(f"{ticker} has {end + 1} bars up to {dates[end]}, and drawing its paths needs "
                             f"{self.scenarios.bars_needed}")
        closes = bars[: end + 1, 3]
        return dates[end], float(closes[-1]), self.scenarios.paths(closes, steps, count, rng)


# Everyday facts questions, as short as typed ones. The 455 stock Q&A templates they replace pushed "how risky is
# infosys" under the routing cut. Not "why is {t} rising" or "how fast is {t} growing": one word from "is tcs rising".
# Nor "is {t} profitable ?", which put "is tcs profitable to trade" on facts past the routing cut.
FACTS_PARAPHRASES = (
    "is {t} priced too high ?", "how leveraged is {t} ?", "what are its borrowings ?", "is {t} making money ?",
    "does {t} generate cash ?", "what could go wrong for {t} ?", "what threatens {t} ?",
    "what problems does {t} face ?", "what is the p/e of {t} ?", "{t} pe ratio ?", "is {t} overvalued ?",
    "what is the valuation of {t} ?", "is {t} debt free ?", "{t} debt to equity",
    "what is {t}'s roe ?", "{t} profit margin", "{t} revenue growth", "is {t}'s revenue growing ?",
    "does {t} pay a dividend ?", "{t} dividend yield", "what business is {t} in ?", "how does {t} make money ?",
    "what does {t} do ?", "what does {t} make ?", "who are {t}'s peers ?", "{t} free cash flow",
)
# Short direction questions, under the intent whose answer holds them, which dev templates such as "why is {t} going
# down?" took them off. As performance, "is {t} going down ?" sent "hey tcs beatne down" there too. A correction is a fall
# from a peak, which the drawdown answers; the dev template on a "normal correction" put "is tcs in a correction" on facts.
MOVE_PARAPHRASES = (("drawdown", "is {t} going down ?"), ("drawdown", "is {t} dropping ?"),
                    ("performance", "how is {t} trending ?"), ("performance", "is {t} strong ?"),
                    ("drawdown", "has {t} gone into a correction ?"))


def examples(paraphrased=True, facts=False):
    """(intent, phrasing, text) for every trained phrasing and every routing-only paraphrase.

    The instrument is stripped out of both sides: the intent is carried by the rest of the wording, and
    leaving one ticker in would make every question about a different instrument slightly unfamiliar.

    A paraphrase carries the intent's canonical phrasing as its key, so a question matched on shape is
    still answered in wording the decoder trained on. `paraphrased=False` is the control the paraphrases
    are measured against, and nothing but a measurement should pass it. `facts` adds the facts route's examples, the
    short direction questions and the dev short templates, each under the label its answer comes from, after the old
    ones.
    """
    known = [(intent, phrasing, _squeeze(advisory.ask(intent, "", phrasing)))
             for intent in advisory.INTENTS
             for phrasing in advisory.trained_phrasings(intent)]
    if paraphrased:
        known += [(intent, advisory.canonical_phrasing(intent), _squeeze(frame.format(t="")))
                  for intent in advisory.INTENTS
                  for frame in advisory.paraphrases(intent)]
    if facts:
        known += [(label, frame if label == FACTS else advisory.canonical_phrasing(label), _squeeze(frame.format(t="")))
                  for label, frame in [(FACTS, frame) for frame in FACTS_PARAPHRASES] + list(MOVE_PARAPHRASES)
                  + short_templates("dev")]
    return known


def short_templates(split):
    """(label, question with {t} for the share) for each of the user's short templates in `split`, "dev" or "test"."""
    held = json.loads(SHORT_TEMPLATES.read_text(encoding="utf-8"))
    labels = {row["id"]: held["categories"][row["category"]].get("except", {}).get(
        row["id"], held["categories"][row["category"]]["label"]) for row in held["templates"]}
    unknown = sorted({label for label in labels.values() if label not in (*advisory.INTENTS, FACTS)})
    if unknown:
        raise ValueError(f"{SHORT_TEMPLATES.name} labels templates {unknown}, which no route answers; use one of "
                         f"{(*advisory.INTENTS, FACTS)}")
    return [(labels[row["id"]], row["question"]) for row in held["templates"] if row["split"] == split]


def stock_questions(split):
    """The rows of one split of the frozen stock Q&A set, as `prepare_stock_qa` wrote them."""
    with open(DATA / STOCK_QA / f"{split}.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _starting_a_word(words):
    """Any of `words` where a word starts, so "rsi" is not read inside "diversified". Searched with a space after."""
    return re.compile(r"(?<![a-z])(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + ")", re.I)


_PRICE_AREA = _starting_a_word(word.lstrip() for area in PRICE_AREAS for word in DOMAIN_WORDS[area])
_FIGURE = _starting_a_word(word.lstrip() for word in DOMAIN_WORDS["fundamental"])
# Words only price questions use, checked by hand against facts_dev and a held-out set's dev half: the price intents'
# and sell templates' words, "exit" and "should i", that no trained unsupported phrasing or facts frame uses, less ones
# like "high" and "model" that facts questions use too ("is its debt high"). Each matches where a word starts: "buying".
# The move words keep "did its price rise on the profit news" off facts. Not "rising" or "gain": facts questions use
# them of figures ("receivables are rising", "treasury gains"), and "rising" took two stock Q&A validation questions.
PRICE_WORDS = ("action", "add", "avoid", "bounce", "bumpy", "buy", "calm", "choppy", "climb", "correction", "crash",
               "deep", "dip", "doing", "done", "drop", "entry", "exhausted", "exit", "extended", "fall", "faring",
               "going up", "grim", "hold", "intraday", "jump", "lately", "lively", "month", "move", "overbought",
               "overdone", "overheated", "oversold", "painful", "peak", "performance", "performing", "plung", "quiet",
               "rall", "range", "react", "restless", "rise", "rose", "rough", "sell", "settled", "shaping", "should i",
               "slid", "slump", "smooth", "spik", "steady", "stopping", "stretch", "summary", "surge", "surging",
               "swings", "tank", "trade", "trading", "turbulence", "week", "wild", "worst")
_PRICE_WORD = _starting_a_word(PRICE_WORDS)
# "what does" opens price questions too ("what does the dip in tcs mean"), so alone it names no profile here, and
# "what does ... do" only when the question ends there: "what does the dip do to tcs" asks about the price
FACTS_ASKED = {**FACTS_DOMAIN_WORDS, "profile": tuple(w for w in FACTS_DOMAIN_WORDS["profile"] if w != "what does")}
_DOES = re.compile(r"\bwhat does (\S+ ){0,3}?(do|make|sell)s?\s*\??\s*$", re.I)
# Indicators no evidence holds, whose nearest examples are the facts route's MACD and RSI templates. Not Bollinger: a
# facts line says when the close is outside its bands.
_UNHELD = re.compile(r"\b(adx|ema)\b", re.I)
# A question asking when ("when does tcs report earnings") asks for a date, and the facts reply gives its figures instead
_WHEN = re.compile(r"\bwhen (does|do|is|are|will|did|was|were)\b", re.I)


def asks_facts(text):
    """Whether a question asks what a company does, makes or sells, or names an area only the facts route reads and no
    price word ("what is its p/e" but not "should i buy for its dividend"). A risk or technical word keeps it off too,
    unless it names a figure: "how volatile is its revenue" asks about the revenue."""
    spaced = f"{text} "
    return _DOES.search(text) is not None or (
        bool(set(named_domains(text, FACTS_ASKED)) & FACTS_AREAS) and _PRICE_WORD.search(spaced) is None
        and (_PRICE_AREA.search(spaced) is None or _FIGURE.search(spaced) is not None) and _WHEN.search(text) is None)


class FactsFirst:
    """A router over `known`, the old intents' examples and those the facts route adds, scored by `score`.

    A question `asks_facts` names gets FACTS_BONUS on each facts example, and is facts when placed on the unsupported
    intent, whose "debt load" and "dividend yield" are what the facts lines hold. An old intent that the old examples
    alone also place keeps their margin when it is wider: the added examples only made the router less sure of the same
    answer, and the routing cut was fitted on the old examples' margins. So does their unsupported placement of a
    question `asks_facts` names, which is then facts. A question placed on facts that names an indicator no evidence
    computes is unsupported.
    """

    def __init__(self, known, score):
        old = set(examples())
        kept = xp.asarray([example in old for example in known])
        bonus = xp.asarray([label == FACTS for label, _, _ in known]) * FACTS_BONUS
        self.router = Router(known, lambda text: score(text) + bonus * asks_facts(text))
        self.old = Router([example for example in known if example in old], lambda text: xp.asarray(score(text))[kept])
        self.labels = self.router.labels

    def route(self, text):
        placed, before, named = self.router.route(text), self.old.route(text), asks_facts(text)
        kept = before.label == placed.label or before.label == UNANSWERABLE and named
        placed = before if kept and before.margin > placed.margin else placed
        if placed.label == FACTS and _UNHELD.search(text):
            return placed._replace(label=UNANSWERABLE, key=advisory.canonical_phrasing(UNANSWERABLE))
        return placed._replace(label=FACTS, key=FACTS) if placed.label == UNANSWERABLE and named else placed


# The words a question opens with. Joined questions are split on " and " only when each part opens with one: that splits
# none of route.py's 385 questions or the 78 user questions, while splitting on any "and" or "," sent 4 preambles astray.
_ASKS = re.compile(r"(how|what|should|is|are|would|do|does|can|could|will|has|have|why|where|when)\b", re.I)
# Each part is a decoder pass under the server's one generating lock: unbounded, one 2000-character message made 86
MOST_PARTS = 2
# A part that names no share points back to the one before with these: "and should i buy it", "and what is its rsi".
# "this" and "that" not before a time: "how is apple doing this week" points at the week, not the share.
_BACK = re.compile(r"\b(it|them|(its)|(this|that)(?!\s+(week|month|year|quarter|time|morning|session|day)\b))\b", re.I)


def parts(question):
    """The questions asked together in one ("how is TCS doing and should i buy it"), or [] when it asks one or too many."""
    found = [part.strip() for part in re.split(r"\s+and\s+", question) if part.strip(" ?.")]
    return found if 1 < len(found) <= MOST_PARTS and all(_ASKS.match(part) for part in found) else []


def points_back(part):
    return _BACK.search(part) is not None


def points_back_only_by_this_or_that(part):
    back = [found.group(3) for found in _BACK.finditer(part)]
    return bool(back) and all(back)


def asks_day_range(text):
    return DAY_RANGE.search(text) is not None


def naming(part, ticker):
    """`part` with the share it points back to named, so the writer reads "should i buy TCS.NS". "its" stays: naming it
    took the lexical router's margin on "how far is it below its high" from 9.68 to 6.45."""
    return _BACK.sub(lambda found: found.group(2) or ticker, part)


def without_subject(text, ticker):
    """The question as the router sees it, with the instrument taken out.

    Both sides go through the same squeeze, so the gap the ticker left behind cannot be the difference
    between a query and the phrasing it should match.
    """
    return _squeeze(re.sub(
        rf"(?i)(?<![a-z]){re.escape(ticker.split('.')[0])}(\.[A-Za-z]+)?(?![a-z])", "", text))


def _squeeze(text):
    return " ".join(text.split())


def last_close(market, now):
    """When `market`'s session last closed at or before `now`, in seconds. Weekends and holidays count as sessions,
    which costs one fetch that changes nothing."""
    local = datetime.fromtimestamp(now, ZoneInfo(market.timezone))
    closed = datetime.combine(local.date(), market.closes_at, local.tzinfo)
    return (closed if closed <= local else closed - timedelta(days=1)).timestamp()


def _written(analysis, path):
    """Written whole or not at all, as a fetched price file is: a question may read it while another fetch runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    # default=str as prepare_stock_qa wrote the training snapshots, so the facts read back are the ones trained on
    part.write_text(json.dumps(analysis, default=str), encoding="utf-8")
    os.replace(part, path)


def _typed_as_symbol(text, found):
    return (found.group(0).isupper() and not _AFTER_NUMBER_OR_ARTICLE.search(text[: found.start()])
            and not text.startswith("-", found.end())
            and not (found.group(0).strip(".") in _TITLES and _BEFORE_NAME.match(text, found.end())))


def ask(intent, phrasing, ticker):
    """The trained wording of a routed question, which is what the decoder is given instead."""
    return advisory.ask(intent, ticker, phrasing)


def needs(intent):
    """The evidence figures this intent's answer quotes, so a missing one is a refusal not an invention."""
    return advisory.NEEDS[intent]


# Words that put a question about money or markets. The reference path reads 519,135 general passages and the
# open web, so without a gate "write me a poem about the sea" was answered out of The Lusiads; the language
# model asked to decline such questions either wrote facts about the sea or, told more firmly, declined
# questions about Titan's results too.
_MONEY = re.compile(
    r"\b(markets?|stocks?|shares?|equit(y|ies)|funds?|mutual|sips?|navs?|nifty|sensex|index|indices|invest\w*|"
    r"prices?|trad(e|es|ed|ing|ers?)|banks?|banking|loans?|rates?|repo|rbi|sebi|inflation|gdp|econom\w*|tax\w*|"
    r"ipos?|dividends?|bonds?|yields?|gold|silver|rupee|dollars?|currenc\w*|crypto\w*|bitcoin|profits?|loss(es)?|"
    r"revenue|earnings|results?|quarter\w*|compan(y|ies)|business\w*|portfolio|risk\w*|volatil\w*|returns?|money|"
    r"salary|savings?|insurance|pension|ppf|epf|deposits?|credit|debt|interest|mortgage|emi|budget|financ\w*|"
    r"capital (gains?|markets?)|balance sheets?|real estate|property|propert(y|ies)|valuation|ratios?|brokers?|demat|futures|options|derivatives?|commodit\w*|crude|oil|sectors?|"
    r"rall(y|ies|ied|ying)|crash\w*|bull\w*|bear\w*|stop[- ]loss|orders?|hedg\w*|etfs?|analysts?|ratings?|buyback|"
    r"merger|acquisitions?|listing|momentum|drawdown|wealth|income|expense\w*|cash|spend\w*|economy|"
    r"rsi|macd|bollinger|stochastic oscillators?|overbought|oversold|(small|mid|large)[- ]?caps?|fiis?|diis?|"
    r"diversif\w*|candlesticks?|chart patterns?|doji|harami|sharpe|sortino|ebitda|ltcg|stcg|elss|cagr|xirr)\b", re.I)
# Words about markets only beside another: "the volume of a sphere", "an acne breakout" and "a pie chart" are not.
_PAIRED = re.compile(r"\b(volume|breakouts?|charts?|candles?|hammer)\b", re.I)


def about_money(text, lesson=""):
    """Whether a question is about money: by its words, or because the curated `lesson` found for it uses every one of
    them. One shared word is not enough: "how do candles burn" found the lesson on candle anatomy."""
    return (_MONEY.search(text) is not None or len(lessons.stems(" ".join(_PAIRED.findall(text)))) > 1
            or bool(lesson) and lessons.covers(lesson, text))
