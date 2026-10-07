"""The one domain this agent is implemented for, behind the seam the core routes through.

Every finance-specific fact lives here: which instruments exist, how a question names one, what the
evidence for a question looks like, and which facts an intent's answer is allowed to quote. The core
holds none of it, which is the whole point of the seam — a second domain would be another module of
this shape, and the plan says explicitly that we build one and leave the interface as the evidence.

Offline by construction. Instruments are the price files on disk and nothing else, so the gate "answers
a free-text question end to end, no network" is met by there being no network call to make. The
reference repo resolves a company name through a Yahoo search endpoint; that is a C4 concern at the
earliest, and until then a question naming something we hold no bars for is answered by saying so.
"""

import re
from collections import namedtuple

from backend.models.data import advisory, prices

from backend.advisory.sentiment.feeds import feed

# Questions are lowercased by the tokenizer anyway, and a ticker is the one token whose case a user
# varies freely: "aapl", "Aapl", "AAPL" all name the same file.
_WORD = re.compile(r"[A-Za-z][A-Za-z.]*")

REFUSAL = advisory.UNSUPPORTED

# The routed intent that means "placed, and the figures do not carry it". Re-exported for the same reason
# the refusal text is: the core acts on it and has no business knowing which dataset named it.
UNANSWERABLE = advisory.UNSUPPORTED_INTENT

# Re-exported rather than read from advisory by the core: the core knows about a domain, not about which
# dataset built it, and a second domain would cap its questions somewhere else.
QUESTION_TOKENS = advisory.QUESTION_TOKENS

NO_SUBJECT = "i need to know which instrument you are asking about."

UNKNOWN_QUESTION = "i am not sure what you are asking. i can talk about price, momentum, volatility, " \
                   "drawdown and week-ahead risk for one instrument."

OFF_TOPIC = "i can only help with money, markets, companies and the economy."

NOT_IN_SOURCES = "i read the recent news and my documents on that, and none of it answers the question."

UNREACHABLE = "i would have to read something outside my own documents to answer that, and i could not " \
              "reach it just now. ask again in a minute."


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
    """

    def __init__(self, price_dir, advisor=None, symbols=None, scenarios=None, listed=None):
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
        so "hdfc bank" is not read as some shorter name inside it.
        """
        for word in _WORD.findall(text):
            symbol = word.upper().strip(".")
            if symbol in self.paths:
                return symbol
            if symbol in self.aliases:
                return self.aliases[symbol]
        named = feed.tag(text, {ticker: (name,) for ticker, name in self.listed.items()})
        return max(named, key=lambda ticker: len(self.listed[ticker])) if named else None

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


def examples(paraphrased=True):
    """(intent, phrasing, text) for every trained phrasing and every routing-only paraphrase.

    The instrument is stripped out of both sides: the intent is carried by the rest of the wording, and
    leaving one ticker in would make every question about a different instrument slightly unfamiliar.

    A paraphrase carries the intent's canonical phrasing as its key, so a question matched on shape is
    still answered in wording the decoder trained on. `paraphrased=False` is the control the paraphrases
    are measured against, and nothing but a measurement should pass it.
    """
    known = [(intent, phrasing, _squeeze(advisory.ask(intent, "", phrasing)))
             for intent in advisory.INTENTS
             for phrasing in advisory.trained_phrasings(intent)]
    if not paraphrased:
        return known
    return known + [(intent, advisory.canonical_phrasing(intent), _squeeze(frame.format(t="")))
                    for intent in advisory.INTENTS
                    for frame in advisory.paraphrases(intent)]


def without_subject(text, ticker):
    """The question as the router sees it, with the instrument taken out.

    Both sides go through the same squeeze, so the gap the ticker left behind cannot be the difference
    between a query and the phrasing it should match.
    """
    return _squeeze(re.sub(
        rf"(?i)(?<![a-z]){re.escape(ticker.split('.')[0])}(\.[A-Za-z]+)?(?![a-z])", "", text))


def _squeeze(text):
    return " ".join(text.split())


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
    r"rall(y|ies)|crash\w*|bull\w*|bear\w*|stop[- ]loss|orders?|hedg\w*|etfs?|analysts?|ratings?|buyback|"
    r"merger|acquisitions?|listing|momentum|drawdown|wealth|income|expense\w*|cash|spend\w*|economy)\b", re.I)


def about_money(text):
    """Whether a question is about money, markets, companies or the economy, by its words alone."""
    return _MONEY.search(text) is not None
