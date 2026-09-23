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

from selfagent.data import advisory, prices

# Questions are lowercased by the tokenizer anyway, and a ticker is the one token whose case a user
# varies freely: "aapl", "Aapl", "AAPL" all name the same file.
_WORD = re.compile(r"[A-Za-z][A-Za-z.]*")

REFUSAL = advisory.UNSUPPORTED

# Re-exported rather than read from advisory by the core: the core knows about a domain, not about which
# dataset built it, and a second domain would cap its questions somewhere else.
QUESTION_TOKENS = advisory.QUESTION_TOKENS

NO_SUBJECT = "i need to know which instrument you are asking about."

UNKNOWN_QUESTION = "i am not sure what you are asking. i can talk about price, momentum, volatility, " \
                   "drawdown and week-ahead risk for one instrument."


class Market:
    """The instruments on disk, and the as-of snapshot of any one of them.

    Bars are read per question rather than held in memory: 101 files of ten years is small, and a cache
    that went stale against a re-download would be a wrong figure served confidently.

    `advisor` is what produces the week-ahead outlook, from `backend/models/price.py`. Given here rather
    than per question so that loading one turns the risk intent on everywhere at once; left out, the
    evidence simply does not carry the figure and the core refuses the one intent that quotes it.
    """

    def __init__(self, price_dir, advisor=None):
        self.paths = {path.stem.upper(): path for path in sorted(price_dir.glob("*.csv"))}
        if not self.paths:
            raise ValueError(f"no price files under {price_dir}; the agent would have no evidence to read")
        # "RELIANCE" should find RELIANCE.NS, but only when no plain RELIANCE.csv exists to prefer.
        self.aliases = {name.split(".")[0]: name for name in self.paths if "." in name}
        self.advisor = advisor

    def resolve(self, text):
        """The ticker a question names, or None. Matched against what is on disk, never guessed."""
        for word in _WORD.findall(text):
            symbol = word.upper().strip(".")
            if symbol in self.paths:
                return symbol
            if symbol in self.aliases:
                return self.aliases[symbol]
        return None

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
            shown.update(advisory.risk_outlook(self.advisor.probabilities(bars, end)))
        return advisory.render_evidence(ticker, shown), shown, dates[end]


def examples():
    """(intent, phrasing, text) for every trained phrasing, as the router's known questions.

    The instrument is stripped out of both sides: the intent is carried by the rest of the wording, and
    leaving one ticker in would make every question about a different instrument slightly unfamiliar.
    """
    return [(intent, phrasing, _squeeze(advisory.ask(intent, "", phrasing)))
            for intent in advisory.INTENTS
            for phrasing in advisory.trained_phrasings(intent)]


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
