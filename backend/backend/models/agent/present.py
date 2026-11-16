"""What a reader sees: a turn laid out as a title, the answer, the figures behind it and where they came from.

Built by our code from the turn, after the agent has decided what to say, so the layout never depends on
the language model: it writes sentences only, and every figure on the "Key figures" line is read straight
out of the evidence. Markdown, which the chat renders.
"""
import json
import re
from datetime import date

from backend.advisory.report import TERM_MEANINGS
from backend.models.agent import phrase
from backend.models.agent.core import BUILT, TOGETHER
from backend.models.agent.finance import FACTS
from backend.models.data.stock_qa import FACT_SEPARATOR
from backend.models.guardrails import figures
from backend.models.guardrails.guardrails import snapshot_values

TITLES = {"performance": "Recent performance", "overbought": "Momentum and RSI", "volatility": "Volatility",
          "drawdown": "Drawdown", "risk": "Week-ahead risk", "buy": "Should I buy?", FACTS: "Company facts"}
# the snapshot's own names, in the order a reader wants them, with the units the advisory formats give them
FIGURES = (("close", "Last close", True), ("return_5d", "5 days", False), ("return_20d", "20 days", False),
           ("volatility_20d", "20-day volatility", False), ("rsi_14", "RSI (14)", False),
           ("drawdown_60d", f"{TERM_MEANINGS['Max drawdown'].removeprefix('the ').capitalize()} in 60 days", False),
           ("outlook", "Week ahead", False))
# ALFA's price files, which it trained on, keep the close as traded: beside our own analysis's RSI and returns, on prices
# adjusted for dividends (report.ADJUSTED), INFY's read 53 and 55.5 on one screen with nothing to say why
KEY_FIGURES = "**Key figures**, on closes not adjusted for dividends"
# The buy intent is asked with "sell" and "hold" too, and "should I sell reliance" was headed "Should I buy?"
_TRADES = re.compile(r"\b(buy|sell|hold)(?:ing|s)?\b(?! up)", re.I)
# Why core._written refused a draft of ALFA's, as a reader would say it
HELD_BACK = {"unsupported figure": "it gave a figure the data does not hold",
             "names another share": "it named another share",
             "wrong trend": "it misread where the price stands against its averages",
             "low confidence": "it was not sure enough of it"}
SITES = (("stackexchange.com", "Stack Exchange"), ("wikipedia.org", "Wikipedia"), ("sec.gov", "SEC filing"),
         ("federalreserve.gov", "Federal Reserve"), ("rbi.org.in", "RBI"), ("sebi.gov.in", "SEBI"),
         ("economictimes.indiatimes.com", "Economic Times"), ("livemint.com", "Mint"),
         ("ndtvprofit.com", "NDTV Profit"), ("thehindubusinessline.com", "Hindu BusinessLine"))
LESSON = "lesson:"
# No facts passage of the 2,752 gave more than three. With the lines a reply adds to its passage, 3 of 373 served replies
# gave five or six, so four keeps a reply short and seldom cuts one
MEANINGS = 4
# longest first, so "Free cash flow yield" is read as itself and not as "Free cash flow"
_TERM = re.compile(r"\b(" + "|".join(sorted(map(re.escape, TERM_MEANINGS), key=len, reverse=True)) + r")\b", re.I)


def readable_day(day):
    """"2026-10-06" as "6 Oct 2026"; anything else unchanged."""
    try:
        return date.fromisoformat(str(day)[:10]).strftime("%-d %b %Y")
    except ValueError:
        return str(day)


def figures_line(evidence, money):
    shown = snapshot_values(evidence)
    parts = [f"{label} {money}{_grouped(shown[name])}" if priced else f"{label} {shown[name]}"
             for name, label, priced in FIGURES if name in shown]
    return f"{KEY_FIGURES} · " + " · ".join(parts) if parts else ""


def _grouped(value):
    """"1217.00" as "1,217.00": display only, the evidence keeps the figure as the model saw it."""
    try:
        return f"{float(value):,.2f}"
    except ValueError:
        return value


def heading(ticker, names, title):
    called = f"{names[ticker]} ({ticker})" if ticker in names else ticker
    return (f"### {called}" + (f" · {title}" if title else "")) if ticker else f"### {title}"


def advisory(turn, names, sources, money):
    """A price-snapshot answer, or one per question asked together: what it is about, the answers, then the figures
    they rest on and the sources of any that read the news."""
    title = " · ".join(dict.fromkeys(_title(label, turn.question) for label in turn.intent.split(TOGETHER)
                                     if label in TITLES))
    held = held_back(turn)
    lines = [heading(turn.ticker, names, title), "", _sentence_case(turn.served, turn.ticker),
             *(["", held] if held else [])]
    prices = re.search(r"prices (.+?) ;; ", turn.evidence)
    figures = figures_line(prices.group(1) if prices else turn.evidence, money)
    return "\n".join(lines + (["", figures] if figures else []) + _cited(turn.evidence, sources))


def _title(label, question):
    """An intent's heading; the buy intent's names the trades the question asks about."""
    trades = dict.fromkeys(trade.lower() for trade in _TRADES.findall(question))
    return f"Should I {' or '.join(trades)}?" if label == "buy" and trades else TITLES[label]


def held_back(turn):
    """That ALFA's draft was held back and why, where our code's sentence speaks for it: for the turn, or for each part
    of questions asked together; empty where ALFA's own words were served."""
    if TOGETHER not in turn.intent:
        return (f"ALFA's draft was held back: {HELD_BACK[turn.because.split(':')[0]]}. This is our code's sentence."
                if turn.phrased_by == BUILT and turn.because else "")
    parts = [part.partition(": ")[::2] for part in turn.because.split("; ")]
    return " ".join(f"ALFA's draft for “{_title(label, turn.question)}” was held back: "
                    f"{HELD_BACK[why.split(':')[0]]}. That part is our code's sentence."
                    for label, why in parts if why.split(":")[0] in HELD_BACK)


def sourced(turn, names, sources, money, title):
    """An answer read out of passages: the answer, the prices if they were part of it, then each source."""
    lines = [heading(turn.ticker, names, title), "", turn.served]
    prices = re.search(r"prices (.+?) ;; ", turn.evidence)
    if prices:
        lines += ["", figures_line(prices.group(1), money)]
    return "\n".join(lines + _cited(turn.evidence, sources))


def from_facts(turn, names):
    """An answer from a share's dated facts: the answer, what its figures mean in TERM_MEANINGS' words and never the
    model's, then every fact it was given and where they came from."""
    facts = turn.evidence.split(FACT_SEPARATOR)
    meant = meanings(facts, turn.served)
    # our code's reply is in sentences already: cased again, "Apple Inc. designs" read "Apple Inc. Designs"
    served = turn.served if turn.phrased_by == BUILT else _sentence_case(turn.served)
    return "\n".join([heading(turn.ticker, names, TITLES[FACTS]), "", served, "",
                      *(["**What these figures mean**", *meant, ""] if meant else []), "**Facts used**",
                      *(f"- {line}" for line in facts), "",
                      f"Source · Yahoo Finance via StockIntel, as of {readable_day(turn.as_of)}"])


def meanings(facts, served):
    """TERM_MEANINGS' entry for each term the facts after the stock line give a figure under, at most MEANINGS, those
    the `served` reply states first."""
    spelt = {term.lower(): term for term in TERM_MEANINGS}
    terms = dict.fromkeys(spelt[term.lower()] for fact in facts[1:] if figures(fact) for term in _TERM.findall(fact))
    stated = {spelt[term.lower()] for term in _TERM.findall(served)}
    return [f"- {term}: {TERM_MEANINGS[term]}." for term in sorted(terms, key=lambda term: term not in stated)[:MEANINGS]]


def _cited(evidence, sources):
    cited = [f"- {link}" for link in (_link(key, day, sources) for key, day in
                                       re.findall(r"document (.+?) ; published (\S+)", evidence)) if link]
    return ["", "**Sources**", *dict.fromkeys(cited)] if cited else []


def present(turn, names, sources):
    """The markdown for a turn that spoke; a refusal is shown as the plain sentence it is, cased as one."""
    if not turn.spoke:
        return _sentence_case(turn.served, turn.ticker)
    money = phrase.currency(turn.ticker)
    if turn.intent.startswith("reference"):
        lesson = re.search(rf"document {LESSON}(.+?) ; published", turn.evidence)
        title = ("What happened lately" if "recent" in turn.intent else lesson.group(1) if lesson
                 else "" if turn.ticker else "From the reference library")
        return sourced(turn, names, sources, money, title)
    if turn.intent == FACTS:
        return from_facts(turn, names)
    return advisory(turn, names, sources, money)


def _sentence_case(text, ticker=""):
    """ALFA's generator writes in lower case, a symbol spaced out ("i do not have a signal for tcs . ns ."); a reader
    expects sentences, "I", and the symbol as it is written. A paragraph starts a sentence: "(ET, 7 Oct 2026)" can end
    the one before it."""
    text = re.sub(r"\s+([.,;:?])", r"\1", text)
    if ticker:
        spelt = r"\s*".join(map(re.escape, re.findall(r"\w+|\W", ticker)))
        text = re.sub(rf"(?<![\w.]){spelt}(?!\w)", ticker, text, flags=re.I)
    text = re.sub(r"(^|[.?!][)\"'”’]*\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text, flags=re.M)
    text = re.sub(r"\bi\b(?!\.\w)", "I", text)
    return re.sub(r"\brsi\b", "RSI", text)


def _link(key, day, sources):
    if key.startswith(LESSON):
        return f"StockIntel lessons · {key[len(LESSON):]}"
    url = key if key.startswith("http") else sources.get(key, "")
    if not url:
        return ""
    site = next((name for host, name in SITES if host in url), url.split("/")[2])
    return f"[{site}, {readable_day(day)}]({url})" if day not in ("", "undated") else f"[{site}]({url})"


def source_map(chunks, manifest):
    """{document key: source URL} for the documents an index holds, read once from the data manifest."""
    wanted = {chunk.document for chunk in chunks}
    found = {}
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            key = entry["sha256"][:16]
            if key in wanted:
                found[key] = entry["url"]
    return found
