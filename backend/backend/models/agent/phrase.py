"""The language model as the agent's last step: what we already decided to say, said plainly.

The passages the reference path chose decide what the answer is; the language model only rewords them. Its
words are served when they add no figure the evidence does not and claim nothing the source does not
(`guardrails.added_claims`, and an entailment model reading each sentence). Otherwise our plain answer is
served and the turn says why, so a reworded answer is never the only version of a fact. The entailment model does
not tie a figure to what it measures, so `guardrails.unbound_figures` does, for names a passage holds: "TCS shares
fell 1.2%" passed it, 1.2% being the Nifty's.

This is the setup that passed 22 of 24 analyst notes (advisory/analyst.py). Composing from raw evidence was
the setup that failed there: a small model handed facts writes fluent misattributions.
"""
import re
from collections import namedtuple
from datetime import date

from backend.advisory.report import CURRENCY
from backend.advisory.sentiment.feeds.sentiment import NEGATORS
from backend.knowledge_base import lessons
from backend.knowledge_base.retrieval.chunk import sentences
from backend.models.agent.finance import market_of
from backend.models.data import advisory
from backend.models.guardrails import added_claims, unbound_figures, unsupported_figures
from backend.models.guardrails.guardrails import BRACKETED, TREND_FIGURES, days, undated

# `text` is empty exactly when the language model's words are not served; `because` then says why. `asked_writer` is
# whether the request reached the language model, which is what a user is charged for
Phrased = namedtuple("Phrased", "text by because asked_writer")
# names our code as the author of words served in place of a language model's or ALFA's
BUILT = "our code"
NOT_ANSWERED = "the language model read the passages and found no answer in them"
HEADLINES_NOT_WHY = "These are headlines, and they do not prove why the market moved."

# No cap on sentence length here: with one, it answered "how is my portfolio doing" with a Treasury headline
PASSAGE_SYSTEM = """Answer the QUESTION for an Indian retail investor in two or three short sentences in everyday \
words, using only the PASSAGES. Report what the passages say happened, and what they say it happened after or \
alongside, naming the source and date as the passage gives them; do not explain beyond what they say. Do not \
answer yes or no, and do not give your own verdict on the question: report what the passages say. Keep \
every number exactly as written. Never refer to passages by number. No advice, no predictions, and never say a \
price will rise or fall. If the passages do not answer the question, reply with the single word NONE and \
nothing else."""

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
# a ticker, or a name in capitals: "RELIEANCE.NS" was written for RELIANCE.NS and passed every other check
_NAMES = re.compile(r"\b[A-Z0-9&-]{2,}\.(?:NS|BO)\b|\b[A-Z]{5,}\b")
# A reply that says the passages do not answer, wherever it says it: "...founded in 1981. The passage does
# not provide a specific reason. None." was half an answer and half a refusal, and only the refusal was true.
_DECLINES = re.compile(r"\bnone\.?\s*$|\b(does|do|did) not (provide|say|mention|explain|give|contain|answer)|"
                       r"\bno (information|mention|reason|details?)\b|\bnot (\w+ly )?(mentioned|provided|specified)\b", re.I)
# a yes or no in place of the report the prompt asks for: "NO" for "are it stocks falling"
_BARE_VERDICT = re.compile(r"^\W*(yes|no)\s*([,.;:!]|$)", re.I)
# Provisional, on sentence.entailment's scale. On 298 replies, 207 by the writer and 91 written to test it, every cut
# from 0.2 to 0.8 was right on 245 to 250 and 0.5 on 249: too narrow a spread to fit.
ENTAILED = 0.5


def currency(ticker):
    """The symbol of the currency a price for `ticker` is in, which a rewrite may not swap for the other: none for an
    index, whose level is in points, or for no ticker."""
    return "" if not ticker or ticker.startswith("^") else CURRENCY[market_of(ticker).currency]


def from_passages(question, chunks, writer, entails, money=None):
    """An answer read out of the retrieved passages, or empty with the reason when a check fails. `entails` is
    `sentence.entailment`, handed in so that this module loads no model."""
    passages = "\n".join(f"- {chunk.text}" for chunk in chunks)
    reply = _ask(writer, PASSAGE_SYSTEM, f"QUESTION: {question}\nPASSAGES:\n{passages}")
    if isinstance(reply, Phrased):
        return reply
    text, by = reply
    if _DECLINES.search(text):
        return Phrased("", "", NOT_ANSWERED, True)
    reasons = (["it answers yes or no"] if _BARE_VERDICT.match(text) else []) + _denials(question, text) + \
        _unentailed(text, chunks, entails) + _checked(text, passages, money, question) + \
        [f"it ties {figure} to {name}, which no passage does" for figure, name in
         unbound_figures(text, [chunk.text for chunk in chunks])]
    return Phrased("", "", "; ".join(reasons), True) if reasons else Phrased(text, by, "", True)


def price_sentence(ticker, prices, money):
    """The snapshot as one line a reader and the language model can both quote from."""
    from backend.models.agent.present import FIGURES, readable_day
    _, shown, day = prices
    parts = [f"{label.lower()} {money if priced else ''}{shown[name]}" for name, label, priced in FIGURES if name in shown]
    return f"{ticker} as of {readable_day(day)}: " + ", ".join(parts)


def recent(question, subject, prices, chunks, writer, entails, money):
    """What happened lately: our sentences on the price move and on when the news is from, then the language model's
    account of the passages.

    The model is not shown the prices. Given both, it wrote "down 0.5% from the previous close of ₹704.80"
    for HDFC Bank: the 5-day change pinned to a quote that said the shares were up, every figure in the
    sources and the claim wrong. So it reads the dated passages only, and the prices are our own words.
    """
    told = from_passages(question, chunks, writer, entails, money)
    lead = f"{price_move(subject, prices)} {news_dated(chunks, prices)}"
    return told._replace(text=f"{lead} {told.text}".strip()) if told.text else told


def price_move(subject, prices):
    """One sentence on where the price stands, from the snapshot; empty without one."""
    from backend.models.agent.present import readable_day
    if not prices:
        return ""
    _, shown, day = prices
    return (f"{subject.ticker} closed at {currency(subject.ticker)}{shown['close']} on {readable_day(day)}: "
            f"{shown['return_5d']} over 5 days and {shown['return_20d']} over 20 days.")


def news_dated(chunks, prices):
    """Our sentence on when the news is from and how long before the close in `prices`, empty without them: a 40-day-old
    preview of results already out was told in the present tense."""
    from backend.models.agent.present import readable_day
    if not prices:
        return ""
    first, last = min(chunk.day for chunk in chunks), latest(chunks).day
    if first == last:
        return f"The news that follows is from {readable_day(last)}, {_age(last, prices)}."
    return f"The news that follows is from {readable_day(first)} to {readable_day(last)}; the newest is from " \
           f"{_age(last, prices)}."


def _age(day, prices):
    """"40 days before that close", "the day of that close" or "1 day after that close", for news of `day` beside the
    close in `prices`: a weekend's news comes after Friday's close."""
    gap = (date.fromisoformat(str(prices[2])[:10]) - date.fromisoformat(day)).days
    count = f"{abs(gap)} day{'s' if abs(gap) != 1 else ''}"
    return f"{count} before that close" if gap > 0 else f"{count} after that close" if gap < 0 else "the day of that close"


def answer_form(intent, phrasing, ticker, shown):
    """The answer form ALFA trained on for a price intent, filled by our code with the snapshot's figures as shown, and
    its trend and RSI words read from those same figures, as the trend guard reads them."""
    facts = {name: float(shown[name]) for name in (*TREND_FIGURES, "rsi_14")}
    return advisory.row(intent, ticker, facts, shown, phrasing)[1]


def recent_plain(subject, prices, chunks, unread=False):
    """The same answer without a language model: the price sentence, then the newest report or the lack of one. With
    no report it says when SEC filings were `unread`, and otherwise claims nothing of filings: not every question reads
    them."""
    said = [price_move(subject, prices)] if prices else []
    said.append(_reported(latest(chunks), prices) if chunks else
                "I found no news about it in the archive and could not read its SEC filings just now, so I cannot "
                "say why." if unread else "I found no news about it in the archive, so I cannot say why.")
    return " ".join(said)


def _reported(chunk, prices):
    """The newest report; beside prices, in our words when it is from and how long before their close."""
    from backend.models.agent.present import readable_day
    if not prices:
        return f"The latest report: {chunk.text}"
    cited, when = _citation(chunk), f"{readable_day(chunk.day)}, {_age(chunk.day, prices)}"
    return (f"On {when}, {cited.rpartition(', ')[0]} wrote: {chunk.text[len(cited) + 2:]}" if cited
            else f"The latest report, from {when}: {chunk.text}")


def latest(chunks):
    """The newest passage, picked by date: they arrive ranked by relevance, with filings ahead of headlines."""
    return max(chunks, key=lambda chunk: chunk.day)


def _ask(writer, system, content):
    """(text, model) from the writer, or a Phrased saying why it could not be asked."""
    from backend.database.sources.provider import DataUnavailable
    from backend.models.external.llm import LLMUnavailable
    try:
        reply = writer.create(system, [{"role": "user", "content": content}])
    except (LLMUnavailable, DataUnavailable) as error:
        return Phrased("", "", f"the language model is unavailable: {error}", False)
    return writer.text(reply).strip(), reply.model


def _unentailed(text, chunks, entails):
    """Why `text` says something no source entails, sentence by sentence, or nothing when every sentence is entailed. A
    sentence naming its sources must be entailed by theirs: "(Economic Times, 28 Oct 2026)" passed on a Mint report."""
    # Only a bracket naming the passages' own sources is left out: kept on its sentence, "(Mint, 6 Oct 2026)" failed 3 of
    # 21 faithful replies. Any other is read without its brackets, which the model takes as an aside: "The Nifty fell
    # (over 500 points)" scored 0.91 for passages where the Sensex fell by them, and 0.07 written without the brackets.
    sources = [chunk.text for chunk in chunks]
    cited = {}
    for chunk in chunks:
        cited.setdefault(_citation(chunk), []).append(chunk.text)
    cited.pop("", None)
    kept = lambda bracket: "" if _named(bracket.group(1), cited) else f" {bracket.group(1)}"
    claims = []
    for part in sentences(text):
        claim = BRACKETED.sub(kept, part).strip()
        named = [source for bracket in BRACKETED.findall(part) for source in _named(bracket, cited)]
        if re.search(r"\w", claim):
            claims.append((claim, named))
        elif claims:
            claims[-1][1].extend(named)
    if not claims:
        return ["it states nothing a passage could support"]
    # the passages' sentences first: short, they entail most true claims, and whole passages took three times as long
    for whole in (False, True):
        read = [(claim, named, _premises([said for source in named for said in cited[source]] or sources, whole))
                for claim, named in claims]
        scores = iter(entails([(premise, claim) for claim, _, premises in read for premise in premises]))
        claims = [(claim, named) for claim, named, premises in read if max(next(scores) for _ in premises) < ENTAILED]
    return [f"{'; '.join(named)} does not say “{claim}”" if named else f"no passage says “{claim}”"
            for claim, named in claims]


def _named(bracket, cited):
    """The sources a bracket names, when each of its parts is one: "Mint, 6 Oct 2026; ET, 7 Oct 2026"."""
    parts = [part.strip() for part in bracket.split(";")]
    return parts if all(part in cited for part in parts) else []


def _premises(texts, whole):
    """The passages whole and joined, or their sentences, each once."""
    return list(dict.fromkeys(texts + [" ".join(texts)] if whole else
                              [part for said in texts for part in sentences(said)]))


def _denials(question, text):
    """Why `text` gives its own verdict: a sentence that is the question's words and a negation, and nothing else. "The
    market is not crashing." passed the entailment model, which fails on negation (Geiger et al., BlackBoxNLP 2020)."""
    asked = lessons.stems(question)
    denials = []
    for part in sentences(BRACKETED.sub("", text)):
        said = re.sub(r"n['’]t\b", " not", part)
        if NEGATORS & set(re.findall(r"[a-z]+", said.lower())) and asked <= lessons.stems(said) <= asked | NEGATORS:
            denials.append(f"it gives its own verdict: “{part}”")
    return denials


def _citation(chunk):
    """"Mint, 6 Oct 2026" for a headline, the source and its own date it opens with (`NewsWire`); else empty."""
    from backend.models.agent.present import readable_day
    head, colon, _ = chunk.text.partition(": ")
    return head if colon and head.endswith(f", {readable_day(chunk.day)}") else ""


def _checked(text, source, money, asked=""):
    from backend.models.agent.present import readable_day
    reasons = ["it breaks into lines, as a list or verse does"] if "\n" in text.strip() else []
    known = f"{source} {asked}".lower()
    reasons += [f"it names {name}, which the source does not" for name in sorted(set(_NAMES.findall(text)))
                if name.lower() not in known]
    reasons += [f"it gives the date {readable_day(day)}, which the source does not" for day in days(text)
                if day not in days(source)]
    reasons += [f"figure {figure} is not in the evidence"
               for figure in unsupported_figures(_plain(undated(text)), _plain(source))]
    reasons += added_claims(text, source)
    other = {"₹": "$", "$": "₹"}.get(money)
    if other and other in text and other not in source:
        reasons.append(f"it writes {other} for a price in {money}")
    return reasons


def _plain(text):
    """Figures without thousands separators, so "2,118.20" and "2118.20" are the same figure."""
    return _THOUSANDS.sub("", text)
