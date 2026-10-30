"""The language model as the agent's last step: what we already decided to say, said plainly.

ALFA's generator, or the passage the reference path chose, decides what the answer is; the language model
only rewords it. Its words are served when they keep every figure the draft states, add no figure the
evidence does not, and claim nothing the source does not (`guardrails.added_claims`, and for passages an
entailment model reading each sentence). Otherwise the draft is served and the turn says why, so a reworded
answer is never the only version of a fact.

This is the setup that passed 22 of 24 analyst notes (advisory/analyst.py). Composing from raw evidence was
the setup that failed there: a small model handed facts writes fluent misattributions.
"""
import re
from collections import namedtuple

from backend.knowledge_base.retrieval.chunk import sentences
from backend.models.guardrails import added_claims, figures, unsupported_figures

# `text` is empty exactly when the language model's words are not served; `because` then says why
Phrased = namedtuple("Phrased", "text by because")
NOT_ANSWERED = "the language model read the passages and found no answer in them"

# No glosses of terms: asked to explain volatility or RSI, the writer added sizes the draft never gives ("has moved a
# lot", "not very risky") and HHEM fell from 0.89 to as low as 0.15 on 16 advisory answers.
ANSWER_SYSTEM = """Rewrite the ANSWER for an Indian retail investor in two or three short sentences, in everyday \
words. Keep each sentence under 15 words. Keep every number exactly as written, with its sign. Keep every judgement \
as it is, and do not call a figure high, low, a lot or a little. Add nothing else: no new facts, no reasons, no \
advice, no predictions. Never say a price will rise or fall. No lists or headings."""

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
# Provisional, on sentence.entailment's scale. On 79 replies, 15 from the user-question eval and 64 written for this, the
# gate was as accurate anywhere from 0.5 to 0.8; more labelled replies from the next eval run would place it.
ENTAILED = 0.5
# a size the draft never gives: "Infosys is quite volatile" for a draft that gave the figure alone
_SIZES = re.compile(r"\b(a lot|quite|very|highly|extremely|fairly|significant(ly)?|moderate(ly)?|sharp(ly)?|little)\b",
                    re.I)


def currency(ticker):
    """The currency a price for `ticker` is in, which a rewrite may not swap for the other."""
    return "₹" if ticker.upper().endswith(".NS") else "$"


def rewrite(question, draft, evidence, writer, money):
    """ALFA's answer reworded, keeping its figures; `draft` is served instead when a check fails."""
    reply = _ask(writer, ANSWER_SYSTEM, f"QUESTION: {question}\nANSWER: {draft}")
    if isinstance(reply, Phrased):
        return reply
    text, by = reply
    kept = set(figures(_plain(draft)))
    reasons = _checked(text, f"{evidence} {draft}", money, question)
    dropped = sorted(kept - set(figures(_plain(text))))
    if dropped:
        reasons.append(f"it leaves out {', '.join(dropped)}")
    sized = {m.group(0).lower() for m in _SIZES.finditer(text)} - {m.group(0).lower() for m in _SIZES.finditer(draft)}
    if sized:
        reasons.append(f"it sizes a figure the draft does not: “{sorted(sized)[0]}”")
    return Phrased("", "", "; ".join(reasons)) if reasons else Phrased(text, by, "")


def from_passages(question, chunks, writer, entails, money=None):
    """An answer read out of the retrieved passages; the first passage is quoted instead when a check fails. `entails`
    is `sentence.entailment`, handed in so that this module loads no model."""
    passages = "\n".join(f"- {chunk.text}" for chunk in chunks)
    reply = _ask(writer, PASSAGE_SYSTEM, f"QUESTION: {question}\nPASSAGES:\n{passages}")
    if isinstance(reply, Phrased):
        return reply
    text, by = reply
    if _DECLINES.search(text):
        return Phrased("", "", NOT_ANSWERED)
    reasons = (["it answers yes or no"] if _BARE_VERDICT.match(text) else []) + \
        _unentailed(text, [chunk.text for chunk in chunks], entails) + _checked(text, passages, money, question)
    return Phrased("", "", "; ".join(reasons)) if reasons else Phrased(text, by, "")


def price_sentence(ticker, prices, money):
    """The snapshot as one line a reader and the language model can both quote from."""
    from backend.models.agent.present import FIGURES, readable_day
    _, shown, day = prices
    parts = [f"{label.lower()} {money if priced else ''}{shown[name]}" for name, label, priced in FIGURES if name in shown]
    return f"{ticker} as of {readable_day(day)}: " + ", ".join(parts)


def recent(question, subject, prices, chunks, writer, entails, money):
    """What happened lately: our sentence on the price move, then the language model's account of the passages.

    The model is not shown the prices. Given both, it wrote "down 0.5% from the previous close of ₹704.80"
    for HDFC Bank: the 5-day change pinned to a quote that said the shares were up, every figure in the
    sources and the claim wrong. So it reads the dated passages only, and the prices are our own words.
    """
    told = from_passages(question, chunks, writer, entails, money)
    lead = price_move(subject, prices)
    return told._replace(text=f"{lead} {told.text}".strip()) if told.text else told


def price_move(subject, prices):
    """One sentence on where the price stands, from the snapshot; empty without one."""
    from backend.models.agent.present import readable_day
    if not prices:
        return ""
    _, shown, day = prices
    return (f"{subject.ticker} closed at {currency(subject.ticker)}{shown['close']} on {readable_day(day)}: "
            f"{shown['return_5d']} over 5 days and {shown['return_20d']} over 20 days.")


def recent_plain(subject, prices, chunks, unread=False):
    """The same answer without a language model: the price sentence, then the newest report or the lack of one. With
    no report it says when SEC filings were `unread`, and otherwise claims nothing of filings: not every question reads
    them."""
    said = [price_move(subject, prices)] if prices else []
    said.append(f"The latest report: {latest(chunks).text}" if chunks else
                "I found no news about it in the archive and could not read its SEC filings just now, so I cannot "
                "say why." if unread else "I found no news about it in the archive, so I cannot say why.")
    return " ".join(said)


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
        return Phrased("", "", f"the language model is unavailable: {error}")
    return writer.text(reply).strip(), reply.model


def _unentailed(text, sources, entails):
    """Why `text` says something no source entails, sentence by sentence, or nothing when every sentence is entailed."""
    claims = []
    for part in sentences(text):
        # a source in brackets after the full stop belongs to the sentence it cites; alone it scored as low as 0.002
        if claims and part.startswith("("):
            claims[-1] += " " + part
        else:
            claims.append(part)
    # the passages' sentences first: short, they entail most true claims, and whole passages took three times as long
    for premises in ([part for source in sources for part in sentences(source)], sources + [" ".join(sources)]):
        premises = list(dict.fromkeys(premises))
        scores = entails([(premise, claim) for claim in claims for premise in premises])
        claims = [claim for at, claim in enumerate(claims)
                  if max(scores[at * len(premises):(at + 1) * len(premises)]) < ENTAILED]
    return [f"no passage says “{claim}”" for claim in claims]


def _checked(text, source, money, asked=""):
    reasons = ["it breaks into lines, as a list or verse does"] if "\n" in text.strip() else []
    known = f"{source} {asked}".lower()
    reasons += [f"it names {name}, which the source does not" for name in sorted(set(_NAMES.findall(text)))
                if name.lower() not in known]
    reasons += [f"figure {figure} is not in the evidence"
               for figure in unsupported_figures(_plain(text), _plain(source))]
    reasons += added_claims(text, source)
    other = {"₹": "$", "$": "₹"}.get(money)
    if other and other in text and other not in source:
        reasons.append(f"it writes {other} for a price in {money}")
    return reasons


def _plain(text):
    """Figures without thousands separators, so "2,118.20" and "2118.20" are the same figure."""
    return _THOUSANDS.sub("", text)
