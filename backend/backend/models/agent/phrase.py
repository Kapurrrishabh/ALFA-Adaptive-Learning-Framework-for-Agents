"""The language model as the agent's last step: what we already decided to say, said plainly.

ALFA's generator, or the passage the reference path chose, decides what the answer is; the language model
only rewords it. Its words are served when they keep every figure the draft states, add no figure the
evidence does not, and claim nothing the source does not (`guardrails.added_claims`). Otherwise the draft is
served and the turn says why, so a reworded answer is never the only version of a fact.

This is the setup that passed 22 of 24 analyst notes (advisory/analyst.py). Composing from raw evidence was
the setup that failed there: a small model handed facts writes fluent misattributions.
"""
import re
from collections import namedtuple

from backend.models.guardrails import added_claims, figures, unsupported_figures

# `text` is empty exactly when the language model's words are not served; `because` then says why
Phrased = namedtuple("Phrased", "text by because")
NOT_ANSWERED = "the language model read the passages and found no answer in them"

ANSWER_SYSTEM = """Rewrite the ANSWER for an Indian retail investor in two or three short, plain sentences. \
Keep every number exactly as written. Keep every judgement as it is. Add nothing: no new facts, no reasons, \
no advice, no predictions. Never say a price will rise or fall. No lists or headings."""

PASSAGE_SYSTEM = """Answer the QUESTION for an Indian retail investor in two or three short, plain sentences, \
using only the PASSAGES. Report what the passages say happened, and what they say it happened after or \
alongside, naming the source and date as the passage gives them; do not explain beyond what they say. Keep \
every number exactly as written. Never refer to passages by number. No advice, no predictions, and never say a \
price will rise or fall. If the passages do not answer the question, reply with the single word NONE and \
nothing else."""

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
# A reply that says the passages do not answer, wherever it says it: "...founded in 1981. The passage does
# not provide a specific reason. None." was half an answer and half a refusal, and only the refusal was true.
_DECLINES = re.compile(r"\bnone\.?\s*$|\b(does|do|did) not (provide|say|mention|explain|give|contain|answer)|"
                       r"\bno (information|mention|reason|details?)\b|\bnot (mentioned|provided|specified)\b", re.I)


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
    reasons = _checked(text, f"{evidence} {draft}", money)
    dropped = sorted(kept - set(figures(_plain(text))))
    if dropped:
        reasons.append(f"it leaves out {', '.join(dropped)}")
    return Phrased("", "", "; ".join(reasons)) if reasons else Phrased(text, by, "")


def from_passages(question, chunks, writer, money=None):
    """An answer read out of the retrieved passages; the first passage is quoted instead when a check fails."""
    passages = "\n".join(f"- {chunk.text}" for chunk in chunks)
    reply = _ask(writer, PASSAGE_SYSTEM, f"QUESTION: {question}\nPASSAGES:\n{passages}")
    if isinstance(reply, Phrased):
        return reply
    text, by = reply
    if _DECLINES.search(text):
        return Phrased("", "", NOT_ANSWERED)
    reasons = _checked(text, passages, money)
    return Phrased("", "", "; ".join(reasons)) if reasons else Phrased(text, by, "")


def _ask(writer, system, content):
    """(text, model) from the writer, or a Phrased saying why it could not be asked."""
    from backend.database.sources.provider import DataUnavailable
    from backend.models.external.llm import LLMUnavailable
    try:
        reply = writer.create(system, [{"role": "user", "content": content}])
    except (LLMUnavailable, DataUnavailable) as error:
        return Phrased("", "", f"the language model is unavailable: {error}")
    return writer.text(reply).strip(), reply.model


def _checked(text, source, money):
    reasons = ["it breaks into lines, as a list or verse does"] if "\n" in text.strip() else []
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
