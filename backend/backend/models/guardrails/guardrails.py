"""Check an answer against the evidence it was given, before anyone reads it.

The generator is a phraser: every figure it states is supposed to be one the evidence already states.
That claim is only worth something if it is checked, because the failure it guards against is the one
that looks fine — a fluent sentence with an invented number in it. Measured on the Stack Exchange
task, 69% of what an answer had to say was absent from its input, and the model duly made it up.

Pure string work, deliberately. It runs on whatever the model produced without re-running the model,
so it can sit in front of serving and in a test over the training set alike.
"""

import re

# Signed decimals and percentages, with thousands separators, which is every shape as_text produces.
# The separated form is tried first and needs a real group, so "5 ," keeps its comma as punctuation
# while a plain "1158.70" still matches whole instead of splitting into 115 and 8.70.
_FIGURE = re.compile(r"[+-]?\d+(?:,\d{3})+(?:\.\d+)?%?|[+-]?\d+(?:\.\d+)?%?")


def figures(text):
    """Every figure the text states, in order. The denominator for an unsupported-figure rate."""
    return _FIGURE.findall(text)


def unsupported_figures(answer, evidence):
    """Figures the answer states that the evidence does not, in the order they appear.

    Both sides are read by the same extractor, which is what makes `unsupported_figures(text, text)`
    empty for every text. Searching the evidence as a string instead did not: a figure ending a sentence
    there is followed by a full stop, and the bounded search that stopped 18.2 passing for 18.25 also
    refused 2017 against evidence reading "for 2017." -- so a quote of the evidence failed against it.
    Comparing extracted figures keeps the 18.25 case, because that is not a figure the evidence states.

    An unsigned answer figure is supported by a signed evidence one of the same magnitude, because the
    model writes its minus sign as a separate token and the extractor then reads "- 5.6%" as unsigned
    while the evidence states "-5.6%". A *signed* figure still has to match exactly, so an answer that
    flips the sign on a grounded number is still caught.
    """
    stated = set(figures(evidence))
    magnitudes = {figure.lstrip("+-") for figure in stated}
    return [figure for figure in figures(answer)
            if figure not in stated and (figure[:1] in "+-" or figure not in magnitudes)]


def is_refusal(text, refusal):
    """Opening words rather than equality, so a model that refuses and then keeps talking still counts.

    Shared because a scorer and a labeller that disagreed about what a refusal is would produce an
    abstention rate and a feedback label that cannot be compared.
    """
    return " ".join(refusal.split()[:7]) in text


def screen(answer, evidence, refusal):
    """(text to serve, unsupported figures). The refusal is returned whenever the list is non-empty.

    Both halves are returned rather than one: swapping in a refusal silently would hide exactly the
    rate this exists to measure.
    """
    unsupported = unsupported_figures(answer, evidence)
    return (refusal if unsupported else answer), unsupported


# a model that claims direction is wrong by the project's own measurements, so the draft is refused
DIRECTION = re.compile(r"\b(will|is likely to|is expected to|should|could)\s+(rise|fall|go up|go down|increase|decrease|"
                       r"climb|drop|rally|decline|gain)|\b(bullish|bearish|upside target|price target|upward trend|"
                       r"downward trend|uptrend|downtrend|positive bias|negative bias)\b", re.I)
# terms a writer may reach for that the note never uses; adding one means it added a claim
NEW_TERMS = re.compile(r"\b(granger|sharpe|sortino|rsi|macd|bollinger|fibonacci|moving average|support level|"
                       r"resistance|p/e|earnings|dividend|beta|alpha|regression|neural|deep learning|sentiment)\b", re.I)
# praise the records do not support
OVERCLAIM = re.compile(r"\b(reliab\w*|trust\w*|accurate|better than random|strong evidence|"
                       r"performing well|confident(ly)?)\b", re.I)


# a cause the source never gives: "infosys is falling because it is a technology company" passed every
# figure check, since a reason invented out of true facts states no new number
REASON = re.compile(r"\b(because|due to|as a result of|driven by|owing to|on account of|thanks to|on the back of)\b",
                    re.I)


def added_claims(text, source):
    """Why `text` says more than `source` does: a direction call, praise, or a finance term it never uses.

    Compared against the source rather than judged alone, so a rewrite may keep a call the draft made
    ("turbulent" from the week-ahead model) and still may not add one.
    """
    reasons = []
    for pattern, what in ((DIRECTION, "it claims a direction"), (OVERCLAIM, "it praises a model beyond its record"),
                          (REASON, "it gives a reason the source does not")):
        made = {m.group(0).lower() for m in pattern.finditer(text)} - {m.group(0).lower() for m in pattern.finditer(source)}
        if made:
            reasons.append(f"{what}: “{sorted(made)[0]}”")
    added = {t.lower() for t in NEW_TERMS.findall(text)} - {t.lower() for t in NEW_TERMS.findall(source)}
    if added:
        reasons.append(f"it adds terms the source does not use: {', '.join(sorted(added))}")
    return reasons
