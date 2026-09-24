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
    """
    stated = set(figures(evidence))
    return [figure for figure in figures(answer) if figure not in stated]


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
