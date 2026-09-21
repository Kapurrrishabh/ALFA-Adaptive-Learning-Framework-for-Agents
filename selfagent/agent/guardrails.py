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
    """Figures the answer states that the evidence does not, in the order they appear."""
    return [figure for figure in figures(answer) if not _states(evidence, figure)]


def _states(evidence, figure):
    # Bounded on both sides, so 18.2 does not count as supported by an evidence line reading 18.25.
    return re.search(rf"(?<![\d.]){re.escape(figure)}(?![\d.])", evidence) is not None


def screen(answer, evidence, refusal):
    """(text to serve, unsupported figures). The refusal is returned whenever the list is non-empty.

    Both halves are returned rather than one: swapping in a refusal silently would hide exactly the
    rate this exists to measure.
    """
    unsupported = unsupported_figures(answer, evidence)
    return (refusal if unsupported else answer), unsupported
