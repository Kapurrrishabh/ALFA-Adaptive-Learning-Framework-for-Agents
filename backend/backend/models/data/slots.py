"""A facts answer in slot form: the share's name, its ticker and each figure become markers that code fills back.

The model still chooses what to say and how, but every exact string comes from the passage, so a garbled name or a
mis-copied digit cannot reach a reader.
"""

import re

from backend.models.data.stock_qa import FACT_SEPARATOR
from backend.models.guardrails import guardrails

# Single vocabulary pieces that no advisory or stock Q&A target holds. Final sigma is left out: lower() makes it from Σ.
NAME = "ι"
TICKER = "χ"
# Then a space and the figure's place among the passage's figures, from 1: "κ 3". WordPiece splits the number off the
# marker and decodes the pair with that space, so model output and this form are one spelling. A sign before it is the
# figure's where the decoder would bind it to digits, not after a number, so a date's "κ 2 - κ 3" keeps its dash.
FIGURE = "κ"
_MARKER = re.compile(rf"{NAME}|{TICKER}|(?:(?<![\d%])(?<![\d%] )([+-]) ?)?{FIGURE}(?: (\d+))?")
# Marked, every advisory source and 87% of facts passages outrun one window, and none needs more than two.
WINDOWS = 2


def mark_source(passage, name, ticker, *other_names):
    """`passage` as the model reads it: the share's names and ticker as their markers, and each figure kept but led by
    the marker an answer copies for it. The figure stays because the answer's words, "rose" or "above both", read it."""
    held = _magnitudes(passage)
    return _each_figure(_named(passage, name, ticker, other_names), held,
                        lambda magnitude, place: f"{FIGURE} {place} {magnitude}")


def windows(marked, count_tokens, room):
    """`marked` as the passages of at most WINDOWS encoder windows of `room` tokens each, each field in the first window
    with room for it. Advisory evidence and facts passages both join their fields with FACT_SEPARATOR, and each figure
    and its marker sit inside one field, so the cut falls between fields only. A source that fits one window is one
    passage. Kept in order, a facts passage of 32, 70 and 27-token fields needed a third window of 93."""
    windows = [[] for _ in range(WINDOWS)]
    for field in marked.split(FACT_SEPARATOR):
        fits = next((w for w in windows if count_tokens(FACT_SEPARATOR.join(w + [field])) <= room), None)
        if fits is None:
            raise ValueError(f"a field of {count_tokens(field)} tokens fits none of {WINDOWS} windows of {room} "
                             f"filled to {[count_tokens(FACT_SEPARATOR.join(w)) for w in windows]}; shorten the "
                             f"source, as a cut field would drop a figure: {marked[:80]!r}")
        fits.append(field)
    return [FACT_SEPARATOR.join(fields) for fields in windows if fields]


def to_slots(answer, passage, name, ticker, *other_names):
    """`answer` with its name, ticker and figures as markers, or None when it states a figure `passage` lacks.
    `other_names` are other spellings of the name, such as the row's short company name: each fills back as `name`."""
    held = _magnitudes(passage)
    return _each_figure(_named(answer, name, ticker, other_names), held, lambda _, place: f"{FIGURE} {place}")


def from_slots(text, passage, name, ticker):
    """`text` with every marker replaced by the exact string it stands for."""
    _require(name, ticker, ())
    held = _magnitudes(passage)
    names = {NAME: name, TICKER: ticker}

    def filled(marker):
        if marker.group() in names:
            return names[marker.group()]
        place = int(marker.group(2) or 0)
        if not 1 <= place <= len(held):
            raise ValueError(f"figure marker {marker.group()!r} names no figure of the {len(held)} in its passage "
                             f"{held}, so the answer cannot be filled; refuse it")
        return (marker.group(1) or "") + held[place - 1]

    return _MARKER.sub(filled, text)


def _named(text, name, ticker, other_names):
    """`text` with the share's names and ticker as their markers, longest first so none is half replaced, and whole
    words only: Mastercard's MA is not the start of MACD, nor Visa's V the end of EV."""
    _require(name, ticker, other_names)
    if _MARKER.search(text):
        raise ValueError(f"the text already holds a slot marker, so its slot form would not read back: {text[:80]!r}")
    for whole, marker in sorted([(n, NAME) for n in (name, *other_names)] + [(ticker, TICKER)],
                                key=lambda pair: len(pair[0]), reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(whole)}(?!\w)", marker, text)
    return text


def _each_figure(text, held, write):
    """`text` with each figure's magnitude replaced by `write(magnitude, its place in held)`, its sign left as text;
    None when `held` lacks one."""
    slotted, cursor = [], 0
    for figure in guardrails.figures(text):
        magnitude = figure.lstrip("+-")
        if magnitude not in held:
            return None
        start = text.index(figure, cursor) + len(figure) - len(magnitude)
        slotted.append(f"{text[cursor:start]}{write(magnitude, held.index(magnitude) + 1)}")
        cursor = start + len(magnitude)
    return "".join(slotted) + text[cursor:]


def _magnitudes(passage):
    """The passage's figures, signs dropped. Answers write "rose 14.1%" for "+14.1%", and the tokenizer splits a sign off
    anyway: matching signed figures exactly dropped 561 of the 2,752 written passage answers, this drops none."""
    return [figure.lstrip("+-") for figure in guardrails.figures(passage)]


def _require(name, ticker, other_names):
    if not name or not ticker or not all(other_names) or ticker in (name, *other_names):
        raise ValueError(f"slots need the share's name and a ticker that is not one of its names, got name={name!r} "
                         f"ticker={ticker!r} other names={other_names!r}")
