"""Turns extracted text into prose paragraphs.

Most of what extraction produces is not prose: numeric table rows, page numbers, navigation
words, exhibit indexes. Training on it teaches the model formatting instead of finance, so it is
dropped here rather than left for the tokenizer to absorb.
"""

import re
import unicodedata

MINIMUM_LINE_CHARACTERS = 25
# Provisional: one full sentence. A news headline plus its feed summary lands just above this, and
# a table caption just below. Confirm against the smoke-tier output before the bulk run.
MINIMUM_PARAGRAPH_CHARACTERS = 100
# A prose line is mostly letters and spaces. A table row is mostly digits, dollar signs, and
# parentheses. This ratio separates them and is the single most effective filter here.
MINIMUM_LETTER_RATIO = 0.65
# The corpus is English. SEBI publishes bilingual circulars whose Hindi half extracts as mangled
# glyphs, so the Hindi lines are dropped and the English ones on the same page are kept.
MINIMUM_LATIN_RATIO = 0.5

_WHITESPACE = re.compile(r"[ \t   ]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_REPEATED_BLANK = re.compile(r"\n{3,}")


def _letter_ratio(line):
    letters = sum(1 for character in line if character.isalpha() or character.isspace())
    return letters / len(line) if line else 0.0


def _latin_ratio(line):
    letters = [character for character in line if character.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for character in letters if character.isascii()) / len(letters)


def _is_prose(line):
    if len(line) < MINIMUM_LINE_CHARACTERS:
        return False
    if _letter_ratio(line) < MINIMUM_LETTER_RATIO:
        return False
    if _latin_ratio(line) < MINIMUM_LATIN_RATIO:
        return False
    # An all-caps line of any length is a heading or a legend, not a sentence.
    return not line.isupper()


def normalize(text):
    """Whitespace and unicode only. For text that is already prose, such as news headlines."""
    text = unicodedata.normalize("NFKC", text)
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return _REPEATED_BLANK.sub("\n\n", "\n".join(lines)).strip()


def clean(text):
    """Returns paragraphs separated by blank lines, or '' if nothing survived."""
    text = unicodedata.normalize("NFKC", text)
    # PDFs break words across lines. Rejoining before line filtering keeps the word intact.
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    paragraphs, current = [], []
    for raw_line in text.splitlines():
        line = _WHITESPACE.sub(" ", raw_line).strip()
        if _is_prose(line):
            current.append(line)
            continue
        if current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    kept = [
        paragraph
        for paragraph in paragraphs
        if len(paragraph) >= MINIMUM_PARAGRAPH_CHARACTERS
    ]
    return _REPEATED_BLANK.sub("\n\n", "\n\n".join(kept))
