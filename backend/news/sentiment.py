"""How a headline reads, from a hand-written finance lexicon.

A lexicon and not a trained classifier, for one reason: nothing on disk labels a headline, so a
classifier here would be fitted on labels somebody invented and would report its own author's
opinion with a confidence interval attached. A lexicon states the opinion openly and can be read.

The words are the finance register, not general English. "Cut" and "miss" are negative here and
neutral anywhere else; "beat" and "raised" are positive here and violent or neutral elsewhere. A
general-purpose sentiment list scores "profit slumps as guidance is cut" as neutral because none of
those words are rude.

The known failure is macro wording, where the good direction depends on the noun: "jobless claims
fall to their lowest since May" scores negative on `fall` and is good news. Nothing here fixes that,
because a word list cannot know what is falling; it is one reason the scored polarity is reported as
description and not fed into advice.

The identifier is a digest of the word lists themselves, so a score can always name what produced
it and editing a single word changes the name. That is the pinning C4 asks for: a stored polarity
carries `VERSION`, and a polarity stored under a version that no longer exists is visibly stale
rather than quietly wrong.
"""

import hashlib
import re

POSITIVE = frozenset("""
    beat beats bullish climbs gain gains gained grew growth higher jumps jumped optimism outperform
    profit profits rally rallies rallied raised raises rebound record recovers rises rising rose soar
    soars strength strong stronger surge surges surged top tops upbeat upgrade upgraded upside win
    wins won approves approved awarded expands expansion launch launches partnership dividend buyback
    accelerates improves improved resilient robust exceeds exceeded advantage momentum
""".split())

NEGATIVE = frozenset("""
    bankruptcy bearish charges charged concern concerns crash cut cuts declines declined decline
    delay delayed downgrade downgraded downturn drop drops dropped fail fails failed fall falls fell
    fine fined fraud halt halted investigation lawsuit layoff layoffs loss losses lower miss misses
    missed plunge plunges plunged pressure probe recall recession risk risks sanctions shortfall
    shrink slump slumps slumping slowdown slides slid sue sued suspended tumble tumbles tumbled
    warns warned weak weaker weakness worse worst writedown default delisting breach
""".split())

# Three tokens, because "not expected to beat" is as long as a flipped phrase gets in a headline and
# a wider window starts flipping the next clause.
NEGATORS = frozenset("no not never without fails failed unable".split())
_NEGATION_REACH = 3

_WORD = re.compile(r"[a-z]+")


def _digest():
    joined = " ".join(sorted(POSITIVE) + ["|"] + sorted(NEGATIVE) + ["|"] + sorted(NEGATORS))
    return hashlib.sha1(joined.encode()).hexdigest()[:12]


VERSION = f"lexicon@{_digest()}"


def matches(text):
    """(positive words, negative words) found, with negated ones counted for the other side."""
    words = _WORD.findall(text.lower())
    positive, negative = [], []
    for index, word in enumerate(words):
        if word not in POSITIVE and word not in NEGATIVE:
            continue
        flipped = any(earlier in NEGATORS
                      for earlier in words[max(0, index - _NEGATION_REACH):index])
        good = (word in POSITIVE) != flipped
        (positive if good else negative).append(word)
    return positive, negative


def polarity(text):
    """-1 to +1, by which side of the lexicon the headline's words fall on. 0 when it says neither.

    The share rather than the count, so a long summary cannot outvote a headline by length alone.
    """
    positive, negative = matches(text)
    total = len(positive) + len(negative)
    return (len(positive) - len(negative)) / total if total else 0.0
