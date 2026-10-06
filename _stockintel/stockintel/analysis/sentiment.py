"""Financial sentiment scoring.

Default scorer is a finance-specific lexicon (Loughran-McDonald style word
classes, not a general-purpose sentiment list) with negation handling and
"direction-inverting" nouns: "losses narrow" and "debt falls" are good news.
The Scorer interface lets a transformer model (e.g. FinBERT) replace it once
it has beaten this baseline on a labelled financial set (docs/DESIGN.md §12).
Every score produced here is a MODEL OUTPUT, never a fact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Protocol

POSITIVE = set("""
beat beats beating exceeded exceeds outperform outperformed outperforms upgrade upgraded upgrades
record strong stronger robust growth grew grow grows gain gains gained surge surged surges
soar soared soars jump jumped jumps rally rallied rallies rise rose rises rising climb climbed
climbs improve improved improves improvement profitable profitability expansion expand
expanded boost boosted boosts win wins won award awarded bullish optimistic upbeat positive
dividend buyback approval approved breakthrough recovery rebound rebounded rebounds accelerate
accelerated momentum raises raised higher highest milestone launch launched order orders contract
""".split())

NEGATIVE = set("""
miss missed misses missing underperform underperformed downgrade downgraded downgrades weak weaker
weakness decline declined declines declining fall fell falls falling drop dropped drops plunge
plunged plunges slump slumped slumps tumble tumbled tumbles crash crashed slide slid slides loss
losses lose losing lost deficit default defaulted fraud probe investigation penalty penalised
penalized fine fined lawsuit litigation sue sued ban banned halt halted suspend suspended
resign resigned resignation bearish pessimistic negative warning warns warned cut cuts cutting
lower lowest layoff layoffs shutdown recall recalled delay delayed impairment writedown
write-off downturn slowdown pressure pressured concern concerns risk risks raid raids
slows slowed slowing slower dip dips dipped shrink shrinks shrank contraction contracted
""".split())

UNCERTAINTY = set("""
uncertain uncertainty unclear may might could possibly volatile volatility unpredictable
speculation speculative rumour rumor rumours rumors pending unconfirmed doubt doubts
""".split())

FEAR = set("""
crisis panic fear fears selloff sell-off crash collapse contagion turmoil meltdown
bankruptcy insolvency insolvent scam fraud emergency
""".split())

NEGATIONS = {"not", "no", "never", "without", "fails", "failed", "neither", "nor", "hardly"}
# Nouns whose movement verbs invert meaning ("debt falls" is positive).
INVERTING_NOUNS = {"debt", "loss", "losses", "cost", "costs", "npa", "npas", "inflation",
                   "deficit", "provisions", "liabilities", "borrowing", "borrowings", "expenses"}
MOVE_WORDS = {"fall", "fell", "falls", "falling", "drop", "dropped", "drops", "decline",
              "declined", "declines", "narrow", "narrowed", "narrows", "shrink", "shrinks",
              "rise", "rose", "rises", "rising", "jump", "jumped", "jumps", "surge", "surged",
              "surges", "widen", "widened", "widens", "grow", "grew", "grows", "higher", "lower",
              "cut", "cuts"}
UP_MOVES = {"rise", "rose", "rises", "rising", "jump", "jumped", "jumps", "surge", "surged",
            "surges", "widen", "widened", "widens", "grow", "grew", "grows", "higher"}
DOWN_MOVES = (MOVE_WORDS - UP_MOVES) | {"slows", "slowed", "slowing", "plunge", "plunged",
                                        "plunges", "slump", "slumps", "slumped", "dip", "dips",
                                        "dipped", "shrank", "contracted"}

TOKEN = re.compile(r"[a-z][a-z\-']*")


@dataclass
class SentimentScore:
    score: float        # -1..+1
    positive: int
    negative: int
    uncertainty: int
    fear: int
    confidence: float   # 0..1, grows with the number of sentiment-bearing words
    model: str = "lexicon-v1"


class Scorer(Protocol):
    name: str

    def score(self, text: str) -> SentimentScore: ...


class LexiconScorer:
    name = "lexicon-v1"

    def score(self, text: str) -> SentimentScore:
        tokens = TOKEN.findall(text.lower())
        pos = neg = unc = fear = 0
        for i, tok in enumerate(tokens):
            ahead = tokens[i + 1:i + 3]
            # The movement verb that follows carries the meaning: "growth slows",
            # "losses narrow". Count it once, at the verb.
            if tok in INVERTING_NOUNS and any(w in MOVE_WORDS for w in ahead):
                continue
            if tok in POSITIVE and any(w in DOWN_MOVES for w in ahead):
                continue
            window = tokens[max(0, i - 3):i]
            negated = any(w in NEGATIONS for w in window)
            polarity = 0
            if tok in MOVE_WORDS and any(w in INVERTING_NOUNS for w in tokens[max(0, i - 2):i]):
                polarity = -1 if tok in UP_MOVES else 1
            elif tok in POSITIVE:
                polarity = 1
            elif tok in NEGATIVE:
                polarity = -1
            if negated:
                polarity = -polarity
            if polarity > 0:
                pos += 1
            elif polarity < 0:
                neg += 1
            if tok in UNCERTAINTY:
                unc += 1
            if tok in FEAR:
                fear += 1
        total = pos + neg
        score = (pos - neg) / total if total else 0.0
        return SentimentScore(score=score, positive=pos, negative=neg, uncertainty=unc,
                              fear=fear, confidence=min(1.0, total / 3.0), model=self.name)


class TransformerScorer:
    """Optional FinBERT-style scorer. Loaded lazily; raises if unavailable."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError("transformers is not installed; `pip install transformers torch` "
                               "to use TransformerScorer") from exc
        self.name = model_name
        self._pipe = pipeline("text-classification", model=model_name, top_k=None)

    def score(self, text: str) -> SentimentScore:
        probs = {d["label"].lower(): d["score"] for d in self._pipe(text[:512])[0]}
        s = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        return SentimentScore(score=s, positive=int(s > 0.2), negative=int(s < -0.2),
                              uncertainty=0, fear=0,
                              confidence=1.0 - probs.get("neutral", 0.0), model=self.name)


def default_scorer() -> Scorer:
    return LexiconScorer()


def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
