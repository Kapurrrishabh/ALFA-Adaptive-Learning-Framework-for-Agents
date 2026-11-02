"""Static financial knowledge retrieval (the "what is RSI?" layer).

The knowledge base holds only timeless definitions and methodology; it
contains no prices, ratios or news for any company, so a retrieved passage
can never be mistaken for current market data. Retrieval is BM25 over
section-level chunks: the corpus is small and definitional, where exact
term matching beats embeddings and needs no model or vector store. Swap in
dense retrieval only if a retrieval eval on real user questions shows BM25
missing relevant sections.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

KB_DIR = Path(__file__).parent / "lessons"
K1, B = 1.5, 0.75
# provisional, set on reviewers' labelled questions. A heading must name question words worth 1.95 idf ("price"
# alone is 1.37); 1.4 to 1.95 scored the same, and the top answers fewest forum questions. Failing that, a named
# heading needs a section score of 5: "unpredictable" in a lesson's text scores 6.8, chance words about 2.2
MIN_TITLE_SCORE, MIN_SECTION_SCORE = 1.95, 5.0
# everyday words carry no finance meaning, yet BM25 scores the rarer ones highly: before "out", "there",
# "you" and "like" were listed, "what is going on out there?" was answered with the lesson on triangles
STOP = set("a an the of to in on for and or is are be what how why does do it its this that "
           "with as by at from me explain tell about mean means "
           "i you your we us our they them he she who whom whose there here this these those "
           "going go goes went like want need know think say said tell told "
           "everything something anything nothing everyone someone anyone "
           "will would shall should can could may might must have has had was were been being am "
           "not no yes very really just also too so but if then than when where which while "
           "all any some more most much many few own same other another now today "
           "out up down over under again still even only please thanks hi hello hey "
           "good bad nice new old thing stuff happen happening happened".split())
SYNONYMS = {"pe": ["p/e", "price", "earnings"], "p/e": ["pe"], "rsi": ["relative", "strength"],
            "sharpe": ["sharpe"], "de": ["debt/equity", "debt"], "roe": ["return", "equity"],
            "fcf": ["free", "cash", "flow"], "sl": ["stop", "loss"],
            "h&s": ["head", "shoulders"], "ltcg": ["long-term", "capital", "gains", "tax"],
            "stcg": ["short-term", "capital", "gains", "tax"], "sip": ["systematic", "investment", "plan"],
            "nav": ["net", "asset", "value"], "etf": ["exchange-traded"], "elss": ["equity", "linked", "savings"]}


@dataclass
class Chunk:
    doc: str
    title: str
    text: str

    @property
    def citation(self) -> str:
        return f"knowledge_base/{self.doc} § {self.title}"


def _tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9/&\-]*", text.lower())
    out = []
    for t in toks:
        # a lone letter is what an apostrophe leaves ("can't" -> "t", "fund's" -> "s"), never a topic, yet it
        # matched lesson text: "what's the fund's p/e ratio" got the expense-ratio lesson
        if t in STOP or (len(t) == 1 and t.isalpha()):
            continue
        if len(t) > 4 and t.endswith("xes"):    # "taxes" -> "tax"; cutting only the "s" left "taxe"
            t = t[:-2]
        out.append(t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t)
    return out


def _names(query: List[str], word: str) -> bool:
    """Whether a question names a heading word, allowing for the crude stemmer above ("taxed" names "tax"). A
    hyphenated word needs every part, so "tax" alone does not name "tax-saving"; "debt" names "debt/equity"."""
    def said(part):
        return any(q == p or (min(len(q), len(p)) >= 3 and (q.startswith(p) or p.startswith(q)))
                   for q in query for p in part.split("/") if p)
    return word in query or all(said(part) for part in word.split("-") if part)


class KnowledgeBase:
    def __init__(self, directory: Path = KB_DIR):
        self.chunks: List[Chunk] = []
        for path in sorted(directory.glob("*.md")):
            parts = re.split(r"^## ", path.read_text(), flags=re.M)
            for part in parts[1:]:
                title, _, body = part.partition("\n")
                self.chunks.append(Chunk(path.name, title.strip(), body.strip()))
        if not self.chunks:
            raise FileNotFoundError(f"no knowledge sections found in {directory}")
        self.docs = [_tokens(c.title + " " + c.title + " " + c.text) for c in self.chunks]
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]
        self.titles = [set(_tokens(c.title)) for c in self.chunks]

    def search(self, query: str, k: int = 3, min_score: float = 1.0) -> List[tuple]:
        q = _tokens(query)
        expanded = list(q)
        for t in q:
            expanded.extend(SYNONYMS.get(t, []))
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in expanded:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / self.avgdl))
            scores.append(s)
        title_scores = [sum(self.idf[w] for w in title if _names(expanded, w)) for title in self.titles]
        # a lesson is about what its heading names, so the sections the question names are ranked on their own:
        # cutting to the top k first let a rare word in another lesson's prose win, and the heading check left nothing
        strong = [i for i, s in enumerate(scores) if s >= min_score and title_scores[i] >= MIN_TITLE_SCORE]
        weak = [i for i, s in enumerate(scores) if s >= max(min_score, MIN_SECTION_SCORE) and title_scores[i] > 0]
        ranked = sorted(strong or weak, key=scores.__getitem__, reverse=True)
        return [(self.chunks[i], round(scores[i], 2)) for i in ranked[:k]]


_KB: Optional[KnowledgeBase] = None


def knowledge_base() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB
