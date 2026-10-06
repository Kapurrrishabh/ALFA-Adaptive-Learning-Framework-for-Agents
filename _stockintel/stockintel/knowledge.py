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

KB_DIR = Path(__file__).parent / "knowledge_base"
K1, B = 1.5, 0.75
STOP = set("a an the of to in on for and or is are be what how why does do it its this that "
           "with as by at from me explain tell about mean means".split())
SYNONYMS = {"pe": ["p/e", "price", "earnings"], "p/e": ["pe"], "rsi": ["relative", "strength"],
            "sharpe": ["sharpe"], "de": ["debt/equity", "debt"], "roe": ["return", "equity"],
            "fcf": ["free", "cash", "flow"], "sl": ["stop", "loss"],
            "h&s": ["head", "shoulders"], "ltcg": ["long-term", "capital", "gains", "tax"],
            "stcg": ["short-term", "capital", "gains", "tax"]}


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
        if t in STOP:
            continue
        out.append(t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t)
    return out


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
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], round(scores[i], 2)) for i in ranked[:k] if scores[i] >= min_score]


_KB: Optional[KnowledgeBase] = None


def knowledge_base() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB
