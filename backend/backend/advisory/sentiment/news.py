"""News intelligence: deduplicate, type, scope, weight, and aggregate news.

Each article becomes a structured NewsEvent. Syndicated copies collapse into
one event (the earliest) with a syndication count. Aggregate sentiment is
weighted by recency, source reliability, company relevance, and event
importance; a shift between the last 7 days and the prior 8-30 is reported
explicitly. The engine can only describe articles it was given — it never
invents events.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from backend.database.sources.provider import NewsItem
from backend.advisory.fusion.evidence import DomainResult, Evidence, Provenance, clamp, unavailable
from backend.advisory.sentiment.sentiment import Scorer, default_scorer

DOMAIN = "news_sentiment"
HALF_LIFE_DAYS = 3.0        # PROVISIONAL: news impact decay; tune via event study
RECENT_DAYS, PRIOR_DAYS = 7, 30
SHIFT_THRESHOLD = 0.30      # PROVISIONAL: change in weighted sentiment that counts as a shift
DUPLICATE_JACCARD = 0.6

# Reliability in [0, 1]. Exchange filings are primary sources.
SOURCE_RELIABILITY: Dict[str, float] = {
    "nse": 1.0, "bse": 1.0, "sebi": 1.0, "rbi": 1.0, "company filing": 1.0,
    "reuters": 0.95, "bloomberg": 0.95, "press trust of india": 0.9, "pti": 0.9,
    "the economic times": 0.85, "economic times": 0.85, "business standard": 0.85,
    "mint": 0.85, "livemint": 0.85, "the hindu businessline": 0.85, "financial express": 0.8,
    "moneycontrol": 0.8, "cnbc": 0.8, "cnbctv18": 0.8, "ndtv profit": 0.75,
    "yahoo finance": 0.7, "zee business": 0.65, "business today": 0.75,
    "investing.com": 0.65, "marketscreener": 0.65, "seeking alpha": 0.5,
    "motley fool": 0.5, "benzinga": 0.55,
}
DEFAULT_RELIABILITY = 0.5

# (event_type, pattern, importance 0..1, typical horizon)
EVENT_RULES = [
    ("earnings", r"\b(results?|earnings|q[1-4]|quarter(ly)?|profit|revenue|ebitda|net income)\b", 0.9, "days-weeks"),
    ("guidance", r"\b(guidance|outlook|forecast|target(s)?)\b", 0.7, "weeks-months"),
    ("dividend", r"\b(dividend|payout)\b", 0.5, "days"),
    ("buyback", r"\b(buy-?back|repurchase)\b", 0.6, "days-weeks"),
    ("corporate_action", r"\b(split|bonus issue|rights issue|record date|demerger)\b", 0.6, "days"),
    ("m_and_a", r"\b(acquire[sd]?|acquisition|merger|merge|stake|takeover|deal)\b", 0.8, "weeks-months"),
    ("capital_raise", r"\b(qip|ipo|fpo|fund ?rais|raises? .{0,20}(crore|billion|million)|ncd|bond issue)\b", 0.6, "days-weeks"),
    ("macro", r"\b(inflation|gdp|repo rate|interest rates?|fed|monetary policy|budget|fiscal|crude|rupee|cpi|wpi)\b", 0.5, "weeks-months"),
    ("regulatory", r"\b(sebi|rbi|regulator|regulatory|licen[cs]e|approval|compliance|norms?)\b", 0.7, "weeks-months"),
    ("legal", r"\b(court|lawsuit|litigation|probe|investigation|penalty|fine[d]?|raid|fraud|tribunal|nclt)\b", 0.8, "weeks-months"),
    ("management", r"\b(ceo|cfo|md|chairman|resign|appoint|board|management)\b", 0.6, "weeks-months"),
    ("analyst_rating", r"\b(upgrade|downgrade|rating|target price|overweight|underweight|brokerage)\b", 0.5, "days"),
    ("contract_order", r"\b(order|contract|wins?|bags?|tender)\b", 0.6, "weeks"),
    ("product", r"\b(launch|product|plant|capacity|expansion)\b", 0.4, "months"),
]
SECTOR_WORDS = r"\b(sector|industry|peers|banks|it stocks|auto stocks|pharma stocks|metal stocks|fmcg)\b"
MARKET_WORDS = r"\b(sensex|nifty|markets?|stocks to watch|dalal street|wall street)\b"

STOP = set("a an the of to in on for and or at by with from as is are be its it this that "
           "after over says said".split())


@dataclass
class NewsEvent:
    company: str
    timestamp: str
    event_type: str
    headline: str
    summary: str
    sentiment: float
    sentiment_model: str
    uncertainty: int
    importance: float
    potential_market_impact: str
    time_horizon: str
    source: str
    source_reliability: float
    scope: str
    url: str = ""
    syndicated_count: int = 1
    weight: float = 0.0
    provenance: Dict[str, str] = field(default_factory=dict)


def _norm_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def deduplicate(items: Sequence[NewsItem]) -> List[tuple]:
    """Group near-identical headlines; returns [(earliest_item, copies)]."""
    ordered = sorted(items, key=lambda it: it.published_at or "")
    groups: List[List] = []
    signatures: List[set] = []
    for it in ordered:
        sig = _norm_tokens(it.headline)
        for g, gs in zip(groups, signatures):
            if _jaccard(sig, gs) >= DUPLICATE_JACCARD:
                g.append(it)
                break
        else:
            groups.append([it])
            signatures.append(sig)
    return [(g[0], len(g)) for g in groups]


def reliability(source: str) -> float:
    s = (source or "").lower()
    for name, score in SOURCE_RELIABILITY.items():
        if name in s:
            return score
    return DEFAULT_RELIABILITY


def classify(text: str) -> tuple:
    t = text.lower()
    for event_type, pattern, importance, horizon in EVENT_RULES:
        if re.search(pattern, t):
            return event_type, importance, horizon
    return "general", 0.3, "unclear"


def scope_of(text: str, company_terms: Sequence[str], event_type: str) -> str:
    t = text.lower()
    if any(term and term.lower() in t for term in company_terms):
        return "company"
    if event_type == "macro":
        return "macro"
    if re.search(SECTOR_WORDS, t):
        return "sector"
    if re.search(MARKET_WORDS, t):
        return "market"
    return "unclear"


RELEVANCE = {"company": 1.0, "sector": 0.6, "macro": 0.4, "market": 0.4, "unclear": 0.3}


def _age_days(ts: str, now: datetime) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 86400.0)


def build_events(symbol: str, company_terms: Sequence[str], items: Sequence[NewsItem],
                 now: Optional[datetime] = None, scorer: Optional[Scorer] = None) -> List[NewsEvent]:
    now = now or datetime.now(timezone.utc)
    scorer = scorer or default_scorer()
    events: List[NewsEvent] = []
    for item, copies in deduplicate(items):
        text = f"{item.headline}. {item.summary}"
        event_type, importance, horizon = classify(text)
        s = scorer.score(text)
        scope = scope_of(text, company_terms, event_type)
        rel = reliability(item.source)
        age = _age_days(item.published_at, now)
        recency = 0.5 ** (age / HALF_LIFE_DAYS) if age is not None else 0.0
        # Syndication widens reach but is not independent confirmation; sub-linear bonus.
        reach = 1.0 + 0.25 * math.log(copies)
        weight = recency * rel * RELEVANCE[scope] * importance * s.confidence * reach
        magnitude = abs(s.score) * importance
        impact = ("high" if magnitude > 0.6 else "moderate" if magnitude > 0.3 else "low")
        if s.score:
            impact += " positive" if s.score > 0 else " negative"
        events.append(NewsEvent(
            company=symbol, timestamp=item.published_at, event_type=event_type,
            headline=item.headline, summary=item.summary[:400], sentiment=round(s.score, 3),
            sentiment_model=s.model, uncertainty=s.uncertainty, importance=importance,
            potential_market_impact=impact, time_horizon=horizon, source=item.source,
            source_reliability=rel, scope=scope, url=item.url, syndicated_count=copies,
            weight=round(weight, 4),
            provenance={"source": item.provenance.source, "retrieved_at": item.provenance.retrieved_at}))
    events.sort(key=lambda e: e.timestamp or "", reverse=True)
    return events


def _weighted(events: Sequence[NewsEvent]) -> Optional[float]:
    wsum = sum(e.weight for e in events)
    if wsum <= 0:
        return None
    return sum(e.sentiment * e.weight for e in events) / wsum


def analyze(symbol: str, company_terms: Sequence[str], items: Sequence[NewsItem],
            now: Optional[datetime] = None, scorer: Optional[Scorer] = None) -> DomainResult:
    if not items:
        return unavailable(DOMAIN, f"no news articles retrieved for {symbol}")
    now = now or datetime.now(timezone.utc)
    events = build_events(symbol, company_terms, items, now, scorer)
    ages = {id(e): _age_days(e.timestamp, now) for e in events}
    recent = [e for e in events if ages[id(e)] is not None and ages[id(e)] <= RECENT_DAYS]
    prior = [e for e in events if ages[id(e)] is not None and RECENT_DAYS < ages[id(e)] <= PRIOR_DAYS]

    # Recency decay is already in weight; the prior window is re-weighted without
    # decay so it can serve as a baseline instead of vanishing.
    def undecayed(es: Sequence[NewsEvent]) -> Optional[float]:
        ws = [(e.sentiment, e.source_reliability * RELEVANCE[e.scope] * e.importance) for e in es]
        tot = sum(w for _, w in ws)
        return sum(s * w for s, w in ws) / tot if tot else None

    current = _weighted(events)
    recent_s, prior_s = undecayed(recent), undecayed(prior)
    as_of = events[0].timestamp if events else None
    ev: List[Evidence] = []
    shift = None
    if recent_s is not None and prior_s is not None and len(recent) >= 2 and len(prior) >= 2:
        delta = recent_s - prior_s
        if abs(delta) >= SHIFT_THRESHOLD:
            shift = "positive" if delta > 0 else "negative"
            ev.append(Evidence(domain=DOMAIN, direction=1 if delta > 0 else -1, strength=0.6,
                               claim=f"News sentiment shifted materially {shift} over the last "
                                     f"{RECENT_DAYS} days ({prior_s:+.2f} -> {recent_s:+.2f}, "
                                     f"{len(recent)} vs {len(prior)} articles).",
                               value=round(delta, 3), is_model_output=True,
                               provenance=Provenance(source="model:news_sentiment", as_of=str(as_of))))
    for e in sorted(events, key=lambda e: e.weight, reverse=True)[:5]:
        if e.weight <= 0 or e.sentiment == 0:
            continue
        ev.append(Evidence(domain=DOMAIN, direction=1 if e.sentiment > 0 else -1,
                           strength=min(1.0, e.weight * 2),
                           claim=f"[{e.event_type}, {e.scope}] \"{e.headline}\" — {e.source}, "
                                 f"{e.timestamp[:10]} (sentiment {e.sentiment:+.2f}"
                                 + (f", {e.syndicated_count} copies" if e.syndicated_count > 1 else "")
                                 + ").",
                           value=e.sentiment, is_model_output=True,
                           provenance=Provenance(source=f"news:{e.source}", as_of=e.timestamp)))
    details = {"events": [asdict(e) for e in events], "article_count": len(items),
               "unique_events": len(events), "recent_count": len(recent), "prior_count": len(prior),
               "weighted_sentiment": None if current is None else round(current, 3),
               "recent_sentiment": None if recent_s is None else round(recent_s, 3),
               "prior_sentiment": None if prior_s is None else round(prior_s, 3),
               "shift": shift,
               "high_importance_recent": [e.headline for e in recent if e.importance >= 0.8]}
    if current is None:
        return DomainResult(domain=DOMAIN, state="no recent relevant news", score=0.0,
                            evidence=ev, details=details, as_of=as_of, confidence=0.0)
    total_weight = sum(e.weight for e in events)
    confidence = round(min(1.0, total_weight / 1.5), 2)  # ~3 fresh, relevant, reliable stories = full
    state = ("positive" if current > 0.2 else "negative" if current < -0.2 else "neutral")
    if shift:
        state += f" (shifting {shift})"
    return DomainResult(domain=DOMAIN, state=state, score=clamp(current), evidence=ev,
                        details=details, as_of=as_of, confidence=confidence)
