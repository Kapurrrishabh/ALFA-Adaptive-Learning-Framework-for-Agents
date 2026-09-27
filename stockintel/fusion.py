"""Evidence fusion and decision engine.

Why this shape (docs/DESIGN.md §17): a transparent weighted vote with explicit
gates, not a learned meta-model. Point-in-time fundamentals and news are not
available historically from the free sources this system uses, so a learned
fusion model could only be trained on the price-derived half of the evidence
and would silently over-weight it. The weighted vote keeps every domain's
contribution inspectable; gates make disagreement and missing data visible
in the label itself instead of averaging them away.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import DECISION, THRESHOLDS, WEIGHTS
from .evidence import DomainResult, Evidence, clamp, utcnow_iso

PRICE_DOMAINS = {"technical", "candlestick", "pattern", "risk", "forecast", "historical", "regime"}
# Investment horizon tilts domain weights; a long-horizon investor should not
# be steered by a candle. Multipliers are PROVISIONAL.
HORIZON_TILT: Dict[str, Dict[str, float]] = {
    "short": {"technical": 1.4, "candlestick": 1.5, "pattern": 1.3, "forecast": 1.3,
              "fundamental": 0.5, "news_sentiment": 1.3},
    "medium": {},
    "long": {"technical": 0.6, "candlestick": 0.3, "pattern": 0.5, "forecast": 0.6,
             "fundamental": 1.6, "news_sentiment": 0.8, "risk": 1.2},
}
HORIZON_TEXT = {"short": "1–4 weeks", "medium": "1–6 months", "long": "1 year or more"}


@dataclass
class Decision:
    overall_state: str
    decision_support_label: str
    score: float
    confidence: float
    uncertainty: float
    conflict: bool
    supporting_evidence: List[Dict[str, object]]
    contradicting_evidence: List[Dict[str, object]]
    risk_factors: List[str]
    key_uncertainties: List[str]
    gates_applied: List[str]
    domain_states: Dict[str, Dict[str, object]]
    time_horizon: str
    data_timestamp: Optional[str]
    analysis_timestamp: str = field(default_factory=utcnow_iso)
    note: str = ("Decision-support classification from structured evidence; not an order "
                 "and not a guarantee. SELL refers to an existing position; AVOID to a new one.")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _ev_dict(e: Evidence) -> Dict[str, object]:
    return {"domain": e.domain, "claim": e.claim, "direction": e.direction,
            "strength": e.strength, "kind": e.kind(), "value": e.value,
            "source": e.provenance.source, "as_of": e.provenance.as_of}


def fuse(results: Dict[str, DomainResult], held: bool = False, horizon: str = "medium",
         stale: bool = False, upcoming_events: Optional[List[str]] = None) -> Decision:
    if horizon not in HORIZON_TILT:
        raise ValueError(f"horizon must be one of {list(HORIZON_TILT)}, got {horizon!r}")
    base = WEIGHTS.as_dict()
    tilt = HORIZON_TILT[horizon]
    weights = {d: base[d] * tilt.get(d, 1.0) for d in base}
    unknown = set(results) - set(weights)
    if unknown:
        raise KeyError(f"no fusion weight configured for domains {sorted(unknown)}; "
                       "add them to config.FusionWeights")

    available = {d: r for d, r in results.items() if r.available}
    voting = {d: r for d, r in available.items() if r.confidence > 0}
    domain_states = {d: {"state": r.state, "score": None if r.score is None else round(r.score, 3),
                         "confidence": r.confidence, "as_of": r.as_of,
                         **({"error": r.error} if r.error else {})}
                     for d, r in results.items()}
    uncertainties: List[str] = []
    for d, r in results.items():
        if not r.available:
            uncertainties.append(f"{d.replace('_', ' ').capitalize()} data unavailable: {r.error}")
    if stale:
        uncertainties.append("Price data is stale; conclusions describe the last available session.")
    fc = results.get("forecast")
    if fc is not None and fc.available and fc.confidence == 0:
        uncertainties.append("Forecast models showed no out-of-sample edge over the base rate.")
    for ev_text in upcoming_events or []:
        uncertainties.append(f"Upcoming event: {ev_text}")

    tech = results.get("technical")
    ts = tech.as_of if tech is not None and tech.available else None
    if len(available) < DECISION.min_domains or "technical" not in available:
        return Decision(
            overall_state="insufficient evidence", decision_support_label="INSUFFICIENT_EVIDENCE",
            score=0.0, confidence=0.0, uncertainty=1.0, conflict=False,
            supporting_evidence=[], contradicting_evidence=[], risk_factors=[],
            key_uncertainties=uncertainties or ["Too few analytical domains produced data."],
            gates_applied=[f"only {len(available)} domains available (need {DECISION.min_domains}, "
                           "including technical)"],
            domain_states=domain_states, time_horizon=HORIZON_TEXT[horizon], data_timestamp=ts)

    w = np.array([weights[d] * r.confidence for d, r in voting.items()])
    s = np.array([r.score for r in voting.values()])
    score = float(clamp((w * s).sum() / w.sum())) if w.sum() > 0 else 0.0
    dispersion = float(np.sqrt((w * (s - score) ** 2).sum() / w.sum())) if w.sum() > 0 else 0.0
    strong_bull = [d for d, r in voting.items() if r.score > 0.25 and r.confidence >= 0.3]
    strong_bear = [d for d, r in voting.items() if r.score < -0.25 and r.confidence >= 0.3]
    conflict = bool(strong_bull and strong_bear) or dispersion >= THRESHOLDS.conflict_dispersion
    if conflict:
        leaning_bull = [d for d, r in voting.items() if r.score > 0.1]
        leaning_bear = [d for d, r in voting.items() if r.score < -0.1]
        uncertainties.insert(0, f"Evidence conflict: {', '.join(leaning_bull) or 'no domain'} lean "
                             f"bullish while {', '.join(leaning_bear) or 'no domain'} lean bearish.")

    missing_weight = sum(weights[d] for d in results if not results[d].available) / sum(
        weights[d] for d in results)
    mean_conf = float(np.mean([r.confidence for r in available.values()]))
    uncertainty = clamp(0.35 * missing_weight + 0.35 * min(1.0, dispersion / 0.8)
                        + 0.15 * stale + 0.15 * (1 - mean_conf), 0.0, 1.0)
    confidence = (1 - uncertainty) * (0.5 + 0.5 * min(1.0, abs(score) / DECISION.strong))
    # Never more confident than the share of evidence weight actually observed.
    confidence = round(min(confidence, 1 - missing_weight), 3)

    gates: List[str] = []
    if score >= DECISION.strong:
        label = "BUY"
    elif score >= DECISION.mild:
        label = "WATCH"
    elif score > -DECISION.mild:
        label = "HOLD"
    elif score > -DECISION.strong:
        label = "HOLD" if held else "AVOID"
    else:
        label = "SELL" if held else "AVOID"

    risk = results.get("risk")
    high_risk = risk is not None and risk.available and risk.score <= THRESHOLDS.high_risk_score
    corroborating = [d for d, r in voting.items() if d not in PRICE_DOMAINS
                     and abs(r.score) >= 0.1 and np.sign(r.score) == np.sign(score)]
    if label in ("BUY", "SELL"):
        reasons = []
        if conflict:
            reasons.append("domains disagree")
        if confidence < DECISION.min_confidence:
            reasons.append(f"confidence {confidence:.2f} < {DECISION.min_confidence}")
        if label == "BUY" and high_risk:
            reasons.append("risk profile is high")
        if not corroborating:
            reasons.append("no fundamental or news evidence agrees with the combined view")
        if reasons:
            # A blocked BUY still leans positive (WATCH); a blocked SELL must not
            # become more favourable than holding.
            fallback = "WATCH" if label == "BUY" else "HOLD"
            gates.append(f"{label} downgraded to {fallback}: " + "; ".join(reasons))
            label = fallback

    direction = 1 if score >= 0 else -1
    ranked = sorted((e for r in available.values() for e in r.evidence if e.direction != 0),
                    key=lambda e: e.strength * weights.get(e.domain, 0.05), reverse=True)
    supporting = [_ev_dict(e) for e in ranked if e.direction == direction][:8]
    contradicting = [_ev_dict(e) for e in ranked if e.direction == -direction][:8]
    risk_factors = [e.claim for d in ("risk", "regime") if d in available
                    for e in available[d].evidence if e.direction < 0]
    news = results.get("news_sentiment")
    if news is not None and news.available:
        for ev in news.details.get("events", [])[:15]:
            if ev["event_type"] in ("legal", "regulatory") and ev["sentiment"] < 0:
                risk_factors.append(f"{ev['event_type'].capitalize()} news: {ev['headline']}")

    lean = ("constructive" if score >= DECISION.mild else "negative" if score <= -DECISION.mild
            else "balanced")
    overall = lean + (", with conflicting evidence" if conflict else "") + (
        ", high uncertainty" if uncertainty > 0.5 else "")
    return Decision(
        overall_state=overall, decision_support_label=label, score=round(score, 3),
        confidence=confidence, uncertainty=round(uncertainty, 3), conflict=conflict,
        supporting_evidence=supporting, contradicting_evidence=contradicting,
        risk_factors=risk_factors[:8], key_uncertainties=uncertainties[:8], gates_applied=gates,
        domain_states=domain_states, time_horizon=HORIZON_TEXT[horizon], data_timestamp=ts)
