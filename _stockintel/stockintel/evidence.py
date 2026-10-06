"""Core evidence contract shared by every analytical engine.

Every engine returns a DomainResult: a state, a score in [-1, +1], and a list
of Evidence items. Each Evidence keeps its provenance and declares whether it
is an observed FACT or a MODEL OUTPUT — downstream layers must never blur the
two. A domain that cannot produce data returns state="unavailable" with an
error string; it never guesses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Provenance:
    source: str                      # e.g. "yahoo:RELIANCE.NS", "computed:rsi(14)"
    as_of: str                       # timestamp/period the data describes
    retrieved_at: str = field(default_factory=utcnow_iso)
    period: Optional[str] = None     # e.g. "FY2025" for statement data


@dataclass(frozen=True)
class Evidence:
    domain: str                      # technical | candlestick | pattern | ...
    claim: str                       # one human-readable sentence
    direction: int                   # +1 bullish, -1 bearish, 0 neutral/informational
    strength: float                  # 0..1, how much this claim should count
    provenance: Provenance
    value: Optional[float] = None    # the number behind the claim, when there is one
    is_model_output: bool = False    # True = prediction, False = observed fact

    def kind(self) -> str:
        return "MODEL OUTPUT" if self.is_model_output else "FACT"


@dataclass
class DomainResult:
    domain: str
    state: str                       # engine-specific summary, or "unavailable"
    score: Optional[float]           # -1..+1; None when unavailable
    evidence: List[Evidence] = field(default_factory=list)
    # 0..1: how much fusion should trust this score. 0 means "ran, but found
    # nothing to say" — it must not be counted as a neutral vote.
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    as_of: Optional[str] = None
    error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.score is not None and self.state != "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def unavailable(domain: str, reason: str) -> DomainResult:
    """The one sanctioned way to report missing data: loud, never a default."""
    return DomainResult(domain=domain, state="unavailable", score=None,
                        error=reason, as_of=utcnow_iso(), confidence=0.0)


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
