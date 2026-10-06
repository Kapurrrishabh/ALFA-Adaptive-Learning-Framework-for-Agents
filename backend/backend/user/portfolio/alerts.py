"""Alert rules. Each alert lists every reason it fired; several independent
signals changing together raise the severity. Rules read only the analysis
payloads, so an alert can always be traced to the same evidence the user
sees in the analysis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


from backend.config import THRESHOLDS
from backend.user.portfolio.portfolio import PortfolioReport
from backend.advisory.service import StockAnalysis

PORTFOLIO_VOL_JUMP = 0.20   # PROVISIONAL: +20% relative rise in portfolio volatility


@dataclass
class Alert:
    symbol: str
    kind: str
    severity: str
    message: str
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _severity(n: int, big_move: bool) -> str:
    if n >= 3 or big_move:
        return "high"
    return "medium" if n == 2 else "low"


def evaluate(a: StockAnalysis, previous: Optional[Dict[str, Any]] = None,
             now: Optional[datetime] = None) -> Optional[Alert]:
    now = now or datetime.now(timezone.utc)
    reasons: List[str] = []
    kinds: List[str] = []
    tech, risk, pat, news = (a.results[k] for k in ("technical", "risk", "pattern", "news_sentiment"))
    big_move = False
    if risk.available:
        day = risk.details["returns"].get("1d_pct")
        if day is not None and abs(day) >= THRESHOLDS.price_move_alert_pct:
            reasons.append(f"price moved {day:+.1f}% in the last session")
            kinds.append("price_move")
            big_move = abs(day) >= 2 * THRESHOLDS.price_move_alert_pct
        if (risk.details.get("vol_percentile_now") or 0) >= 90:
            reasons.append(f"volatility at percentile {risk.details['vol_percentile_now']:.0f} of its own history")
            kinds.append("volatility")
    if tech.available:
        vr = tech.details.get("volume_vs_20d_avg")
        if vr is not None and vr >= THRESHOLDS.volume_spike_mult:
            reasons.append(f"{vr:.1f}x average volume")
            kinds.append("volume_spike")
        if tech.details.get("breakout_status") == "above 20d high":
            reasons.append("close above the prior 20-day high")
            kinds.append("breakout")
    if pat.available:
        for e in pat.details.get("structure_events", []):
            if e["event"] in ("breakout", "breakdown", "false_breakout", "false_breakdown"):
                reasons.append(f"{e['event'].replace('_', ' ')} of the 60-day range at {e['level']:,.2f}")
                kinds.append(e["event"])
    if news.available:
        for ev in news.details.get("events", []):
            try:
                ts = datetime.fromisoformat(ev["timestamp"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if (now - ts).total_seconds() <= 86400 and ev["importance"] >= 0.8:
                reasons.append(f"major {ev['event_type']} news: \"{ev['headline']}\" ({ev['source']})")
                kinds.append("major_news")
                break
        if news.details.get("shift"):
            reasons.append(f"news sentiment shifted {news.details['shift']} over the last 7 days")
            kinds.append("sentiment_shift")
    if previous:
        prev_d = previous["decision"]
        if prev_d["decision_support_label"] != a.decision.decision_support_label:
            reasons.append(f"classification changed {prev_d['decision_support_label']} -> "
                           f"{a.decision.decision_support_label}")
            kinds.append("label_change")
        prev_f = previous["results"].get("fundamental", {})
        cur_f = a.results["fundamental"]
        if prev_f.get("score") is not None and cur_f.available and cur_f.score - prev_f["score"] <= -0.3:
            reasons.append(f"fundamental score fell {prev_f['score']:+.2f} -> {cur_f.score:+.2f}")
            kinds.append("fundamental_deterioration")
    if not reasons:
        return None
    kind = "multi_signal" if len(set(kinds)) >= 2 else kinds[0]
    header = f"ALERT — {a.name} ({a.provider_symbol}) at {a.analysis_timestamp}"
    body = "\n".join(f"- {r}" for r in reasons)
    trigger = ("Multiple independent signals changed simultaneously." if kind == "multi_signal"
               else "Single rule triggered.")
    return Alert(symbol=a.symbol, kind=kind, severity=_severity(len(set(kinds)), big_move),
                 message=f"{header}\nDetected:\n{body}\nTrigger: {trigger}", reasons=reasons)


def portfolio_alert(current: PortfolioReport, previous_payload: Optional[Dict[str, Any]]) -> Optional[Alert]:
    if not previous_payload:
        return None
    before = (previous_payload.get("risk") or {}).get("portfolio_vol_pct")
    after = current.risk.get("portfolio_vol_pct")
    if not before or not after:
        return None
    if after / before - 1 >= PORTFOLIO_VOL_JUMP:
        reason = f"portfolio volatility rose from {before}% to {after}%"
        return Alert(symbol="PORTFOLIO", kind="portfolio_risk", severity="medium",
                     message=f"ALERT — portfolio risk increase\n- {reason}", reasons=[reason])
    return None
