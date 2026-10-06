import pytest

from backend.advisory.fusion.evidence import DomainResult, Evidence, Provenance, unavailable
from backend.advisory.fusion.fusion import fuse

P = Provenance(source="test", as_of="2026-09-25")


def dom(name, score, confidence=1.0, state=None):
    ev = [Evidence(domain=name, claim=f"{name} says {score:+.2f}", direction=(score > 0) - (score < 0),
                   strength=abs(score), provenance=P)] if score else []
    return DomainResult(domain=name, state=state or ("bullish" if score > 0 else "bearish"),
                        score=score, evidence=ev, as_of="2026-09-25", confidence=confidence)


def all_bullish(**over):
    base = {"technical": dom("technical", 0.7), "fundamental": dom("fundamental", 0.6),
            "news_sentiment": dom("news_sentiment", 0.5), "risk": dom("risk", 0.3),
            "pattern": dom("pattern", 0.5), "forecast": dom("forecast", 0.0, 0.0)}
    base.update(over)
    return base


def test_agreeing_strong_evidence_yields_buy():
    d = fuse(all_bullish())
    assert d.decision_support_label == "BUY" and not d.conflict and d.gates_applied == []


def test_conflicting_domains_are_flagged_and_block_a_buy():
    d = fuse(all_bullish(fundamental=dom("fundamental", -0.8), news_sentiment=dom("news_sentiment", -0.6)))
    assert d.conflict
    assert d.decision_support_label != "BUY"
    assert d.contradicting_evidence and d.supporting_evidence
    assert d.key_uncertainties[0].startswith("Evidence conflict")


def test_high_risk_downgrades_buy_to_watch():
    d = fuse(all_bullish(risk=dom("risk", -0.6)))
    assert d.decision_support_label in ("WATCH", "HOLD")
    if d.score >= 0.35:
        assert any("risk profile is high" in g for g in d.gates_applied)


def test_price_only_evidence_cannot_produce_a_buy():
    d = fuse({"technical": dom("technical", 0.9), "pattern": dom("pattern", 0.8), "risk": dom("risk", 0.5),
              "fundamental": unavailable("fundamental", "no statements"),
              "news_sentiment": unavailable("news_sentiment", "no news")})
    assert d.decision_support_label == "WATCH"
    assert any("no fundamental or news evidence" in g for g in d.gates_applied)


def test_missing_domains_are_listed_as_uncertainties_and_cap_confidence():
    d = fuse({"technical": dom("technical", 0.5), "risk": dom("risk", 0.2), "pattern": dom("pattern", 0.4),
              "fundamental": unavailable("fundamental", "no statements")})
    assert any("Fundamental data unavailable: no statements" in u for u in d.key_uncertainties)
    assert d.confidence <= round(1 - 0.22 / (0.20 + 0.15 + 0.10 + 0.22), 3) + 1e-3


def test_too_few_domains_is_insufficient_evidence():
    d = fuse({"technical": dom("technical", 0.9), "risk": dom("risk", 0.5)})
    assert d.decision_support_label == "INSUFFICIENT_EVIDENCE" and d.confidence == 0.0


def test_no_technical_data_is_insufficient_evidence():
    d = fuse({"technical": unavailable("technical", "no prices"), "fundamental": dom("fundamental", 0.8),
              "news_sentiment": dom("news_sentiment", 0.8), "risk": dom("risk", 0.2)})
    assert d.decision_support_label == "INSUFFICIENT_EVIDENCE"


def test_same_negative_evidence_reads_sell_when_held_and_avoid_when_not():
    bearish = {k: dom(k, -abs(v.score) if v.score else 0.0, v.confidence) for k, v in all_bullish().items()}
    assert fuse(bearish, held=True).decision_support_label == "SELL"
    assert fuse(bearish, held=False).decision_support_label == "AVOID"


def test_zero_confidence_domains_do_not_dilute_the_vote():
    with_silent = fuse(all_bullish(candlestick=dom("candlestick", 0.0, 0.0, "no recent formation")))
    without = fuse(all_bullish())
    assert with_silent.score == pytest.approx(without.score)


def test_stale_data_raises_uncertainty():
    assert fuse(all_bullish(), stale=True).uncertainty > fuse(all_bullish()).uncertainty


def test_long_horizon_weights_fundamentals_over_candles():
    mixed = all_bullish(fundamental=dom("fundamental", -0.6), candlestick=dom("candlestick", 0.9))
    assert fuse(mixed, horizon="long").score < fuse(mixed, horizon="short").score


def test_unknown_domain_fails_loudly():
    with pytest.raises(KeyError, match="no fusion weight"):
        fuse({**all_bullish(), "astrology": dom("astrology", 1.0)})


def test_invalid_horizon_fails_loudly():
    with pytest.raises(ValueError):
        fuse(all_bullish(), horizon="forever")


def test_decision_carries_timestamps_and_disclaimer():
    d = fuse(all_bullish())
    assert d.data_timestamp == "2026-09-25" and d.analysis_timestamp and "not an order" in d.note


def test_blocked_sell_reads_hold_never_the_more_favourable_watch():
    price_only_bearish = {"technical": dom("technical", -0.9), "pattern": dom("pattern", -0.8),
                          "risk": dom("risk", -0.5), "fundamental": unavailable("fundamental", "none")}
    d = fuse(price_only_bearish, held=True)
    assert d.decision_support_label == "HOLD"
    assert any("SELL downgraded to HOLD" in g for g in d.gates_applied)


def test_non_price_evidence_must_agree_to_corroborate_a_buy():
    d = fuse({"technical": dom("technical", 0.9), "pattern": dom("pattern", 0.8), "risk": dom("risk", 0.5),
              "fundamental": dom("fundamental", -0.2)})
    assert d.decision_support_label != "BUY"
    assert any("agrees with the combined view" in g for g in d.gates_applied)
