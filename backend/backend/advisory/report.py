"""Deterministic renderers: analysis, full research report, drill-downs,
comparisons, portfolio, screener. Text is assembled only from computed
results, so it is faithful by construction; the optional LLM layer must
pass a numeric faithfulness check against these same payloads.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from backend.advisory.service import StockAnalysis

CURRENCY = {"INR": "₹", "USD": "$"}
UNAVAILABLE = "Data unavailable"
SCORE_SCALE = "from −1 (all negative) to +1 (all positive)"


def _money(v: Optional[float], ccy: str = "INR") -> str:
    if v is None:
        return UNAVAILABLE
    return f"{'-' if v < 0 else ''}{CURRENCY.get(ccy, ccy + ' ')}{abs(v):,.2f}"


def _section(n: int, title: str) -> str:
    return f"\n### {n}. {title}\n"


def _evidence_lines(a: StockAnalysis, domain: str, limit: int = 8) -> List[str]:
    r = a.results.get(domain)
    if r is None:
        return [UNAVAILABLE]
    if not r.available:
        return [f"{UNAVAILABLE}: {r.error}"]
    lines = []
    for e in r.evidence[:limit]:
        lines.append(f"- {_mark(e.direction)} [{e.kind()}] {e.claim}")
    return lines or ["- No notable observations."]


def _mark(direction: int) -> str:
    return "▲" if direction > 0 else "▼" if direction < 0 else "·"


def _metric(m: Dict[str, Any], key: str, suffix: str = "") -> str:
    item = m.get(key) or {}
    v = item.get("value")
    if v is None:
        return item.get("note", UNAVAILABLE)
    unit = item.get("unit")
    if unit == "abs" and item.get("currency") == "INR" and abs(v) >= 1e7:
        return f"₹{v / 1e7:,.0f} crore ({item.get('period', '')})"
    if unit == "abs":
        return f"{item.get('currency', '')} {v:,.0f} ({item.get('period', '')})"
    txt = f"{v:,.2f}{'%' if unit == '%' else 'x' if unit == 'x' else ''}"
    return f"{txt}{suffix} ({item.get('period', '')})"


def render_analysis(a: StockAnalysis, full: bool = False) -> str:
    d = a.decision
    ccy = a.currency
    out = [f"### {a.name} ({a.provider_symbol})",
           f"Last close **{_money(a.price, ccy)}**. Price data as of {a.quality['last_date']}, from "
           f"{a.quality['source']}. Analysis timestamp {a.analysis_timestamp}."]
    if a.quality.get("source") == "synthetic":
        out.append("⚠ SYNTHETIC DEMO DATA — not real market prices.")

    out.append(_section(1, "Executive summary"))
    out.append(f"Label: **{d.decision_support_label}**, looking {d.time_horizon} ahead.")
    out.append(f"Overall view: **{d.overall_state}**.")
    out.append(f"Combined score {d.score:+.2f}, on a scale {SCORE_SCALE}.")
    out.append(f"Confidence {d.confidence:.2f} and uncertainty {d.uncertainty:.2f}, each out of 1. "
               "Uncertainty goes up when data is missing, old, weak or in conflict.")
    for g in d.gates_applied:
        out.append(f"Rule applied: {g}")
    if d.conflict:
        out.append("\nParts of the evidence disagree.")
    out.append("\n**Main positives**" if d.score >= 0 else "\n**Main negatives**")
    out += [f"- {e['claim']}" for e in d.supporting_evidence[:4]] or ["- none"]
    out.append("\n**Main points the other way**")
    out += [f"- {e['claim']}" for e in d.contradicting_evidence[:4]] or ["- none"]
    out.append("Main doubt: " + (d.key_uncertainties[0] if d.key_uncertainties else "none flagged"))

    tech = a.results["technical"]
    out.append(_section(2, "Technical analysis"))
    if tech.available:
        t = tech.details
        out.append(f"Trend (bullish means rising, bearish falling): {tech.state}.")
        out.append(f"Swing structure ({TERM_MEANINGS['Swing structure']}): {t.get('trend_structure')}.")
        out.append(f"Moving averages (average prices over recent days): {t.get('ma_structure', UNAVAILABLE)}.")
        out.append(f"RSI(14), {TERM_MEANINGS['RSI']}: {t.get('rsi_14')}.")
        out.append(f"MACD histogram ({TERM_MEANINGS['MACD histogram']}): {t.get('macd_histogram')}.")
        out.append(f"20-day return: {t.get('return_20d_pct')}%.")
        out.append(f"Volume: {t.get('volume_vs_20d_avg', UNAVAILABLE)}x its 20-day average.")
        out.append(f"Breakout status (has the price left its recent range?): {t.get('breakout_status')}.")
        out.append("Support, where falls have stopped before: "
                   + (", ".join(_money(s, ccy) for s in t.get("support_levels", [])) or "none identified") + ".")
        out.append("Resistance, where rises have stalled before: "
                   + (", ".join(_money(s, ccy) for s in t.get("resistance_levels", [])) or "none identified") + ".")
    out += _evidence_lines(a, "technical")

    out.append(_section(3, "Candlestick analysis"))
    out.append(f"State: {a.results['candlestick'].state}")
    out += _evidence_lines(a, "candlestick", 4)

    out.append(_section(4, "Chart pattern analysis"))
    pat = a.results["pattern"]
    out.append(f"State: {pat.state}")
    out += _evidence_lines(a, "pattern", 5)
    if pat.available and pat.details.get("structure_events"):
        out.append("Recent price events: " + "; ".join(
            f"{e['event'].replace('_', ' ')} ({e['date']})" for e in pat.details["structure_events"]) + ".")

    out.append(_section(5, "Fundamental analysis"))
    f = a.results["fundamental"]
    if f.available:
        m = f.details["metrics"]
        out.append(f"Sector: {f.details['sector']}. Industry: {f.details['industry']}.")
        out.append(f"Trend in the company's results: {f.details['trend']}.")
        out.append(f"Revenue growth: {_metric(m, 'revenue_growth')}.")
        out.append(f"EPS growth ({TERM_MEANINGS['EPS']}): {_metric(m, 'eps_growth')}.")
        out.append(f"Net margin ({TERM_MEANINGS['Margin']}): {_metric(m, 'net_margin')}.")
        out.append(f"Operating margin (the share of revenue left after running costs): {_metric(m, 'operating_margin')}.")
        out.append(f"ROE (profit on {TERM_MEANINGS['Equity']}): {_metric(m, 'roe')}.")
        out.append(f"ROIC (profit on {TERM_MEANINGS['Invested capital']}): {_metric(m, 'roic')}.")
        out.append(f"Debt/equity (debt against {TERM_MEANINGS['Equity']}): {_metric(m, 'debt_to_equity')}.")
        out.append(f"Interest coverage (how many times profit pays the interest): {_metric(m, 'interest_coverage')}.")
        out.append(f"Free cash flow ({TERM_MEANINGS['Free cash flow']}): {_metric(m, 'free_cash_flow')}.")
        out.append(f"P/E ({TERM_MEANINGS['P/E']}): {_metric(m, 'pe')}.")
        out.append(f"P/B (price against what the company owns minus what it owes, per share): "
                   f"{_metric(m, 'price_to_book')}.")
        out.append(f"EV/EBITDA (company value including debt, against profit before interest, tax and depreciation): "
                   f"{_metric(m, 'ev_to_ebitda')}.")
        out.append(f"FCF yield (free cash flow against market value): {_metric(m, 'fcf_yield')}.")
        if f.details["unavailable"]:
            out.append(f"Could not work out from the data: {', '.join(f.details['unavailable'])}.")
    else:
        out.append(f"{UNAVAILABLE}: {f.error}")

    out.append(_section(6, "News & sentiment"))
    n = a.results["news_sentiment"]
    if n.available:
        out.append(f"News tone (sentiment): {n.state}. Weighted score {n.details['weighted_sentiment']}, from "
                   f"{n.details['unique_events']} separate events in {n.details['article_count']} articles. "
                   f"Last 7d {n.details['recent_sentiment']}, before that {n.details['prior_sentiment']}.")
        for ev in n.details["events"][:5]:
            out.append(f"  · {ev['timestamp'][:10]} [{ev['event_type']}/{ev['scope']}] {ev['headline']} "
                       f"— {ev['source']} (sentiment {ev['sentiment']:+.2f}, MODEL OUTPUT)")
    else:
        out.append(f"{UNAVAILABLE}: {n.error}")
    out.append("Upcoming events: " + ("; ".join(a.upcoming_events) or "none found in provider calendar"))

    out.append(_section(7, "Quantitative analysis"))
    r = a.results["risk"]
    if r.available:
        s = r.details
        b = s.get("benchmark") if isinstance(s.get("benchmark"), dict) else {}
        out.append(f"Risk level: {r.state}, measured over {s['period']}.")
        out.append(f"Volatility ({TERM_MEANINGS['Volatility']}): {s['annualized_vol_pct']}%.")
        out.append(f"Max drawdown ({TERM_MEANINGS['Max drawdown']}): {s['max_drawdown_pct']}%.")
        out.append(f"Current drawdown (how far it is below its peak now): {s['current_drawdown_pct']}%.")
        out.append(f"Sharpe ratio ({TERM_MEANINGS['Sharpe ratio']}): {s['sharpe']}.")
        out.append(f"Sortino ratio (the same, counting only falls): {s['sortino']}.")
        out.append(f"Beta ({TERM_MEANINGS['Beta']}): {b.get('beta', UNAVAILABLE)}.")
        out.append(f"Correlation to benchmark (how closely the two move together): "
                   f"{b.get('correlation', UNAVAILABLE)}.")
        out.append(f"1-day 95% VaR: {s['var_95_1d_pct']}%. On 95% of days the loss was smaller than this.")
        out.append(f"CVaR: {s['cvar_95_1d_pct']}%, the average loss on the days worse than that.")
        out.append("Returns: " + ", ".join(f"{k.replace('_pct', '')} {v}%" for k, v in s["returns"].items()
                                          if v is not None) + ".")
    else:
        out.append(f"{UNAVAILABLE}: {r.error}")
    reg = a.results["regime"]
    if reg.available:
        mk = reg.details.get("market")
        out.append(f"Regime (the stock's current phase): {reg.state}.")
        out.append("Pattern in its moves (trending: they carry on; mean-reverting: they reverse; "
                   "no significant serial dependence: no clear pattern): "
                   f"{reg.details['dependence'].get('state')}.")
        if isinstance(mk, dict):
            out.append(f"Whole market: {mk['state']}.")

    out.append(_section(8, "Forecast / scenarios"))
    fc = a.results["forecast"]
    if fc.available:
        for h, e in fc.details["horizons"].items():
            if "error" in e:
                out.append(f"{h}: {UNAVAILABLE} ({e['error']})")
                continue
            rng = e["expected_range"]
            out.append(f"{h}: chance of a rise {e['probability_up']:.2f} (model {e['model']}; skill "
                       f"{'demonstrated' if e['skill_demonstrated'] else 'NOT demonstrated'}). "
                       f"{rng['probability']:.0%} range {_money(rng['low'], ccy)}–{_money(rng['high'], ccy)}; in past "
                       f"tests the price ended inside it {rng['oos_coverage']:.0%} of the time. MODEL OUTPUT.")
    else:
        out.append(f"{UNAVAILABLE}: {fc.error}")
    out += _evidence_lines(a, "historical", 2)

    out.append(_section(9, "Risks"))
    out += [f"  • {x}" for x in d.risk_factors] or ["  • No specific risk flags beyond those above."]

    bulls = [e for r_ in a.results.values() if r_.available for e in r_.evidence if e.direction > 0]
    bears = [e for r_ in a.results.values() if r_.available for e in r_.evidence if e.direction < 0]
    out.append(_section(10, "Bull case"))
    out += [f"  + {e.claim}" for e in sorted(bulls, key=lambda e: -e.strength)[:5]] or ["  (no bullish evidence)"]
    out.append(_section(11, "Bear case"))
    out += [f"  - {e.claim}" for e in sorted(bears, key=lambda e: -e.strength)[:5]] or ["  (no bearish evidence)"]

    out.append(_section(12, "Final evidence-based summary"))
    out.append(_summary_sentence(a))
    out.append("Main doubts:")
    out += [f"  • {u}" for u in d.key_uncertainties] or ["  • none flagged"]
    out.append(f"\n{d.note}")
    if full:
        out.append("\nDomain states: " + "; ".join(
            f"{k}={v['state']} ({v['score']})" for k, v in d.domain_states.items()))
    return "\n".join(out)


def _summary_sentence(a: StockAnalysis) -> str:
    d = a.decision
    avail = [k for k, v in a.results.items() if v.available and v.confidence > 0]
    lean = {"BUY": "favourable", "WATCH": "leaning but not decisive", "HOLD": "balanced",
            "SELL": "unfavourable", "AVOID": "unfavourable for a new position",
            "INSUFFICIENT_EVIDENCE": "too thin to judge"}[d.decision_support_label]
    return (f"Overall, the evidence is {lean}: label {d.decision_support_label}, confidence {d.confidence:.2f}. "
            f"It draws on {len(avail)} areas with usable data: "
            f"{', '.join(DOMAIN_TITLES[k].lower() for k in avail)}."
            + (" These areas disagree." if d.conflict else ""))


DOMAIN_TITLES = {"technical": "Technicals", "candlestick": "Candlesticks", "pattern": "Chart patterns",
                 "fundamental": "Fundamentals", "news_sentiment": "News & sentiment", "risk": "Risk",
                 "regime": "Market regime", "forecast": "Forecast", "historical": "Similar past setups"}
# What a term means, in plain words. Replies that explain a term read it here; a drill-down explains the
# ones its own lines use.
TERM_MEANINGS = {
    "Swing structure": "the run of recent highs and lows",
    "RSI": "a 0 to 100 gauge of recent gains against recent losses",
    "MACD histogram": "positive when short-term momentum is rising",
    "Bollinger band": "a line that marks how far the price usually strays from its recent average",
    "ATRs": "steps of the stock's average daily price range",
    "Volatility": "how much the price swings in a year",
    "Drawdown": "a fall from a peak",
    "Max drawdown": "the biggest fall from a peak",
    "Loaded history": "the run of past prices we hold",
    "Beta": "how far the stock tends to move when the market moves; above 1 means it moves more",
    "Sharpe ratio": "return for the risk taken",
    "EPS": "profit per share",
    "Margin": "the share of revenue kept as profit",
    "Equity": "the owners' money in the business",
    "Invested capital": "all the money put in by owners and lenders",
    "EBIT": "profit before interest and tax",
    "Current ratio": "short-term assets against short-term debts",
    "P/E": "the price against a year's profit per share",
    "PEG": "the P/E divided by profit growth",
    "Free cash flow": "cash left after running and investing in the business",
    "Not a statistically significant difference": "one that could easily be chance",
}


def render_domain(a: StockAnalysis, domain: str) -> str:
    r = a.results.get(domain)
    if r is None:
        return f"No domain named {domain!r}. Available: {', '.join(a.results)}"
    score = "No score" if r.score is None else f"Score {r.score:+.2f}, {SCORE_SCALE}"
    head = [f"### {a.name} · {DOMAIN_TITLES.get(domain, domain.replace('_', ' '))}",
            f"Overall: **{r.state}**. {score}. Confidence {r.confidence:.2f} out of 1. "
            f"Data as of {str(r.as_of)[:10]}.", ""]
    if not r.available:
        return "\n".join(head + [f"{UNAVAILABLE}: {r.error}"])
    events = r.details.get("events") if domain == "news_sentiment" else None
    if events:
        # the headlines themselves, linked, rather than the evidence lines that summarise them
        ranked = sorted(events, key=lambda e: e.get("weight", 0), reverse=True)[:6]
        head += [f"- {_mark(e['sentiment'])} " + (f"[{e['headline']}]({e['url']})" if e.get("url", "").startswith("http")
                                                  else e["headline"]) + f" — {e['source']}, {str(e.get('timestamp', ''))[:10]}"
                 for e in ranked]
    else:
        head += [f"- {_mark(e.direction)} [{e.kind()}] {e.claim}" for e in r.evidence]
        shown = " ".join(e.claim for e in r.evidence)
        meanings = [f"{t}: {m}." for t, m in TERM_MEANINGS.items() if re.search(rf"\b{re.escape(t)}\b", shown, re.I)]
        head += ["", " ".join(meanings)] if meanings else []
    weight_note = ("This part did not count toward the overall label (confidence 0)."
                   if r.confidence == 0 else "")
    return "\n".join(head + (["", weight_note] if weight_note else []))


def render_report(a: StockAnalysis) -> str:
    """Full 18-section research report in Markdown."""
    d = a.decision
    f = a.results["fundamental"]
    profile = f.details.get("profile", {}) if f.available else {}
    sections = [
        f"# Research report: {a.name} ({a.provider_symbol})",
        f"*Analysis timestamp {a.analysis_timestamp}; price data as of {a.quality['last_date']}; "
        f"source {a.quality['source']}.*",
        "## 1. Executive summary", _summary_sentence(a),
        f"Label **{d.decision_support_label}**, score {d.score:+.2f}, confidence "
        f"{d.confidence:.2f}, uncertainty {d.uncertainty:.2f}.",
        "## 2. Company overview",
        str(profile.get("summary", UNAVAILABLE)) if profile else UNAVAILABLE,
        f"Sector: {profile.get('sector', UNAVAILABLE)}; industry: {profile.get('industry', UNAVAILABLE)}; "
        f"employees: {profile.get('employees', UNAVAILABLE)}.",
        "## 3. Current market context",
        "\n".join(_evidence_lines(a, "regime")),
        "## 4. Technical analysis", "\n".join(_evidence_lines(a, "technical", 20)),
        "## 5. Candlestick analysis", "\n".join(_evidence_lines(a, "candlestick", 10)),
        "## 6. Chart pattern analysis", "\n".join(_evidence_lines(a, "pattern", 10)),
        "## 7. Fundamental analysis", "\n".join(_evidence_lines(a, "fundamental", 20)),
        "## 8. Valuation",
        "\n".join(f"- {k}: {_metric(f.details['metrics'], k)}" for k in
                  ("pe", "forward_pe", "price_to_book", "ev_to_ebitda", "ev_to_sales", "peg",
                   "fcf_yield", "dividend_yield")) if f.available else UNAVAILABLE,
        "## 9. News and sentiment", "\n".join(_evidence_lines(a, "news_sentiment", 10)),
        "## 10. Quantitative risk analysis", "\n".join(_evidence_lines(a, "risk", 10)),
        "## 11. Forecast / scenarios",
        "\n".join(_evidence_lines(a, "forecast", 10) + _evidence_lines(a, "historical", 4)),
        "## 12. Portfolio context",
        "Analysed as an existing holding (negative evidence reads SELL)." if a.held else
        "Analysed as a potential new position (negative evidence reads AVOID).",
        "## 13. Bull case",
        "\n".join(f"- {e['claim']}" for e in (d.supporting_evidence if d.score >= 0 else d.contradicting_evidence)) or "- none",
        "## 14. Bear case",
        "\n".join(f"- {e['claim']}" for e in (d.contradicting_evidence if d.score >= 0 else d.supporting_evidence)) or "- none",
        "## 15. Key uncertainties", "\n".join(f"- {u}" for u in d.key_uncertainties) or "- none flagged",
        "## 16. Supporting evidence",
        "\n".join(f"- [{e['kind']}] {e['claim']} ({e['source']}, {e['as_of']})" for e in d.supporting_evidence) or "- none",
        "## 17. Contradicting evidence",
        "\n".join(f"- [{e['kind']}] {e['claim']} ({e['source']}, {e['as_of']})" for e in d.contradicting_evidence) or "- none",
        "## 18. Final decision-support summary",
        f"{d.decision_support_label}: {d.overall_state}. " + " ".join(d.gates_applied), f"_{d.note}_",
    ]
    return "\n\n".join(sections)


def render_compare(analyses: Sequence[StockAnalysis]) -> str:
    rows = [("Metric", *[a.symbol for a in analyses])]

    def m(a: StockAnalysis, key: str) -> str:
        f = a.results["fundamental"]
        if not f.available:
            return "n/a"
        v = f.details["metrics"].get(key, {}).get("value")
        return "n/a" if v is None else f"{v:,.1f}"

    def risk(a: StockAnalysis, key: str) -> str:
        r = a.results["risk"]
        return "n/a" if not r.available else str(r.details.get(key))

    rows += [
        ("Label", *[a.decision.decision_support_label for a in analyses]),
        ("Score, −1 to +1", *[f"{a.decision.score:+.2f}" for a in analyses]),
        ("Confidence, 0 to 1", *[f"{a.decision.confidence:.2f}" for a in analyses]),
        ("Technical", *[a.results["technical"].state for a in analyses]),
        ("Fundamental", *[a.results["fundamental"].state for a in analyses]),
        ("News", *[a.results["news_sentiment"].state for a in analyses]),
        ("Risk", *[a.results["risk"].state for a in analyses]),
        ("Revenue growth %", *[m(a, "revenue_growth") for a in analyses]),
        ("Net margin %", *[m(a, "net_margin") for a in analyses]),
        (f"ROE % (profit on {TERM_MEANINGS['Equity']})", *[m(a, "roe") for a in analyses]),
        (f"P/E ({TERM_MEANINGS['P/E']})", *[m(a, "pe") for a in analyses]),
        (f"Volatility % ({TERM_MEANINGS['Volatility']})", *[risk(a, "annualized_vol_pct") for a in analyses]),
        (f"Max drawdown % ({TERM_MEANINGS['Max drawdown']})", *[risk(a, "max_drawdown_pct") for a in analyses]),
        (f"Sharpe ({TERM_MEANINGS['Sharpe ratio']})", *[risk(a, "sharpe") for a in analyses]),
        ("1-year return %", *[str(a.results["risk"].details["returns"].get("1y_pct"))
                              if a.results["risk"].available else "n/a" for a in analyses]),
    ]
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines = [" | ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    ts = max(a.analysis_timestamp for a in analyses)
    return f"Comparison (analysis timestamp {ts})\n" + "\n".join(lines)


def render_portfolio(rep: Dict[str, Any], ccy: str = "INR") -> str:
    out = [f"Portfolio as of {rep['as_of']}",
           f"Total value {_money(rep['total_value'], ccy)} (invested {_money(rep['invested_value'], ccy)}, "
           f"cash {_money(rep['cash'], ccy)}, utilization {rep['cash_utilization_pct']}%)",
           f"Unrealized P/L {_money(rep['unrealized_pl'], ccy)} ({rep['unrealized_pl_pct']}%) | "
           f"Realized P/L {_money(rep['realized_pl'], ccy)}"]
    out.append("\nHoldings:")
    for h in rep["holdings"]:
        out.append(f"  {h['symbol']:<12} {h['quantity']:>8g} @ {h['avg_cost']:,.2f} → {h['price']:,.2f} "
                   f"| value {h['value']:,.0f} ({h['weight_pct']}%) | P/L {h['unrealized_pl']:,.0f} "
                   f"({h['unrealized_pl_pct']}%) | {h['sector']}")
    out.append("\nSector exposure: " + ", ".join(f"{k} {v}%" for k, v in rep["sector_exposure_pct"].items()))
    c = rep["concentration"]
    out.append(f"Concentration: HHI {c['hhi']}, effective holdings {c['effective_holdings']}, "
               f"top holding {c['top_holding_pct_of_invested']}% of invested")
    r = rep["risk"]
    if "portfolio_vol_pct" in r:
        out.append(f"Risk: volatility {r['portfolio_vol_pct']}%, max drawdown {r['max_drawdown_pct']}%, "
                   f"1-day 95% VaR {r['var_95_1d_pct']}% ({_money(r['var_95_1d_value'], ccy)}), "
                   f"beta {r.get('beta', UNAVAILABLE)}, diversification ratio {r['diversification_ratio']}")
        out.append("Risk contribution: " + ", ".join(
            f"{k} {v}%" for k, v in sorted(r["risk_contribution_pct"].items(), key=lambda kv: -kv[1])))
    elif r:
        out.append(f"Risk: {r.get('note')}")
    w = rep["week_change"]
    out.append(f"\nLast 5 sessions: {_money(w['total_change_value'], ccy)} ({w['total_change_pct']}%)"
               + (f"; best {w['biggest_contributor']}, worst {w['biggest_detractor']}"
                  if "biggest_contributor" in w else ""))
    if rep["flags"]:
        out.append("\nFlags:")
        out += [f"  ⚑ {f_}" for f_ in rep["flags"]]
    out.append("\nAssumptions: " + " ".join(rep["assumptions"]))
    return "\n".join(out)


def render_screen(res: Dict[str, Any], ccy: str = "INR") -> str:
    out = [f"Research candidates for {_money(res['budget'], ccy)} ({res['risk_profile']} profile, "
           f"{res['horizon']} horizon) — {res['timestamp']}",
           f"Universe: {res['universe_size']} stocks; {len(res['excluded'])} excluded by hard constraints.",
           "These are candidates for further research, not recommendations.\n"]
    for i, c in enumerate(res["candidates"], 1):
        out.append(f"{i}. {c['symbol']} — {c['label']} (score {c['score']:+.2f}, confidence "
                   f"{c['confidence']:.2f}) | price {_money(c['price'], ccy)} | up to {c['max_shares']} "
                   f"share{'' if c['max_shares'] == 1 else 's'} within the {c['position_limit_pct']}% position limit")
        out += [f"     · {why}" for why in c["reasons"]]
    if res["excluded"]:
        counts: Dict[str, int] = {}
        for e in res["excluded"]:
            counts[e["reason"].split(":")[0]] = counts.get(e["reason"].split(":")[0], 0) + 1
        out.append("\nExcluded: " + ", ".join(f"{k} ({v})" for k, v in counts.items()))
    out += ["\nAssumptions:"] + [f"  - {x}" for x in res["assumptions"]]
    return "\n".join(out)


def render_what_if(res: Dict[str, Any], ccy: str = "INR") -> str:
    cur = res["current"]
    out = [f"Current: volatility {cur['portfolio_vol_pct']}%, max drawdown {cur['max_drawdown_pct']}%, "
           f"effective holdings {cur['effective_holdings']}"]
    for name, s in res["scenarios"].items():
        out.append(f"{name}: buys {s['shares_bought']} for {_money(s['spent'], ccy)} → volatility "
                   f"{s['portfolio_vol_pct']}%, max drawdown {s['max_drawdown_pct']}%, CVaR "
                   f"{s['cvar_95_1d_pct']}%, effective holdings {s['effective_holdings']}"
                   + (f"; flags: {' | '.join(s['flags'])}" if s["flags"] else ""))
    out.append("Historical backcast with current prices; not a forecast of future risk.")
    return "\n".join(out)
