"""Conversation orchestrator: intent -> tools -> evidence -> answer.

Tools are read-only functions over the analysis service; none can place
orders or change external state. Three generative modes:
  off      deterministic answers only (always faithful)
  narrate  deterministic answer rewritten by Claude, then number-verified
  agent    Claude plans tool calls itself; its answer is number-verified
If the LLM is unreachable, refuses, or fails verification, the reply is the
deterministic answer plus a sentence saying which of those happened.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.advisory.calculators import performance
from backend.advisory import report as R
from backend.user.portfolio.alerts import evaluate as evaluate_alert
from backend.config import RISK_PROFILES, canonical_symbol
from backend.database.sources.provider import DataUnavailable, Provider
from backend.database.sources.quality import DataQualityError
from backend.advisory.fusion.evidence import utcnow_iso
from backend.knowledge_base.lessons import knowledge_base
from backend.api.nlu import Intent, classify
from backend.advisory.statistics.model_registry import ModelRegistry
from backend.user.portfolio.portfolio import Position, Transaction, analyze as analyze_portfolio, build_positions, what_if
from backend.advisory.calculators.screener import CRITERIA, load_universe, research_query, screen
from backend.advisory.service import AnalysisService, StockAnalysis
from backend.database.storage import Store

log = logging.getLogger("stockintel.orchestrator")
SESSION_REUSE_S = 900
SYMBOL_RE = re.compile(r"^\^?[A-Z0-9&\-]{1,20}(\.(NS|BO))?$")
DOMAINS = ("technical", "candlestick", "pattern", "fundamental", "news_sentiment", "risk",
           "regime", "forecast", "historical")


@dataclass
class Session:
    session_id: str
    symbols: List[Tuple[str, str]] = field(default_factory=list)
    analyses: Dict[str, Tuple[float, StockAnalysis]] = field(default_factory=dict)
    budget: Optional[float] = None
    horizon: str = "medium"
    risk_profile: str = "moderate"
    history: List[Dict[str, Any]] = field(default_factory=list)   # LLM agent transcript

    @property
    def current(self) -> Optional[Tuple[str, str]]:
        return self.symbols[-1] if self.symbols else None


@dataclass
class Reply:
    text: str
    intent: str
    data: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    llm: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "intent": self.intent, "data": self.data,
                "tools_used": self.tools_used, "llm": self.llm, "timestamp": self.timestamp}


class Orchestrator:
    def __init__(self, provider: Provider, store: Store, portfolio_name: str = "default",
                 llm_mode: str = "off", use_rss_news: bool = True, llm_client: Any = None):
        if llm_mode not in ("off", "narrate", "agent"):
            raise ValueError(f"llm_mode must be off, narrate or agent; got {llm_mode!r}")
        self.provider, self.store = provider, store
        self.portfolio_name = portfolio_name
        self.llm_mode = llm_mode
        self.use_rss_news = use_rss_news
        self._services: Dict[str, AnalysisService] = {}
        self.sessions: Dict[str, Session] = {}
        import threading
        self._locks: Dict[str, Any] = {}
        self._locks_guard = threading.Lock()
        self.universe = load_universe()
        self.known = set(self.universe["symbol"])
        self.research_phrases = [p for phrases, _, _ in CRITERIA.values() for p in phrases]
        self.llm_client = llm_client
        if llm_mode != "off" and llm_client is None:
            from backend.models.external.llm import LLMClient
            self.llm_client = LLMClient()

    # --- plumbing -----------------------------------------------------------------
    def service(self, market: str = "NSE") -> AnalysisService:
        if market not in self._services:
            deployed = ModelRegistry(self.store).deployed("forecast")
            models = (deployed["hyperparameters"]["models"] if deployed
                      else ("climatology", "logistic"))
            priors = {f: self.store.pattern_priors(f) for f in ("candlestick", "chart")}
            self._services[market] = AnalysisService(self.provider, market, self.use_rss_news,
                                                     forecast_models=models, pattern_priors=priors)
        return self._services[market]

    def session(self, session_id: str) -> Session:
        return self.sessions.setdefault(session_id, Session(session_id))

    def save_portfolio(self, cash: float, risk_profile: str, horizon: str,
                       transactions: List[Dict[str, Any]]) -> int:
        """Validate (FIFO, no oversells) before anything is written."""
        if risk_profile not in RISK_PROFILES:
            raise ValueError(f"risk_profile must be one of {list(RISK_PROFILES)}, got {risk_profile!r}")
        txs = [Transaction(**{k: t[k] for k in ("symbol", "side", "quantity", "price", "trade_date")},
                           fees=t.get("fees", 0.0), sector=t.get("sector")) for t in transactions]
        build_positions(txs)
        self.store.save_portfolio(self.portfolio_name, cash, risk_profile, horizon,
                                  [t.__dict__ for t in txs])
        return len(txs)

    def holdings(self) -> Tuple[Dict[str, Position], float, Dict[str, Any]]:
        p = self.store.portfolio(self.portfolio_name)
        if p is None:
            return {}, 0.0, {}
        txs = [Transaction(**t) for t in p["transactions"]]
        return build_positions(txs), float(p["cash"]), p

    def analysis(self, s: Session, symbol: str, market: str = "NSE", horizon: Optional[str] = None) -> StockAnalysis:
        if not SYMBOL_RE.match(symbol.upper()):
            raise ValueError(f"{symbol!r} is not a valid ticker symbol")
        positions, _, _ = self.holdings()
        pos = positions.get(canonical_symbol(symbol, market))
        held = pos is not None and pos.quantity > 0
        key = f"{market}:{symbol.upper()}:{horizon or s.horizon}:{held}"
        with self._key_lock(key):
            hit = s.analyses.get(key)
            if hit and time.time() - hit[0] < SESSION_REUSE_S:
                return hit[1]
            return self._compute_analysis(s, key, symbol, market, horizon, held)

    def _key_lock(self, key: str):
        # Parallel requests for the same analysis wait for one computation.
        import threading
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _compute_analysis(self, s: Session, key: str, symbol: str, market: str,
                          horizon: Optional[str], held: bool) -> StockAnalysis:
        a = self.service(market).analyze(symbol, held=held, horizon=horizon or s.horizon)
        s.analyses[key] = (time.time(), a)
        previous = self.store.analyses(a.symbol, limit=1)
        alert = evaluate_alert(a, previous[0] if previous else None)
        if alert:
            self.store.save_alert(alert.symbol, alert.kind, alert.severity, alert.message, alert.reasons)
        self.store.save_analysis(a.symbol, a.to_dict())
        self._log_predictions(a, market)
        return a

    def _log_predictions(self, a: StockAnalysis, market: str) -> None:
        """Persist the bars and forecasts so every prediction can be re-run and scored later."""
        fc = a.results["forecast"]
        if not fc.available:
            return
        df = self.service(market).history(a.symbol)
        self.store.save_ohlcv(a.provider_symbol, df, a.quality["source"])
        deployed = ModelRegistry(self.store).deployed("forecast")
        version = deployed["version"] if deployed else "default"
        for h, entry in fc.details["horizons"].items():
            if "probability_up" not in entry:
                continue
            inputs = {"symbol": a.provider_symbol, "last_bar": a.quality["last_date"],
                      "rows": a.quality["rows"], "models": list(self.service(market).forecast_models)}
            self.store.log_prediction(entry["model"], version, a.provider_symbol, int(h.rstrip("d")),
                                      a.quality["last_date"], inputs, entry["probability_up"])

    def _target(self, s: Session, intent: Intent) -> Tuple[str, str]:
        if intent.symbols:
            return intent.symbols[0]
        if s.current:
            return s.current
        raise ValueError("Which stock? Name a ticker, e.g. 'Analyze RELIANCE'.")

    # --- entry point --------------------------------------------------------------
    def handle(self, message: str, session_id: str = "default") -> Reply:
        s = self.session(session_id)
        intent = classify(message, sorted(self.known), s.current is not None, self.research_phrases)
        if intent.horizon:
            s.horizon = intent.horizon
        if intent.risk_profile:
            s.risk_profile = intent.risk_profile
        if self.llm_mode == "agent":
            return self._agent_reply(s, message, intent)
        reply = self._deterministic(s, intent)
        if reply.data.get("no_answer"):          # our patterns or the reference index had nothing; let the LM place it
            routed = self._routed(message, intent)
            if routed.name != intent.name:
                reply = self._deterministic(s, routed)
        if self.llm_mode == "narrate" and reply.data.get("payload") is not None:
            reply = self._narrate(message, reply)
        return reply

    def _routed(self, message: str, intent: Intent) -> Intent:
        """Our patterns did not recognise the question; let the language model pick one of our handlers."""
        from backend.api import intent_lm
        from backend.api.nlu import ROUTABLE
        from backend.models.external.llm import LLMUnavailable
        try:
            name = intent_lm.choose(message, ROUTABLE, self.hub.writer())
        except (LLMUnavailable, DataUnavailable) as exc:
            log.info("no language model to route %r: %s", message, exc)
            return intent
        return Intent(**{**intent.__dict__, "name": name}) if name else intent

    def _deterministic(self, s: Session, intent: Intent) -> Reply:
        handler = getattr(self, f"_h_{intent.name}")
        try:
            reply = handler(s, intent)
        except (DataUnavailable, DataQualityError, ValueError) as exc:
            return Reply(text=f"I couldn't complete that: {exc}", intent=intent.name,
                         data={"error": str(exc)})
        subject = s.current
        for sym in intent.symbols:
            if sym in s.symbols:
                s.symbols.remove(sym)
            s.symbols.append(sym)
        if intent.name == "compare" and subject is not None:
            s.symbols.remove(subject)
            s.symbols.append(subject)
        return reply

    # --- handlers -------------------------------------------------------------------
    def _h_analyze(self, s: Session, i: Intent) -> Reply:
        sym, mkt = self._target(s, i)
        a = self.analysis(s, sym, mkt)
        return Reply(R.render_analysis(a), "analyze", {"analysis": a.to_dict(), "payload": a.to_dict()},
                     ["analyze_stock"])

    def _h_report(self, s: Session, i: Intent) -> Reply:
        sym, mkt = self._target(s, i)
        a = self.analysis(s, sym, mkt)
        body = R.render_report(a)
        rid = self.store.save_report(a.symbol, body)
        return Reply(body, "report", {"report_id": rid}, ["analyze_stock", "save_report"])

    def _h_domain(self, s: Session, i: Intent) -> Reply:
        sym, mkt = self._target(s, i)
        a = self.analysis(s, sym, mkt)
        dom = i.domain or "technical"
        payload = {"symbol": a.symbol, "domain": dom, "result": a.results[dom].to_dict(),
                   "decision_label": a.decision.decision_support_label}
        return Reply(R.render_domain(a, dom), "domain", {"payload": payload}, ["domain_evidence"])

    def _h_explain_decision(self, s: Session, i: Intent) -> Reply:
        sym, mkt = self._target(s, i)
        a = self.analysis(s, sym, mkt)
        d = a.decision
        lines = [f"Why {a.symbol} is classified {d.decision_support_label} (score {d.score:+.2f}, "
                 f"confidence {d.confidence:.2f}, uncertainty {d.uncertainty:.2f}):",
                 "\nHow each domain voted:"]
        for dom, st in d.domain_states.items():
            lines.append(f"  {dom:<15} {st['state']:<32} score {st['score']}  confidence {st['confidence']}")
        lines.append("\nSupporting evidence:")
        lines += [f"  + [{e['kind']}] {e['claim']}" for e in d.supporting_evidence] or ["  none"]
        lines.append("Contradicting evidence:")
        lines += [f"  - [{e['kind']}] {e['claim']}" for e in d.contradicting_evidence] or ["  none"]
        if d.gates_applied:
            lines.append("Gates: " + "; ".join(d.gates_applied))
        lines.append("Key uncertainties: " + ("; ".join(d.key_uncertainties) or "none"))
        lines.append("Ask 'why technical?', 'why fundamentals?' etc. to see any domain's raw evidence.")
        return Reply("\n".join(lines), "explain_decision", {"payload": {"decision": d.to_dict()}},
                     ["analyze_stock"])

    def _h_compare(self, s: Session, i: Intent) -> Reply:
        syms = list(i.symbols)
        if (len(syms) < 2 or re.search(r"\b(it|this)\b", i.raw.lower())) and s.current and s.current not in syms:
            syms.insert(0, s.current)
        if len(syms) < 2:
            raise ValueError("Name at least two stocks to compare, e.g. 'compare TCS vs INFY'.")
        analyses = [self.analysis(s, sym, mkt) for sym, mkt in syms[:6]]
        payload = {"comparison": [{"symbol": a.symbol, "decision": a.decision.to_dict(),
                                   "states": {k: v.state for k, v in a.results.items()}} for a in analyses]}
        return Reply(R.render_compare(analyses), "compare", {"payload": payload}, ["compare_stocks"])

    def _h_screen(self, s: Session, i: Intent) -> Reply:
        s.budget = i.amount
        positions, _, _ = self.holdings()
        res = screen(self.service(), i.amount, risk_profile=s.risk_profile, horizon=s.horizon,
                     positions=positions, include_sectors=i.include_sectors,
                     exclude_sectors=i.exclude_sectors)
        return Reply(R.render_screen(res), "screen", {"payload": {k: v for k, v in res.items()
                                                                  if k != "also_scored"}},
                     ["screen_budget"])

    def _h_research_query(self, s: Session, i: Intent) -> Reply:
        res = research_query(self.service(), i.raw)
        if res.get("error"):
            return Reply(res["error"], "research_query", {"payload": res})
        lines = ["Stocks matching: " + "; ".join(res["criteria_applied"].values())
                 + f" (universe {res['universe_size']}, {res['timestamp']})"]
        for m in res["matches"]:
            lines.append(f"  {m['symbol']:<12} {m['label']:<8} score {m['score']:+.2f} | technical "
                         f"{m['technical']}, fundamental {m['fundamental']}, risk {m['risk']}")
        if not res["matches"]:
            lines.append("  No stock in the universe met every criterion.")
        lines += [f"Not supported: {u}" for u in res["unsupported"]]
        return Reply("\n".join(lines), "research_query", {"payload": res}, ["research_query"])

    def _h_budget_analysis(self, s: Session, i: Intent) -> Reply:
        sym, mkt = self._target(s, i)
        s.budget = i.amount
        a = self.analysis(s, sym, mkt)
        prof = RISK_PROFILES[s.risk_profile]
        limit = i.amount * prof.max_position_weight
        shares = int(limit // a.price) if a.price else 0
        risk = a.results["risk"]
        lines = [f"{a.symbol} with a budget of {R._money(i.amount, a.currency)} ({prof.name} profile):",
                 f"- Last close {R._money(a.price, a.currency)}; the {prof.max_position_weight:.0%} position "
                 f"limit is {R._money(limit, a.currency)}, i.e. up to {shares} shares "
                 f"({R._money(shares * a.price, a.currency)})."]
        if shares == 0:
            lines.append("- One share exceeds the position limit for this budget; the stock does not fit "
                         "a diversified allocation at this size.")
        if risk.available:
            var = risk.details["var_95_1d_pct"]
            lines.append(f"- Historical 1-day 95% VaR is {var}%: on 1 day in 20 the position has lost "
                         f"more than {R._money(abs(var) / 100 * shares * a.price, a.currency)}.")
            vol = risk.details["annualized_vol_pct"]
            if vol / 100 > prof.max_holding_vol:
                lines.append(f"- Volatility {vol}% is above the {prof.max_holding_vol:.0%} ceiling for "
                             f"a {prof.name} profile.")
        lines.append(f"- Evidence classification remains {a.decision.decision_support_label}; the budget "
                     "changes position size, not the evidence.")
        return Reply("\n".join(lines), "budget_analysis", {"payload": {"analysis_decision": a.decision.to_dict(),
                     "budget": i.amount, "max_shares": shares, "position_limit": limit}},
                     ["analyze_stock"])

    def _h_explain_move(self, s: Session, i: Intent) -> Reply:
        sym, mkt = self._target(s, i)
        a = self.analysis(s, sym, mkt)
        days = i.days or 5
        svc = self.service(mkt)
        df = svc.history(sym)
        if len(df) <= days:
            raise DataUnavailable(f"only {len(df)} bars of history")
        move = float(df["close"].iloc[-1] / df["close"].iloc[-1 - days] - 1) * 100
        lines = [f"{a.symbol} moved {move:+.2f}% over the last {days} session(s) "
                 f"(to {R._money(a.price, a.currency)}, {a.quality['last_date']})."]
        bench = svc.benchmark()
        risk = a.results["risk"]
        beta = (risk.details.get("benchmark") or {}).get("beta") if risk.available and isinstance(
            risk.details.get("benchmark"), dict) else None
        if bench is not None and beta is not None and len(bench) > days:
            bmove = float(bench["close"].iloc[-1] / bench["close"].iloc[-1 - days] - 1) * 100
            market_part = beta * bmove
            lines.append(f"- Market: the benchmark moved {bmove:+.2f}%; at beta {beta:.2f} that explains "
                         f"about {market_part:+.2f}%, leaving {move - market_part:+.2f}% specific to the stock.")
        else:
            lines.append("- Market comparison: Data unavailable (no benchmark or beta).")
        news = a.results["news_sentiment"]
        if news.available:
            cutoff = str(df.index[-1 - days].date())
            recent = [e for e in news.details["events"] if (e["timestamp"] or "")[:10] >= cutoff]
            if recent:
                lines.append("- News in the window (sentiment is a MODEL OUTPUT):")
                lines += [f"    {e['timestamp'][:10]} [{e['event_type']}] {e['headline']} — {e['source']} "
                          f"({e['sentiment']:+.2f})" for e in recent[:6]]
            else:
                lines.append("- No news articles found in the window; the move may be flow- or market-driven.")
        else:
            lines.append(f"- News: Data unavailable ({news.error}).")
        tech = a.results["technical"]
        if tech.available:
            lines.append(f"- Technical context: {tech.state}; breakout status {tech.details.get('breakout_status')}; "
                         f"volume {tech.details.get('volume_vs_20d_avg', 'n/a')}x its 20-day average.")
        for e in a.results["pattern"].details.get("structure_events", []) if a.results["pattern"].available else []:
            lines.append(f"- Structure: {e['event'].replace('_', ' ')} ({e['date']}).")
        lines.append("This attribution is descriptive; it does not establish cause.")
        return Reply("\n".join(lines), "explain_move", {"payload": {"move_pct": move, "days": days,
                     "analysis": a.to_dict()}}, ["analyze_stock", "benchmark"])

    def _h_what_changed(self, s: Session, i: Intent) -> Reply:
        sym, mkt = self._target(s, i)
        previous = self.store.analyses(sym.upper(), limit=5)
        a = self.analysis(s, sym, mkt)
        earlier = next((p for p in previous if p["analysis_timestamp"] < a.analysis_timestamp), None)
        lines = [f"What changed for {a.symbol}:"]
        if earlier:
            pd_, cd = earlier["decision"], a.decision
            lines.append(f"- Since the stored analysis of {earlier['analysis_timestamp']}: classification "
                         f"{pd_['decision_support_label']} -> {cd.decision_support_label}, score "
                         f"{pd_['score']:+.2f} -> {cd.score:+.2f}; price {R._money(earlier['price'], a.currency)} -> "
                         f"{R._money(a.price, a.currency)}.")
            for dom in DOMAINS:
                before = earlier["results"].get(dom, {}).get("state")
                now = a.results[dom].state
                if before != now:
                    lines.append(f"  · {dom}: {before} -> {now}")
            seen = {e["headline"] for e in (earlier["results"].get("news_sentiment", {}).get("details") or {}).get("events", [])}
        else:
            lines.append("- No earlier stored analysis to compare against; showing recent activity instead.")
            seen = set()
        r = a.results["risk"]
        if r.available:
            ret = r.details["returns"]
            lines.append(f"- Returns: 1 week {ret['1w_pct']}%, 1 month {ret['1m_pct']}%.")
        news = a.results["news_sentiment"]
        if news.available:
            fresh = [e for e in news.details["events"] if e["headline"] not in seen][:6]
            lines.append("- New events:" if fresh else "- No new news events.")
            lines += [f"    {e['timestamp'][:10]} [{e['event_type']}] {e['headline']} ({e['source']})" for e in fresh]
        f = a.results["fundamental"]
        if f.available:
            lines.append(f"- Latest reported fundamentals: {f.state}, trend {f.details['trend']} (period {f.as_of}).")
        return Reply("\n".join(lines), "what_changed", {"payload": {"analysis": a.to_dict(),
                     "previous": earlier and earlier["decision"]}}, ["analyze_stock", "stored_analyses"])

    def _h_knowledge(self, s: Session, i: Intent) -> Reply:
        hits = knowledge_base().search(i.raw)
        if not hits:
            return Reply("I don't have a reference entry for that. I can explain indicators, ratios, risk "
                         "metrics, patterns, methodology and Indian market basics.", "knowledge",
                         {"no_answer": True})
        best = hits[0][0]
        body = re.sub(r"(?<!\n)\n(?!\n)", " ", best.text)   # unwrap hard-wrapped source lines
        text = f"{best.title}\n\n{body}\n\n(Static knowledge: {best.citation}. Contains no live market data.)"
        if len(hits) > 1:
            text += "\nRelated: " + ", ".join(c.title for c, _ in hits[1:])
        return Reply(text, "knowledge", {"payload": {"passages": [{"title": c.title, "text": c.text,
                     "citation": c.citation} for c, _ in hits]}}, ["explain_concept"])

    def _h_concept_with_data(self, s: Session, i: Intent) -> Reply:
        concept = self._h_knowledge(s, i)
        dom_intent = Intent(**{**i.__dict__, "domain": i.domain or "fundamental"})
        live = self._h_domain(s, dom_intent)
        return Reply(concept.text + "\n\nApplied to the current stock:\n" + live.text, "concept_with_data",
                     {"payload": {"knowledge": concept.data.get("payload"), "live": live.data["payload"]}},
                     concept.tools_used + live.tools_used)

    # portfolio ---------------------------------------------------------------------
    def portfolio_report(self, s: Session):
        positions, cash, meta = self.holdings()
        if not positions:
            raise ValueError(f"No portfolio named {self.portfolio_name!r} is stored. Add holdings with "
                             "`stockintel portfolio import holdings.json` or POST /portfolio.")
        svc = self.service()
        prices = {sym: svc.history(sym)["close"] for sym, p in positions.items() if p.quantity > 0}
        bench = svc.benchmark()
        rep = analyze_portfolio(positions, cash, prices, meta.get("risk_profile", s.risk_profile),
                                bench=None if bench is None else bench["close"], rf=svc.market.risk_free_rate)
        return rep, positions, cash, prices, meta

    def _h_portfolio_summary(self, s: Session, i: Intent) -> Reply:
        rep, *_ = self.portfolio_report(s)
        d = rep.to_dict()
        self.store.save_snapshot(self.portfolio_name, rep.total_value, d)
        return Reply(R.render_portfolio(d), i.name, {"payload": d}, ["portfolio_summary"])

    _h_portfolio_risk = _h_portfolio_exposure = _h_portfolio_changes = _h_portfolio_cash = \
        _h_portfolio_volatility = _h_portfolio_summary

    def _per_holding(self, s: Session, i: Intent, domain: str, title: str,
                     key: Callable[[StockAnalysis], Any]) -> Reply:
        rep, positions, *_ = self.portfolio_report(s)
        rows = []
        for h in rep.holdings:
            a = self.analysis(s, h["symbol"])
            rows.append((h["symbol"], a.results[domain], key(a)))
        rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))
        lines = [title]
        for sym, res, k in rows:
            lines.append(f"  {sym:<12} {res.state:<28} " + (f"{k:+.2f}" if isinstance(k, (int, float)) else "n/a")
                         + ("" if res.available else f"  ({res.error})"))
        return Reply("\n".join(lines), i.name, {"payload": [{"symbol": r[0], "state": r[1].state,
                                                             "value": r[2]} for r in rows]},
                     ["portfolio_summary", "analyze_stock"])

    def _h_portfolio_momentum(self, s: Session, i: Intent) -> Reply:
        return self._per_holding(s, i, "technical", "Holdings ranked by 20-day return (technical state):",
                                 lambda a: a.results["technical"].details.get("return_20d_pct")
                                 if a.results["technical"].available else None)

    def _h_portfolio_fundamentals(self, s: Session, i: Intent) -> Reply:
        reply = self._per_holding(s, i, "fundamental", "Holdings by fundamental score (lowest = weakest):",
                                  lambda a: a.results["fundamental"].score)
        worsening = [r["symbol"] for r in reply.data["payload"] if r["value"] is not None and r["value"] < 0]
        reply.text += "\nWeak or deteriorating: " + (", ".join(worsening) or "none")
        return reply

    def _h_portfolio_sentiment(self, s: Session, i: Intent) -> Reply:
        return self._per_holding(s, i, "news_sentiment", "Holdings by weighted news sentiment (MODEL OUTPUT):",
                                 lambda a: a.results["news_sentiment"].score)

    def _h_portfolio_what_if(self, s: Session, i: Intent) -> Reply:
        rep, positions, cash, prices, meta = self.portfolio_report(s)
        amount = i.amount
        svc = self.service()
        candidates = [sym for sym, _ in i.symbols] or [h["symbol"] for h in rep.holdings[:3]]
        for sym in candidates:
            prices.setdefault(sym, svc.history(sym)["close"])
        scenarios = {f"all in {c}": {c: amount} for c in candidates}
        if len(candidates) > 1:
            scenarios["split equally"] = {c: amount / len(candidates) for c in candidates}
        res = what_if(positions, cash, prices, scenarios, meta.get("risk_profile", s.risk_profile))
        return Reply(R.render_what_if(res), i.name, {"payload": res}, ["portfolio_what_if"])

    def _signal_plan(self, s: Session, i: Intent):
        from backend.database.sources.panel import index_constituents, universe_panel
        from backend.advisory.signals.advisor import Holding, plan
        panel = universe_panel("nifty500", "10y")
        bench = self.service().history("^NSEI")["close"]
        positions, cash, meta = self.holdings()
        first_buy = {}
        for t in (meta or {}).get("transactions", []):
            if t["side"] == "BUY":
                first_buy.setdefault(t["symbol"], t["trade_date"])
        holdings = [Holding(k, p.quantity, p.avg_cost, first_buy.get(k)) for k, p in positions.items()
                    if p.quantity > 0]
        capital = i.amount if i.amount else cash
        if not capital and not holdings:
            raise ValueError("Tell me how much you want to invest (e.g. 'what should I buy with ₹1 lakh?') "
                             "or import your portfolio first.")
        t = i.raw.lower()
        strategy = "lowvol" if re.search(r"low[- ]?vol|safe|defensive stocks|stable", t) else \
            "blend" if re.search(r"blend|mix|balanced portfolio", t) else "momentum"
        risk = {"conservative": "defensive", "aggressive": "full"}.get(s.risk_profile, "balanced")
        from backend.database.sources import insider
        from backend.database.sources.provider import DataUnavailable
        try:
            trades = insider.load()
        except DataUnavailable:
            trades = None
        p = plan(panel, bench, float(capital or 0.0) or 1.0, holdings, strategy=strategy, risk_mode=risk,
                 universe=list(index_constituents("nifty200")["symbol"]), insider_trades=trades,
                 measured=performance.momentum_evidence(performance.track_record()))
        return p

    def _h_signals(self, s: Session, i: Intent) -> Reply:
        from backend.api.jsonutil import sanitize
        from backend.advisory.signals.advisor import render
        p = self._signal_plan(s, i)
        payload = sanitize(p.to_dict())
        self.store.save_plan(payload)
        return Reply(render(p), "signals", {"payload": payload}, ["signal_plan"])

    def _h_market_regime(self, s: Session, i: Intent) -> Reply:
        from backend.advisory.signals.advisor import CRASH_EXPOSURE, market_state
        bench = self.service().history("^NSEI")["close"]
        m = market_state(bench, bench.index[-1])
        lines = [f"Nifty market state as of {bench.index[-1].date()}:",
                 f"- Trend: {m['trend'].upper()} — month-end {m['month_end_close']:,.0f} vs 10-month average "
                 f"{m['sma_10m']:,.0f} (Faber rule: invested above, cash below).",
                 f"- Volatility: {m['ewma_vol_pct']}% (EWMA); a 15% volatility target suggests "
                 f"{m['vol_target_exposure']:.0%} exposure.",
                 f"- 24-month return {m['return_24m_pct']}%; volatility percentile {m['vol_percentile']}; "
                 f"momentum crash risk: {'YES' if m['momentum_crash_risk'] else 'no'}."]
        defensive = (1.0 if m["trend"] == "up" else 0.0) * m["vol_target_exposure"]
        lines.append(f"- Suggested equity exposure: defensive {defensive:.0%}, balanced "
                     f"{CRASH_EXPOSURE if m['momentum_crash_risk'] else 1:.0%}, full 100%.")
        rec = performance.track_record()
        cut = (f"cut our NSE momentum backtest's worst fall from {performance.live_edge(rec)['max_dd']:.0f}% to "
               f"{performance.live_edge(rec)['overlay_max_dd']:.0f}%" if rec else "is measured once `stockintel build-performance` has run")
        lines.append("Evidence: the 10-month trend filter halved the worst drawdown on US stocks over a century "
                     f"(Faber); combined with a volatility target it {cut}, but it also "
                     "lowered returns and whipsaws in sideways markets. It is a risk control, not a forecast.")
        return Reply("\n".join(lines), "market_regime", {"payload": m}, ["market_state"])

    def _h_stops(self, s: Session, i: Intent) -> Reply:
        from backend.advisory.signals.advisor import STOP_PCT
        positions, _, _ = self.holdings()
        if not positions:
            raise ValueError("No stored portfolio to check stops on.")
        planned = {a["symbol"]: a["stop_level"] for a in (self.store.latest_plan() or {}).get("actions", [])
                   if a.get("stop_level")}
        lines = [f"Stop check ({STOP_PCT:.0%} stops):"]
        for sym, p in positions.items():
            if p.quantity <= 0:
                continue
            df = self.service().history(sym)
            level = planned.get(sym, round(p.avg_cost * (1 - STOP_PCT), 2))
            low, px = float(df["low"].iloc[-1]), float(df["close"].iloc[-1])
            lines.append(f"- {sym}: close ₹{px:,.2f}, stop ₹{level:,.2f} — "
                         + ("STOP HIT, sell at next open" if low <= level else f"{(px / level - 1) * 100:+.1f}% above"))
        return Reply("\n".join(lines), "stops", {}, ["stops"])

    @property
    def hub(self):
        if getattr(self, "_hub", None) is None:
            from backend.api.hub import Hub
            self._hub = Hub(self)
        return self._hub

    def _h_should_buy(self, s: Session, i: Intent) -> Reply:
        sym, mkt = self._target(s, i)
        if mkt != "NSE":
            raise ValueError("The buy/sell signal covers NSE stocks only; try 'Analyze' for other markets.")
        v = self.hub.verdict(sym)
        lines = [f"{sym}: {v['verdict']}", v["headline"], ""]
        if v["why_buy"]:
            lines.append("Why you might buy:")
            lines += [f"  ✓ {x}" for x in v["why_buy"]]
        if v["why_not"]:
            lines.append("Why you might not:")
            lines += [f"  ✕ {x}" for x in v["why_not"]]
        if v["plan"]["stop_loss"]:
            lines.append(f"\nIf you buy: stop-loss ₹{v['plan']['stop_loss']:,.2f} (10% below today), "
                         f"{v['plan']['position_size']}.")
        lines.append(f"Next review: {v['plan']['next_review']}.")
        lines.append(f"Research engines: {v['research_label']} (score {v['research_score']:+.2f}"
                     + (", engines disagree)" if v["conflict"] else ")") + ". Ask 'Why?' for the full breakdown.")
        lines.append("\n" + v["evidence_note"])
        return Reply("\n".join(lines), "should_buy", {"payload": v}, ["verdict"])

    def _h_alerts(self, s: Session, i: Intent) -> Reply:
        alerts = self.store.alerts()
        if not alerts:
            return Reply("No unacknowledged alerts.", "alerts", {"payload": []})
        text = "\n\n".join(f"[{a['severity'].upper()}] {a['message']}" for a in alerts[:10])
        return Reply(text, "alerts", {"payload": alerts}, ["alerts"])

    def _h_greeting(self, s: Session, i: Intent) -> Reply:
        return Reply("Hi! Ask me about any NSE stock (\u201cShould I buy ITC?\u201d), the market as a whole "
                     "(\u201cWhy is the market going down?\u201d), your own portfolio, or what a term means. "
                     "Every number I give comes from our tested models.", "greeting")

    def _h_help(self, s: Session, i: Intent) -> Reply:
        return Reply("I can help with: 'What should I buy with ₹1 lakh?', 'Rebalance my portfolio', "
                     "'Is the market in a downtrend?', 'Check my stops', 'Analyze RELIANCE', 'Why is the technical state positive?', "
                     "'What about fundamentals?', 'Compare it with HDFC Bank', 'I have ₹50,000 — which "
                     "stocks are worth researching?', 'How is my portfolio performing?', 'Which holding "
                     "contributes most to my risk?', 'If I add ₹20,000 to TCS and INFY how does risk change?', "
                     "'Find fundamentally strong companies with positive momentum', 'What is a Sharpe ratio?', "
                     "'Why did INFY fall this week?', 'Generate a report on TCS'.", "help", {"no_answer": True})

    # --- LLM modes --------------------------------------------------------------------
    def _narrate(self, question: str, reply: Reply) -> Reply:
        from backend.models.external.llm import LLMUnavailable, Narrator, compact_payload
        payload = reply.data["payload"]
        if isinstance(payload, dict) and "results" in payload:
            payload = compact_payload(payload)
        try:
            text, meta = Narrator(self.llm_client).narrate(question, payload, reply.text)
        except LLMUnavailable as exc:
            reply.text += f"\n\n(Conversational layer unavailable: {exc}. Showing the computed answer.)"
            reply.llm = {"error": str(exc)}
            return reply
        reply.text, reply.llm = text, meta
        return reply

    def agent_tools(self, s: Session) -> Dict[str, Tuple[Dict[str, Any], Callable[..., Dict[str, Any]]]]:
        from backend.models.external.llm import compact_payload

        def run(intent_name: str, **kw) -> Dict[str, Any]:
            syms = [(x.upper(), kw.get("market", "NSE")) for x in kw.pop("symbols", [])]
            intent = Intent(name=intent_name, symbols=syms, raw=kw.pop("raw", ""), **kw)
            r = self._deterministic(s, intent)
            payload = r.data.get("payload")
            if isinstance(payload, dict) and "results" in payload:
                payload = compact_payload(payload)
            return {"text": r.text, "data": payload, "error": r.data.get("error")}

        sym = {"type": "string", "description": "NSE ticker such as RELIANCE or TCS"}
        return {
            "analyze_stock": ({"description": "Full multi-domain analysis and decision-support label for one stock.",
                               "input_schema": {"type": "object", "properties": {"symbol": sym},
                                                "required": ["symbol"]}},
                              lambda symbol: run("analyze", symbols=[symbol])),
            "domain_evidence": ({"description": "Raw evidence for one analytical domain of a stock.",
                                 "input_schema": {"type": "object", "properties": {
                                     "symbol": sym, "domain": {"type": "string", "enum": list(DOMAINS)}},
                                     "required": ["symbol", "domain"]}},
                                lambda symbol, domain: run("domain", symbols=[symbol], domain=domain)),
            "compare_stocks": ({"description": "Side-by-side comparison of 2-6 stocks.",
                                "input_schema": {"type": "object", "properties": {
                                    "symbols": {"type": "array", "items": sym, "minItems": 2, "maxItems": 6}},
                                    "required": ["symbols"]}},
                               lambda symbols: run("compare", symbols=symbols)),
            "screen_budget": ({"description": "Research candidates that fit a budget and risk profile.",
                               "input_schema": {"type": "object", "properties": {
                                   "budget": {"type": "number", "exclusiveMinimum": 0},
                                   "exclude_sectors": {"type": "array", "items": {"type": "string"}},
                                   "include_sectors": {"type": "array", "items": {"type": "string"}}},
                                   "required": ["budget"]}},
                              lambda budget, exclude_sectors=(), include_sectors=():
                              run("screen", amount=float(budget), exclude_sectors=list(exclude_sectors),
                                  include_sectors=list(include_sectors))),
            "research_query": ({"description": "Screen the universe by criteria phrased in words, e.g. "
                                               "'fundamentally strong with positive momentum'.",
                                "input_schema": {"type": "object", "properties": {"question": {"type": "string"}},
                                                 "required": ["question"]}},
                               lambda question: run("research_query", raw=question)),
            "portfolio": ({"description": "The user's stored portfolio: value, P/L, exposure, risk, weekly change.",
                           "input_schema": {"type": "object", "properties": {}}},
                          lambda: run("portfolio_summary")),
            "portfolio_what_if": ({"description": "Risk impact of adding money to given stocks.",
                                   "input_schema": {"type": "object", "properties": {
                                       "amount": {"type": "number", "exclusiveMinimum": 0},
                                       "symbols": {"type": "array", "items": sym}}, "required": ["amount"]}},
                                  lambda amount, symbols=(): run("portfolio_what_if", amount=float(amount),
                                                                 symbols=list(symbols))),
            "explain_move": ({"description": "Why a stock moved: market vs stock-specific part, news, technicals.",
                              "input_schema": {"type": "object", "properties": {
                                  "symbol": sym, "days": {"type": "integer", "minimum": 1, "maximum": 252}},
                                  "required": ["symbol"]}},
                             lambda symbol, days=5: run("explain_move", symbols=[symbol], days=int(days))),
            "explain_concept": ({"description": "Static finance knowledge (definitions, methodology). No live data.",
                                 "input_schema": {"type": "object", "properties": {"question": {"type": "string"}},
                                                  "required": ["question"]}},
                                lambda question: run("knowledge", raw=question)),
        }

    def _agent_reply(self, s: Session, message: str, intent: Intent) -> Reply:
        from backend.models.external.llm import LLMUnavailable, ToolAgent
        try:
            text, meta = ToolAgent(self.llm_client, self.agent_tools(s)).run(message, s.history[-8:])
        except LLMUnavailable as exc:
            reply = self._deterministic(s, intent)
            reply.text += f"\n\n(Agent unavailable: {exc}. Answered with the rule-based router.)"
            reply.llm = {"error": str(exc)}
            return reply
        if meta["unsupported_numbers"]:
            reply = self._deterministic(s, intent)
            reply.text += ("\n\n(The agent's answer was discarded because it contained figures not found in "
                           f"its tool results: {', '.join(meta['unsupported_numbers'][:6])}.)")
            reply.llm = {"rejected": meta["unsupported_numbers"], "tool_calls": meta["tool_calls"]}
            return reply
        s.history += [{"role": "user", "content": message}, {"role": "assistant", "content": text}]
        return Reply(text, "agent", {}, meta["tool_calls"], {"tool_calls": meta["tool_calls"]})
