"""REST API + dashboard host.

Auth: every endpoint except /health and the static dashboard requires the
X-API-Key header to equal STOCKINTEL_API_KEY (compared in constant time).
The server refuses to start without a key. Rate limit: token bucket per key.
All handlers are read-only over market data; the only writes are the user's
own portfolio, reports and alert acknowledgements in the local database.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import report as R
from .backtest import strategies as S
from .config import resolve_symbol
from .data.provider import DataUnavailable
from .data.quality import DataQualityError
from .jsonutil import sanitize
from .knowledge import knowledge_base
from .nlu import ALIASES
from .orchestrator import DOMAINS, Orchestrator, SYMBOL_RE
from .portfolio import what_if
from .screener import screen

WEB_DIR = Path(__file__).parent / "web"
RATE_PER_MIN = 120          # PROVISIONAL: generous for one user, blocks runaway scripts
DOMAIN_ALIASES = {"fundamentals": "fundamental", "news": "news_sentiment", "patterns": "pattern",
                  "candlesticks": "candlestick", "analogues": "historical"}


class TokenBucket:
    def __init__(self, rate_per_min: int):
        self.capacity = float(rate_per_min)
        self.rate = rate_per_min / 60.0
        self.state: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            tokens, last = self.state.get(key, [self.capacity, now])
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self.state[key] = [tokens, now]
                return False
            self.state[key] = [tokens - 1, now]
            return True


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", pattern=r"^[A-Za-z0-9_\-]{1,64}$")


class TransactionIn(BaseModel):
    symbol: str = Field(pattern=r"^[A-Za-z0-9&\-\.]{1,20}$")
    side: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    trade_date: date
    fees: float = Field(default=0.0, ge=0)
    sector: Optional[str] = Field(default=None, max_length=64)


class PortfolioIn(BaseModel):
    cash: float = Field(ge=0)
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    horizon: Literal["short", "medium", "long"] = "medium"
    transactions: List[TransactionIn]


class CashIn(BaseModel):
    cash: float = Field(ge=0, lt=1e10)
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    horizon: Literal["short", "medium", "long"] = "medium"


class BuilderHolding(BaseModel):
    symbol: str = Field(pattern=r"^[A-Za-z0-9&\-\.]{1,20}$")
    shares: int = Field(gt=0)
    price: float = Field(gt=0)


class BuilderApplyIn(BaseModel):
    holdings: List[BuilderHolding] = Field(min_length=1, max_length=30)


class WhatIfIn(BaseModel):
    amount: float = Field(gt=0, lt=1e10)
    symbols: List[str] = Field(min_length=1, max_length=6)


def create_app(orch: Orchestrator, api_key: Optional[str] = None, preload: bool = False) -> FastAPI:
    key = api_key if api_key is not None else os.environ.get("STOCKINTEL_API_KEY", "")
    if len(key) < 16:
        raise RuntimeError("STOCKINTEL_API_KEY must be set to a random string of at least 16 "
                           "characters before starting the API (e.g. `openssl rand -hex 24`).")
    bucket = TokenBucket(RATE_PER_MIN)
    app = FastAPI(title="StockIntel", version="0.1.0",
                  description="Evidence-first stock research API. Decision-support only.")

    def auth(x_api_key: str = Header(default="")) -> str:
        if not hmac.compare_digest(x_api_key.encode(), key.encode()):
            raise HTTPException(401, "missing or invalid X-API-Key header")
        if not bucket.allow(x_api_key):
            raise HTTPException(429, f"rate limit of {RATE_PER_MIN} requests/minute exceeded")
        return x_api_key

    @app.exception_handler(DataUnavailable)
    async def _unavailable(_: Request, exc: DataUnavailable):
        return JSONResponse(status_code=404, content={"error": "data_unavailable", "detail": str(exc)})

    @app.exception_handler(DataQualityError)
    async def _quality(_: Request, exc: DataQualityError):
        return JSONResponse(status_code=422, content={"error": "data_quality", "detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": str(exc)})

    def ticker(t: str) -> str:
        t = t.upper()
        if not SYMBOL_RE.match(t):
            raise HTTPException(400, f"{t!r} is not a valid ticker")
        return t

    def ok(obj: Any) -> JSONResponse:
        return JSONResponse(sanitize(obj))

    def session():
        return orch.session("api")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    from .hub import Hub
    from .selfagent_client import SelfAgentClient
    agent_client = SelfAgentClient()
    hub = Hub(orch, preload=preload)
    app.state.hub = orch._hub = hub      # one hub (and one cached panel) for the API and chat
    app.mount("/static", StaticFiles(directory=WEB_DIR / "app"), name="static")

    @app.get("/")
    def home():
        return FileResponse(WEB_DIR / "app" / "index.html")

    @app.get("/classic")
    def dashboard():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/ui/search", dependencies=[Depends(auth)])
    def ui_search(q: str = Query(min_length=1, max_length=40)):
        return ok(hub.search(q))

    @app.get("/ui/explore", dependencies=[Depends(auth)])
    def ui_explore():
        return ok(hub.explore())

    @app.get("/ui/performance", dependencies=[Depends(auth)])
    def ui_performance():
        from . import performance
        try:
            return ok(performance.load())
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.get("/ui/learn", dependencies=[Depends(auth)])
    def ui_learn():
        return ok(hub.learn_index())

    @app.get("/ui/stock/{t}", dependencies=[Depends(auth)])
    def ui_stock(t: str):
        return ok(hub.overview(ticker(t)))

    @app.get("/ui/stock/{t}/chart", dependencies=[Depends(auth)])
    def ui_chart(t: str, range: str = "1Y"):
        return ok(hub.chart(ticker(t), range))

    @app.get("/ui/stock/{t}/verdict", dependencies=[Depends(auth)])
    def ui_verdict(t: str):
        return ok(hub.verdict(ticker(t)))

    @app.get("/ui/stock/{t}/technical", dependencies=[Depends(auth)])
    def ui_technical(t: str, bars: int = Query(default=180, ge=20, le=750), horizon: int = Query(default=20, ge=5, le=60)):
        return ok(hub.technical(ticker(t), bars, horizon))

    @app.get("/ui/stock/{t}/intraday", dependencies=[Depends(auth)])
    def ui_intraday(t: str, range: str = "5D"):
        return ok(hub.intraday(ticker(t), range))

    @app.get("/ui/stock/{t}/indicators", dependencies=[Depends(auth)])
    def ui_indicators(t: str, bars: int = Query(default=126, ge=30, le=1000)):
        return ok(hub.indicators(ticker(t), bars))

    @app.get("/ui/stock/{t}/ai", dependencies=[Depends(auth)])
    def ui_ai(t: str, model: Literal["chronos", "kronos"] = "chronos"):
        return ok(hub.ai_forecast(ticker(t), model))

    @app.get("/ui/ticker", dependencies=[Depends(auth)])
    def ui_ticker():
        return ok(hub.ticker())

    @app.get("/ui/sectors", dependencies=[Depends(auth)])
    def ui_sectors():
        n = hub.names()
        return sorted(n.loc[n["nifty200"], "sector"].unique().tolist())

    @app.get("/ui/features", dependencies=[Depends(auth)])
    def ui_features():
        from .analysis import kronos_model, remote_models
        return {"ml": hub.ml_enabled(), "kronos": hub.ml_enabled() and (bool(remote_models.space()) or kronos_model.available())}

    @app.get("/manifest.webmanifest")
    def manifest():
        return JSONResponse({"name": "StockIntel", "short_name": "StockIntel", "start_url": "/#/home", "display": "standalone",
                             "background_color": "#06080b", "theme_color": "#06080b",
                             "icons": [{"src": "/static/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"}]},
                            media_type="application/manifest+json")

    @app.get("/sw.js")
    def service_worker():
        return FileResponse(WEB_DIR / "app" / "sw.js", media_type="application/javascript")

    @app.get("/ui/builder", dependencies=[Depends(auth)])
    def ui_builder(capital: float = Query(gt=999, lt=1e9), strategy: Literal["momentum", "lowvol", "blend"] = "momentum",
                   risk: Literal["full", "balanced", "defensive"] = "balanced", n: int = Query(default=10, ge=3, le=25),
                   exclude: str = ""):
        return ok(hub.build_portfolio(capital, strategy, risk, n, [x for x in exclude.split(",") if x]))

    @app.post("/ui/builder/apply", dependencies=[Depends(auth)])
    def ui_builder_apply(body: BuilderApplyIn):
        p = orch.store.portfolio(orch.portfolio_name) or {"cash": 0.0, "risk_profile": "moderate",
                                                         "horizon": "medium", "transactions": []}
        today = date.today().isoformat()
        txs = p["transactions"] + [{"symbol": h.symbol, "side": "BUY", "quantity": h.shares, "price": h.price,
                                    "trade_date": today} for h in body.holdings]
        n = orch.save_portfolio(p["cash"], p["risk_profile"], p["horizon"], txs)
        return {"status": "saved", "transactions": n}

    @app.get("/ui/agent/status", dependencies=[Depends(auth)])
    def ui_agent_status():
        return agent_client.status()

    @app.post("/ui/agent/ask", dependencies=[Depends(auth)])
    def ui_agent_ask(body: ChatIn):
        from .selfagent_client import AgentUnavailable
        try:
            return ok(agent_client.ask(body.message, ""))
        except AgentUnavailable as exc:
            raise HTTPException(503, str(exc)) from None

    @app.get("/ui/stock/{t}/insider", dependencies=[Depends(auth)])
    def ui_insider(t: str):
        return ok(hub.insider(ticker(t)))

    @app.post("/portfolio/transactions", dependencies=[Depends(auth)])
    def add_transaction(tx: TransactionIn):
        p = orch.store.portfolio(orch.portfolio_name) or {"cash": 0.0, "risk_profile": "moderate",
                                                         "horizon": "medium", "transactions": []}
        txs = p["transactions"] + [{**tx.model_dump(), "trade_date": tx.trade_date.isoformat()}]
        n = orch.save_portfolio(p["cash"], p["risk_profile"], p["horizon"], txs)
        return {"status": "saved", "transactions": n}

    @app.delete("/portfolio/transactions/{index}", dependencies=[Depends(auth)])
    def delete_transaction(index: int):
        p = orch.store.portfolio(orch.portfolio_name)
        if p is None or not 0 <= index < len(p["transactions"]):
            raise HTTPException(404, f"no transaction #{index}")
        txs = [t for i, t in enumerate(p["transactions"]) if i != index]
        n = orch.save_portfolio(p["cash"], p["risk_profile"], p["horizon"], txs)
        return {"status": "deleted", "transactions": n}

    @app.post("/portfolio/cash", dependencies=[Depends(auth)])
    def set_cash(body: CashIn):
        p = orch.store.portfolio(orch.portfolio_name) or {"risk_profile": "moderate", "horizon": "medium",
                                                         "transactions": []}
        orch.save_portfolio(body.cash, body.risk_profile, body.horizon, p["transactions"])
        return {"status": "saved"}

    @app.get("/search", dependencies=[Depends(auth)])
    def search(q: str = Query(min_length=1, max_length=40)):
        ql = q.lower()
        hits = [s for s in sorted(orch.known) if ql in s.lower()]
        hits += [sym for name, (sym, _) in ALIASES.items() if ql in name and sym not in hits]
        return {"query": q, "results": hits[:15]}

    @app.get("/stocks/{t}", dependencies=[Depends(auth)])
    def stock(t: str):
        t = ticker(t)
        svc = orch.service()
        df = svc.history(t)
        info = svc.info(t)
        keep = ("longName", "sector", "industry", "marketCap", "currency", "exchange", "longBusinessSummary")
        return ok({"symbol": t, "provider_symbol": resolve_symbol(t), "last_close": float(df["close"].iloc[-1]),
                   "as_of": str(df.index[-1].date()), "profile": {k: info.get(k, "Data unavailable") for k in keep}})

    @app.get("/stocks/{t}/analysis", dependencies=[Depends(auth)])
    def analysis(t: str, horizon: Literal["short", "medium", "long"] = "medium"):
        a = orch.analysis(session(), ticker(t), horizon=horizon)
        return ok({**a.to_dict(), "text": R.render_analysis(a)})

    @app.get("/stocks/{t}/chart", dependencies=[Depends(auth)])
    def chart(t: str, bars: int = Query(default=260, ge=30, le=1500)):
        a = orch.analysis(session(), ticker(t))
        df = orch.service().history(t).iloc[-bars:]
        close = orch.service().history(t)["close"]
        sma50, sma200 = close.rolling(50).mean().iloc[-bars:], close.rolling(200).mean().iloc[-bars:]
        tech = a.results["technical"].details
        return ok({"symbol": a.symbol, "dates": [str(d.date()) for d in df.index],
                   "open": df["open"].round(2).tolist(), "high": df["high"].round(2).tolist(),
                   "low": df["low"].round(2).tolist(), "close": df["close"].round(2).tolist(),
                   "volume": df["volume"].tolist(), "sma50": sma50.round(2).tolist(),
                   "sma200": sma200.round(2).tolist(),
                   "support": tech.get("support_levels", []), "resistance": tech.get("resistance_levels", []),
                   "patterns": a.results["pattern"].details.get("patterns", []) if a.results["pattern"].available else []})

    @app.get("/stocks/{t}/backtest", dependencies=[Depends(auth)])
    def backtest(t: str):
        df = orch.service().history(ticker(t))
        strategies = {**S.BASELINES, "technical_engine": S.technical_engine,
                      "forecast_logistic": lambda d: S.forecast_model(d, bench=orch.service().benchmark())}
        start = str(df.index[min(len(df) - 2, 300)].date())
        res = S.compare(df, strategies, start=start)
        return ok({k: v for k, v in res.items() if k != "results"})

    @app.get("/stocks/{t}/{domain}", dependencies=[Depends(auth)])
    def domain(t: str, domain: str):
        dom = DOMAIN_ALIASES.get(domain, domain)
        if dom not in DOMAINS:
            raise HTTPException(404, f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
        a = orch.analysis(session(), ticker(t))
        return ok({"symbol": a.symbol, "analysis_timestamp": a.analysis_timestamp,
                   "result": a.results[dom].to_dict(), "text": R.render_domain(a, dom)})

    @app.get("/compare", dependencies=[Depends(auth)])
    def compare(symbols: str = Query(description="comma-separated, 2-6 tickers")):
        syms = [ticker(s) for s in symbols.split(",") if s.strip()]
        if not 2 <= len(syms) <= 6:
            raise HTTPException(400, "compare needs 2 to 6 symbols")
        analyses = [orch.analysis(session(), s) for s in syms]
        return ok({"text": R.render_compare(analyses),
                   "items": [{"symbol": a.symbol, "decision": a.decision.to_dict()} for a in analyses]})

    @app.get("/screener", dependencies=[Depends(auth)])
    def screener(budget: float = Query(gt=0, lt=1e10),
                 risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate",
                 horizon: Literal["short", "medium", "long"] = "medium",
                 exclude: str = "", include: str = ""):
        positions, _, _ = orch.holdings()
        res = screen(orch.service(), budget, risk_profile, horizon, positions,
                     include_sectors=[s for s in include.split(",") if s],
                     exclude_sectors=[s for s in exclude.split(",") if s])
        return ok({**{k: v for k, v in res.items() if k != "also_scored"}, "text": R.render_screen(res)})

    @app.get("/portfolio", dependencies=[Depends(auth)])
    def portfolio():
        p = orch.store.portfolio(orch.portfolio_name)
        if p is None:
            raise HTTPException(404, "no portfolio stored; POST /portfolio first")
        return ok(p)

    @app.post("/portfolio", dependencies=[Depends(auth)])
    def set_portfolio(body: PortfolioIn):
        n = orch.save_portfolio(body.cash, body.risk_profile, body.horizon, [
            {**t.model_dump(), "trade_date": t.trade_date.isoformat()} for t in body.transactions])
        return {"status": "saved", "transactions": n}

    @app.get("/portfolio/analysis", dependencies=[Depends(auth)])
    def portfolio_analysis():
        rep, *_ = orch.portfolio_report(session())
        d = rep.to_dict()
        return ok({**d, "text": R.render_portfolio(d)})

    @app.get("/ui/portfolio/history", dependencies=[Depends(auth)])
    def ui_portfolio_history():
        return ok(hub.portfolio_history())

    @app.get("/portfolio/holdings", dependencies=[Depends(auth)])
    def holdings():
        rep, *_ = orch.portfolio_report(session())
        return ok({"as_of": rep.as_of, "holdings": rep.holdings})

    @app.get("/portfolio/risk", dependencies=[Depends(auth)])
    def portfolio_risk():
        rep, *_ = orch.portfolio_report(session())
        return ok({"as_of": rep.as_of, "risk": rep.risk, "concentration": rep.concentration,
                   "sector_exposure_pct": rep.sector_exposure_pct, "flags": rep.flags})

    @app.post("/portfolio/what-if", dependencies=[Depends(auth)])
    def portfolio_what_if(body: WhatIfIn):
        rep, positions, cash, prices, meta = orch.portfolio_report(session())
        syms = [ticker(s) for s in body.symbols]
        for s in syms:
            prices.setdefault(s, orch.service().history(s)["close"])
        scenarios = {f"all in {s}": {s: body.amount} for s in syms}
        if len(syms) > 1:
            scenarios["split equally"] = {s: body.amount / len(syms) for s in syms}
        res = what_if(positions, cash, prices, scenarios, meta.get("risk_profile", "moderate"))
        return ok({**res, "text": R.render_what_if(res)})

    @app.post("/chat", dependencies=[Depends(auth)])
    def chat(body: ChatIn):
        return ok(orch.handle(body.message, body.session_id).to_dict())

    @app.get("/reports", dependencies=[Depends(auth)])
    def reports(symbol: Optional[str] = None):
        return ok(orch.store.reports(ticker(symbol) if symbol else None))

    @app.post("/reports/{t}", dependencies=[Depends(auth)])
    def make_report(t: str):
        a = orch.analysis(session(), ticker(t))
        body = R.render_report(a)
        return {"id": orch.store.save_report(a.symbol, body), "report": body}

    @app.get("/alerts", dependencies=[Depends(auth)])
    def alerts(include_acknowledged: bool = False):
        return ok(orch.store.alerts(include_acknowledged))

    @app.post("/alerts/{alert_id}/ack", dependencies=[Depends(auth)])
    def ack(alert_id: int):
        orch.store.acknowledge_alert(alert_id)
        return {"status": "acknowledged", "id": alert_id}

    @app.get("/signals", dependencies=[Depends(auth)])
    def signals(capital: float = Query(default=0.0, ge=0, lt=1e10),
                strategy: Literal["momentum", "lowvol", "blend"] = "momentum",
                risk: Literal["full", "balanced", "defensive"] = "balanced"):
        from .nlu import Intent
        from .signals.advisor import render
        s = session()
        s.risk_profile = {"full": "aggressive", "defensive": "conservative"}.get(risk, "moderate")
        words = {"lowvol": "low volatility", "blend": "blend", "momentum": ""}[strategy]
        p = orch._signal_plan(s, Intent(name="signals", amount=capital or None, raw=words))
        payload = sanitize(p.to_dict())
        orch.store.save_plan(payload)
        return ok({**payload, "text": render(p)})

    @app.get("/market", dependencies=[Depends(auth)])
    def market():
        from .signals.advisor import market_state
        bench = orch.service().history("^NSEI")["close"]
        return ok({"as_of": str(bench.index[-1].date()), **market_state(bench, bench.index[-1])})

    @app.get("/knowledge", dependencies=[Depends(auth)])
    def knowledge(q: str = Query(min_length=2, max_length=200)):
        return {"query": q, "results": [{"title": c.title, "text": c.text, "citation": c.citation,
                                         "score": s} for c, s in knowledge_base().search(q, k=5)]}

    return app
