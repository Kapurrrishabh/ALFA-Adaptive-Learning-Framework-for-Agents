"""REST API + dashboard host.

Auth: fastapi-users sign-in (a session cookie naming a database row). Every route outside
PUBLIC_PATHS and /auth/ needs a verified user, a request from our own site, and stays under
a per-user rate limit; the sign-in routes have per-IP and per-email limits.
All handlers are read-only over market data; the only writes are the user's
own portfolio, reports, alert acknowledgements and daily model-call counts in
the local database.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic.networks import validate_email

from backend.advisory import report as R
from backend.advisory.calculators.backtest import strategies as S
from backend.advisory.fusion.fusion import HORIZON_TEXT
from backend.config import BUILDER_MIN_CAPITAL, BUILDER_STOCKS, FORECAST_DAYS, RISK_MODE, RISK_PROFILES, resolve_symbol
from backend.database.sources import panel
from backend.database.sources.provider import DataUnavailable
from backend.database.sources.quality import DataQualityError
from backend.api.accounts import Accounts, owner_of
from backend.api.jsonutil import sanitize
from backend.api.quota import QuotaExceeded
from backend.knowledge_base.lessons import knowledge_base
from backend.paths import FRONTEND
from backend.api.nlu import ALIASES
from backend.api.orchestrator import DOMAINS, Orchestrator, SYMBOL_RE
from backend.user.portfolio import cas, funds, importers
from backend.user.portfolio.portfolio import what_if
from backend.advisory.calculators.screener import screen


RATE_PER_MIN = 120          # PROVISIONAL: generous for one user, blocks runaway scripts
SIGN_INS_PER_MIN = 10       # PROVISIONAL, per IP; behind a proxy the real IP needs FORWARDED_ALLOW_IPS
# PROVISIONAL, per email: NIST allows up to 100 failed tries. It also bounds the emails one address can trigger, and
# probing which addresses have accounts (forgot-password answers a known one more slowly; sign-up tells anyway)
EMAIL_TRIES, EMAIL_WINDOW_S = 10, 900
MAX_KEYS = 10_000           # PROVISIONAL: past this many callers, buckets that have refilled are dropped
MAX_UPLOAD = 10_000_000     # PROVISIONAL: room for a long CAS PDF; check against real statements
SET_UP_FIRST = "Set up your portfolio first: choose your risk level, time horizon and cash."
PUBLIC_PATHS = ("/", "/health", "/manifest.webmanifest", "/sw.js", "/login/google", "/privacy")
# Strict, because the web app runs no inline script or style and loads nothing from another site
SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "same-origin", "X-Frame-Options": "DENY",
                    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; "
                    "base-uri 'none'; form-action 'self'"}
HSTS = "max-age=31536000"
DOMAIN_ALIASES = {"fundamentals": "fundamental", "news": "news_sentiment", "patterns": "pattern",
                  "candlesticks": "candlestick", "analogues": "historical"}


class TokenBucket:
    def __init__(self, capacity: int, per_seconds: float):
        self.capacity = float(capacity)
        self.rate = capacity / per_seconds
        self.state: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if len(self.state) > MAX_KEYS:
                self.state = {k: v for k, v in self.state.items() if v[0] + (now - v[1]) * self.rate < self.capacity}
            tokens, last = self.state.get(key, [self.capacity, now])
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self.state[key] = [tokens, now]
                return False
            self.state[key] = [tokens - 1, now]
            return True


def email_key(email: str) -> str:
    """The address as the sign-in routes read it: their EmailStr drops spaces and a 'Name <...>' wrapper."""
    try:
        return validate_email(email)[1].lower()
    except ValueError:      # the route refuses it too, so it only needs a bucket of its own
        return email.lower()


def theme_tokens(selector: str) -> Dict[str, str]:
    """The tokens a theme block in app.css declares, so colours set outside the stylesheet follow it."""
    blocks = re.findall(re.escape(selector) + r"\s*\{([^}]*)\}", (FRONTEND / "app.css").read_text())
    if not blocks:
        raise RuntimeError(f"app.css has no {selector} block; the install colours are read from it.")
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", "".join(blocks)))


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


class FundTransactionIn(BaseModel):
    scheme_code: int = Field(gt=0)
    side: Literal["BUY", "SELL"]
    units: float = Field(gt=0)
    nav: float = Field(gt=0)
    trade_date: date
    fees: float = Field(default=0.0, ge=0)


class PortfolioIn(BaseModel):
    cash: float = Field(ge=0)
    risk_profile: Literal["conservative", "moderate", "aggressive"]
    horizon: Literal["short", "medium", "long"]
    transactions: List[TransactionIn]


class CashIn(BaseModel):
    cash: float = Field(ge=0, lt=1e10)
    risk_profile: Literal["conservative", "moderate", "aggressive"]
    horizon: Literal["short", "medium", "long"]


class ImportIn(BaseModel):
    mode: Literal["add", "replace"]
    stocks: List[TransactionIn] = Field(min_length=1)


class BuilderHolding(BaseModel):
    symbol: str = Field(pattern=r"^[A-Za-z0-9&\-\.]{1,20}$")
    shares: int = Field(gt=0)
    price: float = Field(gt=0)


class BuilderApplyIn(BaseModel):
    holdings: List[BuilderHolding] = Field(min_length=1, max_length=30)
    paid_from_cash: bool       # true takes the cost off the stored cash in the same save, so it is not counted twice


class WhatIfIn(BaseModel):
    amount: float = Field(gt=0, lt=1e10)
    symbols: List[str] = Field(min_length=1, max_length=6)


def create_app(orch: Orchestrator, accounts: Accounts, preload: bool = False) -> FastAPI:
    if not orch.charges_model_calls:
        raise RuntimeError("The API serves many users, so build its Orchestrator with charges_model_calls=True; "
                           "otherwise their model calls skip the daily allowance.")
    bucket, per_ip, per_email = (TokenBucket(RATE_PER_MIN, 60), TokenBucket(SIGN_INS_PER_MIN, 60),
                                 TokenBucket(EMAIL_TRIES, EMAIL_WINDOW_S))
    # No /openapi.json, /docs or /redoc: they skip sign-in and would map every route to anyone
    app = FastAPI(title="StockIntel", version="0.1.0", openapi_url=None,
                  description="Evidence-first stock research API. Decision-support only.")
    headers = {**SECURITY_HEADERS, **({"Strict-Transport-Security": HSTS} if accounts.secure else {})}
    light = theme_tokens(':root:not([data-theme="dark"])')
    # One colour pair for the install screen: phones start in light mode, and theme.js recolours the bar on load
    install_colours = {"theme_color": light["--bg"], "background_color": light["--bg-2"]}

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(headers)
        if response.headers.get("content-type", "").startswith("application/json"):
            response.headers["Cache-Control"] = "no-store"      # JSON answers carry the user's own data
        return response

    def same_origin(request: Request) -> None:
        # The browser sends our cookie with requests another site starts; these headers say who started it
        site, origin = request.headers.get("sec-fetch-site"), request.headers.get("origin")
        if site in ("cross-site", "same-site") or origin not in (None, accounts.public_url):
            raise HTTPException(403, f"This request came from another site. Open {accounts.public_url} and try there.")

    def auth(request: Request, user=Depends(accounts.current_user)) -> str:
        same_origin(request)
        if not bucket.allow(str(user.id)):
            raise HTTPException(429, f"rate limit of {RATE_PER_MIN} requests/minute exceeded")
        return owner_of(user)

    def limit_ip(request: Request) -> None:
        if not per_ip.allow(request.client.host):
            raise HTTPException(429, f"More than {SIGN_INS_PER_MIN} sign-in requests a minute came from your "
                                     "network. Wait a minute and try again.")

    async def limit_email(request: Request) -> None:
        email = (await request.form()).get("username")      # the login form; an empty form for other bodies
        if "json" in request.headers.get("content-type", "").lower() and (raw := await request.body()):
            try:
                body = json.loads(raw)
            except ValueError:
                raise HTTPException(400, "The request body is not valid JSON.") from None
            email = body.get("email") if isinstance(body, dict) else None
        if isinstance(email, str) and not per_email.allow(email_key(email)):
            raise HTTPException(429, f"More than {EMAIL_TRIES} tries for this email address in "
                                     f"{EMAIL_WINDOW_S // 60} minutes. Wait a few minutes and try again.")

    accounts.include(app, dependencies=[Depends(limit_ip), Depends(limit_email), Depends(same_origin)])

    @app.exception_handler(DataUnavailable)
    async def _unavailable(_: Request, exc: DataUnavailable):
        return JSONResponse(status_code=404, content={"error": "data_unavailable", "detail": str(exc)})

    @app.exception_handler(DataQualityError)
    async def _quality(_: Request, exc: DataQualityError):
        return JSONResponse(status_code=422, content={"error": "data_quality", "detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": str(exc)})

    @app.exception_handler(importers.ImportUnsupported)
    async def _unsupported(_: Request, exc: importers.ImportUnsupported):
        return JSONResponse(status_code=501, content={"error": "unsupported", "detail": str(exc)})

    @app.exception_handler(QuotaExceeded)
    async def _quota(_: Request, exc: QuotaExceeded):
        return JSONResponse(status_code=429, content={"error": "quota", "detail": str(exc), "limit": exc.limit,
                                                      "resets_at": exc.resets_at.isoformat()})

    def ticker(t: str) -> str:
        t = t.upper()
        if not SYMBOL_RE.match(t):
            raise HTTPException(400, f"{t!r} is not a valid ticker")
        return t

    def ok(obj: Any) -> JSONResponse:
        return JSONResponse(sanitize(obj))

    def session(owner: str):
        return orch.session(owner, "api")

    def require_set_up_portfolio(owner: str) -> Dict[str, Any]:
        """No row means the user has not answered the setup questions, and there is no default to guess them."""
        p = orch.store.portfolio(owner)
        if p is None:
            raise HTTPException(400, SET_UP_FIRST)
        return p

    @app.get("/health")
    def health():
        return {"status": "ok"}

    from backend.api.hub import Hub
    from backend.models.agent.client import SelfAgentClient
    agent_client = SelfAgentClient()
    hub = Hub(orch, preload=preload)
    app.state.hub = orch._hub = hub      # one hub (and one cached panel) for the API and chat
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    @app.get("/login/google")
    def home():
        return FileResponse(FRONTEND / "index.html")

    @app.get("/privacy")
    def privacy():
        return FileResponse(FRONTEND / "privacy.html")

    @app.get("/auth/methods")
    def auth_methods():
        return {"google": accounts.google is not None, "signup": accounts.mailer is not None}

    @app.get("/users/me")
    def me(_: str = Depends(auth), user=Depends(accounts.current_user)):
        return {"email": user.email, "is_verified": user.is_verified, "google": bool(user.oauth_accounts)}

    @app.delete("/users/me", status_code=204)
    async def delete_me(owner: str = Depends(auth), user=Depends(accounts.current_user),
                        manager=Depends(accounts.users.get_user_manager)):
        # India's DPDP Act gives a right to erasure. Erase before the account goes, so a failure part-way can be
        # retried, and after, because another tab's request that signed in just before may have saved since
        def erase():
            orch.store.delete_portfolio(owner)
            orch.store.delete_usage(owner)
            orch.forget(owner)
        await run_in_threadpool(erase)
        await manager.delete(user)
        await run_in_threadpool(erase)

    @app.get("/ui/search", dependencies=[Depends(auth)])
    def ui_search(q: str = Query(min_length=1, max_length=40)):
        return ok(hub.search(q))

    @app.get("/ui/explore", dependencies=[Depends(auth)])
    def ui_explore():
        return ok(hub.explore())

    @app.get("/ui/performance", dependencies=[Depends(auth)])
    def ui_performance():
        from backend.advisory.calculators import performance
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

    @app.get("/ui/stock/{t}/verdict")
    def ui_verdict(t: str, owner: str = Depends(auth)):
        return ok(hub.verdict(ticker(t), session(owner)))

    @app.get("/ui/stock/{t}/technical", dependencies=[Depends(auth)])
    def ui_technical(t: str, bars: int = Query(default=180, ge=20, le=750), horizon: int = Query(default=FORECAST_DAYS, ge=5, le=60)):
        return ok(hub.technical(ticker(t), bars, horizon))

    @app.get("/ui/stock/{t}/intraday", dependencies=[Depends(auth)])
    def ui_intraday(t: str, range: str = "5D"):
        return ok(hub.intraday(ticker(t), range))

    @app.get("/ui/stock/{t}/indicators", dependencies=[Depends(auth)])
    def ui_indicators(t: str, bars: int = Query(default=126, ge=30, le=1000)):
        return ok(hub.indicators(ticker(t), bars))

    @app.get("/ui/stock/{t}/ai")
    def ui_ai(t: str, model: Literal["chronos", "kronos", "alfa", "gru"] = "chronos", owner: str = Depends(auth)):
        return ok(hub.ai_forecast(ticker(t), model, owner))

    @app.get("/ui/ticker", dependencies=[Depends(auth)])
    def ui_ticker():
        return ok(hub.ticker())

    @app.get("/ui/stock/{t}/analyst")
    def ui_analyst(t: str, owner: str = Depends(auth)):
        return ok(hub.analyst_note(ticker(t), session(owner)))

    @app.get("/ui/sectors", dependencies=[Depends(auth)])
    def ui_sectors():
        n = hub.names()
        return sorted(n.loc[n["nifty200"], "sector"].unique().tolist())

    @app.get("/ui/features", dependencies=[Depends(auth)])
    def ui_features():
        from backend.models.serving import alfa_fan as alfa_model, gru_line
        from backend.models.external import kronos_model, remote_models
        space = bool(remote_models.space())
        return {"ml": hub.ml_enabled(), "kronos": hub.ml_enabled() and (space or kronos_model.available()),
                "alfa": space or alfa_model.available(), "gru": gru_line.available()}

    @app.get("/manifest.webmanifest")
    def manifest():
        return JSONResponse({"name": "StockIntel", "short_name": "StockIntel", "start_url": "/#/home", "display": "standalone",
                             **install_colours,
                             "icons": [{"src": "/static/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"}]},
                            media_type="application/manifest+json")

    @app.get("/sw.js")
    def service_worker():
        return FileResponse(FRONTEND / "sw.js", media_type="application/javascript")

    @app.get("/ui/builder", dependencies=[Depends(auth)])
    def ui_builder(capital: float = Query(ge=BUILDER_MIN_CAPITAL, lt=1e9),
                   strategy: Literal["momentum", "lowvol", "blend"] = "momentum",
                   risk: Literal["full", "balanced", "defensive"] = "balanced",
                   n: int = Query(default=10, ge=BUILDER_STOCKS[0], le=BUILDER_STOCKS[1]),
                   exclude: str = ""):
        return ok(hub.build_portfolio(capital, strategy, risk, n, [x for x in exclude.split(",") if x]))

    @app.get("/ui/profiles", dependencies=[Depends(auth)])
    def ui_profiles():
        return {"risk": {name: {"max_position_pct": round(p.max_position_weight * 100, 2),
                                "max_sector_pct": round(p.max_sector_weight * 100, 2), "mode": RISK_MODE[name]}
                         for name, p in RISK_PROFILES.items()},
                "horizon": HORIZON_TEXT,
                "builder": {"min_capital": BUILDER_MIN_CAPITAL, "min_stocks": BUILDER_STOCKS[0],
                            "max_stocks": BUILDER_STOCKS[1]}}

    @app.post("/ui/builder/apply")
    def ui_builder_apply(body: BuilderApplyIn, owner: str = Depends(auth)):
        with orch.portfolio_lock(owner):
            p = require_set_up_portfolio(owner)
            today = date.today().isoformat()
            txs = p["transactions"] + [{"symbol": h.symbol, "side": "BUY", "quantity": h.shares, "price": h.price,
                                        "trade_date": today} for h in body.holdings]
            cost = sum(h.shares * h.price for h in body.holdings)
            cash = round(p["cash"] - cost, 2) if body.paid_from_cash else p["cash"]
            if cash < 0:
                raise HTTPException(400, f"These stocks cost ₹{cost:,.2f}, more than your ₹{p['cash']:,.2f} of cash. "
                                         "Check the prices you typed, or raise your cash first.")
            n = orch.save_portfolio(owner, cash, p["risk_profile"], p["horizon"], txs)
        return {"status": "saved", "transactions": n}

    @app.get("/ui/agent/status", dependencies=[Depends(auth)])
    def ui_agent_status():
        return agent_client.status()

    @app.post("/ui/agent/ask")
    def ui_agent_ask(body: ChatIn, owner: str = Depends(auth)):
        from backend.models.agent.client import AgentUnavailable
        try:
            with orch.model_call(owner):
                return ok(agent_client.ask(body.message, "", session=owner + ":" + body.session_id))
        except AgentUnavailable as exc:
            raise HTTPException(503, str(exc)) from None

    @app.get("/ui/usage")
    def ui_usage(owner: str = Depends(auth)):
        return orch.usage(owner)

    @app.get("/ui/stock/{t}/insider", dependencies=[Depends(auth)])
    def ui_insider(t: str):
        return ok(hub.insider(ticker(t)))

    @app.get("/ui/funds/search", dependencies=[Depends(auth)])
    def ui_fund_search(q: str = Query(default="", max_length=80)):
        return ok(hub.fund_search(q))

    @app.get("/ui/funds/{code}", dependencies=[Depends(auth)])
    def ui_fund(code: int):
        return ok(hub.fund(code))

    @app.post("/portfolio/transactions")
    def add_transaction(tx: TransactionIn, owner: str = Depends(auth)):
        with orch.portfolio_lock(owner):
            p = require_set_up_portfolio(owner)
            txs = p["transactions"] + [{**tx.model_dump(), "trade_date": tx.trade_date.isoformat()}]
            n = orch.save_portfolio(owner, p["cash"], p["risk_profile"], p["horizon"], txs)
        return {"status": "saved", "transactions": n}

    @app.delete("/portfolio/transactions/{index}")
    def delete_transaction(index: int, owner: str = Depends(auth)):
        with orch.portfolio_lock(owner):
            p = orch.store.portfolio(owner)
            if p is None or not 0 <= index < len(p["transactions"]):
                raise HTTPException(404, f"no transaction #{index}")
            txs = [t for i, t in enumerate(p["transactions"]) if i != index]
            n = orch.save_portfolio(owner, p["cash"], p["risk_profile"], p["horizon"], txs)
        return {"status": "deleted", "transactions": n}

    @app.get("/portfolio/funds")
    def portfolio_funds(owner: str = Depends(auth)):
        from backend.database.sources import amfi
        from dataclasses import asdict
        txs, schemes = orch.store.fund_transactions(owner), {}
        try:
            # with no funds the page must not wait on, or fail with, AMFI's daily download
            schemes = {s.code: s for s in amfi.schemes()} if txs else {}
            valued = {**funds.holdings(txs, schemes), "valuation_error": None}
        except DataUnavailable as exc:      # the transactions still list, so a fund AMFI dropped can be deleted
            valued = {"valuation_error": str(exc)}
        named = [{**t, "scheme": asdict(schemes[t["scheme_code"]]) if t["scheme_code"] in schemes else None} for t in txs]
        return ok({**valued, "transactions": named, "source": amfi.NAV_SOURCE})

    @app.post("/portfolio/funds/transactions")
    def add_fund_transaction(tx: FundTransactionIn, owner: str = Depends(auth)):
        from backend.database.sources import amfi
        if tx.scheme_code not in {s.code for s in amfi.schemes()}:
            raise HTTPException(400, f"AMFI lists no scheme {tx.scheme_code} with a NAV in the last {amfi.LIVE_DAYS} "
                                     "days. Find the fund with the search box and add it from its page.")
        with orch.portfolio_lock(owner):
            txs = orch.store.fund_transactions(owner) + [{**tx.model_dump(), "trade_date": tx.trade_date.isoformat()}]
            funds.positions(txs)
            if orch.store.portfolio(owner) is None:
                orch.save_portfolio(owner, 0.0, "moderate", "medium", [])
            orch.store.save_fund_transactions(owner, txs)
        return {"status": "saved", "fund_transactions": len(txs)}

    @app.delete("/portfolio/funds/transactions/{tx_id}")
    def delete_fund_transaction(tx_id: int, owner: str = Depends(auth)):
        with orch.portfolio_lock(owner):
            txs = orch.store.fund_transactions(owner)
            kept = [t for t in txs if t["id"] != tx_id]
            if len(kept) == len(txs):
                raise HTTPException(404, f"no fund transaction #{tx_id}")
            funds.positions(kept)
            orch.store.save_fund_transactions(owner, kept)
        return {"status": "deleted", "fund_transactions": len(kept)}

    @app.post("/portfolio/cash")
    def set_cash(body: CashIn, owner: str = Depends(auth)):
        """The setup call: with no row yet, it makes one from the user's own answers."""
        with orch.portfolio_lock(owner):
            p = orch.store.portfolio(owner)
            orch.save_portfolio(owner, body.cash, body.risk_profile, body.horizon, p["transactions"] if p else [])
        return {"status": "saved"}

    @app.post("/portfolio/import/preview")
    def import_preview(file: UploadFile, password: str = Form(""), owner: str = Depends(auth)):
        # read here, never saved: CACHE_DIR is backed up to Hugging Face, and these files name what a person owns
        data = file.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"This file is over {MAX_UPLOAD // 1_000_000} MB. {importers.SHORTER_RANGE}")
        statement = cas.statement(cas.read(data, password)) if data.startswith(b"%PDF") else importers.parse(data)
        names = panel.company_names()
        matches = importers.match(statement.rows, panel.isin_directory(), names)
        positions, _, _ = orch.holdings(owner)
        funds = [f"This file's {len(statement.funds)} mutual fund rows are left out: a portfolio here holds only "
                 "shares, ETFs and REITs for now."] if statement.funds else []

        def listed(listing):
            return {**listing._asdict(), "name": names.get(listing.symbol)}
        return ok({"source": statement.source, "as_of": statement.as_of, "counts": importers.counts(matches),
                   "warnings": statement.warnings + funds,
                   "held": {m.listing.symbol: positions[m.listing.symbol].quantity for m in matches
                            if m.listing and m.listing.symbol in positions},
                   "rows": [{**asdict(m.row), "state": m.state, "listing": m.listing and listed(m.listing),
                             "choices": [listed(c) for c in m.choices]} for m in matches]})

    @app.post("/portfolio/import/confirm")
    def import_confirm(body: ImportIn, owner: str = Depends(auth)):
        with orch.portfolio_lock(owner):
            p = require_set_up_portfolio(owner)
            kept = p["transactions"] if body.mode == "add" else []
            n = orch.save_portfolio(owner, p["cash"], p["risk_profile"], p["horizon"], kept + [
                {**t.model_dump(), "trade_date": t.trade_date.isoformat()} for t in body.stocks])
        return {"status": "saved", "transactions": n}

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

    @app.get("/stocks/{t}/analysis")
    def analysis(t: str, horizon: Literal["short", "medium", "long"] = "medium", owner: str = Depends(auth)):
        a = orch.analysis(session(owner), ticker(t), horizon=horizon)
        return ok({**a.to_dict(), "text": R.render_analysis(a)})

    @app.get("/stocks/{t}/chart")
    def chart(t: str, bars: int = Query(default=260, ge=30, le=1500), owner: str = Depends(auth)):
        a = orch.analysis(session(owner), ticker(t))
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

    @app.get("/stocks/{t}/{domain}")
    def domain(t: str, domain: str, owner: str = Depends(auth)):
        dom = DOMAIN_ALIASES.get(domain, domain)
        if dom not in DOMAINS:
            raise HTTPException(404, f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
        a = orch.analysis(session(owner), ticker(t))
        return ok({"symbol": a.symbol, "analysis_timestamp": a.analysis_timestamp,
                   "result": a.results[dom].to_dict(), "text": R.render_domain(a, dom)})

    @app.get("/compare")
    def compare(symbols: str = Query(description="comma-separated, 2-6 tickers"), owner: str = Depends(auth)):
        syms = [ticker(s) for s in symbols.split(",") if s.strip()]
        if not 2 <= len(syms) <= 6:
            raise HTTPException(400, "compare needs 2 to 6 symbols")
        analyses = [orch.analysis(session(owner), s) for s in syms]
        return ok({"text": R.render_compare(analyses),
                   "items": [{"symbol": a.symbol, "decision": a.decision.to_dict()} for a in analyses]})

    @app.get("/screener")
    def screener(budget: float = Query(gt=0, lt=1e10),
                 risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate",
                 horizon: Literal["short", "medium", "long"] = "medium",
                 exclude: str = "", include: str = "", owner: str = Depends(auth)):
        positions, _, _ = orch.holdings(owner)
        res = screen(orch.service(), budget, risk_profile, horizon, positions,
                     include_sectors=[s for s in include.split(",") if s],
                     exclude_sectors=[s for s in exclude.split(",") if s])
        return ok({**{k: v for k, v in res.items() if k != "also_scored"}, "text": R.render_screen(res)})

    @app.get("/portfolio")
    def portfolio(owner: str = Depends(auth)):
        p = orch.store.portfolio(owner)
        if p is None:
            raise HTTPException(404, "no portfolio stored; POST /portfolio first")
        return ok(p)

    @app.post("/portfolio")
    def set_portfolio(body: PortfolioIn, owner: str = Depends(auth)):
        with orch.portfolio_lock(owner):
            n = orch.save_portfolio(owner, body.cash, body.risk_profile, body.horizon, [
                {**t.model_dump(), "trade_date": t.trade_date.isoformat()} for t in body.transactions])
        return {"status": "saved", "transactions": n}

    @app.get("/portfolio/analysis")
    def portfolio_analysis(owner: str = Depends(auth)):
        rep, *_ = orch.portfolio_report(session(owner))
        d = rep.to_dict()
        return ok({**d, "text": R.render_portfolio(d)})

    @app.get("/ui/portfolio/history")
    def ui_portfolio_history(owner: str = Depends(auth)):
        return ok(hub.portfolio_history(owner))

    @app.get("/portfolio/holdings")
    def holdings(owner: str = Depends(auth)):
        rep, *_ = orch.portfolio_report(session(owner))
        return ok({"as_of": rep.as_of, "holdings": rep.holdings})

    @app.get("/portfolio/risk")
    def portfolio_risk(owner: str = Depends(auth)):
        rep, *_ = orch.portfolio_report(session(owner))
        return ok({"as_of": rep.as_of, "risk": rep.risk, "concentration": rep.concentration,
                   "sector_exposure_pct": rep.sector_exposure_pct, "flags": rep.flags})

    @app.post("/portfolio/what-if")
    def portfolio_what_if(body: WhatIfIn, owner: str = Depends(auth)):
        rep, positions, cash, prices, meta = orch.portfolio_report(session(owner))
        syms = [ticker(s) for s in body.symbols]
        for s in syms:
            prices.setdefault(s, orch.service().history(s)["close"])
        scenarios = {f"all in {s}": {s: body.amount} for s in syms}
        if len(syms) > 1:
            scenarios["split equally"] = {s: body.amount / len(syms) for s in syms}
        res = what_if(positions, cash, prices, scenarios, meta.get("risk_profile", "moderate"))
        return ok({**res, "text": R.render_what_if(res)})

    @app.post("/chat")
    def chat(body: ChatIn, owner: str = Depends(auth)):
        return ok(orch.handle(body.message, owner, body.session_id).to_dict())

    @app.get("/reports")
    def reports(symbol: Optional[str] = None, owner: str = Depends(auth)):
        return ok(orch.store.reports(owner, ticker(symbol) if symbol else None))

    @app.get("/reports/{report_id}")
    def report(report_id: int, owner: str = Depends(auth)):
        r = orch.store.report(owner, report_id)
        if r is None:
            raise HTTPException(404, f"no report #{report_id}")
        return ok(r)

    @app.post("/reports/{t}")
    def make_report(t: str, owner: str = Depends(auth)):
        a = orch.analysis(session(owner), ticker(t))
        body = R.render_report(a)
        return {"id": orch.store.save_report(owner, a.symbol, body), "report": body}

    @app.get("/alerts")
    def alerts(include_acknowledged: bool = False, owner: str = Depends(auth)):
        return ok(orch.alerts(owner, include_acknowledged))

    @app.post("/alerts/{alert_id}/ack")
    def ack(alert_id: int, owner: str = Depends(auth)):
        orch.store.acknowledge_alert(owner, alert_id)
        return {"status": "acknowledged", "id": alert_id}

    @app.get("/signals")
    def signals(capital: float = Query(default=0.0, ge=0, lt=1e10),
                strategy: Literal["momentum", "lowvol", "blend"] = "momentum",
                risk: Literal["full", "balanced", "defensive"] = "balanced", owner: str = Depends(auth)):
        from backend.api.nlu import Intent
        from backend.advisory.signals.advisor import render
        s = session(owner)
        words = {"lowvol": "low volatility", "blend": "blend", "momentum": ""}[strategy]
        p = orch._signal_plan(s, Intent(name="signals", amount=capital or None, raw=words), risk)
        payload = sanitize(p.to_dict())
        orch.store.save_plan(owner, payload)
        return ok({**payload, "text": render(p)})

    @app.get("/market", dependencies=[Depends(auth)])
    def market():
        from backend.advisory.signals.advisor import market_state
        bench = orch.service().history("^NSEI")["close"]
        return ok({"as_of": str(bench.index[-1].date()), **market_state(bench, bench.index[-1])})

    @app.get("/knowledge", dependencies=[Depends(auth)])
    def knowledge(q: str = Query(min_length=2, max_length=200)):
        return {"query": q, "results": [{"title": c.title, "text": c.text, "citation": c.citation,
                                         "score": s} for c, s in knowledge_base().search(q, k=5)]}

    return app
