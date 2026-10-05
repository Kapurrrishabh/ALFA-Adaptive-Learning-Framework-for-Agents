"""UI-facing aggregations for the web app (search, stock page, verdict, explore, learn).

The verdict answers "should I buy this?" with the one signal that has
measured edge — NSE-method momentum rank inside the Nifty 200 — and puts
everything else (research engines, market regime, insider flags,
valuation) into explicit "why buy" / "why not" lists. It never invents a
confidence it hasn't measured; the momentum edge it relies on is modest
(≈ +1.5 to +2.5 pts/yr for the live index ETF, very uneven by year) and it says so.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data.panel import CACHE_DIR, index_constituents, universe_panel
from .data.provider import DataUnavailable
from .knowledge import knowledge_base
from .nlu import ALIASES
from .orchestrator import Orchestrator
from .signals import factors as F
from .signals.advisor import STOP_PCT, market_state, next_rebalance

log = logging.getLogger("stockintel.hub")

INDICES = [("^NSEI", "NIFTY 50"), ("^BSESN", "SENSEX"), ("^NSEBANK", "BANK NIFTY"),
           ("^CRSLDX", "NIFTY 500")]
BUY_RANK, KEEP_RANK = 20, 40
PANEL_TTL_S = 6 * 3600


class Hub:
    def __init__(self, orch: Orchestrator, preload: bool = False):
        self.orch = orch
        self._lock = threading.Lock()
        self._panel = None
        self._panel_at = 0.0
        self._rank: Dict[str, Any] = {}
        self._names: Optional[pd.DataFrame] = None
        if preload:
            threading.Thread(target=self.panel, daemon=True).start()

    # --- shared data ------------------------------------------------------------------
    def names(self) -> pd.DataFrame:
        if self._names is None:
            index_constituents("nifty500")          # refreshes the cached CSV if stale
            raw = pd.read_csv(CACHE_DIR / "nifty500.csv")
            nifty200 = set(index_constituents("nifty200")["symbol"])
            self._names = pd.DataFrame({"symbol": raw["Symbol"].str.strip(),
                                        "name": raw["Company Name"].str.strip(),
                                        "sector": raw["Industry"].str.strip()})
            self._names["nifty200"] = self._names["symbol"].isin(nifty200)
        return self._names

    def panel(self):
        with self._lock:
            if self._panel is None or time.time() - self._panel_at > PANEL_TTL_S:
                self._panel = universe_panel("nifty500", "10y")
                self._panel_at = time.time()
                self._rank = {}
            return self._panel

    def ranking(self) -> pd.DataFrame:
        p = self.panel()
        asof = p.close.index[-1]
        if self._rank.get("asof") != asof:
            n = self.names()
            uni = pd.Index([s for s in n.loc[n["nifty200"], "symbol"] if s in p.close.columns])
            mom = F.momentum_score(p.close, asof, uni)
            mom["rank"] = np.arange(1, len(mom) + 1)
            self._rank = {"asof": asof, "table": mom}
        return self._rank["table"]

    def bench(self) -> pd.Series:
        return self.orch.service().history("^NSEI")["close"]

    # --- search -------------------------------------------------------------------------
    def search(self, q: str, limit: int = 10) -> List[Dict[str, Any]]:
        ql = q.strip().lower()
        if not ql:
            return []
        n = self.names()
        sym_hit = n[n["symbol"].str.lower().str.startswith(ql)]
        name_hit = n[n["name"].str.lower().str.contains(ql, regex=False)]
        out = pd.concat([sym_hit, name_hit]).drop_duplicates("symbol")
        rows = out.head(limit).to_dict("records")
        known = set(n["symbol"])
        for alias, (sym, mkt) in ALIASES.items():
            if ql in alias and sym not in known and not sym.startswith("^") and len(rows) < limit:
                rows.append({"symbol": sym, "name": alias.title(), "sector": mkt, "nifty200": False})
        return rows

    # --- stock page ---------------------------------------------------------------------
    def overview(self, symbol: str) -> Dict[str, Any]:
        svc = self.orch.service()
        df = svc.history(symbol)
        try:
            info = svc.info(symbol)
        except DataUnavailable:
            info = {}
        c = df["close"]
        last, prev = float(c.iloc[-1]), float(c.iloc[-2])
        yr = df.iloc[-252:]

        def ret(n: int) -> Optional[float]:
            return round((last / float(c.iloc[-1 - n]) - 1) * 100, 2) if len(c) > n else None
        n = self.names()
        row = n[n["symbol"] == symbol.upper()]
        return {
            "symbol": symbol.upper(), "as_of": str(df.index[-1].date()),
            "name": info.get("longName") or (row["name"].iloc[0] if len(row) else symbol.upper()),
            "sector": info.get("sector") or (row["sector"].iloc[0] if len(row) else None),
            "industry": info.get("industry"), "price": round(last, 2),
            "change": round(last - prev, 2), "change_pct": round((last / prev - 1) * 100, 2),
            "returns": {"1W": ret(5), "1M": ret(21), "6M": ret(126), "1Y": ret(252), "3Y": ret(756),
                        # providers return ~1,240 sessions for "5y"; use the full span when it's that close
                        "5Y": ret(min(1250, len(c) - 1)) if len(c) > 1200 else None},
            "high_52w": round(float(yr["high"].max()), 2), "low_52w": round(float(yr["low"].min()), 2),
            "stats": {"Market cap": info.get("marketCap"), "P/E (TTM)": info.get("trailingPE"),
                      "P/B": info.get("priceToBook"),
                      # computed, not read: providers disagree on whether dividendYield is a fraction or a percent
                      "Dividend yield": (float(info["dividendRate"]) / last * 100) if info.get("dividendRate") else None,
                      "ROE": info.get("returnOnEquity"), "Debt to equity": info.get("debtToEquity"),
                      "Beta": info.get("beta"), "Book value": info.get("bookValue")},
            "about": info.get("longBusinessSummary"), "website": info.get("website"),
            "in_nifty200": bool(len(row) and row["nifty200"].iloc[0]),
        }

    def chart(self, symbol: str, rng: str) -> Dict[str, Any]:
        days = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1249}
        if rng not in days:
            raise ValueError(f"range must be one of {list(days)}")
        df = self.orch.service().history(symbol).iloc[-(days[rng] + 1):]
        return {"symbol": symbol.upper(), "range": rng, "dates": [str(d.date()) for d in df.index],
                "close": [round(float(x), 2) for x in df["close"]],
                "volume": [float(x) for x in df["volume"]]}

    def technical(self, symbol: str, bars: int = 180, horizon: int = 20) -> Dict[str, Any]:
        """Everything the candlestick chart draws. Every overlay carries its tested
        track record so a clean-looking shape is never shown as a proven signal."""
        from .analysis import candlesticks, indicators as ind, patterns, technical
        from .analysis.forecast import ewma_sigma
        from .analysis.validation import event_study, with_prior
        from .analysis.stats import norm_ppf
        df = self.orch.service().history(symbol)
        if len(df) < 120:
            raise DataUnavailable(f"only {len(df)} bars for {symbol}; need 120 for chart overlays")
        bars = max(20, min(bars, len(df)))
        view = df.iloc[-bars:]
        dates = [str(d.date()) for d in view.index]
        close = df["close"]
        priors = {f: self.orch.store.pattern_priors(f) for f in ("candlestick", "chart")}

        # candle markers inside the view, each with its pooled or own-history record
        det = candlesticks.detect_all(df)
        records: Dict[str, Dict[str, Any]] = {}
        markers = []
        for name in det.columns:
            hits = det[name].iloc[-bars:]
            if not hits.any():
                continue
            direction = candlesticks.DIRECTION[name]
            if name not in records:
                own = event_study(close, det[name], candlesticks.VALIDATION_HORIZON, direction)
                records[name] = _record_summary(with_prior(own, priors["candlestick"].get(name)))
            for d in hits.index[hits.values]:
                markers.append({"date": str(d.date()), "pattern": name, "label": name.replace("_", " "),
                                "direction": direction, "high": round(float(df.loc[d, "high"]), 2),
                                "low": round(float(df.loc[d, "low"]), 2), "record": records[name]})

        # chart patterns with geometry and tested record
        pr = patterns.analyze(df, priors["chart"])
        pats = []
        for p_ in (pr.details.get("patterns", []) if pr.available else []):
            rec = _record_summary(p_.get("historical_outcomes", {}))
            pats.append({k: p_[k] for k in ("pattern", "direction", "start_date", "end_date", "confidence",
                                            "confirmation_status", "breakout_level", "target",
                                            "supporting_features", "geometry")} | {"record": rec})

        tech = technical.analyze(df)
        sma50, sma200 = ind.sma(close, 50).iloc[-bars:], ind.sma(close, 200).iloc[-bars:]

        # forward range: no directional forecast (none beat the base rate), only
        # the volatility cone that was calibrated out of sample (≈79% in the 80% band)
        last = float(close.iloc[-1])
        sigma = float(ewma_sigma(close).iloc[-1])
        future = [str(d.date()) for d in pd.bdate_range(view.index[-1], periods=horizon + 1)[1:]]
        z80, z50 = norm_ppf(0.9), norm_ppf(0.75)
        cone = [{"date": d, "mid": round(last, 2),
                 "lo80": round(last * float(np.exp(-z80 * sigma * np.sqrt(k))), 2),
                 "hi80": round(last * float(np.exp(z80 * sigma * np.sqrt(k))), 2),
                 "lo50": round(last * float(np.exp(-z50 * sigma * np.sqrt(k))), 2),
                 "hi50": round(last * float(np.exp(z50 * sigma * np.sqrt(k))), 2)} for k, d in enumerate(future, 1)]
        fwd = close.shift(-horizon) / close - 1
        base_up = float((fwd.dropna() > 0).mean())
        # trendlines of the current pattern, extended over the projection window
        projections = []
        for p_ in pats:
            for ln in p_["geometry"].get("lines", []):
                i0 = df.index.get_loc(pd.Timestamp(ln["from"][0]))
                i1 = df.index.get_loc(pd.Timestamp(ln["to"][0]))
                if i1 <= i0 or ln["label"] in ("Neckline", "Rim"):
                    continue
                slope = (ln["to"][1] - ln["from"][1]) / (i1 - i0)
                projections.append({"label": f"{ln['label']} extended", "from": [ln["to"][0], ln["to"][1]],
                                    "to": [future[-1], round(ln["to"][1] + slope * horizon, 2)]})
        return {
            "symbol": symbol.upper(), "as_of": dates[-1], "dates": dates,
            "open": [round(float(x), 2) for x in view["open"]], "high": [round(float(x), 2) for x in view["high"]],
            "low": [round(float(x), 2) for x in view["low"]], "close": [round(float(x), 2) for x in view["close"]],
            "volume": [float(x) for x in view["volume"]],
            "sma50": [None if np.isnan(x) else round(float(x), 2) for x in sma50],
            "sma200": [None if np.isnan(x) else round(float(x), 2) for x in sma200],
            "support": tech.details.get("support_levels", []), "resistance": tech.details.get("resistance_levels", []),
            "markers": markers, "patterns": pats, "projections": projections,
            "cone": cone, "cone_note": (f"Grey band: where {horizon} trading days could plausibly end, from current "
                                        f"volatility ({sigma * np.sqrt(252) * 100:.0f}% a year). Out of sample it held "
                                        f"about 79% of outcomes in the 80% band. It shows size of move, not direction: "
                                        f"historically this stock was up after {horizon} days {base_up:.0%} of the time, "
                                        "and no model we tested predicted direction better than that."),
            "base_rate_up": round(base_up, 3), "horizon": horizon,
        }

    INTRADAY = {"1D": ("1d", "5m"), "5D": ("5d", "15m"), "1M": ("1mo", "60m")}

    def intraday(self, symbol: str, rng: str = "5D") -> Dict[str, Any]:
        """Intraday candles with the daily analysis drawn on top: support/resistance,
        previous close, VWAP, the 10% stop and the next-5-day 80% range."""
        from .analysis import technical
        from .analysis.forecast import ewma_sigma
        from .analysis.stats import norm_ppf
        from .config import resolve_symbol
        if rng not in self.INTRADAY:
            raise ValueError(f"range must be one of {list(self.INTRADAY)}")
        period, interval = self.INTRADAY[rng]
        bars = self.orch.provider.intraday(resolve_symbol(symbol), period, interval)
        daily = self.orch.service().history(symbol)
        prior = daily[daily.index < bars.index[-1].normalize()]
        prev_close = float(prior["close"].iloc[-1]) if len(prior) else None
        tp = (bars["high"] + bars["low"] + bars["close"]) / 3
        day = bars.index.normalize()
        vwap = (tp * bars["volume"]).groupby(day).cumsum() / bars["volume"].groupby(day).cumsum().replace(0, np.nan)
        tech = technical.analyze(daily)
        last = float(bars["close"].iloc[-1])
        sigma = float(ewma_sigma(daily["close"]).iloc[-1])
        z = float(norm_ppf(0.9))
        return {"symbol": symbol.upper(), "range": rng, "interval": interval,
                "times": [d.strftime("%Y-%m-%d %H:%M") for d in bars.index],
                "open": [round(float(x), 2) for x in bars["open"]], "high": [round(float(x), 2) for x in bars["high"]],
                "low": [round(float(x), 2) for x in bars["low"]], "close": [round(float(x), 2) for x in bars["close"]],
                "volume": [float(x) for x in bars["volume"]],
                "vwap": [None if np.isnan(x) else round(float(x), 2) for x in vwap],
                "levels": {"prev_close": None if prev_close is None else round(prev_close, 2),
                           "support": tech.details.get("support_levels", [])[:2],
                           "resistance": tech.details.get("resistance_levels", [])[:2],
                           "stop_10pct": round(last * 0.9, 2),
                           "range5_lo": round(last * float(np.exp(-z * sigma * np.sqrt(5))), 2),
                           "range5_hi": round(last * float(np.exp(z * sigma * np.sqrt(5))), 2)},
                "note": "Intraday bars from Yahoo, may lag ~15 minutes."}

    def indicators(self, symbol: str, bars: int = 126) -> Dict[str, Any]:
        """Series for the analytics panels: RSI, MACD, drawdown, rolling volatility."""
        from .analysis import indicators as ind
        df = self.orch.service().history(symbol)
        c = df["close"]
        macd, sig, hist = ind.macd(c)
        dd = c / c.cummax() - 1
        vol = ind.historical_volatility(c, 20)
        view = slice(-bars, None)
        clean = lambda s_: [None if np.isnan(x) else round(float(x), 4) for x in s_.iloc[view]]  # noqa: E731
        return {"symbol": symbol.upper(), "dates": [str(d.date()) for d in c.index[view]],
                "close": clean(c), "rsi": clean(ind.rsi(c)), "macd": clean(macd), "signal": clean(sig), "hist": clean(hist),
                "drawdown_pct": [round(float(x) * 100, 2) for x in dd.iloc[view]],
                "vol20_pct": [None if np.isnan(x) else round(float(x) * 100, 2) for x in vol.iloc[view]],
                "max_drawdown_5y_pct": round(float(dd.min()) * 100, 2)}

    def insider(self, symbol: str, days: int = 365) -> List[Dict[str, Any]]:
        from .data import insider
        try:
            df = insider.load()
        except DataUnavailable:
            return []
        cut = pd.Timestamp(date.today()) - pd.Timedelta(days=days)
        mine = df[(df["symbol"] == symbol.upper()) & (df["broadcast"] >= cut)].sort_values("broadcast", ascending=False)
        return [{"date": str(r.broadcast.date()), "category": r.category,
                 "action": "Bought" if r.direction > 0 else "Sold",
                 "value_cr": round(float(r.value) / 1e7, 2), "shares": r.shares}
                for r in mine.head(20).itertuples()]

    # --- verdict --------------------------------------------------------------------------
    def verdict(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.upper()
        s = self.orch.session("api")      # shared with /stocks/{t}/analysis, so one computation serves both
        a = self.orch.analysis(s, sym)
        d = a.decision
        rank_tbl = self.ranking()
        m = market_state(self.bench(), self.bench().index[-1])
        positions, _, _ = self.orch.holdings()
        held = sym in positions and positions[sym].quantity > 0
        why_buy: List[str] = []
        why_not: List[str] = []
        in_uni = sym in rank_tbl.index
        rank = int(rank_tbl.loc[sym, "rank"]) if in_uni else None
        total = len(rank_tbl)
        if in_uni:
            r = rank_tbl.loc[sym]
            line = (f"Momentum rank {rank} of {total} in the Nifty 200 (6-month {r.r6 * 100:+.1f}%, "
                    f"12-month {r.r12 * 100:+.1f}%, risk-adjusted by {r.sigma * 100:.0f}% volatility).")
            (why_buy if rank <= KEEP_RANK else why_not).append(line)
        for e in d.supporting_evidence[:3] if d.score >= 0 else d.contradicting_evidence[:3]:
            why_buy.append(e["claim"])
        for e in d.contradicting_evidence[:3] if d.score >= 0 else d.supporting_evidence[:3]:
            why_not.append(e["claim"])
        fund = a.results.get("fundamental")
        if fund is not None and fund.available:
            (why_buy if fund.score > 0.1 else why_not if fund.score < -0.1 else why_buy).append(
                f"Fundamentals look {fund.state} (trend: {fund.details.get('trend')}).")
        risk = a.results.get("risk")
        if risk is not None and risk.available:
            vol = risk.details["annualized_vol_pct"]
            if vol > 40:
                why_not.append(f"High volatility: {vol}% a year — expect large swings.")
            dd = risk.details["current_drawdown_pct"]
            if dd < -25:
                why_not.append(f"Price is {abs(dd):.0f}% below its peak in the loaded history.")
        if m["trend"] == "down":
            why_not.append(f"The overall market is in a downtrend (Nifty below its 10-month average "
                           f"{m['sma_10m']:,.0f}); new buys carry more risk.")
        if m["momentum_crash_risk"]:
            why_not.append("Market is in a bear-then-rebound regime where momentum stocks have crashed before.")
        ins = self.insider(sym, 90)
        bought = sum(x["value_cr"] for x in ins if x["action"] == "Bought" and "romoter" in x["category"])
        sold = sum(x["value_cr"] for x in ins if x["action"] == "Sold" and "romoter" in x["category"])
        if bought:
            why_buy.append(f"Promoters bought ₹{bought:,.2f} cr in the last 90 days (weak signal on its own).")
        if sold:
            why_not.append(f"Promoters sold ₹{sold:,.2f} cr in the last 90 days.")

        if not in_uni:
            verdict, tone = "NO SIGNAL", "neutral"
            headline = ("Outside the Nifty 200, so the tested buy signal doesn't cover this stock. "
                        f"Research view: {d.decision_support_label}.")
        elif rank <= BUY_RANK:
            if m["momentum_crash_risk"]:
                verdict, tone = "BUY — HALF SIZE", "positive"
                headline = "Top-20 momentum stock, but the market regime calls for half position size."
            else:
                verdict, tone = ("HOLD / ADD" if held else "BUY"), "positive"
                headline = "Top-20 momentum stock in the Nifty 200 — it qualifies for the signal portfolio."
        elif rank <= KEEP_RANK:
            verdict, tone = ("HOLD" if held else "WAIT"), "neutral"
            headline = (f"Rank {rank}: good enough to keep if you own it (top {KEEP_RANK}), "
                        f"not strong enough to buy fresh (top {BUY_RANK}).")
        else:
            verdict, tone = ("SELL AT REBALANCE" if held else "DON'T BUY"), "negative"
            headline = f"Rank {rank} of {total}: momentum is weak, which is where the signal says to avoid."
        price = a.price
        return {
            "symbol": sym, "verdict": verdict, "tone": tone, "headline": headline, "held": held,
            "momentum_rank": rank, "universe_size": total, "research_label": d.decision_support_label,
            "research_score": d.score, "research_confidence": d.confidence, "conflict": d.conflict,
            "why_buy": why_buy[:7], "why_not": why_not[:7],
            "plan": {"stop_loss": round(price * (1 - STOP_PCT), 2) if tone == "positive" else None,
                     "next_review": next_rebalance(date.today()),
                     "position_size": "equal slice of your momentum portfolio (≈5% with 20 stocks)"
                     if tone == "positive" else None},
            "market": m, "analysis_timestamp": a.analysis_timestamp, "data_as_of": a.quality["last_date"],
            "evidence_note": ("The buy/sell rule is NSE's momentum method. Its live ETF beat the Nifty 500 by "
                              "about 1.5–2.5 points a year (Sep 2022–Sep 2026), but unevenly: +15 points in 2023, −10 in 2025, "
                              "and backtests fell up to 35% in bad stretches. "
                              "Pattern, candle and forecast signals were tested and showed no edge, so they "
                              "only appear as context."),
        }

    def portfolio_history(self) -> Dict[str, Any]:
        """Daily value of the holdings as they actually were, rebuilt from transactions,
        next to the money put in. Cash is excluded (its history isn't recorded)."""
        p = self.orch.store.portfolio(self.orch.portfolio_name)
        if not p or not p["transactions"]:
            return {"dates": [], "value": [], "invested": []}
        txs = sorted(p["transactions"], key=lambda t: t["trade_date"])
        start = pd.Timestamp(txs[0]["trade_date"])
        closes = {}
        for sym in {t["symbol"] for t in txs}:
            try:
                closes[sym] = self.orch.service().history(sym)["close"]
            except DataUnavailable:
                continue
        if not closes:
            return {"dates": [], "value": [], "invested": []}
        px = pd.DataFrame(closes).sort_index().ffill()
        px = px[px.index >= start]
        qty = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        invested = pd.Series(0.0, index=px.index)
        for t in txs:
            if t["symbol"] not in qty.columns:
                continue
            d = pd.Timestamp(t["trade_date"])
            sign = 1 if t["side"] == "BUY" else -1
            qty.loc[qty.index >= d, t["symbol"]] += sign * t["quantity"]
            invested[invested.index >= d] += sign * t["quantity"] * t["price"]
        value = (qty * px.fillna(0)).sum(axis=1)
        step = max(1, len(value) // 400)
        return {"dates": [str(d.date()) for d in value.index[::step]],
                "value": [round(float(v), 2) for v in value.iloc[::step]],
                "invested": [round(float(v), 2) for v in invested.iloc[::step]]}

    # --- open-source models --------------------------------------------------------------
    def ml_enabled(self) -> bool:
        import importlib.util
        import os
        return os.environ.get("STOCKINTEL_ML", "1") != "0" and importlib.util.find_spec("torch") is not None

    def ai_forecast(self, symbol: str, model: str = "chronos") -> Dict[str, Any]:
        """Chronos quantile fan or Kronos projected candles, cached for the day, with the
        model's measured out-of-sample record attached."""
        from .performance import MODELS_PATH
        if model not in ("chronos", "kronos"):
            raise ValueError("model must be chronos or kronos")
        if not self.ml_enabled():
            raise DataUnavailable("open-source models are disabled on this server (STOCKINTEL_ML=0 or torch missing)")
        df = self.orch.service().history(symbol)
        key = (symbol.upper(), model, str(df.index[-1].date()))
        cache = self.__dict__.setdefault("_ai_cache", {})
        if key not in cache:
            with self._lock:
                if model == "chronos":
                    from .analysis import tsfm
                    horizon = 10
                    q = tsfm.fan(df["close"], horizon)
                    dates = [str(d.date()) for d in pd.bdate_range(df.index[-1], periods=horizon + 1)[1:]]
                    cache[key] = {"dates": dates, **q}
                else:
                    from .analysis import kronos_model
                    cache[key] = kronos_model.next_candles(df, pred_len=5, samples=8)
        record = {}
        if MODELS_PATH.exists():
            import json as _json
            ev = _json.loads(MODELS_PATH.read_text()).get(model, {})
            record = ev
        if model == "chronos":
            h5 = next((h for h in record.get("horizons", []) if h["horizon"] == 5), None)
            note = ("Chronos-Bolt (open-source AI model) forecast range. "
                    + (f"Tested on {h5['n']} past 5-day forecasts: direction right {h5['chronos_direction_acc']:.0%} vs {h5['base_rate_acc']:.0%} for the base rate, "
                       f"80% band held {h5['chronos_cover80']:.0%} of outcomes — as good as plain volatility, not better."
                       if h5 else "evaluation not run yet (`stockintel evaluate-models`)."))
        else:
            note = ("Kronos (open-source candlestick AI) projected candles — EXPERIMENTAL. Tested on "
                    + (f"{record['n']} past forecasts: direction right {record['direction_acc']:.0%} vs {record['base_rate_acc']:.0%} base rate, "
                       f"and its 5-day price error was {record['mae_pct']:.1f}% vs {record['no_change_mae_pct']:.1f}% for assuming no change. "
                       "Treat these candles as a picture of what the model imagines, not a forecast."
                       if record else "evaluation not run yet."))
        return {"symbol": symbol.upper(), "model": model, "as_of": key[2], **cache[key], "record": record, "note": note}

    # --- portfolio builder ---------------------------------------------------------------
    def build_portfolio(self, capital: float, strategy: str = "momentum", risk_mode: str = "balanced",
                        n: int = 10, exclude_sectors: Optional[List[str]] = None) -> Dict[str, Any]:
        """A fresh portfolio for `capital`: the plan's buys, plus what that basket's risk
        and last year would have looked like (a backcast, not a forecast)."""
        from .portfolio import Position, analyze as analyze_portfolio
        from .signals.advisor import plan
        if not 3 <= n <= 25:
            raise ValueError("number of stocks must be between 3 and 25")
        names = self.names()
        excl = {x.lower() for x in exclude_sectors or []}
        uni = [r.symbol for r in names.itertuples() if r.nifty200 and r.sector.lower() not in excl]
        p = plan(self.panel(), self.bench(), capital, [], strategy=strategy, risk_mode=risk_mode, n=n, universe=uni)
        buys = [a for a in p.actions if a.action == "BUY"]
        sector = dict(zip(names["symbol"], names["sector"]))
        name_of = dict(zip(names["symbol"], names["name"]))
        close = self.panel().close
        positions = {a.symbol: Position(a.symbol, a.shares, a.value, 0.0, sector.get(a.symbol)) for a in buys}
        spent = sum(a.value for a in buys)
        risk, curve = {}, {"dates": [], "basket": [], "nifty": []}
        if positions:
            prices = {s: close[s].dropna() for s in positions}
            rep = analyze_portfolio(positions, capital - spent, prices, "moderate", bench=self.bench())
            risk = {k: rep.risk.get(k) for k in ("portfolio_vol_pct", "max_drawdown_pct", "var_95_1d_pct", "beta",
                                                  "diversification_ratio")}
            risk["sector_exposure_pct"] = rep.sector_exposure_pct
            risk["effective_holdings"] = rep.concentration["effective_holdings"]
            w = pd.Series({a.symbol: a.value for a in buys}) / spent
            px = close[list(w.index)].iloc[-253:].ffill()
            basket = (px / px.iloc[0] * w).sum(axis=1)
            b = self.bench().reindex(px.index).ffill()
            step = 3
            curve = {"dates": [str(d.date()) for d in px.index[::step]],
                     "basket": [round(float(v) * capital, 0) for v in basket.iloc[::step]],
                     "nifty": [round(float(v) * capital, 0) for v in (b / b.iloc[0]).iloc[::step]]}
        return {"capital": capital, "strategy": strategy, "risk_mode": risk_mode, "n": n,
                "invest_fraction": p.invest_fraction, "market": p.market,
                "holdings": [{"symbol": a.symbol, "name": name_of.get(a.symbol, a.symbol),
                              "sector": sector.get(a.symbol, "Unknown"), "shares": a.shares, "price": a.price,
                              "value": a.value, "weight_pct": round(a.value / capital * 100, 2), "rank": a.rank,
                              "stop": a.stop_level, "why": a.reasons[0] if a.reasons else ""} for a in buys],
                "cash_left": round(capital - spent, 2), "risk": risk, "backcast": curve,
                "warnings": p.warnings, "next_rebalance": p.next_rebalance,
                "note": ("The backcast shows how this exact basket moved over the past year — the stocks were picked "
                         "because they rose, so it will look good by construction. It is not a forecast.")}

    # --- explore ----------------------------------------------------------------------------
    def indices(self) -> List[Dict[str, Any]]:
        out = []
        for sym, label in INDICES:
            try:
                c = self.orch.service().history(sym)["close"]
            except DataUnavailable:
                continue
            last, prev = float(c.iloc[-1]), float(c.iloc[-2])
            out.append({"symbol": sym, "name": label, "value": round(last, 2),
                        "change": round(last - prev, 2), "change_pct": round((last / prev - 1) * 100, 2),
                        "spark": [round(float(x), 2) for x in c.iloc[-60:]]})
        return out

    def explore(self) -> Dict[str, Any]:
        p = self.panel()
        n = self.names().set_index("symbol")
        uni = [s for s in n.index[n["nifty200"]] if s in p.close.columns]
        c = p.close[uni].iloc[-2:]
        chg = (c.iloc[-1] / c.iloc[-2] - 1).dropna().sort_values()

        def rows(syms):
            return [{"symbol": s, "name": n.loc[s, "name"], "price": round(float(p.close[s].iloc[-1]), 2),
                     "change_pct": round(float(chg[s]) * 100, 2)} for s in syms]
        rk = self.ranking()
        top = [{"symbol": s, "name": n.loc[s, "name"] if s in n.index else s, "rank": int(rk.loc[s, "rank"]),
                "r6_pct": round(float(rk.loc[s, "r6"]) * 100, 1), "r12_pct": round(float(rk.loc[s, "r12"]) * 100, 1),
                "price": round(float(p.close[s].iloc[-1]), 2)} for s in rk.index[:10]]
        from .data import insider
        try:
            buys = insider.recent_buys(insider.load(), days=30, min_value=1e7)
            agg = buys.groupby("symbol").agg(value=("value", "sum"), day=("day", "max"),
                                             filings=("filings", "sum")).sort_values("value", ascending=False).head(8)
            ins = [{"symbol": s_, "name": n.loc[s_, "name"] if s_ in n.index else s_,
                    "date": str(r.day.date()), "value_cr": round(float(r.value) / 1e7, 2), "filings": int(r.filings)}
                   for s_, r in agg.iterrows()]
        except DataUnavailable:
            ins = []
        return {"as_of": str(p.close.index[-1].date()), "indices": self.indices(),
                "market": market_state(self.bench(), self.bench().index[-1]),
                "gainers": rows(list(chg.index[-5:][::-1])), "losers": rows(list(chg.index[:5])),
                "momentum_top": top, "insider_buys": ins}

    # --- learn --------------------------------------------------------------------------------
    def learn_index(self) -> List[Dict[str, Any]]:
        kb = knowledge_base()
        topics: Dict[str, List[Dict[str, str]]] = {}
        for ch in kb.chunks:
            text = " ".join(ch.text.split("\n"))
            topics.setdefault(ch.doc, []).append({"title": ch.title, "text": text})
        pretty = {"technical_indicators.md": "Technical indicators", "candlesticks.md": "Candlesticks",
                  "chart_patterns.md": "Chart patterns", "fundamentals.md": "Fundamentals & valuation",
                  "risk_metrics.md": "Risk & performance", "methodology.md": "How this system decides",
                  "india_market.md": "Indian market basics"}
        return [{"topic": pretty.get(doc, doc), "lessons": lessons} for doc, lessons in topics.items()]


def _record_summary(h: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, chart-friendly version of an event-study result."""
    pooled = h.get("pooled")
    src = pooled or h
    verdict = h.get("verdict", "no historical occurrences")
    edge = verdict.startswith("historically significant")
    return {"verdict": verdict, "has_edge": edge,
            "events": src.get("events", 0), "stocks": src.get("stocks"),
            "excess_pct": src.get("excess_mean_pct"), "z": src.get("z"),
            "scope": "48 NSE stocks pooled" if pooled else "this stock's history",
            "summary": (f"Tested {src.get('events', 0)} times"
                        + (f" across {src['stocks']} stocks" if src.get("stocks") else " on this stock")
                        + (f": average excess return {src['excess_mean_pct']:+.2f}%" if src.get("excess_mean_pct") is not None else "")
                        + f" — {verdict}.")}
