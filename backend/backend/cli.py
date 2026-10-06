"""Command-line interface. `stockintel --help` lists every command."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from backend.paths import ARTIFACTS, DATA
from typing import List, Optional

from backend.advisory.calculators import performance
from backend.advisory import report as R
from backend.api.jsonutil import sanitize

DEFAULT_DB = os.environ.get("STOCKINTEL_DB", str(DATA / "stockintel.db"))


def _provider(args):
    if args.demo:
        from backend.database.sources.synthetic import SyntheticProvider
        return SyntheticProvider()
    from backend.database.sources.yahoo import YahooProvider
    return YahooProvider()


def _orchestrator(args, llm_mode: str = "off"):
    from backend.api.orchestrator import Orchestrator
    from backend.database.storage import Store
    return Orchestrator(_provider(args), Store(args.db), llm_mode=llm_mode,
                        use_rss_news=not args.demo)


def cmd_analyze(args) -> int:
    orch = _orchestrator(args, args.llm)
    s = orch.session("cli")
    a = orch.analysis(s, args.symbol.upper(), horizon=args.horizon)
    if args.json:
        print(json.dumps(sanitize(a.to_dict()), indent=2))
    elif args.report:
        body = R.render_report(a)
        orch.store.save_report(a.symbol, body)
        print(body)
    else:
        if args.llm == "narrate":
            print(orch.handle(f"Analyze {args.symbol}", "cli").text)
        else:
            print(R.render_analysis(a))
    return 0


def cmd_chat(args) -> int:
    orch = _orchestrator(args, args.llm)
    print("StockIntel research chat. Type 'help' for examples, 'quit' to exit.")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if msg.lower() in ("quit", "exit"):
            return 0
        if msg:
            reply = orch.handle(msg, "cli")
            print(f"\n{reply.text}\n\n[{reply.intent} · {reply.timestamp}]")


def cmd_screen(args) -> int:
    from backend.advisory.calculators.screener import screen
    orch = _orchestrator(args)
    positions, _, _ = orch.holdings()
    res = screen(orch.service(), args.budget, args.risk, args.horizon, positions,
                 include_sectors=args.include, exclude_sectors=args.exclude)
    print(json.dumps(sanitize({k: v for k, v in res.items() if k != "also_scored"}), indent=2)
          if args.json else R.render_screen(res))
    return 0


def cmd_query(args) -> int:
    orch = _orchestrator(args)
    print(orch.handle(args.question, "cli").text)
    return 0


def cmd_portfolio(args) -> int:
    orch = _orchestrator(args)
    if args.action == "import":
        data = json.loads(Path(args.file).read_text())
        n = orch.save_portfolio(float(data.get("cash", 0)), data.get("risk_profile", "moderate"),
                                data.get("horizon", "medium"), data["transactions"])
        print(f"Imported {n} transactions into portfolio '{orch.portfolio_name}'.")
        return 0
    if args.action == "what-if":
        if not args.amount or not args.symbols:
            raise SystemExit("what-if needs --amount and --symbols")
        print(orch.handle(f"If I add {args.amount:.0f} rupees to {' '.join(args.symbols)} how does my "
                          "portfolio risk change?", "cli").text)
        return 0
    print(orch.handle("How is my portfolio performing?", "cli").text)
    return 0


def cmd_backtest(args) -> int:
    import numpy as np
    from backend.advisory.calculators.backtest import strategies as S
    from backend.config import SLIPPAGE_BPS, TRANSACTION_COST_BPS
    from backend.database.sources.provider import DataUnavailable
    from backend.advisory.calculators.screener import load_universe
    orch = _orchestrator(args)
    svc = orch.service()
    bench = svc.benchmark()
    strategies = dict(S.BASELINES)
    strategies["technical_engine"] = S.technical_engine
    strategies["forecast_logistic"] = lambda d: S.forecast_model(d, bench=bench)
    if args.fusion:
        strategies["price_fusion"] = lambda d: S.price_fusion(d, bench)
    symbols = (list(load_universe()["symbol"]) if args.universe else
               [x.strip().upper() for x in (args.symbols or args.symbol or "").split(",") if x.strip()])
    if not symbols:
        raise SystemExit("give a SYMBOL, --symbols A,B or --universe")
    cols = ["strategy", "cagr_pct", "sharpe", "sortino", "max_drawdown_pct", "win_rate",
            "profit_factor", "trade_count", "exposure_pct", "turnover_per_year", "costs_paid_pct"]
    per_stock = {}
    for sym in symbols:
        try:
            df = svc.history(sym)
        except DataUnavailable as exc:
            print(f"skipping {sym}: {exc}")
            continue
        start = str(df.index[min(len(df) - 2, 300)].date())
        res = S.compare(df, strategies, start=start)
        per_stock[sym] = {row["strategy"]: row for row in res["table"]}
        if len(symbols) == 1:
            print(f"{sym} backtest {res['window']} — next-open execution, "
                  f"{TRANSACTION_COST_BPS + SLIPPAGE_BPS:g} bps cost per unit of turnover")
            print(" | ".join(f"{c:>14}" for c in cols))
            for row in res["table"]:
                print(" | ".join(f"{str(row.get(c)):>14}" for c in cols))
            print("\n" + res["caveat"])
    if len(per_stock) > 1:
        print(f"Cross-sectional summary over {len(per_stock)} stocks (net of "
              f"{TRANSACTION_COST_BPS + SLIPPAGE_BPS:g} bps per unit turnover, next-open fills)")
        print(f"{'strategy':<22}{'median CAGR%':>13}{'median Sharpe':>15}{'mean MaxDD%':>13}"
              f"{'beats B&H Sharpe':>18}")
        for name in strategies:
            rows = [per_stock[s][name] for s in per_stock]
            bh = [per_stock[s]["buy_and_hold"]["sharpe"] for s in per_stock]
            wins = sum(r["sharpe"] > b for r, b in zip(rows, bh))
            print(f"{name:<22}{np.median([r['cagr_pct'] for r in rows]):>13.2f}"
                  f"{np.median([r['sharpe'] for r in rows]):>15.3f}"
                  f"{np.mean([r['max_drawdown_pct'] for r in rows]):>13.2f}"
                  f"{wins:>12}/{len(rows)}")
        print("\nSame window per stock; today's large caps only (survivorship bias favours buy-and-hold).")
    return 0


def cmd_validate_patterns(args) -> int:
    from backend.advisory.statistics import candlesticks, patterns
    from backend.advisory.statistics.stats import norm_ppf
    from backend.advisory.statistics.validation import pooled_event_study
    from backend.database.sources.provider import DataUnavailable
    from backend.advisory.calculators.screener import load_universe
    orch = _orchestrator(args)
    svc = orch.service()
    symbols = list(load_universe()["symbol"]) if args.universe else [s.strip().upper() for s in args.symbols.split(",")]
    data = {}
    for sym in symbols:
        try:
            data[sym] = svc.history(sym)
        except DataUnavailable as exc:
            print(f"skipping {sym}: {exc}")
    candle = {s: candlesticks.detect_all(df) for s, df in data.items()}
    chart = {s: patterns.scan_history(df) for s, df in data.items()}
    chart_keys = sorted(set().union(*[set(c) for c in chart.values()])) if chart else []
    tests = len(candlesticks.DIRECTION) + len(chart_keys)
    z_bonf = norm_ppf(1 - 0.025 / max(1, tests))
    print(f"Pooled event study over {len(data)} stocks; {tests} patterns tested, so the Bonferroni "
          f"threshold is |z| >= {z_bonf:.2f} (5% family-wise error).")
    print(f"\nCandlestick formations, {candlesticks.VALIDATION_HORIZON}-day forward excess return")
    rows = []
    for name, direction in candlesticks.DIRECTION.items():
        res = pooled_event_study([(data[s]["close"], candle[s][name]) for s in data],
                                 candlesticks.VALIDATION_HORIZON, direction, z_bonf)
        rows.append(("candlestick", name, candlesticks.VALIDATION_HORIZON, res))
        print(f"  {name:<22} events {res['events']:>5} in {res['stocks']:>3} stocks  excess "
              f"{res.get('excess_mean_pct', 'n/a')!s:>7}%  z {res.get('z', 'n/a')!s:>6}  {res['verdict']}")
    print(f"\nChart patterns (causal replay), {patterns.VALIDATION_HORIZON}-day forward excess return")
    for key in chart_keys:
        direction = 1 if key.endswith(":up") else -1 if key.endswith(":down") else 0
        res = pooled_event_study([(data[s]["close"], chart[s][key]) for s in data if key in chart[s]],
                                 patterns.VALIDATION_HORIZON, direction, z_bonf)
        rows.append(("chart", key, patterns.VALIDATION_HORIZON, res))
        print(f"  {key:<32} events {res['events']:>5} in {res['stocks']:>3} stocks  excess "
              f"{res.get('excess_mean_pct', 'n/a')!s:>7}%  z {res.get('z', 'n/a')!s:>6}  {res['verdict']}")
    if args.save:
        for family, name, horizon, res in rows:
            orch.store.save_pattern_validation(family, name, horizon, res, list(data))
        print(f"\nSaved {len(rows)} pooled results; analyses now use them when a stock's own history is thin.")
    print("\nCaveat: the universe is today's large caps (survivorship bias); excess returns are "
          "gross of costs.")
    return 0


def cmd_forecast_eval(args) -> int:
    from backend.advisory.statistics import forecast
    from backend.advisory.statistics.model_registry import ModelRegistry
    orch = _orchestrator(args)
    svc = orch.service()
    models = tuple(args.models.split(","))
    if "climatology" not in models:
        models = ("climatology",) + models
    symbols = args.symbols.split(",")
    horizons = [int(h) for h in args.horizons.split(",")]
    pooled = {h: {m: [] for m in models} for h in horizons}
    for sym in symbols:
        df = svc.history(sym)
        for h in horizons:
            wf = forecast.walk_forward(df, h, models, svc.benchmark())
            if "error" in wf:
                print(f"{sym} h={h}: {wf['error']}")
                continue
            for m, met in wf["metrics"].items():
                pooled[h][m].append(met)
                print(f"{sym:<12} h={h:<3} {m:<12} n={met['n']:<5} brier={met['brier']:.4f} "
                      f"skill={met['brier_skill']!s:<8} auc={met['auc']!s:<6} "
                      f"(high-vol auc {met['by_vol_regime']['high_vol']['auc']}, low-vol "
                      f"{met['by_vol_regime']['low_vol']['auc']})  range coverage {wf['interval_coverage']}")
    summary = {}
    print("\nPooled (prediction-weighted) Brier skill vs climatology:")
    for h in horizons:
        for m in models:
            rows = pooled[h][m]
            if not rows or m == "climatology":
                continue
            n = sum(r["n"] for r in rows)
            skill = sum((r["brier_skill"] or 0) * r["n"] for r in rows) / n
            auc = sum((r["auc"] or 0.5) * r["n"] for r in rows) / n
            summary[f"{m}_h{h}"] = {"brier_skill": round(skill, 4), "auc": round(auc, 3), "n": n}
            print(f"  h={h:<3} {m:<12} skill {skill:+.4f}  auc {auc:.3f}  (n={n}, {len(rows)} stocks)")
    if args.register:
        best = max(summary.items(), key=lambda kv: kv[1]["brier_skill"]) if summary else None
        if best is None:
            print("Nothing to register.")
            return 1
        reg = ModelRegistry(orch.store)
        version = f"{date.today().isoformat()}-{'-'.join(models)}"
        reg.register("forecast", version, training_period="expanding walk-forward, per stock",
                     features=list(forecast.features(svc.history(symbols[0])).columns),
                     hyperparameters={"models": list(models), "min_train": forecast.MIN_TRAIN,
                                      "step": forecast.STEP, "symbols": symbols},
                     metrics={"brier_skill": best[1]["brier_skill"], "by_config": summary},
                     backtest=None, limitations="Per-stock training; survivorship-biased universe.")
        print(f"Registered forecast {version} as a candidate (best {best[0]}). Promote with "
              f"`stockintel models approve forecast {version} --by <your name>`.")
    return 0


def cmd_models(args) -> int:
    from backend.advisory.statistics.model_registry import ModelRegistry, PromotionError
    from backend.database.storage import Store
    reg = ModelRegistry(Store(args.db))
    if args.action == "approve":
        try:
            print(reg.approve(args.name, args.version, args.by))
        except PromotionError as exc:
            print(f"Not approved: {exc}")
            return 1
    elif args.action == "deploy":
        reg.deploy(args.name, args.version)
        print(f"Deployed {args.name} {args.version}.")
    elif args.action == "reject":
        reg.reject(args.name, args.version)
        print(f"Rejected {args.name} {args.version}.")
    else:
        rows = reg.conn.execute("SELECT name, version, status, created_at, approved_by FROM model_versions "
                                "ORDER BY id DESC").fetchall()
        for r in rows:
            print(dict(r))
    return 0


def cmd_alerts(args) -> int:
    orch = _orchestrator(args)
    if args.symbols:
        from backend.user.portfolio.alerts import evaluate
        for sym in args.symbols:
            previous = orch.store.analyses(sym.upper(), limit=1)
            a = orch.service().analyze(sym)
            alert = evaluate(a, previous[0] if previous else None)
            orch.store.save_analysis(a.symbol, a.to_dict())
            if alert:
                orch.store.save_alert(alert.symbol, alert.kind, alert.severity, alert.message, alert.reasons)
                print(f"[{alert.severity.upper()}] {alert.message}\n")
            else:
                print(f"{a.symbol}: no alert conditions.")
        return 0
    print(orch.handle("show alerts", "cli").text)
    return 0


def cmd_monitor(args) -> int:
    from backend.advisory.statistics.monitoring import data_drift, live_skill, resolve_predictions
    orch = _orchestrator(args)
    symbols = sorted({p["symbol"] for p in orch.store.predictions()})
    closes = {}
    for sym in symbols:
        fresh = orch.provider.ohlcv(sym)
        orch.store.save_ohlcv(sym, fresh, orch.provider.name)
        closes[sym] = fresh["close"]
    n = resolve_predictions(orch.store, closes)
    print(f"Resolved {n} matured predictions across {len(symbols)} symbols.")
    skill = live_skill(orch.store)
    if not skill:
        print("No resolved predictions yet; live skill needs at least one horizon to pass.")
    for key, m in skill.items():
        verdict = "better" if m["brier"] < m["brier_if_0.5"] else "NOT better"
        print(f"  {key:<18} n={m['n']:<5} live Brier {m['brier']} ({verdict} than a coin at "
              f"{m['brier_if_0.5']}); predicted {m['mean_predicted']} vs realized {m['hit_rate']}")
    print("\nReturn-distribution drift (PSI, last 60 days vs prior 500):")
    for d in data_drift(closes)[:15]:
        print(f"  {d['symbol']:<14} PSI {d['returns_psi']:<6} {'DRIFT' if d['drift'] else 'stable'}")
    return 0


def _signal_inputs(args, orch=None):
    import yfinance as yf
    from backend.database.sources.panel import CACHE_DIR, index_constituents, universe_panel
    panel = universe_panel("nifty500", "10y", refresh=getattr(args, "refresh", False))
    bench = yf.Ticker("^NSEI").history(period="5y", auto_adjust=True)["Close"]
    bench.index = bench.index.tz_localize(None)
    uni = list(index_constituents(args.universe)["symbol"])
    return panel, bench, uni


def _insider_or_none():
    from backend.database.sources import insider
    from backend.database.sources.provider import DataUnavailable
    try:
        return insider.load()
    except DataUnavailable:
        return None


def cmd_insider_fetch(args) -> int:
    from datetime import date
    from backend.database.sources import insider
    if args.since:
        n = insider.fetch_range(date.fromisoformat(args.since), insider.LEGACY_END)
        print(f"Legacy feed: added {n} promoter/director trades.")
    n = insider.fetch_recent(days=args.days)
    print(f"XBRL feed (last {args.days} days): added {n} promoter/director trades.")
    df = insider.recent_buys(insider.load(), days=30, min_value=args.min_value)
    print(f"\nPromoter market purchases ≥ ₹{args.min_value / 1e7:g} cr in the last 30 days:")
    for r in df.head(25).itertuples():
        print(f"  {r.day.date()}  {r.symbol:<14} ₹{r.value / 1e7:8.2f} cr  ({r.filings} filing(s))")
    print("\nEvidence: +0.7% average 20-day excess return in liquid stocks, median negative; "
          "treat as a weak flag, not a buy signal.")
    return 0


def _stored_holdings(orch):
    from backend.advisory.signals.advisor import Holding
    positions, cash, meta = orch.holdings()
    txs = (meta or {}).get("transactions", [])
    first_buy = {}
    for t in txs:
        if t["side"] == "BUY":
            first_buy.setdefault(t["symbol"], t["trade_date"])
    return [Holding(s, p.quantity, p.avg_cost, first_buy.get(s)) for s, p in positions.items()
            if p.quantity > 0], cash


def cmd_signals(args) -> int:
    from backend.advisory.signals.advisor import plan, render
    orch = _orchestrator(args)
    holdings, stored_cash = _stored_holdings(orch)
    capital = args.capital if args.capital is not None else stored_cash
    if not capital and not holdings:
        raise SystemExit("give --capital (new money to invest) or import a portfolio first")
    panel, bench, uni = _signal_inputs(args)
    p = plan(panel, bench, float(capital or 0.0) or 1.0, holdings, strategy=args.strategy,
             risk_mode=args.risk, n=args.positions, universe=uni, insider_trades=_insider_or_none(),
             measured=performance.momentum_evidence(performance.track_record()))
    orch.store.save_plan(sanitize(p.to_dict()))
    print(json.dumps(sanitize(p.to_dict()), indent=2) if args.json else render(p))
    return 0


def cmd_stops(args) -> int:
    from backend.advisory.signals.advisor import STOP_PCT
    orch = _orchestrator(args)
    holdings, _ = _stored_holdings(orch)
    if not holdings:
        raise SystemExit("no stored portfolio; import one with `stockintel portfolio import`")
    last = orch.store.latest_plan() or {}
    planned = {a["symbol"]: a["stop_level"] for a in last.get("actions", []) if a.get("stop_level")}
    svc = orch.service()
    print(f"Stop check ({STOP_PCT:.0%} stops; levels from the plan of {last.get('as_of', 'n/a')}, "
          "else 10% below your average cost):")
    for h in holdings:
        df = svc.history(h.symbol)
        px, low = float(df["close"].iloc[-1]), float(df["low"].iloc[-1])
        level = planned.get(h.symbol, round(h.avg_cost * (1 - STOP_PCT), 2))
        status = ("STOP HIT — sell at next open" if low <= level
                  else f"ok ({(px / level - 1) * 100:+.1f}% above stop)")
        print(f"  {h.symbol:<12} close ₹{px:,.2f}  stop ₹{level:,.2f}  {status}")
    return 0


def cmd_evaluate_signals(args) -> int:
    import pandas as pd
    import yfinance as yf
    from backend.database.sources.panel import universe_panel
    from backend.advisory.signals import evaluate as E
    panel = universe_panel("nifty500", "10y")
    bm = {}
    for t, name in [("^NSEI", "Nifty 50 (price)"), ("^CRSLDX", "Nifty 500 (price)"),
                    ("MOMENTUM.NS", "Real Nifty200 Momentum 30 ETF")]:
        h = yf.Ticker(t).history(period="12y", auto_adjust=True)["Close"]
        h.index = h.index.tz_localize(None)
        bm[name] = h
    res = E.run(panel, bm["Nifty 50 (price)"], start=args.start, cost_bps=args.cost_bps)
    pd.set_option("display.width", 250)
    tbl = E.table(res, {k: v for k, v in bm.items() if "ETF" not in k})
    print(f"Backtest {args.start} → {panel.close.index[-1].date()}, costs {args.cost_bps} bps per side, "
          "next-open fills, universe = top 200 by traded value among today's Nifty 500")
    print(tbl.to_string())
    print("\nCalendar-year returns (%):")
    print(E.yearly_table(res, {k: v for k, v in bm.items() if "ETF" not in k}).to_string())
    etf = bm["Real Nifty200 Momentum 30 ETF"].dropna().iloc[5:]   # listing-day price is an artefact
    a, b = etf.index[0], etf.index[-1]

    def cagr(s):
        s = s.loc[a:b].dropna()
        return ((s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1) * 100
    print(f"\nSurvivorship check over the live ETF's life ({a.date()} → {b.date()}):")
    print(f"  real momentum ETF {cagr(etf):.1f}% vs Nifty 500 {cagr(bm['Nifty 500 (price)']):.1f}% "
          f"→ real edge {cagr(etf) - cagr(bm['Nifty 500 (price)']):+.1f} pts/yr")
    print(f"  backtest momentum {cagr(res['momentum_semiannual'].equity):.1f}% vs backtest equal-weight "
          f"{cagr(res['equal_weight_universe'].equity):.1f}% → backtest edge "
          f"{cagr(res['momentum_semiannual'].equity) - cagr(res['equal_weight_universe'].equity):+.1f} pts/yr")
    print("  Absolute backtest returns are inflated by survivorship bias; only the edge is comparable.")
    return 0


def cmd_app(args) -> int:
    import secrets
    import threading
    import webbrowser
    import uvicorn
    from backend.api.app import create_app
    # the key is kept between runs so a phone that logged in once stays logged in
    key_file = Path.home() / ".stockintel" / "app_key"
    key = os.environ.get("STOCKINTEL_API_KEY") or (key_file.read_text().strip() if key_file.exists() else "")
    if not key:
        key = secrets.token_hex(24)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(key)
        key_file.chmod(0o600)
    app = create_app(_orchestrator(args, args.llm), api_key=key, preload=True)
    url = f"http://{args.host}:{args.port}/#key={key}"
    print(f"StockIntel is running at {url}\n(the key in the link logs you in; keep it private)")
    if args.lan:
        print(f"On your phone (same Wi-Fi): http://{_lan_ip()}:{args.port}/#key={key}\n"
              "Use this on your home network only: the traffic is not encrypted.")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="0.0.0.0" if args.lan else args.host, port=args.port, log_level="warning")
    return 0


def _lan_ip() -> str:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))        # UDP connect sends nothing; it only picks the Wi-Fi interface
        return s.getsockname()[0]


def cmd_build_performance(args) -> int:
    logging.getLogger("stockintel.performance").setLevel(logging.INFO)
    out = performance.build(_orchestrator(args))
    print(f"Saved {performance.PATH}: {len(out['patterns'])} patterns, "
          f"{len(out['forecast']['horizons'])} forecast horizons, {len(out['momentum']['curves'])} equity curves, "
          f"{len(out['insider'])} insider tests.")
    return 0


def cmd_train_chronos(args) -> int:
    from backend.models.external import tsfm_train
    from backend.database.sources.panel import CACHE_DIR, index_constituents, universe_panel
    logging.getLogger("stockintel.tsfm_train").setLevel(logging.INFO)
    out_dir = ARTIFACTS / "external"
    results = tsfm_train.run(universe_panel("nifty500", "10y").close, list(index_constituents("nifty200")["symbol"]), out_dir)
    path = CACHE_DIR / "chronos_training.json"
    path.write_text(json.dumps(sanitize(results), indent=2))
    print(f"chosen {results['chosen']}; model saved to {out_dir / 'chronos-2-nse'}; results in {path}")
    return 0


def cmd_train_kronos(args) -> int:
    import pandas as pd
    from backend.models.external import kronos_train
    from backend.database.sources.panel import CACHE_DIR, index_constituents, universe_panel
    logging.getLogger("stockintel.kronos_train").setLevel(logging.INFO)
    p = universe_panel("nifty500", "10y")
    frames = {s: pd.DataFrame({"open": p.open[s], "high": p.high[s], "low": p.low[s], "close": p.close[s],
                               "volume": p.volume[s]}).dropna() for s in p.close.columns}
    # sampled inference is the slow part, so it is scored on the most-traded names only
    traded = (p.close * p.volume).iloc[-250:].mean()
    names = [s for s in index_constituents("nifty200")["symbol"] if s in frames]
    eval_names = sorted(names, key=lambda s: -traded.get(s, 0))[:args.eval_stocks]
    out_dir = ARTIFACTS / "external"
    results = kronos_train.run(frames, {s: frames[s] for s in eval_names}, out_dir)
    path = CACHE_DIR / "kronos_training.json"
    path.write_text(json.dumps(sanitize(results), indent=2))
    print(f"chosen lr {results['chosen_lr']}; model saved to {out_dir / 'kronos-nse'}; results in {path}")
    return 0


def cmd_evaluate_models(args) -> int:
    out = performance.evaluate_models(_orchestrator(args).service())
    print(json.dumps(out, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    from backend.database import hfstate
    from backend.api.app import create_app
    saver = None
    if hfstate.repo():                  # restore before the database is opened
        logging.getLogger("stockintel.state").setLevel(logging.INFO)
        hfstate.pull(Path(args.db))
        saver = hfstate.Autosave(Path(args.db), hfstate.PUSH_MINUTES).start()
    app = create_app(_orchestrator(args, args.llm))
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        if saver:
            saver.stop()
    return 0


def cmd_state(args) -> int:
    from backend.database import hfstate
    if not hfstate.repo():
        raise SystemExit("set STOCKINTEL_STATE_REPO (e.g. you/stockintel-state) and HF_TOKEN first")
    if args.action == "push":
        print(f"saved to {hfstate.repo()} at commit {hfstate.push(Path(args.db))}")
    else:
        print("restored:", ", ".join(hfstate.pull(Path(args.db), overwrite=args.overwrite)) or "nothing new")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stockintel", description="Evidence-first stock research assistant.")
    p.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database path (default {DEFAULT_DB})")
    p.add_argument("--demo", action="store_true", help="offline mode with SYNTHETIC data (no network)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    llm = dict(choices=["off", "narrate", "agent"], default="off",
               help="generative layer: off (deterministic), narrate, or agent (needs Claude credentials)")

    a = sub.add_parser("analyze", help="full analysis of one stock")
    a.add_argument("symbol")
    a.add_argument("--horizon", choices=["short", "medium", "long"], default="medium")
    a.add_argument("--report", action="store_true", help="18-section Markdown research report")
    a.add_argument("--json", action="store_true")
    a.add_argument("--llm", **{**llm, "choices": ["off", "narrate"]})
    a.set_defaults(fn=cmd_analyze)

    c = sub.add_parser("chat", help="interactive multi-turn research chat")
    c.add_argument("--llm", **llm)
    c.set_defaults(fn=cmd_chat)

    s = sub.add_parser("screen", help="budget-aware research candidates")
    s.add_argument("--budget", type=float, required=True)
    s.add_argument("--risk", choices=["conservative", "moderate", "aggressive"], default="moderate")
    s.add_argument("--horizon", choices=["short", "medium", "long"], default="medium")
    s.add_argument("--include", nargs="*", default=[])
    s.add_argument("--exclude", nargs="*", default=[])
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_screen)

    q = sub.add_parser("ask", help="one natural-language question")
    q.add_argument("question")
    q.set_defaults(fn=cmd_query)

    pf = sub.add_parser("portfolio", help="import/show portfolio, what-if")
    pf.add_argument("action", choices=["show", "import", "what-if"])
    pf.add_argument("file", nargs="?")
    pf.add_argument("--amount", type=float)
    pf.add_argument("--symbols", nargs="*")
    pf.set_defaults(fn=cmd_portfolio)

    b = sub.add_parser("backtest", help="baselines vs engine strategies, net of costs")
    b.add_argument("symbol", nargs="?")
    b.add_argument("--symbols", help="comma-separated; prints a cross-sectional summary")
    b.add_argument("--universe", action="store_true")
    b.add_argument("--fusion", action="store_true", help="also replay the price-domain fusion (slow)")
    b.set_defaults(fn=cmd_backtest)

    vp = sub.add_parser("validate-patterns", help="pooled historical edge of every candle/chart pattern")
    target = vp.add_mutually_exclusive_group(required=True)
    target.add_argument("--symbols", help="comma-separated tickers")
    target.add_argument("--universe", action="store_true", help="the whole research universe")
    vp.add_argument("--save", action="store_true", help="store results as priors for live analysis")
    vp.set_defaults(fn=cmd_validate_patterns)

    fe = sub.add_parser("forecast-eval", help="walk-forward evaluation of forecast models")
    fe.add_argument("--symbols", required=True, help="comma-separated")
    fe.add_argument("--models", default="logistic,gbm")
    fe.add_argument("--horizons", default="1,5,10,20")
    fe.add_argument("--register", action="store_true", help="register the result as a candidate model")
    fe.set_defaults(fn=cmd_forecast_eval)

    m = sub.add_parser("models", help="model registry: list / approve / deploy / reject")
    m.add_argument("action", choices=["list", "approve", "deploy", "reject"])
    m.add_argument("name", nargs="?")
    m.add_argument("version", nargs="?")
    m.add_argument("--by", default="")
    m.set_defaults(fn=cmd_models)

    al = sub.add_parser("alerts", help="scan symbols for alerts, or list stored alerts")
    al.add_argument("symbols", nargs="*")
    al.set_defaults(fn=cmd_alerts)

    mo = sub.add_parser("monitor", help="score matured predictions and check data drift")
    mo.set_defaults(fn=cmd_monitor)

    sg = sub.add_parser("signals", help="this period's buy/sell/hold plan sized to your capital")
    sg.add_argument("--capital", type=float, help="new money to invest (default: stored portfolio cash)")
    sg.add_argument("--strategy", choices=["momentum", "lowvol", "blend"], default="momentum")
    sg.add_argument("--risk", choices=["full", "balanced", "defensive"], default="balanced")
    sg.add_argument("--positions", type=int, help="number of stocks (default: by capital, max 20)")
    sg.add_argument("--universe", choices=["nifty100", "nifty200", "nifty500"], default="nifty200")
    sg.add_argument("--refresh", action="store_true", help="force a fresh price download")
    sg.add_argument("--json", action="store_true")
    sg.set_defaults(fn=cmd_signals)

    ins = sub.add_parser("insider-fetch", help="update promoter/director trade disclosures from NSE")
    ins.add_argument("--days", type=int, default=45, help="recent days to fetch from the XBRL feed")
    ins.add_argument("--since", help="also backfill the legacy feed from this date (YYYY-MM-DD)")
    ins.add_argument("--min-value", type=float, default=1e7)
    ins.set_defaults(fn=cmd_insider_fetch)

    st = sub.add_parser("stops", help="check stop levels on stored holdings")
    st.set_defaults(fn=cmd_stops)

    es = sub.add_parser("evaluate-signals", help="reproduce the cross-sectional backtest and survivorship check")
    es.add_argument("--start", default="2018-01-01")
    es.add_argument("--cost-bps", type=float, default=16.0)
    es.set_defaults(fn=cmd_evaluate_signals)

    bp = sub.add_parser("build-performance", help="compute the app's Performance page (a few minutes)")
    bp.set_defaults(fn=cmd_build_performance)

    em = sub.add_parser("evaluate-models", help="walk-forward test of Chronos and Kronos (needs torch; minutes)")
    em.set_defaults(fn=cmd_evaluate_models)

    ap = sub.add_parser("app", help="open the web app (creates a login key automatically)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--lan", action="store_true", help="also listen on your Wi-Fi so your phone can open it")
    ap.add_argument("--llm", **llm)
    ap.set_defaults(fn=cmd_app)

    sv = sub.add_parser("serve", help="REST API + dashboard (needs STOCKINTEL_API_KEY)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--llm", **llm)
    sv.set_defaults(fn=cmd_serve)

    tc = sub.add_parser("train-chronos", help="fine-tune Chronos-2 on NSE prices and test it (CPU, about an hour)")
    tc.set_defaults(fn=cmd_train_chronos)

    tk = sub.add_parser("train-kronos", help="fine-tune the Kronos predictor on NSE candles and test it (CPU)")
    tk.add_argument("--eval-stocks", type=int, default=50)
    tk.set_defaults(fn=cmd_train_kronos)

    st = sub.add_parser("state", help="save or restore the portfolio and caches in a private Hugging Face dataset")
    st.add_argument("action", choices=["push", "pull"])
    st.add_argument("--overwrite", action="store_true", help="pull: replace local files too")
    st.set_defaults(fn=cmd_state)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
