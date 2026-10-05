"""Builds the numbers behind the app's Performance page, once, sequentially.

Everything here is out-of-sample and reproducible from the CLI commands that
first produced it; this module only gathers the results into one JSON file so
the page loads instantly and never recomputes studies on a web request.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .analysis import forecast
from .analysis.stats import auc, brier, norm_ppf
from .data.panel import CACHE_DIR, universe_panel
from .evidence import utcnow_iso
from .storage import Store

log = logging.getLogger("stockintel.performance")
PATH = CACHE_DIR / "performance.json"
FORECAST_STOCKS = ["RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "INFY", "ITC", "LT", "SBIN",
                   "BHARTIARTL", "MARUTI", "SUNPHARMA", "TITAN"]
BONFERRONI_Z = float(norm_ppf(1 - 0.025 / 43))


def _patterns(store: Store) -> List[Dict[str, Any]]:
    rows = []
    for fam in ("candlestick", "chart"):
        for name, r in store.pattern_priors(fam).items():
            ex, z = r.get("excess_mean_pct"), r.get("z")
            se = abs(ex / z) if ex is not None and z not in (None, 0) else None
            rows.append({"family": fam, "pattern": name.replace("_", " ").replace(":up", " (up-break)")
                         .replace(":down", " (down-break)").replace(":flat", ""),
                         "events": r.get("events", 0), "stocks": r.get("stocks"), "excess_pct": ex, "z": z,
                         "ci_lo": None if se is None else round(ex - 1.96 * se, 3),
                         "ci_hi": None if se is None else round(ex + 1.96 * se, 3),
                         "horizon": r.get("horizon_days"), "verdict": r.get("verdict")})
    return sorted(rows, key=lambda r: (r["family"], -(r["excess_pct"] or -99)))


def _forecasts(service) -> Dict[str, Any]:
    out: Dict[str, Any] = {"horizons": [], "calibration": {}, "coverage": []}
    for h in (1, 5, 20):
        ys, ps = [], []
        for sym in FORECAST_STOCKS:
            df = service.history(sym)
            wf = forecast.walk_forward(df, h, ("climatology", "logistic"))
            if "error" in wf:
                continue
            p = wf["predictions"]["logistic"]
            y = forecast.labels(df["close"], h).reindex(p.index)
            ys.append(y.to_numpy())
            ps.append(p.to_numpy())
            if h in (5, 20):
                out["coverage"].append({"symbol": sym, "horizon": h, "coverage": wf["interval_coverage"]})
        y, p = np.concatenate(ys), np.concatenate(ps)
        base = np.full(len(y), y.mean())
        out["horizons"].append({"horizon": h, "n": int(len(y)), "accuracy": round(float(((p > 0.5) == y).mean()), 4),
                                "always_up_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
                                "auc": round(auc(y, p), 4), "brier_skill": round(1 - brier(y, p) / brier(y, base), 4)})
        bins = np.linspace(0.3, 0.7, 9)
        idx = np.digitize(p, bins)
        cal = []
        for b in range(1, len(bins)):
            mask = idx == b
            if mask.sum() >= 50:
                cal.append({"predicted": round(float(p[mask].mean()), 3), "actual": round(float(y[mask].mean()), 3),
                            "n": int(mask.sum())})
        out["calibration"][str(h)] = cal
    return out


def _momentum(bench_symbols: Dict[str, pd.Series]) -> Dict[str, Any]:
    from .signals import evaluate as E
    panel = universe_panel("nifty500", "10y")
    res = E.run(panel, bench_symbols["^NSEI"], start="2018-01-01",
                only=["equal_weight_universe", "momentum_semiannual", "momentum_trend_vol_overlay"])
    idx = res["momentum_semiannual"].equity.index
    weekly = idx[::5]

    def series(s: pd.Series) -> List[float]:
        s = s.reindex(idx).ffill().bfill()
        return [round(float(v), 4) for v in (s / s.iloc[0]).reindex(weekly)]
    curves = {"Momentum 20 (backtest)": series(res["momentum_semiannual"].equity),
              "Momentum + risk overlay (backtest)": series(res["momentum_trend_vol_overlay"].equity),
              "Equal-weight same stocks (backtest)": series(res["equal_weight_universe"].equity),
              "Nifty 500 index (real)": series(bench_symbols["^CRSLDX"])}
    # skip the listing week: the first print (19 Aug 2022) is 12% above the next day's close
    etf = bench_symbols["MOMENTUM.NS"].dropna().iloc[5:]
    a, b = etf.index[0], etf.index[-1]

    def cagr(s: pd.Series) -> float:
        s = s.loc[a:b].dropna()
        return round(float(((s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1) * 100), 2)
    yearly = []
    for y in range(a.year + 1, b.year + 1):
        e_, n_ = etf.loc[str(y)], bench_symbols["^CRSLDX"].loc[str(y)]
        if len(e_) > 20 and len(n_) > 20:
            yearly.append({"year": y, "etf": round(float(e_.iloc[-1] / e_.iloc[0] - 1) * 100, 1),
                           "nifty500": round(float(n_.iloc[-1] / n_.iloc[0] - 1) * 100, 1)})
    live = {"window": f"{a.date()} to {b.date()}", "yearly": yearly,
            "bars": [{"label": "Real momentum ETF", "cagr": cagr(etf), "real": True},
                     {"label": "Real Nifty 500", "cagr": cagr(bench_symbols["^CRSLDX"]), "real": True},
                     {"label": "Backtest momentum", "cagr": cagr(res["momentum_semiannual"].equity), "real": False},
                     {"label": "Backtest equal-weight", "cagr": cagr(res["equal_weight_universe"].equity), "real": False}]}
    from .signals.simulate import metrics
    return {"dates": [str(d.date()) for d in weekly], "curves": curves, "live_check": live,
            "table": {k: metrics(v.equity) for k, v in res.items()}}


def _insider() -> List[Dict[str, Any]]:
    from .data import insider
    from .data.provider import DataUnavailable
    from .signals.factors import liquid_universe, month_ends
    from .signals.insider_study import event_returns, summarize
    try:
        df = insider.load()
    except DataUnavailable:
        return []
    panel = universe_panel("nifty500", "10y")
    P_ = {"Promoters", "Promoter Group", "Promoter"}
    buys = insider.events(df, 1, 0, P_)
    me = month_ends(panel.close.index)
    tops: Dict[Any, set] = {}

    def liquid(day) -> bool:
        m = me[me <= day]
        if not len(m):
            return False
        if m[-1] not in tops:
            tops[m[-1]] = set(liquid_universe(panel.close, panel.volume, m[-1], top=300))
        return True
    liq = buys[[liquid(d) and s in tops[me[me <= d][-1]] for s, d in zip(buys.symbol, buys.day)]]
    rng = np.random.default_rng(0)
    placebo = buys.assign(day=pd.to_datetime(rng.choice(panel.close.index[300:-70], len(buys))))
    tests = [("Promoter buys, all panel stocks", buys), ("Same stocks on random dates (placebo)", placebo),
             ("Promoter buys, stocks liquid at the time", liq),
             ("Promoter sales of ₹1 cr or more", insider.events(df, -1, 1e7, P_))]
    rows = []
    for label, ev in tests:
        r = summarize(event_returns(panel, ev, 20), 20)
        rows.append({"test": label, "events": r["events"], "excess_20d": r.get("mean_excess_pct"),
                     "median_20d": r.get("median_excess_pct"), "t": r.get("t_month_clustered")})
    return rows


def build(orch) -> Dict[str, Any]:
    svc = orch.service()
    log.info("patterns")
    out: Dict[str, Any] = {"built_at": utcnow_iso(), "bonferroni_z": round(BONFERRONI_Z, 2),
                           "patterns": _patterns(orch.store)}
    log.info("forecasts")
    out["forecast"] = _forecasts(svc)
    log.info("momentum")
    import yfinance as yf
    benches = {}
    for t in ("^NSEI", "^CRSLDX", "MOMENTUM.NS"):
        h = yf.Ticker(t).history(period="12y", auto_adjust=True)["Close"]
        h.index = h.index.tz_localize(None)
        benches[t] = h
    out["momentum"] = _momentum(benches)
    log.info("insider")
    out["insider"] = _insider()
    PATH.write_text(json.dumps(out, default=str))
    return out


MODELS_PATH = CACHE_DIR / "model_evals.json"


def evaluate_models(service) -> Dict[str, Any]:
    """Open-source forecasting models, walk-forward on the forecast stocks. Slow (minutes, torch)."""
    from .analysis import kronos_model, tsfm
    out: Dict[str, Any] = {"built_at": utcnow_iso()}
    closes = {s: service.history(s)["close"] for s in FORECAST_STOCKS}
    out["chronos"] = tsfm.evaluate(closes)
    if kronos_model.available():
        out["kronos"] = kronos_model.evaluate({s: service.history(s) for s in FORECAST_STOCKS})
    MODELS_PATH.write_text(json.dumps(out))
    return out


# pinball skill within ±1% of the volatility band counts as a tie (provisional: the
# 660-forecast sample cannot separate smaller differences from noise)
TIE_SKILL = 0.01


def _vs_base(acc: float, base: float, n: int) -> str:
    """Direction accuracy against the base rate, calling gaps under two standard errors noise."""
    se = (base * (1 - base) / n) ** 0.5
    return ("more often than" if acc - base > 2 * se else "less often than" if base - acc > 2 * se
            else "about as often as")


def model_verdicts(evals: Dict[str, Any]) -> Dict[str, str]:
    """One plain sentence per open-source model, read off its measured record."""
    out = {}
    h5 = next((h for h in evals.get("chronos", {}).get("horizons", []) if h["horizon"] == 5), None)
    if h5:
        skill = h5["pinball_skill"]
        out["chronos"] = ("its range was " + ("better than" if skill > TIE_SKILL else "worse than" if skill < -TIE_SKILL else "about as good as")
                          + " plain volatility, and its direction was right "
                          + _vs_base(h5["chronos_direction_acc"], h5["base_rate_acc"], h5["n"]) + " the base rate")
    k = evals.get("kronos")
    if k:
        out["kronos"] = ("its price error was " + ("larger" if k["mae_pct"] > k["no_change_mae_pct"] else "smaller")
                         + " than assuming no change, and its direction was right "
                         + _vs_base(k["direction_acc"], k["base_rate_acc"], k["n"]) + " the base rate")
    return out


def load() -> Dict[str, Any]:
    if not PATH.exists():
        raise FileNotFoundError("performance results not built yet; run `stockintel build-performance`")
    out = json.loads(PATH.read_text())
    if MODELS_PATH.exists():
        out["open_models"] = json.loads(MODELS_PATH.read_text())
        out["open_models"]["verdicts"] = model_verdicts(out["open_models"])
    return out


def track_record() -> Optional[Dict[str, Any]]:
    """The measured results if they have been built, else None."""
    return load() if PATH.exists() else None


def live_edge(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The momentum facts every explanation quotes: live ETF edge, its best and worst
    year, and the backtest's worst falls with and without the risk overlay."""
    live, table = rec["momentum"]["live_check"], rec["momentum"]["table"]
    years = {y["year"]: y["etf"] - y["nifty500"] for y in live["yearly"]}
    best, worst = max(years, key=years.get), min(years, key=years.get)
    return {"gap": live["bars"][0]["cagr"] - live["bars"][1]["cagr"], "window": live["window"],
            "best": (best, years[best]), "worst": (worst, years[worst]),
            "max_dd": float(table["momentum_semiannual"]["max_dd_pct"]),
            "overlay_max_dd": float(table["momentum_trend_vol_overlay"]["max_dd_pct"])}


def momentum_evidence(rec: Optional[Dict[str, Any]]) -> List[str]:
    """Measured lines shown with every signal plan."""
    if rec is None:
        return ["Measured track record not built yet; run `stockintel build-performance`."]
    e = live_edge(rec)
    return [f"Live Nifty200 Momentum 30 ETF beat the Nifty 500 by {e['gap']:+.1f} pts/yr ({e['window']}): "
            f"{e['best'][1]:+.0f} pts in {e['best'][0]}, {e['worst'][1]:+.0f} pts in {e['worst'][0]}.",
            f"Backtest on today's index list (survivorship-inflated): worst fall {e['max_dd']:.0f}%, "
            f"{e['overlay_max_dd']:.0f}% with the trend + volatility overlay."]


def evidence_note(rec: Optional[Dict[str, Any]]) -> str:
    """What the momentum rule has actually done, read off the built track record."""
    rule = "The buy/sell rule is NSE's momentum method. "
    if rec is None:
        return rule + "Its measured track record is not built yet; run `stockintel build-performance`."
    e = live_edge(rec)
    edges = sum(abs(p["z"] or 0) >= rec["bonferroni_z"] for p in rec["patterns"])
    beats = sum(h["accuracy"] > h["always_up_accuracy"] for h in rec["forecast"]["horizons"])
    return (rule + f"Its live ETF beat the Nifty 500 by {e['gap']:+.1f} points a year ({e['window']}), but unevenly: "
            f"{e['best'][1]:+.0f} points in {e['best'][0]}, {e['worst'][1]:+.0f} in {e['worst'][0]}, "
            f"and the backtest fell up to {abs(e['max_dd']):.0f}% in bad stretches. {edges} of {len(rec['patterns'])} candle and "
            f"chart patterns showed an edge, and the direction forecast beat always-guessing-up at {beats} of "
            f"{len(rec['forecast']['horizons'])} horizons, so they only appear as context.")
