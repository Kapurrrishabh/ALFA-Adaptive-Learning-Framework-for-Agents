#!/usr/bin/env python3
"""Train the path GRU (models/networks/path_gru.py) and hold it against "no change", the base rate
and the EWMA volatility band.

Protocol, with the splits in models/learning/protocol.py:
- train each hidden size in HIDDEN on origins whose last forecast day ends by TRAIN_END, scoring the
  likelihood on VAL_START..VAL_END every EVAL_EVERY steps;
- choose the size and step count with the best validation likelihood (early stopping: trained to
  the end, the first squared-error GRU did 59% worse than "no change" on validation);
- retrain that size for that many steps on everything up to VAL_END; score it once from TEST_START.

Scored on the test window: the median path's error and direction (against "no change" and the base
rate, grouped by date because stocks move together), its 80% range for the cumulative move against
the EWMA band's (coverage and width), its per-day likelihood against the EWMA band's, and the
precision of its most confident 10% of up calls.

    python -m backend.models.training.train_path_gru
"""
import argparse
import json
import logging
import time

import pandas as pd

from backend.models.core.autograd.tensor import no_grad
from backend.models.core.backend import default_rng, dtype, xp
from backend.models.core.optim import AdamW, clip_grad_norm
from backend.models.core.optim.schedule import warmup_cosine
from backend.models.data import returns
from backend.models.learning.protocol import (SCORE_STRIDE, TEST_START, TRAIN_END, VAL_END, VAL_START, direction_edge,
                                              selective)
from backend.models.networks import path_gru as G
from backend.paths import ARTIFACTS

log = logging.getLogger("alfa.train_path_gru")
HIDDEN = (32, 64)
TRAIN_STRIDE = 5          # training origins a week apart: more samples, overlapping outcomes are fine to fit on
EVAL_EVERY = 50
EWMA_LAMBDA = 0.94        # the same RiskMetrics decay the app's volatility band uses
Z80 = 1.2816              # two-sided 80% range of a normal
REPORT_DAYS = sorted({5, G.HORIZON})


def ewma_sd(r):
    """Trailing EWMA daily standard deviation known at each return index."""
    var, out = float(r[:20].var()), xp.empty(len(r))
    for i, x in enumerate(r):
        var = EWMA_LAMBDA * var + (1 - EWMA_LAMBDA) * x * x
        out[i] = var ** 0.5
    return out


def samples(panel, symbols, market, first, last, stride):
    """Windows, daily targets and per-row facts for origins whose last forecast day is in range."""
    X, Y, rows, skipped = [], [], [], 0
    lo, hi = pd.Timestamp(first), pd.Timestamp(last) if last else None
    calendar = pd.DatetimeIndex(sorted(set().union(*[panel.close[s].dropna().index for s in symbols])))
    origins = set(calendar[calendar >= lo][::stride])      # one shared calendar, so date groups compare like with like
    for sym in symbols:
        frame = pd.DataFrame({"c": panel.close[sym], "h": panel.high[sym], "l": panel.low[sym]}).dropna()
        frame["m"] = market.reindex(frame.index).ffill()
        frame = frame.dropna()
        if len(frame) < returns.SCALE_BARS + G.WINDOW + G.HORIZON + 2:
            continue
        cv = frame.c.to_numpy(dtype=float)
        r, rm = returns.log_returns(cv), returns.log_returns(frame.m.to_numpy(dtype=float))
        ranges = xp.log(frame.h.to_numpy(dtype=float) / frame.l.to_numpy(dtype=float))[1:]
        sd = ewma_sd(r)
        for end in range(returns.SCALE_BARS + G.WINDOW - 1, len(r) - G.HORIZON):
            origin, outcome = frame.index[end + 1], frame.index[end + 1 + G.HORIZON]
            if origin not in origins or (hi is not None and outcome > hi):
                continue
            try:
                window, scale = G.window_at(r, rm, ranges, end)
            except ValueError:            # a year of zero returns (a suspended stock) has no scale
                skipped += 1
                continue
            X.append(window)
            Y.append(G.targets_at(r, end, scale))
            row = {"symbol": sym, "date": str(origin.date()), "scale": scale, "ewma_sd": float(sd[end])}
            for k in REPORT_DAYS:
                past = cv[k: end + 2] / cv[: end + 2 - k]
                row[f"base_up_{k}"] = float((past > 1).mean())
            rows.append(row)
    if skipped:
        log.info("skipped %d windows with no volatility scale", skipped)
    return xp.asarray(X, dtype=dtype()), xp.asarray(Y, dtype=dtype()), rows


def predict(model, X, batch=2048):
    with no_grad():
        parts = [model(X[i:i + batch]) for i in range(0, len(X), batch)]
    return xp.concatenate([m.data for m, _ in parts]), xp.concatenate([s.data for _, s in parts])


def nll_of(mean, log_sd, Y):
    return float((log_sd + 0.5 * ((Y - mean) / xp.exp(log_sd)) ** 2).mean())


def fit(X, Y, hidden, steps, batch, lr, seed=0, watch=None):
    """Train for `steps`; with `watch=(Xv, Yv)`, also return the validation likelihood every EVAL_EVERY steps."""
    model = G.PathGRU(hidden, seed)
    optimizer = AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.01)
    rng, curve, started = default_rng(seed), [], time.time()
    for step in range(steps):
        pick = rng.integers(0, len(X), batch)
        optimizer.lr = warmup_cosine(step, steps, lr)
        mean, log_sd = model(X[pick])
        loss = G.nll(mean, log_sd, Y[pick])
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(optimizer.parameters, 1.0)
        optimizer.step()
        if watch and (step + 1) % EVAL_EVERY == 0:
            model.eval()
            curve.append((step + 1, round(nll_of(*predict(model, watch[0]), watch[1]), 4)))
            model.train()
        if step % 250 == 0:
            log.info("hidden %d step %d loss %.4f (%.0fs)", hidden, step, float(loss.data), time.time() - started)
    model.eval()
    return (model, curve) if watch else model


def by_date(rows, guess, actual, base, draws=2000, seed=0):
    """Stocks on one date move together, so the honest unit is the date: a date-resampled 95% range
    for (direction accuracy minus base-rate accuracy), and the mean within-date rank correlation of
    forecast and outcome (market-neutral)."""
    dates = [r["date"] for r in rows]
    lo, hi = direction_edge(dates, guess > 0, actual > 0, base, draws, seed)
    frame = pd.DataFrame({"date": dates, "g": guess, "a": actual})
    ic = frame.groupby("date").apply(lambda d: d.g.rank().corr(d.a.rank()), include_groups=False)
    return {"direction_edge_lo": lo, "direction_edge_hi": hi,
            "rank_ic": round(float(ic.mean()), 4), "rank_ic_t": round(float(ic.mean() / (ic.std() / len(ic) ** 0.5)), 2),
            "dates": int(len(ic))}


def report(mean, log_sd, Y, rows):
    scale = xp.asarray([r["scale"] for r in rows])
    ewma = xp.asarray([r["ewma_sd"] for r in rows]) / scale          # in the same scaled units as Y
    ewma_log_sd = xp.repeat(xp.log(ewma)[:, None], Y.shape[1], axis=1)
    out = {"n": len(Y), "stocks": len({r["symbol"] for r in rows}), "origins": len({r["date"] for r in rows}),
           "nll": round(nll_of(mean, log_sd, Y), 4), "ewma_nll": round(nll_of(xp.zeros_like(Y), ewma_log_sd, Y), 4),
           **nll_edge(rows, mean, log_sd, ewma_log_sd, Y), "days": []}
    for k in REPORT_DAYS:
        y_cum, centre = Y[:, :k].sum(1), mean[:, :k].sum(1)
        actual, guess = xp.exp(y_cum * scale) - 1, xp.exp(centre * scale) - 1
        spread = Z80 * xp.sqrt((xp.exp(log_sd[:, :k]) ** 2).sum(1))
        ewma_spread = Z80 * ewma * k ** 0.5
        base = xp.asarray([r[f"base_up_{k}"] for r in rows]) > 0.5
        up = actual > 0
        day = {"day": k, "mae_pct": round(float(xp.abs(guess - actual).mean()) * 100, 3),
               "no_change_mae_pct": round(float(xp.abs(actual).mean()) * 100, 3),
               "direction_acc": round(float(((guess > 0) == up).mean()), 4),
               "base_rate_acc": round(float((base == up).mean()), 4),
               "cover80": round(float((xp.abs(y_cum - centre) <= spread).mean()), 4),
               "ewma_cover80": round(float((xp.abs(y_cum) <= ewma_spread).mean()), 4),
               "width_vs_ewma": round(float((spread / ewma_spread).mean()), 4)}
        day.update(by_date(rows, guess, actual, base))
        out["days"].append(day)
    return out


def nll_edge(rows, mean, log_sd, ewma_log_sd, Y, draws=2000, seed=0):
    """Date-resampled 95% range for how much lower the GRU's likelihood loss is than the EWMA band's."""
    per_row = lambda m, s: (s + 0.5 * ((Y - m) / xp.exp(s)) ** 2).mean(1)
    gain = pd.Series(per_row(xp.zeros_like(Y), ewma_log_sd) - per_row(mean, log_sd)).groupby([r["date"] for r in rows])
    total, count = gain.sum().to_numpy(), gain.count().to_numpy()
    pick = default_rng(seed).integers(0, len(total), (draws, len(total)))
    boot = total[pick].sum(1) / count[pick].sum(1)
    return {"nll_gain_lo": round(float(xp.percentile(boot, 2.5)), 4), "nll_gain_hi": round(float(xp.percentile(boot, 97.5)), 4)}


def call_rows(mean, Y, rows):
    return [{"horizon": k, "date": r["date"], "y": float(Y[i, :k].sum()), "pred": float(mean[i, :k].sum())}
            for i, r in enumerate(rows) for k in REPORT_DAYS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=str(ARTIFACTS / "path_gru.npz"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from backend.database.sources.panel import index_constituents, universe_panel
    import yfinance as yf
    panel = universe_panel("nifty500", "10y")
    market = yf.Ticker("^NSEI").history(period="10y", auto_adjust=True)["Close"]
    market.index = market.index.tz_localize(None).normalize()
    every = list(panel.close.columns)
    scored = [s for s in index_constituents("nifty200")["symbol"] if s in panel.close.columns]

    Xt, Yt, _ = samples(panel, every, market, "2000-01-01", TRAIN_END, TRAIN_STRIDE)
    Xv, Yv, rv = samples(panel, scored, market, VAL_START, VAL_END, SCORE_STRIDE)
    log.info("train %d windows, validation %d", len(Xt), len(Xv))
    selection = []
    for hidden in HIDDEN:
        _, curve = fit(Xt, Yt, hidden, args.steps, args.batch, args.lr, watch=(Xv, Yv))
        step, value = min(curve, key=lambda c: c[1])
        selection.append({"hidden": hidden, "best_step": step, "val_nll": value, "curve": curve})
        log.info("hidden %d: best validation NLL %.4f at step %d (last %.4f)", hidden, value, step, curve[-1][1])
    chosen = min(selection, key=lambda s: s["val_nll"])
    best, best_steps = chosen["hidden"], chosen["best_step"]

    Xf, Yf, _ = samples(panel, every, market, "2000-01-01", VAL_END, TRAIN_STRIDE)
    final = fit(Xf, Yf, best, best_steps, args.batch, args.lr)
    Xs, Ys, rs = samples(panel, scored, market, TEST_START, None, SCORE_STRIDE)
    mean_s, sd_s = predict(final, Xs)
    mean_v, _ = predict(final, Xv)
    record = {"model": "path_gru", "protocol": __doc__, "hidden": best, "steps": best_steps, "window": G.WINDOW,
              "horizon": G.HORIZON, "selection": selection, "test": report(mean_s, sd_s, Ys, rs), "test_from": TEST_START,
              "precision_calls": selective(call_rows(mean_v, Yv, rv), call_rows(mean_s, Ys, rs), REPORT_DAYS, key="pred")}
    G.save(args.out, final, record)
    print(json.dumps({k: record[k] for k in ("hidden", "steps", "test", "precision_calls")}, indent=2))


if __name__ == "__main__":
    main()
