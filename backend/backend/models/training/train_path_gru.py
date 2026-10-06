#!/usr/bin/env python3
"""Train the path GRU (models/networks/path_gru.py) and hold it against "no change" and the base rate.

Protocol, with the splits in models/learning/protocol.py:
- train each hidden size in HIDDEN on origins whose 20-day outcome ends by TRAIN_END, scoring
  VAL_START..VAL_END every EVAL_EVERY steps;
- choose the size and step count with the lowest validation squared error (early stopping);
- retrain that size for that many steps on everything up to VAL_END; score it once from TEST_START.
Changed after the first validation run, never after a test: trained the full 2,000 steps, the
32-unit model's validation error was 59% above "no change", so the step count is now chosen too.

Reported against two baselines that a forecast has to beat to be worth a line on a chart: "no change"
(error) and the stock's own base rate of going up (direction). The confident-call check is the
project's usual one: the cut that fires on 10% of validation, applied unchanged to the test window.

    python -m backend.models.training.train_path_gru
"""
import argparse
import json
import logging
import time

import pandas as pd

from backend.models.core.autograd import ops
from backend.models.core.autograd.tensor import no_grad
from backend.models.core.backend import default_rng, dtype, xp
from backend.models.core.optim import AdamW, clip_grad_norm
from backend.models.core.optim.schedule import warmup_cosine
from backend.models.data import returns
from backend.models.learning.protocol import SCORE_STRIDE, TEST_START, TRAIN_END, VAL_END, VAL_START, selective
from backend.models.networks import path_gru as G
from backend.paths import ARTIFACTS

log = logging.getLogger("alfa.train_path_gru")
HIDDEN = (32, 64)
TRAIN_STRIDE = 5          # training origins a week apart: more samples, overlapping outcomes are fine to fit on
REPORT_DAYS = (5, 20)
EVAL_EVERY = 50


def samples(closes, market, first, last, stride, with_base=False):
    """Windows, targets and per-row facts for origins whose outcome date is in [first, last]."""
    X, Y, rows, skipped = [], [], [], 0
    lo, hi = pd.Timestamp(first), pd.Timestamp(last) if last else None
    # one shared calendar of origin dates, so a date-grouped range compares like with like
    calendar = pd.DatetimeIndex(sorted(set().union(*[c.index for c in closes.values()])))
    origins = set(calendar[calendar >= lo][::stride])
    for sym, c in closes.items():
        m = market.reindex(c.index).ffill()
        c, m = c[m.notna()], m[m.notna()]
        if len(c) < returns.SCALE_BARS + G.WINDOW + G.HORIZON + 2:
            continue
        cv = c.to_numpy(dtype=float)
        r, rm = returns.log_returns(cv), returns.log_returns(m.to_numpy(dtype=float))
        for end in range(returns.SCALE_BARS + G.WINDOW - 1, len(r) - G.HORIZON):
            origin, outcome = c.index[end + 1], c.index[end + 1 + G.HORIZON]
            if origin not in origins or (hi is not None and outcome > hi):
                continue
            try:
                window, scale = G.window_at(r, rm, end)
            except ValueError:            # a year of zero returns (a suspended stock) has no scale
                skipped += 1
                continue
            X.append(window)
            Y.append(G.targets_at(cv, end, scale))
            row = {"symbol": sym, "date": str(origin.date()), "scale": scale, "close": cv[end + 1]}
            if with_base:
                for k in REPORT_DAYS:
                    past = cv[k: end + 2] / cv[: end + 2 - k]
                    row[f"base_up_{k}"] = float((past > 1).mean())
            rows.append(row)
    if skipped:
        log.info("skipped %d windows with no volatility scale", skipped)
    return xp.asarray(X, dtype=dtype()), xp.asarray(Y, dtype=dtype()), rows


def fit(X, Y, hidden, steps, batch, lr, seed=0, watch=None):
    """Train for `steps`; with `watch=(Xv, Yv)`, also return the validation skill every EVAL_EVERY steps."""
    model = G.PathGRU(hidden, seed)
    curve = []
    optimizer = AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.01)
    rng = default_rng(seed)
    started = time.time()
    for step in range(steps):
        pick = rng.integers(0, len(X), batch)
        optimizer.lr = warmup_cosine(step, steps, lr)
        error = ops.sub(model(X[pick]), ops.as_tensor(Y[pick]))
        loss = ops.mean(ops.mul(error, error))
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(optimizer.parameters, 1.0)
        optimizer.step()
        if watch and (step + 1) % EVAL_EVERY == 0:
            model.eval()
            p = predict(model, watch[0])
            curve.append((step + 1, round(1 - float(((p - watch[1]) ** 2).mean()) / float((watch[1] ** 2).mean()), 4)))
            model.train()
        if step % 250 == 0:
            log.info("hidden %d step %d loss %.4f (%.0fs)", hidden, step, float(loss.data), time.time() - started)
    model.eval()
    return (model, curve) if watch else model


def predict(model, X, batch=2048):
    with no_grad():
        return xp.concatenate([model(X[i:i + batch]).data for i in range(0, len(X), batch)])


def report(pred, Y, rows):
    """Squared-error skill against "no change", and per reported day its error and direction."""
    out = {"n": len(Y), "stocks": len({r["symbol"] for r in rows}), "origins": len({r["date"] for r in rows}),
           "mse_skill": round(1 - float(((pred - Y) ** 2).mean()) / float((Y ** 2).mean()), 4), "days": []}
    scale = xp.asarray([r["scale"] for r in rows])
    for k in REPORT_DAYS:
        actual = xp.exp(Y[:, k - 1] * scale * k ** 0.5) - 1
        guess = xp.exp(pred[:, k - 1] * scale * k ** 0.5) - 1
        base = xp.asarray([r[f"base_up_{k}"] for r in rows]) > 0.5
        up = actual > 0
        day = {"day": k, "mae_pct": round(float(xp.abs(guess - actual).mean()) * 100, 3),
               "no_change_mae_pct": round(float(xp.abs(actual).mean()) * 100, 3),
               "direction_acc": round(float(((guess > 0) == up).mean()), 4),
               "base_rate_acc": round(float((base == up).mean()), 4),
               "corr": round(float(xp.corrcoef(guess, actual)[0, 1]), 4)}
        day.update(by_date(rows, guess, actual, base))
        out["days"].append(day)
    return out


def by_date(rows, guess, actual, base, draws=2000, seed=0):
    """Stocks on one date move together, so the honest unit is the date. Two numbers per day:
    a date-resampled 95% range for (direction accuracy minus base-rate accuracy), and the mean
    within-date rank correlation of forecast and outcome (market-neutral: it ignores the market's swing)."""
    frame = pd.DataFrame({"date": [r["date"] for r in rows], "g": guess, "a": actual, "b": base})
    frame["hit"], frame["base_hit"] = (frame.g > 0) == (frame.a > 0), frame.b == (frame.a > 0)
    per = frame.groupby("date").agg(hit=("hit", "sum"), base_hit=("base_hit", "sum"), n=("hit", "count"))
    ic = frame.groupby("date").apply(lambda d: d.g.rank().corr(d.a.rank()), include_groups=False)
    rng = default_rng(seed)
    pick = rng.integers(0, len(per), (draws, len(per)))
    edge = (per.hit.to_numpy()[pick].sum(1) - per.base_hit.to_numpy()[pick].sum(1)) / per.n.to_numpy()[pick].sum(1)
    return {"direction_edge_lo": round(float(xp.percentile(edge, 2.5)), 4),
            "direction_edge_hi": round(float(xp.percentile(edge, 97.5)), 4),
            "rank_ic": round(float(ic.mean()), 4), "rank_ic_t": round(float(ic.mean() / (ic.std() / len(ic) ** 0.5)), 2),
            "dates": int(len(per))}


def call_rows(pred, Y, rows):
    return [{"horizon": k, "date": r["date"], "y": float(Y[i, k - 1]), "pred": float(pred[i, k - 1])}
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
    closes = universe_panel("nifty500", "10y").close
    market = yf.Ticker("^NSEI").history(period="10y", auto_adjust=True)["Close"]
    market.index = market.index.tz_localize(None).normalize()
    train_closes = {s: closes[s].dropna() for s in closes.columns}
    scored = {s: closes[s].dropna() for s in index_constituents("nifty200")["symbol"] if s in closes.columns}

    Xt, Yt, _ = samples(train_closes, market, "2000-01-01", TRAIN_END, TRAIN_STRIDE)
    Xv, Yv, rv = samples(scored, market, VAL_START, VAL_END, SCORE_STRIDE, with_base=True)
    log.info("train %d windows, validation %d", len(Xt), len(Xv))
    selection = []
    for hidden in HIDDEN:
        _, curve = fit(Xt, Yt, hidden, args.steps, args.batch, args.lr, watch=(Xv, Yv))
        step, skill = max(curve, key=lambda c: c[1])
        selection.append({"hidden": hidden, "best_step": step, "val_mse_skill": skill, "curve": curve})
        log.info("hidden %d: best validation skill %.4f at step %d (last %.4f)", hidden, skill, step, curve[-1][1])
    chosen = max(selection, key=lambda s: s["val_mse_skill"])
    best, best_steps = chosen["hidden"], chosen["best_step"]

    Xf, Yf, _ = samples(train_closes, market, "2000-01-01", VAL_END, TRAIN_STRIDE)
    final = fit(Xf, Yf, best, best_steps, args.batch, args.lr)
    Xs, Ys, rs = samples(scored, market, TEST_START, None, SCORE_STRIDE, with_base=True)
    test_pred = predict(final, Xs)
    record = {"model": "path_gru", "protocol": __doc__, "hidden": best, "steps": best_steps, "window": G.WINDOW, "horizon": G.HORIZON,
              "selection": selection, "test": report(test_pred, Ys, rs), "test_from": TEST_START,
              "precision_calls": selective(call_rows(predict(final, Xv), Yv, rv), call_rows(test_pred, Ys, rs),
                                           REPORT_DAYS, key="pred")}
    G.save(args.out, final, record)
    print(json.dumps({k: record[k] for k in ("hidden", "steps", "test", "precision_calls")}, indent=2))


if __name__ == "__main__":
    main()
