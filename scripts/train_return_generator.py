#!/usr/bin/env python3
"""Train the return generator and hold it against GARCH and against a trailing-volatility Gaussian.

Three periods, split by date, because each number has one job:

  train      every return before --cutoff; the model and the GARCH fit both learn here and nowhere else
  selection  --cutoff to --test-from; picks the checkpoint
  test       --test-from onwards; the number that is quoted, and it never chose anything

A scored return is always predicted from returns before it, and every window's scale comes from the year
before the window starts (`data/returns.scale_before`), so no period's figures are measured with anything
from a later one. A held-out window scores only its second half, so each scored return has between half a
context and a full one behind it -- serving always has a full one, and scoring position 0 would handicap the
model against a GARCH filter that has read the whole history.

The comparison is on the log score in return units, per scored return, with each window as one block in the
standard error. The model is saved with the numbers, and `backend/models/scenarios.py` serves it only if the
file says it beat GARCH by more than that error -- otherwise GARCH answers.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent import pretrained  # noqa: E402
from selfagent.autograd import functional as F  # noqa: E402
from selfagent.autograd import no_grad  # noqa: E402
from selfagent.config import ModelConfig  # noqa: E402
from selfagent.data import prices, returns  # noqa: E402
from selfagent.learn.forecast import paired  # noqa: E402
from selfagent.models import ReturnGenerator  # noqa: E402
from selfagent.models.garch import Garch, t_log_density  # noqa: E402
from selfagent.optim import AdamW, clip_grad_norm  # noqa: E402

# The rule the price head is served against, as a density: a Gaussian at the trailing 20-day volatility.
TRAILING = 20


def load_series(price_dir):
    """[(ticker, return dates, log returns)] for every ticker with a usable file."""
    loaded = []
    for path in sorted(Path(price_dir).glob("*.csv")):
        try:
            dates, bars = prices.load_bars(path)
        except ValueError:
            continue
        loaded.append((path.stem, np.array(dates[1:]), returns.log_returns(bars[:, 3])))
    if not loaded:
        raise SystemExit(f"no usable price files under {price_dir}")
    return loaded


def windows(series, length, low, high, stride):
    """(series index, start) for every window whose next-return targets all fall in [low, high).

    A window reads returns[start : start + length] and is scored on returns[start + 1 : start + length + 1],
    so it is the target dates that must sit inside the period, not the inputs: a forecast made on the first
    day of the test period is allowed to read the day before it.
    """
    found = []
    for index, (_, dates, values) in enumerate(series):
        # A year of flat closes has no unit to scale by; some early histories hold one, and they are stale
        # quotes, not a year in which the stock did not move.
        moved = np.concatenate([[0.0], np.cumsum(np.abs(values))])
        for start in range(returns.SCALE_BARS, len(values) - length, stride):
            if (low <= dates[start + 1] and dates[start + length] < high
                    and moved[start] > moved[start - returns.SCALE_BARS]):
                found.append((index, start))
    return found


def batch(series, chosen, length):
    """(inputs, targets, scales, raw targets) for a list of (series index, start)."""
    inputs, targets, scales, raw = [], [], [], []
    for index, start in chosen:
        values = series[index][2]
        scale = returns.scale_before(values, start)
        inputs.append(returns.encode(values[start : start + length], scale))
        targets.append(returns.encode(values[start + 1 : start + length + 1], scale))
        scales.append(scale)
        raw.append(values[start + 1 : start + length + 1])
    return np.stack(inputs), np.stack(targets), np.array(scales), np.stack(raw)


def model_log_density(model, series, chosen, length, batch_size):
    """log density of every scored return in `chosen` windows, in return units: (windows, length // 2)."""
    scored = []
    with no_grad():
        for start in range(0, len(chosen), batch_size):
            inputs, _, scales, raw = batch(series, chosen[start : start + batch_size], length)
            logits = model(inputs).data
            logits = logits - logits.max(axis=-1, keepdims=True)
            probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            density = returns.density(probabilities, raw, scales[:, None])
            scored.append(np.log(density)[:, length // 2 :])
    return np.concatenate(scored)


def baseline_log_density(series, chosen, length, garch, levels):
    """(GARCH-t, trailing Gaussian) log densities on exactly the returns `model_log_density` scores."""
    filtered = {index: garch.variance(series[index][2], levels[index]) for index in {i for i, _ in chosen}}
    fitted, trailing = [], []
    for index, start in chosen:
        values = series[index][2]
        at = np.arange(start + 1 + length // 2, start + length + 1)
        realised = values[at]
        fitted.append(t_log_density(realised, filtered[index][at], garch.nu))
        spread = np.array([values[t - TRAILING : t].std(ddof=1) for t in at])
        trailing.append(-0.5 * np.log(2 * math.pi * spread ** 2) - realised ** 2 / (2 * spread ** 2))
    return np.stack(fitted), np.stack(trailing)


def compare(model, series, chosen, length, batch_size, garch, levels):
    """Mean negative log score of each forecaster and the model's paired gaps, per scored return."""
    ours = model_log_density(model, series, chosen, length, batch_size)
    fitted, trailing = baseline_log_density(series, chosen, length, garch, levels)
    blocks = np.repeat(np.arange(len(chosen)), ours.shape[1])
    report = dict(windows=len(chosen), scored=int(ours.size), model=float(-ours.mean()),
                  garch=float(-fitted.mean()), trailing=float(-trailing.mean()))
    report["vs_garch"], report["vs_garch_error"] = paired(-ours.ravel(), -fitted.ravel(), blocks)
    report["vs_trailing"], report["vs_trailing_error"] = paired(-ours.ravel(), -trailing.ravel(), blocks)
    return report


def show(name, report):
    print(f"  {name}: {report['scored']:,} returns in {report['windows']:,} windows; negative log score "
          f"per return  model {report['model']:.4f}  GARCH-t {report['garch']:.4f}  "
          f"trailing Gaussian {report['trailing']:.4f}")
    print(f"    model minus GARCH {report['vs_garch']:+.4f} +/- {report['vs_garch_error']:.4f}   "
          f"model minus trailing {report['vs_trailing']:+.4f} +/- {report['vs_trailing_error']:.4f}  "
          f"(negative is the model winning)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/prices")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--save", default="returns.npz")
    parser.add_argument("--cutoff", default="2021-01-01", help="training ends before this date")
    parser.add_argument("--test-from", default="2023-01-01", help="the quoted period starts here")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    # A month of trading on each side of the smallest shape that learns anything. Provisional: set to fit a
    # CPU step under a second, not by a sweep.
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()
    if not args.cutoff < args.test_from:
        raise SystemExit(f"--cutoff {args.cutoff} must come before --test-from {args.test_from}")

    series = load_series(args.prices)
    length = args.context
    train = windows(series, length, "", args.cutoff, 1)
    selection = windows(series, length, args.cutoff, args.test_from, length // 2)
    test = windows(series, length, args.test_from, "9999", length // 2)
    print(f"{len(series)} tickers; {len(train):,} training windows before {args.cutoff}, "
          f"{len(selection):,} selection windows to {args.test_from}, {len(test):,} test windows after")

    # GARCH learns from the training period alone, and each ticker's long-run level is its training
    # variance, so the baseline knows no more about the future than the model does.
    history = [values[dates < args.cutoff] for _, dates, values in series]
    levels = [float(np.var(one)) if len(one) > 1 else float(np.var(values[:returns.SCALE_BARS]))
              for one, (_, _, values) in zip(history, series)]
    started = time.monotonic()
    garch = Garch.fit([one for one in history if len(one) > returns.SCALE_BARS])
    print(f"GARCH-t fitted on the training period in {time.monotonic() - started:.0f}s: {garch.as_dict()}")

    config = ModelConfig(vocab_size=returns.BINS, price_window=length, dim=args.dim, num_heads=args.heads,
                         num_layers=args.layers, ffn_dim=4 * args.dim)
    model = ReturnGenerator(config)
    optimizer = AdamW(model.trainable_parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(config.seed)
    print(f"{sum(p.size for p in model.parameters()):,} parameters; uniform baseline loss is "
          f"ln({returns.BINS}) = {math.log(returns.BINS):.3f}")

    best, best_step, started = math.inf, 0, time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        chosen = [train[i] for i in rng.integers(0, len(train), args.batch_size)]
        inputs, targets, _, _ = batch(series, chosen, length)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape((-1, returns.BINS)), targets.reshape(-1))
        loss.backward()
        clip_grad_norm(optimizer.parameters, 1.0)
        optimizer.step()
        if step % args.log_every == 0:
            rate = step / (time.monotonic() - started)
            print(f"step {step:,}/{args.steps:,}  loss {loss.item():.4f}  {rate:.2f} step/s  "
                  f"{(args.steps - step) / rate / 60:.0f} min left")
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            chosen_report = compare(model, series, selection, length, args.batch_size, garch, levels)
            show(f"selection at step {step:,}", chosen_report)
            if chosen_report["model"] < best:
                best, best_step = chosen_report["model"], step
                pretrained.save(Path(args.artifacts) / f"{Path(args.save).stem}.best.npz", model, config)

    config, weights = pretrained.load(Path(args.artifacts) / f"{Path(args.save).stem}.best.npz")
    model.load_state_dict(weights)
    model.eval()
    quoted = compare(model, series, test, length, args.batch_size, garch, levels)
    print(f"\ncheckpoint from step {best_step:,}, chosen on the selection period")
    show("test", quoted)
    metadata = dict(cutoff=args.cutoff, test_from=args.test_from, garch=garch.as_dict(), step=best_step,
                    test=quoted, selection_best=best, scale_bars=returns.SCALE_BARS, bins=returns.BINS,
                    limit=returns.LIMIT)
    pretrained.save(Path(args.artifacts) / args.save, model, config, metadata)
    print(f"saved {args.artifacts}/{args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
