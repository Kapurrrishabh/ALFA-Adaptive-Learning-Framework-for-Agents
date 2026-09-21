#!/usr/bin/env python3
"""Train the price head, and settle GRU against the transformer tower by measurement.

Both towers see identical windows, labels, split and step count, so the accuracy difference is the
architecture and nothing else. Every number is reported against the majority-class baseline: three
balanced classes make 33% the score of knowing nothing, and a head that cannot beat it has no
business informing advice.

--target picks what the head classifies. Direction scored 35.7% against a 36.0% baseline with a loss
that never fell, so volatility is the default: trailing volatility explains 22.1% of the next 5 days'
volatility where trailing return explains 0.3% of the next return.

The split is by date with a gap of one horizon, so no training label reaches into the validation
period. Class boundaries come from the training period alone.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent.autograd import functional as F  # noqa: E402
from selfagent.autograd import no_grad  # noqa: E402
from selfagent.config import ModelConfig  # noqa: E402
from selfagent.data import prices  # noqa: E402
from selfagent.models import PriceWindowClassifier  # noqa: E402
from selfagent.optim import AdamW, clip_grad_norm  # noqa: E402


def build_index(price_dir, window, horizon, stride, cutoff):
    """(bars per ticker, train samples, validation samples) as (ticker, end) references.

    Windows are never materialised here. At 101 tickers a strided index is ~100k windows, and
    holding them all as arrays would cost gigabytes for data that is cheap to slice per batch.
    """
    bars, train, validation = [], [], []
    for path in sorted(Path(price_dir).glob("*.csv")):
        try:
            dates, series = prices.load_bars(path)
        except ValueError:
            continue
        if len(series) < window + horizon + 2:
            continue
        index = len(bars)
        bars.append(series)
        for end in prices.sample_ends(len(series), window, horizon, stride):
            # The label runs to end + horizon, so a training window must finish its label before
            # the cutoff or it has already seen the validation period.
            if dates[end + horizon] < cutoff:
                train.append((index, end))
            elif dates[end] >= cutoff:
                validation.append((index, end))
    return bars, train, validation


def target_of(bars, end, horizon, target):
    """Direction is kept selectable so the abstention claim stays measured, not assumed."""
    if target == "volatility":
        return prices.forward_volatility(bars, end, horizon)
    return prices.label_at(bars, end, horizon)


def batch_of(bars, samples, rows, window, horizon, edges, target):
    windows = np.stack([prices.window_at(bars[t], end, window) for t, end in (samples[r] for r in rows)])
    values = [target_of(bars[t], end, horizon, target) for t, end in (samples[r] for r in rows)]
    return windows, prices.to_tertile(values, edges)


def accuracy(model, bars, samples, window, horizon, edges, batch_size, batches, target):
    model.eval()
    right = total = 0
    with no_grad():
        for start in range(0, min(len(samples), batches * batch_size), batch_size):
            rows = range(start, min(start + batch_size, len(samples)))
            windows, wanted = batch_of(bars, samples, list(rows), window, horizon, edges, target)
            predicted = model(windows).data.argmax(axis=-1)
            right += int((predicted == wanted).sum())
            total += len(wanted)
    model.train()
    return right / max(total, 1)


def persistence(bars, train, validation, args, edges):
    """What one line of arithmetic already scores: bucket the label by the trailing value alone.

    This, not the majority class, is the bar. Volatility persists, so a head that only rediscovers
    persistence has added nothing to a system that could compute it directly.
    """
    trailing = lambda t, end: float(  # noqa: E731
        np.diff(np.log(bars[t][end - 20 : end + 1, 3])).std(ddof=1)
    )
    own_edges = prices.tertile_edges([trailing(t, end) for t, end in train[::7]])
    predicted = prices.to_tertile([trailing(t, end) for t, end in validation[::7]], own_edges)
    wanted = prices.to_tertile(
        [target_of(bars[t], end, args.horizon, args.target) for t, end in validation[::7]], edges
    )
    return float(np.mean(np.asarray(predicted) == np.asarray(wanted)))


def run(name, recurrent, config, bars, train, validation, args, edges):
    target = args.target
    model = PriceWindowClassifier(config, recurrent=recurrent)
    optimizer = AdamW(model.trainable_parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(config.seed)
    parameters = sum(p.size for p in model.parameters())
    print(f"\n=== {name}: {parameters:,} parameters ===")

    started = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        rows = rng.integers(0, len(train), args.batch_size).tolist()
        windows, wanted = batch_of(bars, train, rows, config.price_window, args.horizon, edges, target)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(windows), wanted)
        loss.backward()
        clip_grad_norm(optimizer.parameters, 1.0)
        optimizer.step()
        if step % args.log_every == 0:
            print(f"  step {step}/{args.steps}  loss {loss.item():.4f}  "
                  f"{step / (time.monotonic() - started):.2f} step/s")

    held = accuracy(model, bars, validation, config.price_window, args.horizon, edges,
                    args.batch_size, args.eval_batches, target)
    print(f"  {name} held-out accuracy {held:.1%}")
    return name, held, parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/prices")
    parser.add_argument("--cutoff", default="2021-01-01", help="validation starts on this date")
    parser.add_argument("--horizon", type=int, default=5, help="bars ahead the label looks")
    parser.add_argument("--stride", type=int, default=5, help="bars between consecutive windows")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=60)
    parser.add_argument("--target", choices=("direction", "volatility"), default="volatility",
                        help="volatility is the default because direction measured unpredictable")
    args = parser.parse_args()

    config = ModelConfig()
    print(f"indexing {args.prices}")
    bars, train, validation = build_index(
        args.prices, config.price_window, args.horizon, args.stride, args.cutoff
    )
    # The channel count comes from the data, so adding a feature cannot silently mismatch the tower.
    channels = prices.window_at(bars[0], train[0][1], config.price_window).shape[-1]
    config = ModelConfig(price_channels=channels)
    print(f"  {len(bars)} tickers, {len(train):,} train windows, {len(validation):,} held out")
    print(f"  train ends before {args.cutoff}, validation starts on it")

    training_values = [target_of(bars[t], end, args.horizon, args.target) for t, end in train[::7]]
    edges = prices.tertile_edges(training_values)
    print(f"  {args.target} class edges from training period: {edges[0]:+.4f}, {edges[1]:+.4f}")

    held_labels = prices.to_tertile(
        [target_of(bars[t], end, args.horizon, args.target) for t, end in validation[::7]], edges
    )
    counts = np.bincount(held_labels, minlength=3)
    baseline = counts.max() / counts.sum()
    print(f"  held-out class counts {counts.tolist()}, majority baseline {baseline:.1%}")

    trailing = persistence(bars, train, validation, args, edges)
    print(f"  persistence baseline from the trailing value alone {trailing:.1%}")

    results = [
        run("transformer", False, config, bars, train, validation, args, edges),
        run("gru", True, config, bars, train, validation, args, edges),
    ]
    print("\n=== verdict ===")
    for name, held, parameters in results:
        print(f"  {name:12s} {held:.1%}  ({held - baseline:+.1%} vs majority, "
              f"{held - trailing:+.1%} vs persistence)  {parameters:,} params")


if __name__ == "__main__":
    sys.exit(main())
