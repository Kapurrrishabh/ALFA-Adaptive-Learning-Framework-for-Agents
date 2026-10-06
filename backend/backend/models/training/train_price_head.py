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

Every held-out window is scored, because the gap between the head and the arithmetic is around a point
and a sampled slice cannot read that. Three things come out of that one pass: each tower's accuracy, the
accuracy of averaging both towers with the rule, and what the winner's stated confidence is actually
worth -- a Platt fit on the first half of held-out, scored on the second.
"""
from backend.paths import ARTIFACTS, DATA

import argparse
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from backend.models import serving as models  # noqa: E402
from backend.models.core import pretrained  # noqa: E402
from backend.models.core.autograd import functional as F  # noqa: E402
from backend.models.core.autograd import no_grad  # noqa: E402
from backend.models.core.config import ModelConfig  # noqa: E402
from backend.models.data import advisory, prices  # noqa: E402
from backend.models.learning import Calibrator, brier, expected_calibration_error  # noqa: E402
from backend.models.networks import PriceWindowClassifier  # noqa: E402
from backend.models.core.optim import AdamW, clip_grad_norm  # noqa: E402


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


def held_out_probabilities(model, bars, samples, window, horizon, edges, batch_size, target):
    """The model's class distribution for every held-out window, in `samples` order.

    The distribution rather than the argmax, because the ensemble and the calibrator both need it and a
    second pass over 28,676 windows costs as much as the first.
    """
    model.eval()
    out = []
    with no_grad():
        for start in range(0, len(samples), batch_size):
            rows = list(range(start, min(start + batch_size, len(samples))))
            windows, _ = batch_of(bars, samples, rows, window, horizon, edges, target)
            out.append(F.softmax(model(windows)).data)
    model.train()
    return np.concatenate(out)


def persistence(bars, train, validation, args, edges, wanted):
    """What one line of arithmetic already scores, and the table that lets it state a confidence.

    This, not the majority class, is the bar. Volatility persists, so a head that only rediscovers
    persistence has added nothing to a system that could compute it directly.

    The table is the row-normalised confusion over the **training** period: how often each trailing
    bucket was followed by each forward class. That is what `backend/models/price.py` quotes as the
    outlook confidence, so it is a measured hit rate rather than an uncalibrated softmax. On this data it
    comes out diagonal -- 58.3%, 41.5%, 61.3% -- so the rule is "the bucket you are in", measured.
    """
    trailing = lambda t, end: prices.trailing_volatility(bars[t], end)  # noqa: E731
    own_edges = prices.tertile_edges([trailing(t, end) for t, end in train[::7]])
    buckets = prices.to_tertile([trailing(t, end) for t, end in train[::7]], own_edges)
    truth = prices.to_tertile(
        [target_of(bars[t], end, args.horizon, args.target) for t, end in train[::7]], edges
    )
    counts = np.zeros((3, 3))
    for bucket, actual in zip(buckets, truth):
        counts[bucket, actual] += 1
    if (counts.sum(axis=1) == 0).any():
        raise ValueError(f"a trailing bucket has no training rows: {counts.sum(axis=1).tolist()}; the "
                         f"edges {own_edges.tolist()} are degenerate and the table would be a guess")
    table = counts / counts.sum(axis=1, keepdims=True)

    # Scored the way serving reads it -- the class the bucket's table row peaks on, over every held-out
    # row. The rule's number has to be the number the served rule gets, on the rows the head is scored on.
    held = prices.to_tertile([trailing(t, end) for t, end in validation], own_edges)
    predicted = table.argmax(axis=1)[held]
    return (float(np.mean(predicted == np.asarray(wanted))), own_edges, table, table[held])


Measured = namedtuple("Measured", "name accuracy evaluated parameters model probabilities")


def score(name, probabilities, wanted):
    """One candidate's held-out accuracy from its distributions, so every row is scored the same way."""
    return Measured(name, float(np.mean(probabilities.argmax(axis=1) == wanted)), len(wanted),
                    0, None, probabilities)


def blended(members, wanted):
    """The mean of the candidates' distributions, scored on the same rows.

    An unweighted mean rather than a fitted weight per member: with three members that is three
    parameters fitted on the very rows the gain would then be claimed on.
    """
    return score("ensemble", sum(members) / len(members), wanted)


def calibration(probabilities, wanted, bins=10):
    """What the top-class probability is worth as a probability, before and after a Platt fit.

    Fitted on the first half of the held-out period and scored on the second, because a calibrator
    scored on its own rows always looks calibrated. This is the number the evidence states: "44%" has to
    mean the call comes true 44 times in 100, or the answer is quoting a decoration.
    """
    confidence = probabilities.max(axis=1)
    is_right = probabilities.argmax(axis=1) == wanted
    half = len(confidence) // 2
    fitted = Calibrator().fit(confidence[:half], is_right[:half])
    raw, held = confidence[half:], is_right[half:]
    mapped = np.asarray(fitted(raw))
    print(f"  raw        ece {expected_calibration_error(raw, held, bins):.3f}  "
          f"brier {brier(raw, held):.3f}  mean stated {raw.mean():.1%} against {held.mean():.1%} right")
    print(f"  calibrated ece {expected_calibration_error(mapped, held, bins):.3f}  "
          f"brier {brier(mapped, held):.3f}  mean stated {mapped.mean():.1%}")
    return fitted


def run(name, recurrent, config, bars, train, validation, args, edges, wanted):
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
        windows, labels = batch_of(bars, train, rows, config.price_window, args.horizon, edges, target)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(windows), labels)
        loss.backward()
        clip_grad_norm(optimizer.parameters, 1.0)
        optimizer.step()
        if step % args.log_every == 0:
            print(f"  step {step}/{args.steps}  loss {loss.item():.4f}  "
                  f"{step / (time.monotonic() - started):.2f} step/s")

    probabilities = held_out_probabilities(model, bars, validation, config.price_window, args.horizon,
                                           edges, args.batch_size, target)
    measured = score(name, probabilities, wanted)
    print(f"  {name} held-out accuracy {measured.accuracy:.1%} on {measured.evaluated:,} windows")
    return measured._replace(parameters=parameters, model=model)


def save(path, results, config, args, edges, trailing, trailing_edges, table, baseline, fitted):
    """The served artifact: the head's weights, and everything serving needs to decide what to do.

    Both accuracies go in the same file so `backend/models/price.py` compares the head against the
    baseline it was actually measured against, rather than against a number typed in somewhere else.
    """
    measured = next(row for row in results if row.name == models.RECURRENT_TOWER)
    pretrained.save(path, measured.model, config, metadata={
        "tower": measured.name, "target": args.target, "horizon": args.horizon, "cutoff": args.cutoff,
        "window": config.price_window, "channels": config.price_channels, "steps": args.steps,
        "classes": list(advisory.RISK_NAMES), "edges": [float(edge) for edge in edges],
        "accuracy": measured.accuracy, "evaluated": measured.evaluated, "majority": float(baseline),
        "persistence": trailing, "trailing_edges": [float(edge) for edge in trailing_edges],
        "table": [[float(value) for value in row] for row in table],
        "calibration": {"weight": fitted.weight, "bias": fitted.bias},
    })
    advisor = models.load(path)
    print(f"\nsaved {measured.name} to {path}; serving will use {advisor.version}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA / "prices"))
    parser.add_argument("--artifacts", default=str(ARTIFACTS))
    parser.add_argument("--towers", nargs="+", choices=models.TOWERS, default=list(models.TOWERS),
                        help="both by default, so the tower comparison stays reproducible")
    parser.add_argument("--save", default="", help="filename under --artifacts for the served "
                                                   f"artifact; needs the {models.RECURRENT_TOWER} tower")
    parser.add_argument("--cutoff", default="2021-01-01", help="validation starts on this date")
    parser.add_argument("--horizon", type=int, default=5, help="bars ahead the label looks")
    parser.add_argument("--stride", type=int, default=5, help="bars between consecutive windows")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--target", choices=("direction", "volatility"), default="volatility",
                        help="volatility is the default because direction measured unpredictable")
    args = parser.parse_args()
    # Checked before indexing rather than after training: the head is the only tower that gets served,
    # so asking to save without it is an hour of work that ends in a KeyError.
    if args.save and models.RECURRENT_TOWER not in args.towers:
        parser.error(f"--save writes the {models.RECURRENT_TOWER} tower, which --towers "
                     f"{' '.join(args.towers)} does not train")

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
        [target_of(bars[t], end, args.horizon, args.target) for t, end in validation], edges
    )
    counts = np.bincount(held_labels, minlength=3)
    baseline = counts.max() / counts.sum()
    print(f"  held-out class counts {counts.tolist()}, majority baseline {baseline:.1%}")

    trailing, trailing_edges, table, rule_rows = persistence(bars, train, validation, args, edges,
                                                             held_labels)
    print(f"  persistence baseline from the trailing value alone {trailing:.1%}")

    results = [run(name, name == models.RECURRENT_TOWER, config, bars, train, validation, args, edges,
                   held_labels) for name in args.towers]
    # The rule is a candidate in the blend, not just the bar it has to clear: it is the one member whose
    # confidence is already a measured hit rate, and averaging cannot be scored without it in the mean.
    results.append(blended([row.probabilities for row in results] + [rule_rows], held_labels))
    print("\n=== verdict ===")
    for row in results:
        print(f"  {row.name:12s} {row.accuracy:.1%}  ({row.accuracy - baseline:+.1%} vs majority, "
              f"{row.accuracy - trailing:+.1%} vs persistence)  {row.parameters:,} params")

    print("\n=== what the stated confidence is worth, held-out second half ===")
    fitted = {}
    for row in results:
        print(f"  [{row.name}]")
        fitted[row.name] = calibration(row.probabilities, held_labels)

    if args.save:
        save(Path(args.artifacts) / args.save, results, config, args, edges, trailing,
             trailing_edges, table, baseline, fitted[models.RECURRENT_TOWER])


if __name__ == "__main__":
    sys.exit(main())
