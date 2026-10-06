#!/usr/bin/env python3
"""The three questions a generative return model has to answer, each against GARCH-t and a trailing Gaussian.

  1. Does it write realistic data?  Paths of --steps returns from real test-period contexts, scored on
     Cont's stylized facts with Quant GANs' distances, against the real returns that followed.
  2. Does it forecast the path?     CRPS of the 5- and 20-day cumulative return over sampled paths.
  3. Does it learn from outcomes?   The whole test period replayed day by day with `OnlineTemperature`
     updating after every realised return and the weights frozen, scored prequentially -- each forecast
     before the day it forecasts is seen -- against the same model with no feedback.

Everything reads the checkpoint's own recorded periods, so the test period here is the one the training
run held out and the GARCH parameters are the ones it fitted on the training period.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfagent import pretrained  # noqa: E402
from selfagent.autograd import no_grad  # noqa: E402
from selfagent.data import returns  # noqa: E402
from selfagent.learn.forecast import (OnlineTemperature, crps, paired, stylized, stylized_gap,  # noqa: E402
                                      wasserstein)
from selfagent.models import ReturnGenerator  # noqa: E402
from selfagent.models.garch import Garch  # noqa: E402
from train_return_generator import TRAILING, load_series  # noqa: E402

HORIZONS = (5, 20)


def origins(series, length, test_from, steps, count, rng):
    """(series index, context start) for `count` forecast dates in the test period with `steps` bars after."""
    found = []
    for index, (_, dates, values) in enumerate(series):
        for start in range(returns.SCALE_BARS, len(values) - length - steps):
            if dates[start + length] >= test_from:
                found.append((index, start))
    if len(found) < count:
        raise SystemExit(f"only {len(found)} test-period forecast dates have {steps} bars after them")
    return [found[i] for i in rng.choice(len(found), count, replace=False)]


def draw(model, series, chosen, length, steps, per_origin, rng):
    """(origins, per_origin, steps) sampled returns in return units, and each origin's scale."""
    contexts, scales = [], []
    for index, start in chosen:
        values = series[index][2]
        scale = returns.scale_before(values, start)
        contexts.append(returns.encode(values[start : start + length], scale))
        scales.append(scale)
    rows = np.repeat(np.stack(contexts), per_origin, axis=0)
    tokens = model.sample_paths(rows, steps, rng)
    drawn = returns.decode(tokens, np.repeat(scales, per_origin)[:, None], rng)
    return drawn.reshape(len(chosen), per_origin, steps), np.array(scales)


def baselines(series, chosen, length, steps, per_origin, garch, levels, rng):
    """GARCH-t and trailing-Gaussian samples shaped like `draw`, each started from its forecast date."""
    fitted, trailing = [], []
    for index, start in chosen:
        values = series[index][2]
        origin = start + length
        variance = garch.variance(values[: origin + 1], levels[index])[origin]
        fitted.append(garch.simulate(steps, levels[index], rng, per_origin, start=variance))
        spread = values[origin - TRAILING : origin].std(ddof=1)
        trailing.append(rng.normal(0.0, spread, (per_origin, steps)))
    return np.stack(fitted), np.stack(trailing)


def realistic(model, series, garch, levels, length, test_from, steps, count, rng):
    chosen = origins(series, length, test_from, steps, count, rng)
    ours, scales = draw(model, series, chosen, length, steps, 1, rng)
    fitted, trailing = baselines(series, chosen, length, steps, 1, garch, levels, rng)
    real = np.stack([series[i][2][s + length : s + length + steps] for i, s in chosen])
    # In scale units, so a quiet stock and a wild one pool into one set of facts.
    unit = scales[:, None]
    facts = {name: stylized(list(paths[:, 0] / unit if paths.ndim == 3 else paths / unit))
             for name, paths in (("real", real), ("model", ours), ("GARCH-t", fitted),
                                 ("trailing Gaussian", trailing))}
    print(f"\n1. realistic data: {count} paths of {steps} returns from test-period contexts after "
          f"{test_from}, compared with the returns that actually followed")
    print(f"   {'':<18}{'kurtosis':>9}{'tail idx':>9}{'|acf|':>8}{'acf|r|':>8}{'acf r^2':>8}{'lev':>8}"
          f"{'EMD 1d':>8}{'EMD 5d':>8}{'acf|r| 1-5':>12}{'lev 1-5':>9}")
    flat_real = (real / unit).ravel()
    week_real = (real / unit)[:, : steps // 5 * 5].reshape(count, -1, 5).sum(axis=-1).ravel()
    for name, fact in facts.items():
        gap = stylized_gap(facts["real"], fact)
        paths = {"real": real, "model": ours[:, 0], "GARCH-t": fitted[:, 0],
                 "trailing Gaussian": trailing[:, 0]}[name] / unit
        week = paths[:, : steps // 5 * 5].reshape(count, -1, 5).sum(axis=-1).ravel()
        print(f"   {name:<18}{fact['excess_kurtosis']:>9.2f}{fact['tail_index']:>9.2f}{gap['acf']:>8.3f}"
              f"{gap['acf_abs']:>8.3f}{gap['acf_squared']:>8.3f}{gap['leverage']:>8.3f}"
              f"{wasserstein(flat_real, paths.ravel()):>8.3f}{wasserstein(week_real, week):>8.3f}"
              f"{fact['acf_abs'][:5].mean():>12.3f}{fact['leverage'][:5].mean():>9.3f}")
    print("   |acf| to lev are L2 distances from the real row over 20 lags, so real reads 0 and lower is closer;"
          " a distance cannot say too much from too little, so the last two columns are the values themselves")


def forecast(model, series, garch, levels, length, test_from, count, per_origin, rng):
    steps = max(HORIZONS)
    chosen = origins(series, length, test_from, steps, count, rng)
    ours, _ = draw(model, series, chosen, length, steps, per_origin, rng)
    fitted, trailing = baselines(series, chosen, length, steps, per_origin, garch, levels, rng)
    print(f"\n2. path forecast: CRPS of the cumulative return, {count} test-period dates x {per_origin} "
          f"sampled paths each (lower is better)")
    for horizon in HORIZONS:
        realised = [series[i][2][s + length : s + length + horizon].sum() for i, s in chosen]
        scores = {name: np.array([crps(paths[row, :, :horizon].sum(axis=-1), realised[row])
                                  for row in range(count)])
                  for name, paths in (("model", ours), ("GARCH-t", fitted), ("trailing", trailing))}
        blocks = np.arange(count)
        against_garch = paired(scores["model"], scores["GARCH-t"], blocks)
        against_trailing = paired(scores["model"], scores["trailing"], blocks)
        print(f"   {horizon:>2} days: model {scores['model'].mean():.5f}  GARCH-t {scores['GARCH-t'].mean():.5f}"
              f"  trailing {scores['trailing'].mean():.5f}   model minus GARCH {against_garch[0]:+.5f} +/- "
              f"{against_garch[1]:.5f}, minus trailing {against_trailing[0]:+.5f} +/- {against_trailing[1]:.5f}")


def feedback(model, series, length, test_from, step):
    """Replay the test period in date order; score each day before learning from it."""
    days = sorted({day for _, dates, _ in series for day in dates if day >= test_from})
    position = [{day: t for t, day in enumerate(dates)} for _, dates, _ in series]
    online = OnlineTemperature(step)
    frozen_scores, online_scores, months, frozen_pit, online_pit, temperatures = [], [], [], [], [], []
    called_up, went_up = [], []
    for day in days:
        rows, contexts, scales, realised = [], [], [], []
        for index, (_, _, values) in enumerate(series):
            t = position[index].get(day)
            if t is None or t - length < returns.SCALE_BARS:
                continue
            start = t - length
            if not np.abs(values[start - returns.SCALE_BARS : start]).sum() > 0:
                continue
            scale = returns.scale_before(values, start)
            rows.append(index)
            contexts.append(returns.encode(values[start:t], scale))
            scales.append(scale)
            realised.append(values[t])
        if not rows:
            continue
        with no_grad():
            logits = model(np.stack(contexts)).data[:, -1].astype(np.float64)
        realised, scales = np.array(realised), np.array(scales)
        for logit, outcome, scale in zip(logits, realised, scales):
            for probabilities, scores, pits in ((online.probabilities(logit), online_scores, online_pit),
                                                (OnlineTemperature().probabilities(logit), frozen_scores,
                                                 frozen_pit)):
                scores.append(-np.log(returns.density(probabilities[None], [outcome], scale)[0]))
                pits.append(returns.cdf(probabilities[None], [outcome], scale)[0])
            months.append(day[:7])
            # Bins are symmetric about zero with an edge on it, so the upper half is exactly P(r > 0).
            called_up.append(OnlineTemperature().probabilities(logit)[returns.BINS // 2 :].sum() > 0.5)
            went_up.append(outcome > 0)
            # Learning happens after the day is scored, so no forecast has seen its own outcome.
            online.observe(logit, int(returns.encode([outcome], scale)[0]))
        temperatures.append(online.temperature)

    gap, error = paired(online_scores, frozen_scores, months)
    print(f"\n3. learning from outcomes, weights frozen: {len(frozen_scores):,} test-period returns over "
          f"{len(days):,} days, each forecast scored before its day is learned from")
    print(f"   negative log score per return  frozen {np.mean(frozen_scores):.4f}  with feedback "
          f"{np.mean(online_scores):.4f}   difference {gap:+.4f} +/- {error:.4f} (blocked by month; "
          f"negative is feedback helping)")
    for name, pits in (("frozen", frozen_pit), ("with feedback", online_pit)):
        pits = np.array(pits)
        inside = ((pits >= 0.05) & (pits <= 0.95)).mean()
        deciles = np.histogram(pits, bins=10, range=(0, 1))[0] / len(pits)
        print(f"   {name:<14} 90% interval holds {inside:.1%} of outcomes; PIT deciles "
              f"{' '.join(f'{d:.3f}' for d in deciles)} (0.100 each when calibrated)")
    called_up, went_up = np.array(called_up), np.array(went_up)
    hit, majority = (called_up == went_up).mean(), max(went_up.mean(), 1 - went_up.mean())
    error = (hit * (1 - hit) / len(went_up)) ** 0.5
    print(f"   direction, P(up) > 0.5 as the call: right {hit:.1%} +/- {error:.1%} against {majority:.1%} for "
          f"always calling the more common side")
    quartiles = np.quantile(temperatures, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"   temperature over the period: min {quartiles[0]:.3f}  q1 {quartiles[1]:.3f}  median "
          f"{quartiles[2]:.3f}  q3 {quartiles[3]:.3f}  max {quartiles[4]:.3f}  final {temperatures[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/prices")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--checkpoint", default="returns.npz")
    parser.add_argument("--paths", type=int, default=128, help="realistic-data paths, one per context")
    parser.add_argument("--steps", type=int, default=250, help="returns per realistic-data path: a year")
    parser.add_argument("--dates", type=int, default=150, help="forecast dates for the CRPS comparison")
    parser.add_argument("--per-date", type=int, default=50, help="paths sampled per forecast date")
    parser.add_argument("--step", type=float, default=OnlineTemperature.STEP,
                        help="the online temperature's step size on log T")
    parser.add_argument("--only", choices=("realistic", "forecast", "feedback"), default=None)
    args = parser.parse_args()

    path = Path(args.artifacts) / args.checkpoint
    config, weights = pretrained.load(path)
    recorded = pretrained.metadata(path)
    model = ReturnGenerator(config)
    model.load_state_dict(weights)
    model.eval()
    garch = Garch(**recorded["garch"])
    series = load_series(args.prices)
    levels = [float(np.var(values[dates < recorded["cutoff"]])) if (dates < recorded["cutoff"]).sum() > 1
              else float(np.var(values[: returns.SCALE_BARS])) for _, dates, values in series]
    rng = np.random.default_rng(config.seed)
    length, test_from = config.price_window, recorded["test_from"]
    print(f"{path}: trained before {recorded['cutoff']}, chosen on the period to {test_from}, "
          f"evaluated after it; GARCH-t {recorded['garch']}")

    if args.only in (None, "realistic"):
        realistic(model, series, garch, levels, length, test_from, args.steps, args.paths, rng)
    if args.only in (None, "forecast"):
        forecast(model, series, garch, levels, length, test_from, args.dates, args.per_date, rng)
    if args.only in (None, "feedback"):
        feedback(model, series, length, test_from, args.step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
