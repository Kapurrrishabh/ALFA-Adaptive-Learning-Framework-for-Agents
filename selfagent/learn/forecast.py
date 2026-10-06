"""How a distribution over returns is judged, and how it keeps learning from outcomes with its weights frozen.

**Scored, not eyeballed.** Two strictly proper scoring rules, so a forecast only scores best in expectation
by stating the distribution it actually believes:

  log score   -log density at the realised return; what the model is trained on, so the headline
  CRPS        E|X - y| - E|X - X'|/2 over sampled paths; scores a whole multi-day path in return units,
              where the log score needs a density the paths do not have

A score means nothing alone, so every one is paired against a baseline on the same returns. `paired` gives
the mean difference and its standard error with each window as one block: returns in one window share a
volatility regime, and treating them as independent would shrink the error bar several times over.

**Generated, then checked.** A synthetic path is only as good as the regularities it reproduces, and Cont
(2001) lists the ones every equity series shows. `stylized` measures each on a set of paths, and the
distance scores are the ones Quant GANs defines -- the L2 gap between historical and generated
autocorrelation vectors -- so a number here means what it means in that paper.

**Learning from outcomes without retraining.** `OnlineTemperature` rescales the model's logits and moves the
scale after every realised return by one gradient step on the log score, whose derivative is exact:

    d(-log p_y) / d(log T) = (z_y - E_p[z]) / T

so a model that has become overconfident as a regime changes is widened by its own mistakes, day by day, and
nothing about the network is retrained. It is the project's question in one number: whether the agent's
forecasts improve from feedback while the model under them stays fixed.
"""

import math

from ..backend import xp


def crps(samples, outcome):
    """CRPS of an ensemble against one outcome, by the sorted form of E|X - X'| so it costs m log m."""
    ordered = xp.sort(xp.asarray(samples, dtype=xp.float64))
    count = len(ordered)
    if not count:
        raise ValueError("CRPS needs at least one sample")
    spread = 2.0 * float(((2 * xp.arange(1, count + 1) - count - 1) * ordered).sum()) / count ** 2
    return float(xp.abs(ordered - outcome).mean()) - 0.5 * spread


def coverage(samples, outcome, level):
    """Whether `outcome` falls inside the central `level` interval of the samples."""
    low, high = xp.quantile(xp.asarray(samples), [(1 - level) / 2, (1 + level) / 2])
    return bool(low <= outcome <= high)


def paired(scores, baseline, blocks):
    """(mean of scores - baseline, its standard error), with every row of one block treated as one draw.

    Lower is better for both rules here, so a negative mean is the model winning.
    """
    difference = xp.asarray(scores, dtype=xp.float64) - xp.asarray(baseline, dtype=xp.float64)
    blocks = xp.asarray(blocks)
    labels, inverse = xp.unique(blocks, return_inverse=True)
    if len(labels) < 2:
        raise ValueError(f"a standard error needs at least two blocks, got {len(labels)}")
    means = xp.bincount(inverse, weights=difference) / xp.bincount(inverse)
    return float(difference.mean()), float(means.std(ddof=1) / math.sqrt(len(labels)))


def autocorrelation(series, lags):
    """Correlation of the series with itself `lag` bars later, for lag 1..lags."""
    centred = xp.asarray(series, dtype=xp.float64) - xp.mean(series)
    variance = float((centred ** 2).sum())
    if not variance > 0.0:
        raise ValueError("a constant series has no autocorrelation")
    return xp.array([(centred[:-lag] * centred[lag:]).sum() / variance for lag in range(1, lags + 1)])


def leverage(series, lags):
    """Corr(r[t + lag]^2, r[t]) for lag 1..lags. Negative in equities: falls are followed by turbulence."""
    series = xp.asarray(series, dtype=xp.float64)
    return xp.array([xp.corrcoef(series[:-lag], series[lag:] ** 2)[0, 1] for lag in range(1, lags + 1)])


def hill_tail_index(series, tail=0.05):
    """Hill's estimate of alpha in P(|r| > x) ~ x^-alpha, from the largest `tail` share of |r|.

    Equities sit between 2 and 5 (Cont 2001); a Gaussian has thin tails and scores high. `tail` is provisional,
    and the estimate is known to drift with it, so it is only ever compared at one fixed share.
    """
    ordered = xp.sort(xp.abs(xp.asarray(series, dtype=xp.float64)))[::-1]
    count = max(int(len(ordered) * tail), 2)
    top, floor = ordered[:count], ordered[count]
    if not floor > 0.0:
        raise ValueError("the tail sits on zeros, so it has no index")
    return float(1.0 / xp.log(top / floor).mean())


def wasserstein(first, second, points=1000):
    """Earth mover distance between two 1-d samples, read off their quantile functions."""
    grid = (xp.arange(points) + 0.5) / points
    return float(xp.abs(xp.quantile(first, grid) - xp.quantile(second, grid)).mean())


# Quant GANs scores autocorrelation to 250 lags on 4,000-bar paths. The generated paths here are shorter, so
# the lag count is too; 20 is a trading month, the horizon over which clustering is strongest.
LAGS = 20


def stylized(paths):
    """The stylized facts of a set of return paths, each fact pooled over all of them."""
    paths = [xp.asarray(path, dtype=xp.float64) for path in paths]
    pooled = xp.concatenate(paths)
    standard = (pooled - pooled.mean()) / pooled.std()
    return dict(
        excess_kurtosis=float((standard ** 4).mean() - 3.0),
        tail_index=hill_tail_index(pooled),
        acf=xp.mean([autocorrelation(path, LAGS) for path in paths], axis=0),
        acf_abs=xp.mean([autocorrelation(xp.abs(path), LAGS) for path in paths], axis=0),
        acf_squared=xp.mean([autocorrelation(path ** 2, LAGS) for path in paths], axis=0),
        leverage=xp.mean([leverage(path, LAGS) for path in paths], axis=0),
    )


def stylized_gap(real, generated):
    """How far generated facts are from real ones: an L2 distance per autocorrelation vector, as in Quant
    GANs, and the plain difference for the two scalars."""
    return {name: (float(xp.linalg.norm(real[name] - generated[name])) if xp.ndim(real[name])
                   else float(generated[name] - real[name]))
            for name in real}


class OnlineTemperature:
    """A logit temperature that moves after every outcome by one step of gradient descent on the log score.

    Started at 1.0, which is the model as trained, so before any feedback the forecast is unchanged.
    """

    # Step size on log T. Chosen, not tuned: at 0.01 it lowered the test-period log score by 0.0078 +/- 0.0017
    # nats per return, and no other value has been tried. A sweep of `eval_return_generator.py --step` would.
    STEP = 0.01

    def __init__(self, step=STEP):
        self.log_temperature = 0.0
        self.step = step

    @property
    def temperature(self):
        return math.exp(self.log_temperature)

    def probabilities(self, logits):
        scaled = xp.asarray(logits, dtype=xp.float64) / self.temperature
        weights = xp.exp(scaled - scaled.max(axis=-1, keepdims=True))
        return weights / weights.sum(axis=-1, keepdims=True)

    def observe(self, logits, outcome_bin):
        """Learn from one realised return: its bin under the logits the forecast was made from."""
        logits = xp.asarray(logits, dtype=xp.float64)
        expected = float((self.probabilities(logits) * logits).sum())
        self.log_temperature -= self.step * (float(logits[outcome_bin]) - expected) / self.temperature
