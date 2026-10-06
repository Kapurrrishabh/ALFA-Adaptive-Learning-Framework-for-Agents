"""GARCH(1,1) with Student-t innovations: the fifty-year-old model the return generator has to beat.

    sigma2[t] = omega + alpha * r[t-1]^2 + beta * sigma2[t-1]        r[t] = sigma[t] * z[t],  z ~ t(nu)

It is the reference Quant GANs and most of the synthetic-returns literature measure against, and on the
log score it is hard to beat, because it encodes the two regularities that matter most by construction:
volatility clusters (the beta term) and tails are fat (the t). A neural model that cannot beat it has
learned less than two lines of arithmetic, and should not be served in its place.

Fitted by variance targeting and a grid, not a numerical optimiser. Targeting fixes omega at
v * (1 - alpha - beta), with v the sample variance, which is the one parameter an optimiser handles badly;
alpha, beta and nu then move on a grid fine enough that the log score is flat across a cell. The fit is
deterministic, needs no library this project has banned, and its whole search is visible in `GRID`.
"""

import math

from backend.models.core.backend import xp

# alpha + beta must stay below 1 for the variance to have a finite long-run level. Daily equity fits sit at
# alpha 0.05-0.10 and beta 0.85-0.93; the grid covers well past both ends so the fit can land outside them.
ALPHAS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20)
BETAS = (0.70, 0.75, 0.80, 0.84, 0.87, 0.90, 0.92, 0.94, 0.96, 0.98)
# Degrees of freedom above 2 so the variance exists. Daily stock returns fit at 3-8.
NUS = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 30.0)
GRID = (ALPHAS, BETAS, NUS)


def filtered_variance(returns, omega, alpha, beta, start):
    """sigma2 for every bar, each computed from returns strictly before it. `start` seeds the first."""
    returns = xp.asarray(returns, dtype=xp.float64)
    variance = xp.empty(len(returns))
    previous = start
    for t in range(len(returns)):
        variance[t] = previous
        previous = omega + alpha * returns[t] ** 2 + beta * previous
    return variance


def t_log_density(returns, variance, nu):
    """log density of a Student-t scaled to have `variance`, so nu changes the tails and not the spread."""
    squared_scale = variance * (nu - 2.0) / nu
    constant = (math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi))
    return (constant - 0.5 * xp.log(squared_scale)
            - (nu + 1.0) / 2.0 * xp.log1p(xp.asarray(returns) ** 2 / (squared_scale * nu)))


class Garch:
    """alpha, beta and nu shared across tickers; each series brings its own long-run variance."""

    def __init__(self, alpha, beta, nu):
        if not alpha + beta < 1.0:
            raise ValueError(f"alpha {alpha} + beta {beta} >= 1 has no long-run variance")
        if not nu > 2.0:
            raise ValueError(f"nu {nu} <= 2 has no variance to scale to")
        self.alpha, self.beta, self.nu = alpha, beta, nu

    @classmethod
    def fit(cls, series):
        """The grid point with the highest total log likelihood over `series`, a list of return arrays.

        Pass training-period returns only. Every series is targeted at its own variance, so a quiet stock
        and a wild one share the dynamics without sharing a level.
        """
        series = [xp.asarray(one, dtype=xp.float64) for one in series if len(one) > 1]
        if not series:
            raise ValueError("no return series with more than one bar to fit GARCH on")
        best, chosen = -math.inf, None
        for alpha in ALPHAS:
            for beta in BETAS:
                if alpha + beta >= 1.0:
                    continue
                variances = [cls._variance(one, alpha, beta) for one in series]
                for nu in NUS:
                    total = sum(float(t_log_density(one, variance, nu).sum())
                                for one, variance in zip(series, variances))
                    if total > best:
                        best, chosen = total, (alpha, beta, nu)
        return cls(*chosen)

    @staticmethod
    def _variance(returns, alpha, beta, level=None):
        level = float(xp.var(returns)) if level is None else level
        return filtered_variance(returns, level * (1.0 - alpha - beta), alpha, beta, level)

    def variance(self, returns, level):
        """Each bar's one-step-ahead variance, from bars before it, around the long-run `level`."""
        return self._variance(returns, self.alpha, self.beta, level)

    def log_density(self, returns, level):
        return t_log_density(returns, self.variance(returns, level), self.nu)

    def simulate(self, steps, level, rng, paths=1, start=None):
        """(paths, steps) returns drawn from the fitted process, from variance `start` or the long-run level.

        A forecast has to start from the filtered variance at the forecast date; starting every path at the
        long-run level would hand the comparison to any model that reads the recent past.
        """
        omega = level * (1.0 - self.alpha - self.beta)
        variance = xp.full(paths, level if start is None else start, dtype=xp.float64)
        drawn = xp.empty((paths, steps))
        # A standard t has variance nu / (nu - 2); dividing that out leaves unit variance.
        unit = math.sqrt((self.nu - 2.0) / self.nu)
        for t in range(steps):
            drawn[:, t] = xp.sqrt(variance) * rng.standard_t(self.nu, paths) * unit
            variance = omega + self.alpha * drawn[:, t] ** 2 + self.beta * variance
        return drawn

    def as_dict(self):
        return dict(alpha=self.alpha, beta=self.beta, nu=self.nu)
