"""Sampled future price paths for one instrument, and which of two models draws them.

  `Generative`   the return generator, `selfagent/models/return_generator.py`
  `GarchPaths`   GARCH(1,1)-t with the parameters fitted in the same training run

`load` serves the generator only when the file records it beating GARCH on the test period's log score by
more than one standard error of that difference -- the margin rule `price.beats` applies to the risk head,
here on a score that is lower-is-better. Otherwise GARCH draws the paths, because a model that has not
beaten two lines of arithmetic has no business drawing the future in its place.

The paths are what might happen and not what will: neither model forecasts direction, and every client
that shows them has to say so. `DIRECTION` carries that sentence so no page has to write its own.

Point in time: a path starts from bars up to the as-of date and nothing later, the scale is the year before
the context (`data/returns.scale_before`), and GARCH's long-run level is the variance of the returns up to
that date.
"""

from selfagent import pretrained
from selfagent.backend import xp
from selfagent.data import returns
from selfagent.models import ReturnGenerator
from selfagent.models.garch import Garch

DIRECTION = ("These are possible paths drawn from a model of how this stock's returns vary. They say how "
             "wide the future is, not which way it goes: no model here beats a coin on direction.")

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

REQUIRED = ("garch", "test", "cutoff", "test_from")


class Generative:
    name = "generative"

    def __init__(self, model, recorded):
        self.model, self.recorded = model, recorded
        self.bars_needed = returns.SCALE_BARS + model.config.price_window + 1

    def paths(self, closes, steps, count, rng):
        """(count, steps) daily log returns continuing `closes`."""
        values = returns.log_returns(closes)
        start = len(values) - self.model.config.price_window
        scale = returns.scale_before(values, start)
        context = xp.repeat(returns.encode(values[start:], scale)[None], count, axis=0)
        return returns.decode(self.model.sample_paths(context, steps, rng), scale, rng)

    def describe(self):
        return dict(name=self.name, measured=self.recorded["test"], test_from=self.recorded["test_from"])


class GarchPaths:
    name = "garch"

    def __init__(self, garch, recorded):
        self.garch, self.recorded = garch, recorded
        self.bars_needed = returns.SCALE_BARS + 1

    def paths(self, closes, steps, count, rng):
        values = returns.log_returns(closes)
        level = float(xp.var(values))
        # One step past the last return is the variance tomorrow starts from.
        start = self.garch.variance(xp.append(values, 0.0), level)[-1]
        return self.garch.simulate(steps, level, rng, count, start=start)

    def describe(self):
        return dict(name=self.name, parameters=self.garch.as_dict(), measured=self.recorded["test"],
                    test_from=self.recorded["test_from"])


def beats(recorded):
    """Whether the generator's test-period log score is lower than GARCH's by more than its standard error."""
    test = recorded["test"]
    return test["vs_garch"] < -test["vs_garch_error"]


def load(path):
    """The path drawer for `path`: the generator when its own record says it beat GARCH, else GARCH."""
    config, weights = pretrained.load(path)
    recorded = pretrained.metadata(path)
    missing = [name for name in REQUIRED if name not in recorded]
    if missing:
        raise ValueError(f"{path} records no {', '.join(missing)}; re-save it with "
                         f"scripts/train_return_generator.py, which writes what serving needs")
    if not beats(recorded):
        return GarchPaths(Garch(**recorded["garch"]), recorded)
    model = ReturnGenerator(config)
    model.load_state_dict(weights)
    model.eval()
    return Generative(model, recorded)


def priced(close, drawn):
    """(count, steps) prices from (count, steps) daily log returns starting at `close`."""
    return close * xp.exp(xp.cumsum(drawn, axis=-1))


def fan(close, drawn, quantiles=QUANTILES):
    """Price quantiles at every step ahead: {quantile: [price per step]}."""
    return {str(q): [float(v) for v in xp.quantile(priced(close, drawn), q, axis=0)] for q in quantiles}
