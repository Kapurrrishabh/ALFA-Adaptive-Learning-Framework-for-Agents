"""A GRU that reads the last WINDOW days of a stock and forecasts each of the next HORIZON days.

For every day ahead it gives a distribution, not a single number: the expected log return and how
large that day's move is likely to be (its standard deviation). The median path is the dashed line
on the chart; paths drawn from the distributions zigzag the way prices do, with bigger swings on the
days it expects to be volatile. Direction is not something it has shown it can call; the size of
moves is, because volatility clusters.

Inputs per day, each in units of that series' mean absolute return over the year before the window
(`returns.scale_before`), so a quiet utility and a volatile small cap read on one scale: the stock's
log return, its absolute value, its high-low range, and the Nifty's log return. Trained on the
Gaussian negative log-likelihood of each day's scaled return.
"""
import json
import math

from backend.config import FORECAST_DAYS
from backend.models.core.autograd import ops
from backend.models.core.backend import default_rng, xp
from backend.models.core.nn.layers import Linear
from backend.models.core.nn.module import Module
from backend.models.core.nn.recurrent import GRU
from backend.models.data import returns

WINDOW, HORIZON, FEATURES = 60, FORECAST_DAYS, 4


class PathGRU(Module):
    def __init__(self, hidden, seed=0):
        super().__init__()
        rng = default_rng(seed)
        self.hidden = hidden
        self.gru = GRU(FEATURES, hidden, rng)
        self.head = Linear(hidden, 2 * HORIZON, rng)

    def forward(self, windows):
        """(batch, WINDOW, FEATURES) -> (mean, log_sd), each (batch, HORIZON), in scaled daily returns."""
        out = self.head(ops.getitem(self.gru(ops.as_tensor(windows)), (slice(None), -1)))
        return ops.getitem(out, (slice(None), slice(0, HORIZON))), ops.getitem(out, (slice(None), slice(HORIZON, None)))


def nll(mean, log_sd, target):
    """Gaussian negative log-likelihood per scaled daily return (constant dropped)."""
    z = ops.div(ops.sub(ops.as_tensor(target), mean), ops.exp(log_sd))
    return ops.mean(ops.add(log_sd, ops.mul(0.5, ops.mul(z, z))))


def window_at(stock_returns, market_returns, ranges, end):
    """Features for the WINDOW days ending at return index `end` (inclusive), and the stock's scale."""
    start = end - WINDOW + 1
    scale = returns.scale_before(stock_returns, start)
    market_scale = returns.scale_before(market_returns, start)
    r, m = stock_returns[start:end + 1] / scale, market_returns[start:end + 1] / market_scale
    return xp.stack([r, xp.abs(r), ranges[start:end + 1] / scale, m], axis=-1), scale


def targets_at(stock_returns, end, scale):
    """Scaled daily log returns for the HORIZON days after return index `end`."""
    return stock_returns[end + 1:end + 1 + HORIZON] / scale


def median_prices(last_close, mean, scale):
    """The path through each day's median: compounding the expected daily log returns."""
    return last_close * xp.exp(xp.cumsum(mean * scale))


def sample_paths(last_close, mean, log_sd, scale, count, seed=0):
    """`count` possible price paths, each day drawn from that day's forecast distribution."""
    draws = mean + xp.exp(log_sd) * default_rng(seed).standard_normal((count, len(mean)))
    return last_close * xp.exp(xp.cumsum(draws * scale, axis=1))


def interval(mean, log_sd, scale, k, z):
    """The (low, high) log-return range for the cumulative move over the first k days, at z sigmas."""
    centre = float(mean[:k].sum()) * scale
    spread = z * math.sqrt(float((xp.exp(log_sd[:k]) ** 2).sum())) * scale
    return centre - spread, centre + spread


def save(path, model, record):
    xp.savez(path, hidden=model.hidden, horizon=HORIZON, record=json.dumps(record), **model.state_dict())


def load(path):
    stored = xp.load(path)
    if int(stored["horizon"]) != HORIZON:
        raise ValueError(f"{path} forecasts {int(stored['horizon'])} days but FORECAST_DAYS is {HORIZON}; retrain it "
                         "with python -m backend.models.training.train_path_gru")
    model = PathGRU(int(stored["hidden"]))
    model.load_state_dict({k: stored[k] for k in stored.files if k not in ("hidden", "horizon", "record")})
    model.eval()
    return model, json.loads(str(stored["record"]))
