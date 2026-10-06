"""A GRU that reads the last WINDOW days of a stock and projects the next HORIZON days' price path.

It is the dashed line drawn beside the return generator's median dots: a second model, built the
other way (a point forecast trained on squared error, where the generator draws whole distributions),
so the two can be read against each other.

Inputs per day: the stock's log return, its absolute value, and the Nifty's log return, each in units
of that series' mean absolute return over the year before the window (`returns.scale_before`), so one
model reads a quiet utility and a volatile small cap on the same scale. The target for day k is the
cumulative log return over k days divided by scale * sqrt(k): a random walk gives every k the same
spread, so no horizon dominates the loss.
"""
import json

from backend.models.core.autograd import ops
from backend.models.core.backend import default_rng, xp
from backend.models.core.nn.layers import Linear
from backend.models.core.nn.module import Module
from backend.models.core.nn.recurrent import GRU
from backend.models.data import returns

WINDOW, HORIZON, FEATURES = 60, 20, 3


class PathGRU(Module):
    def __init__(self, hidden, seed=0):
        super().__init__()
        rng = default_rng(seed)
        self.hidden = hidden
        self.gru = GRU(FEATURES, hidden, rng)
        self.head = Linear(hidden, HORIZON, rng)

    def forward(self, windows):
        """(batch, WINDOW, FEATURES) -> (batch, HORIZON) scaled cumulative returns."""
        states = self.gru(ops.as_tensor(windows))
        return self.head(ops.getitem(states, (slice(None), -1)))


def window_at(stock_returns, market_returns, end):
    """Features for the WINDOW returns ending at index `end` (inclusive), and the stock's scale."""
    start = end - WINDOW + 1
    scale = returns.scale_before(stock_returns, start)
    market_scale = returns.scale_before(market_returns, start)
    r, m = stock_returns[start:end + 1] / scale, market_returns[start:end + 1] / market_scale
    return xp.stack([r, xp.abs(r), m], axis=-1), scale


def targets_at(closes, end, scale):
    """Scaled cumulative log returns for days 1..HORIZON after close index `end + 1`."""
    k = xp.arange(1, HORIZON + 1)
    return xp.log(closes[end + 1 + k] / closes[end + 1]) / (scale * xp.sqrt(k))


def to_prices(last_close, predicted, scale):
    """A (HORIZON,) scaled forecast back to prices."""
    k = xp.arange(1, HORIZON + 1)
    return last_close * xp.exp(predicted * scale * xp.sqrt(k))


def save(path, model, record):
    xp.savez(path, hidden=model.hidden, record=json.dumps(record), **model.state_dict())


def load(path):
    stored = xp.load(path)
    model = PathGRU(int(stored["hidden"]))
    model.load_state_dict({k: stored[k] for k in stored.files if k not in ("hidden", "record")})
    model.eval()
    return model, json.loads(str(stored["record"]))
