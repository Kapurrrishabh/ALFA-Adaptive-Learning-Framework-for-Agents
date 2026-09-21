"""A gated recurrent layer, so the price window has a second reader to be measured against.

A GRU carries one hidden state along the window instead of attending across it, so reading 128 bars
costs 128 small matmuls rather than a 128x128 attention matrix, and it needs no patching to stay
affordable. Whether it also reads a price window *better* than the transformer tower is an open
question, which is why both exist rather than one being chosen up front.
"""

from ..autograd import functional as F
from ..autograd import ops
from ..backend import xp
from .layers import Linear
from .module import Module


class GRU(Module):
    """(batch, timesteps, features) -> (batch, timesteps, hidden_size), every state returned."""

    def __init__(self, in_features, hidden_size, rng):
        super().__init__()
        self.hidden_size = hidden_size
        # All three gates for all timesteps in one matmul before the loop. In NumPy the per-step
        # overhead dominates, so this is the difference between one matmul and three per timestep.
        self.input_gates = Linear(in_features, 3 * hidden_size, rng)
        # No bias on the recurrent half: it would only duplicate the input projection's own.
        self.hidden_gates = Linear(hidden_size, 3 * hidden_size, rng, bias=False)

    def forward(self, sequence):
        batch, timesteps = sequence.shape[0], sequence.shape[1]
        size = self.hidden_size
        projected = self.input_gates(sequence)
        hidden = ops.as_tensor(xp.zeros((batch, size), dtype=projected.data.dtype))

        states = []
        for step in range(timesteps):
            from_input = ops.getitem(projected, (slice(None), step))
            from_hidden = self.hidden_gates(hidden)
            update = F.sigmoid(ops.add(_gate(from_input, 0, size), _gate(from_hidden, 0, size)))
            reset = F.sigmoid(ops.add(_gate(from_input, 1, size), _gate(from_hidden, 1, size)))
            # Resetting after the recurrent matmul rather than before it is what lets all three
            # gates share the single hidden projection above.
            candidate = ops.tanh(
                ops.add(_gate(from_input, 2, size), ops.mul(reset, _gate(from_hidden, 2, size)))
            )
            hidden = ops.add(ops.mul(ops.sub(1.0, update), hidden), ops.mul(update, candidate))
            states.append(ops.reshape(hidden, (batch, 1, size)))
        return ops.concat(states, axis=1)


def _gate(gates, index, size):
    return ops.getitem(gates, (slice(None), slice(index * size, (index + 1) * size)))
