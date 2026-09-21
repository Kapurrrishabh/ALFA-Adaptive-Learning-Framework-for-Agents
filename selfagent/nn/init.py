"""Parameter initialisation. One place so every layer starts from the same scheme."""

from ..autograd.tensor import Tensor
from ..backend import xp

# BERT's initializer_range. Small enough that a pre-LayerNorm stack stays stable without warmup.
INIT_STD = 0.02


def normal(rng, *shape, std=INIT_STD):
    return Tensor(rng.normal(0.0, std, shape), requires_grad=True)


def zeros(*shape):
    return Tensor(xp.zeros(shape), requires_grad=True)


def ones(*shape):
    return Tensor(xp.ones(shape), requires_grad=True)
