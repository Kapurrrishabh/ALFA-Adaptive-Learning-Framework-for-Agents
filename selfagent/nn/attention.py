"""Multi-head attention, used both for self-attention and for cross-modal attention."""

import math

from ..autograd import functional as F
from ..autograd import ops
from ..autograd.tensor import Tensor
from ..backend import xp
from .layers import Dropout, Linear
from .module import Module

# Large finite value rather than -inf: exp() of it is 0 in float32 without risking nan
MASKED_SCORE = -1e9


def padding_mask(is_real_token):
    """Turns a (batch, length) 1/0 array into an additive mask that broadcasts over heads."""
    keep = xp.asarray(is_real_token)
    if keep.ndim != 2:
        raise ValueError(f"expected (batch, length) token mask, got shape {keep.shape}")
    return Tensor((1.0 - keep)[:, None, None, :] * MASKED_SCORE)


def causal_mask(length):
    """Additive mask stopping a position from attending to later ones. Broadcasts over batch/heads."""
    return Tensor(xp.triu(xp.ones((length, length)), k=1)[None, None] * MASKED_SCORE)


class MultiHeadAttention(Module):
    """`context=None` gives self-attention; passing another sequence gives cross-attention."""

    def __init__(self, dim, num_heads, dropout, rng):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query = Linear(dim, dim, rng)
        self.key = Linear(dim, dim, rng)
        self.value = Linear(dim, dim, rng)
        self.out = Linear(dim, dim, rng)
        self.dropout = Dropout(dropout, rng)

    def _to_heads(self, x):
        batch, length, _ = x.shape
        split = ops.reshape(x, (batch, length, self.num_heads, self.head_dim))
        return ops.transpose(split, (0, 2, 1, 3))

    def _from_heads(self, x):
        batch, _, length, _ = x.shape
        merged = ops.transpose(x, (0, 2, 1, 3))
        return ops.reshape(merged, (batch, length, self.num_heads * self.head_dim))

    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
        queries = self._to_heads(self.query(x))
        keys = self._to_heads(self.key(context))
        values = self._to_heads(self.value(context))

        scores = ops.mul(ops.matmul(queries, ops.transpose(keys, (0, 1, 3, 2))), self.scale)
        if mask is not None:
            scores = ops.add(scores, mask)
        weights = self.dropout(F.softmax(scores))
        return self.out(self._from_heads(ops.matmul(weights, values)))
