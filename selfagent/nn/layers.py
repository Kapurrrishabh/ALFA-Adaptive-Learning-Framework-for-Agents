"""The elementary layers: Linear, Embedding, LayerNorm, Dropout."""

from ..autograd import functional as F
from ..autograd import ops
from . import init
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, rng, bias=True):
        super().__init__()
        self.weight = init.normal(rng, in_features, out_features)
        self.bias = init.zeros(out_features) if bias else None

    def forward(self, x):
        projected = ops.matmul(x, self.weight)
        return projected if self.bias is None else ops.add(projected, self.bias)


class Embedding(Module):
    def __init__(self, num_embeddings, dim, rng):
        super().__init__()
        self.weight = init.normal(rng, num_embeddings, dim)

    def forward(self, ids):
        return F.embedding(self.weight, ids)


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = init.ones(dim)
        self.bias = init.zeros(dim)
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class Dropout(Module):
    def __init__(self, probability, rng):
        super().__init__()
        self.probability = probability
        self.rng = rng

    def forward(self, x):
        return F.dropout(x, self.probability, self.rng, self.training)
