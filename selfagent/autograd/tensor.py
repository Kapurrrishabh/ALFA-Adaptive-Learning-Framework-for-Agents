"""Reverse-mode autodiff over a graph rebuilt on every forward pass.

A Tensor always holds floats. Integer inputs such as token ids stay plain arrays, so
a wrong dtype cannot silently reach a gradient.
"""

from .. import backend
from ..backend import xp

_grad_enabled = True


def grad_enabled():
    return _grad_enabled


class no_grad:
    """Stops graph building, for evaluation and for frozen-backbone forward passes."""

    def __enter__(self):
        global _grad_enabled
        self._previous = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *exc_info):
        global _grad_enabled
        _grad_enabled = self._previous
        return False


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward")

    def __init__(self, data, requires_grad=False):
        self.data = xp.asarray(data, dtype=backend.dtype())
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = ()
        self._backward = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def detach(self):
        """A value with no history. Used for reference policies and stored targets."""
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    def item(self):
        return float(self.data.reshape(-1)[0])

    def backward(self):
        if self.data.size != 1:
            raise ValueError(
                f"backward() needs a scalar loss, got shape {self.data.shape}. "
                "Reduce with sum() or mean() first."
            )
        self.grad = xp.ones_like(self.data)
        for node in reversed(_topological_order(self)):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"


def _topological_order(root):
    """Children before parents, iteratively — a deep graph must not blow the Python stack."""
    order = []
    seen = set()
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for parent in node._parents:
            if id(parent) not in seen:
                stack.append((parent, False))
    return order
