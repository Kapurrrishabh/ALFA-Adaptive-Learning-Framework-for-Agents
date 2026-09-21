"""Differentiable array operations. Every op derives its own backward pass.

Correctness here is checked against central finite differences in tests/test_gradcheck.py.
A wrong derivative in this file is silent and would poison every result downstream.
"""

import builtins

from .. import backend
from ..backend import xp
from .tensor import Tensor, grad_enabled


def as_tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


def _unbroadcast(grad, shape):
    """Sums a gradient back down to `shape`, undoing numpy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _accumulate(tensor, grad):
    if not tensor.requires_grad:
        return
    grad = _unbroadcast(grad, tensor.data.shape)
    tensor.grad = grad if tensor.grad is None else tensor.grad + grad


def result(data, parents, vjp):
    """Wraps `data` as a graph node. `vjp(grad)` returns one gradient per parent."""
    if backend.check_numerics() and not xp.isfinite(data).all():
        raise FloatingPointError(
            f"op produced a non-finite value, output shape {xp.shape(data)}. "
            "The traceback above names the op."
        )
    if not grad_enabled() or not builtins.any(p.requires_grad for p in parents):
        return Tensor(data)
    out = Tensor(data, requires_grad=True)
    out._parents = parents

    # The gradient arrives as an argument rather than being read off `out`. Closing over `out`
    # made every node part of a reference cycle, so a graph holding gigabytes of activations was
    # freed only when gc happened to run: memory grew about 1.4 GB per training step.
    def backward(gradient):
        for parent, grad in zip(parents, vjp(gradient)):
            if grad is not None:
                _accumulate(parent, grad)

    out._backward = backward
    return out


def add(a, b):
    a, b = as_tensor(a), as_tensor(b)
    return result(a.data + b.data, (a, b), lambda g: (g, g))


def sub(a, b):
    a, b = as_tensor(a), as_tensor(b)
    return result(a.data - b.data, (a, b), lambda g: (g, -g))


def mul(a, b):
    a, b = as_tensor(a), as_tensor(b)
    return result(a.data * b.data, (a, b), lambda g: (g * b.data, g * a.data))


def div(a, b):
    a, b = as_tensor(a), as_tensor(b)
    return result(
        a.data / b.data, (a, b), lambda g: (g / b.data, -g * a.data / (b.data * b.data))
    )


def neg(a):
    a = as_tensor(a)
    return result(-a.data, (a,), lambda g: (-g,))


def matmul(a, b):
    a, b = as_tensor(a), as_tensor(b)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(f"matmul needs 2-D or higher, got {a.shape} @ {b.shape}")

    # Apple's Accelerate BLAS raises divide-by-zero and overflow flags on float32 sgemm even
    # when inputs and outputs are finite, so the flags are muted and result() checks values.
    with xp.errstate(all="ignore"):
        product = a.data @ b.data

    def vjp(g):
        with xp.errstate(all="ignore"):
            return (g @ b.data.swapaxes(-1, -2), a.data.swapaxes(-1, -2) @ g)

    return result(product, (a, b), vjp)


def _restore_reduced(grad, shape, axis, keepdims):
    if axis is not None and not keepdims:
        grad = xp.expand_dims(grad, axis)
    # copy() because broadcast_to returns a read-only view that later steps may scale in place
    return xp.broadcast_to(grad, shape).copy()


def sum(tensor, axis=None, keepdims=False):
    tensor = as_tensor(tensor)
    shape = tensor.data.shape
    return result(
        tensor.data.sum(axis=axis, keepdims=keepdims),
        (tensor,),
        lambda g: (_restore_reduced(g, shape, axis, keepdims),),
    )


def mean(tensor, axis=None, keepdims=False):
    tensor = as_tensor(tensor)
    shape = tensor.data.shape
    out = tensor.data.mean(axis=axis, keepdims=keepdims)
    count = tensor.data.size // builtins.max(out.size, 1)
    return result(
        out,
        (tensor,),
        lambda g: (_restore_reduced(g / count, shape, axis, keepdims),),
    )


def exp(tensor):
    tensor = as_tensor(tensor)
    out = xp.exp(tensor.data)
    return result(out, (tensor,), lambda g: (g * out,))


def log(tensor):
    tensor = as_tensor(tensor)
    return result(xp.log(tensor.data), (tensor,), lambda g: (g / tensor.data,))


def sqrt(tensor):
    tensor = as_tensor(tensor)
    out = xp.sqrt(tensor.data)
    return result(out, (tensor,), lambda g: (g * 0.5 / out,))


def tanh(tensor):
    tensor = as_tensor(tensor)
    out = xp.tanh(tensor.data)
    return result(out, (tensor,), lambda g: (g * (1.0 - out * out),))


def reshape(tensor, shape):
    tensor = as_tensor(tensor)
    original = tensor.data.shape
    return result(
        tensor.data.reshape(shape), (tensor,), lambda g: (g.reshape(original),)
    )


def transpose(tensor, axes):
    tensor = as_tensor(tensor)
    inverse = [0] * len(axes)
    for position, axis in enumerate(axes):
        inverse[axis] = position
    return result(
        tensor.data.transpose(axes),
        (tensor,),
        lambda g: (g.transpose(tuple(inverse)),),
    )


def concat(tensors, axis=-1):
    tensors = tuple(as_tensor(t) for t in tensors)
    splits = xp.cumsum([t.data.shape[axis] for t in tensors[:-1]])
    return result(
        xp.concatenate([t.data for t in tensors], axis=axis),
        tensors,
        lambda g: tuple(xp.split(g, splits, axis=axis)),
    )


def getitem(tensor, key):
    tensor = as_tensor(tensor)

    def vjp(g):
        grad = xp.zeros_like(tensor.data)
        xp.add.at(grad, key, g)
        return (grad,)

    return result(tensor.data[key], (tensor,), vjp)


# Bound here rather than in tensor.py so that module cannot import this one back.
Tensor.__add__ = add
Tensor.__radd__ = lambda self, other: add(other, self)
Tensor.__sub__ = sub
Tensor.__rsub__ = lambda self, other: sub(other, self)
Tensor.__mul__ = mul
Tensor.__rmul__ = lambda self, other: mul(other, self)
Tensor.__truediv__ = div
Tensor.__rtruediv__ = lambda self, other: div(other, self)
Tensor.__neg__ = neg
Tensor.__matmul__ = matmul
Tensor.sum = sum
Tensor.mean = mean
Tensor.reshape = reshape
Tensor.transpose = transpose
Tensor.__getitem__ = getitem
