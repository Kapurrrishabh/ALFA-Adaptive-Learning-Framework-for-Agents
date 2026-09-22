"""Fused activations, normalisation and losses.

These are fused rather than composed from ops.py because the composed versions allocate
several intermediates per call and lose precision in the softmax and log-softmax paths.
"""

import math

from ..backend import xp
from . import ops
from .tensor import Tensor

GELU_COEFFICIENT = math.sqrt(2.0 / math.pi)


def sigmoid(tensor):
    tensor = ops.as_tensor(tensor)
    # via tanh rather than 1/(1+exp(-x)), which overflows for large negative x
    out = 0.5 * (1.0 + xp.tanh(0.5 * tensor.data))
    return ops.result(out, (tensor,), lambda g: (g * out * (1.0 - out),))


def softmax(tensor, axis=-1):
    tensor = ops.as_tensor(tensor)
    shifted = tensor.data - tensor.data.max(axis=axis, keepdims=True)
    # A masked position is shifted to a large negative, so underflowing to zero is the intended result.
    # Left unsilenced the flag surfaces on whatever op numpy checks next, blaming the wrong line.
    with xp.errstate(under="ignore"):
        exponentiated = xp.exp(shifted)
    out = exponentiated / exponentiated.sum(axis=axis, keepdims=True)
    return ops.result(
        out,
        (tensor,),
        lambda g: (((g - (g * out).sum(axis=axis, keepdims=True)) * out),),
    )


def gelu(tensor):
    """Tanh approximation, as used by BERT and GPT-2.

    The exact form needs erf, which lives in scipy and scipy is evaluation-only here.
    """
    tensor = ops.as_tensor(tensor)
    x = tensor.data
    inner = GELU_COEFFICIENT * (x + 0.044715 * x * x * x)
    tanh_inner = xp.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_inner)

    def vjp(g):
        d_inner = GELU_COEFFICIENT * (1.0 + 3.0 * 0.044715 * x * x)
        return (g * (0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * d_inner),)

    return ops.result(out, (tensor,), vjp)


def layer_norm(tensor, weight, bias, eps=1e-5):
    tensor, weight, bias = (ops.as_tensor(t) for t in (tensor, weight, bias))
    centred = tensor.data - tensor.data.mean(axis=-1, keepdims=True)
    inverse_std = 1.0 / xp.sqrt((centred * centred).mean(axis=-1, keepdims=True) + eps)
    normalised = centred * inverse_std
    out = normalised * weight.data + bias.data

    def vjp(g):
        g_normalised = g * weight.data
        grad_input = inverse_std * (
            g_normalised
            - g_normalised.mean(axis=-1, keepdims=True)
            - normalised * (g_normalised * normalised).mean(axis=-1, keepdims=True)
        )
        return (grad_input, g * normalised, g)

    return ops.result(out, (tensor, weight, bias), vjp)


def cross_entropy(logits, targets, ignore_index=-100):
    """Mean loss over positions whose target is not `ignore_index`.

    Masked-language-model batches leave most positions unlabelled, so the ignore path is
    the normal case, not an edge case.
    """
    logits = ops.as_tensor(logits)
    if logits.ndim != 2:
        raise ValueError(f"cross_entropy needs (positions, classes) logits, got {logits.shape}")
    targets = xp.asarray(targets)
    if targets.shape != (logits.shape[0],):
        raise ValueError(
            f"got {logits.shape[0]} rows of logits but {targets.shape} targets"
        )

    counted = targets != ignore_index
    count = int(counted.sum())
    if count == 0:
        raise ValueError("cross_entropy got a batch with no labelled positions")

    # Only the labelled rows are softmaxed. Unlabelled ones contribute nothing to the loss and
    # nothing to the gradient, and in an MLM batch they are ~85% of the positions.
    labelled = xp.flatnonzero(counted)
    selected = logits.data[labelled]
    picked_targets = targets[labelled]
    rows = xp.arange(count)

    shifted = selected - selected.max(axis=-1, keepdims=True)
    log_probabilities = shifted - xp.log(xp.exp(shifted).sum(axis=-1, keepdims=True))
    loss = -log_probabilities[rows, picked_targets].sum() / count

    def vjp(g):
        probabilities = xp.exp(log_probabilities)
        probabilities[rows, picked_targets] -= 1.0
        grad = xp.zeros_like(logits.data)
        grad[labelled] = probabilities * (g / count)
        return (grad,)

    return ops.result(loss, (logits,), vjp)


def dropout(tensor, probability, rng, training=True):
    if not training or probability == 0.0:
        return tensor
    if not 0.0 <= probability < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {probability}")
    keep = (rng.random(tensor.shape) >= probability) / (1.0 - probability)
    return ops.mul(tensor, Tensor(keep))


def embedding(weight, ids):
    weight = ops.as_tensor(weight)
    ids = xp.asarray(ids)

    def vjp(g):
        grad = xp.zeros_like(weight.data)
        xp.add.at(grad, ids, g)
        return (grad,)

    return ops.result(weight.data[ids], (weight,), vjp)
