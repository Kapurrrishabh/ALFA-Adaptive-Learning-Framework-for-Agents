"""S1: every backward pass matches central finite differences.

This is the project's foundation gate. A hand-derived gradient that is subtly wrong still
trains to a plausible-looking loss curve, so nothing downstream is trustworthy until this
file is green.
"""

import math

import numpy as np
import pytest

from selfagent import backend
from selfagent.autograd import functional as F
from selfagent.autograd import ops
from selfagent.autograd.tensor import Tensor

TOLERANCE = 1e-6
STEP = 1e-6


@pytest.fixture(autouse=True)
def float64_precision():
    """float32 finite differences are only accurate to ~1e-3, far too coarse to check against."""
    backend.set_dtype("float64")
    yield
    backend.set_dtype("float32")


def numeric_gradient(scalar_fn, tensor):
    flat = tensor.data.reshape(-1)
    gradient = np.zeros(flat.size)
    for index in range(flat.size):
        original = flat[index]
        flat[index] = original + STEP
        plus = scalar_fn().item()
        flat[index] = original - STEP
        minus = scalar_fn().item()
        flat[index] = original
        gradient[index] = (plus - minus) / (2.0 * STEP)
    return gradient.reshape(tensor.data.shape)


def assert_gradients_match(build_scalar, inputs):
    """`build_scalar(*inputs)` must return a scalar Tensor and re-read input data each call."""
    for tensor in inputs:
        tensor.grad = None
    build_scalar(*inputs).backward()

    for position, tensor in enumerate(inputs):
        expected = numeric_gradient(lambda: build_scalar(*inputs), tensor)
        actual = tensor.grad
        assert actual is not None, f"input {position} received no gradient"
        scale = max(1.0, float(np.abs(expected).max()), float(np.abs(actual).max()))
        error = float(np.abs(actual - expected).max()) / scale
        assert error < TOLERANCE, f"input {position}: relative error {error:.2e}"


def rand(*shape, seed=0):
    return Tensor(np.random.default_rng(seed).standard_normal(shape), requires_grad=True)


def weighted_sum(tensor, seed=99):
    """Reduces to a scalar with random weights, so symmetric errors cannot cancel out."""
    weights = np.random.default_rng(seed).standard_normal(tensor.shape)
    return ops.sum(ops.mul(tensor, Tensor(weights)))


# --- ops.py -----------------------------------------------------------------


def test_add():
    assert_gradients_match(lambda a, b: weighted_sum(ops.add(a, b)), [rand(3, 4), rand(3, 4, seed=1)])


def test_add_broadcast():
    assert_gradients_match(lambda a, b: weighted_sum(ops.add(a, b)), [rand(3, 4), rand(4, seed=1)])


def test_sub():
    assert_gradients_match(lambda a, b: weighted_sum(ops.sub(a, b)), [rand(2, 3), rand(2, 3, seed=1)])


def test_mul():
    assert_gradients_match(lambda a, b: weighted_sum(ops.mul(a, b)), [rand(3, 4), rand(3, 4, seed=1)])


def test_mul_broadcast():
    assert_gradients_match(lambda a, b: weighted_sum(ops.mul(a, b)), [rand(2, 3, 4), rand(1, 4, seed=1)])


def test_div():
    denominator = Tensor(np.random.default_rng(1).standard_normal((3, 4)) + 3.0, requires_grad=True)
    assert_gradients_match(lambda a, b: weighted_sum(ops.div(a, b)), [rand(3, 4), denominator])


def test_neg():
    assert_gradients_match(lambda a: weighted_sum(ops.neg(a)), [rand(3, 4)])


def test_matmul():
    assert_gradients_match(lambda a, b: weighted_sum(ops.matmul(a, b)), [rand(3, 4), rand(4, 5, seed=1)])


def test_matmul_batched():
    assert_gradients_match(
        lambda a, b: weighted_sum(ops.matmul(a, b)), [rand(2, 3, 4, 5), rand(2, 3, 5, 6, seed=1)]
    )


def test_matmul_rejects_1d():
    with pytest.raises(ValueError, match="2-D or higher"):
        ops.matmul(rand(4), rand(4, 3))


def test_sum_all():
    assert_gradients_match(lambda a: ops.mul(ops.sum(a), Tensor(2.0)), [rand(3, 4)])


def test_sum_axis():
    assert_gradients_match(lambda a: weighted_sum(ops.sum(a, axis=1)), [rand(3, 4, 5)])


def test_sum_axis_keepdims():
    assert_gradients_match(lambda a: weighted_sum(ops.sum(a, axis=-1, keepdims=True)), [rand(3, 4)])


def test_mean_all():
    assert_gradients_match(lambda a: ops.mul(ops.mean(a), Tensor(2.0)), [rand(3, 4)])


def test_mean_axis():
    assert_gradients_match(lambda a: weighted_sum(ops.mean(a, axis=-1)), [rand(3, 4, 5)])


def test_exp():
    assert_gradients_match(lambda a: weighted_sum(ops.exp(a)), [rand(3, 4)])


def test_log():
    positive = Tensor(np.random.default_rng(0).random((3, 4)) + 0.5, requires_grad=True)
    assert_gradients_match(lambda a: weighted_sum(ops.log(a)), [positive])


def test_sqrt():
    positive = Tensor(np.random.default_rng(0).random((3, 4)) + 0.5, requires_grad=True)
    assert_gradients_match(lambda a: weighted_sum(ops.sqrt(a)), [positive])


def test_tanh():
    assert_gradients_match(lambda a: weighted_sum(ops.tanh(a)), [rand(3, 4)])


def test_reshape():
    assert_gradients_match(lambda a: weighted_sum(ops.reshape(a, (4, 3))), [rand(3, 4)])


def test_transpose():
    assert_gradients_match(lambda a: weighted_sum(ops.transpose(a, (0, 2, 1, 3))), [rand(2, 3, 4, 5)])


def test_concat():
    assert_gradients_match(
        lambda a, b: weighted_sum(ops.concat([a, b], axis=-1)), [rand(3, 4), rand(3, 2, seed=1)]
    )


def test_getitem_slice():
    assert_gradients_match(lambda a: weighted_sum(ops.getitem(a, (slice(None), 0))), [rand(3, 4)])


def test_getitem_repeated_index():
    """Duplicate indices must accumulate, not overwrite."""
    assert_gradients_match(lambda a: weighted_sum(ops.getitem(a, [1, 1, 2])), [rand(4, 3)])


# --- functional.py ----------------------------------------------------------


def test_sigmoid():
    assert_gradients_match(lambda a: weighted_sum(F.sigmoid(a)), [rand(3, 4)])


def test_softmax():
    assert_gradients_match(lambda a: weighted_sum(F.softmax(a)), [rand(3, 4)])


def test_softmax_middle_axis():
    assert_gradients_match(lambda a: weighted_sum(F.softmax(a, axis=1)), [rand(2, 3, 4)])


def test_gelu():
    assert_gradients_match(lambda a: weighted_sum(F.gelu(a)), [rand(3, 4)])


def test_layer_norm():
    inputs = [rand(3, 5), rand(5, seed=1), rand(5, seed=2)]
    assert_gradients_match(lambda x, w, b: weighted_sum(F.layer_norm(x, w, b)), inputs)


def test_layer_norm_3d():
    inputs = [rand(2, 3, 5), rand(5, seed=1), rand(5, seed=2)]
    assert_gradients_match(lambda x, w, b: weighted_sum(F.layer_norm(x, w, b)), inputs)


def test_cross_entropy():
    targets = np.array([0, 3, 1])
    assert_gradients_match(lambda a: F.cross_entropy(a, targets), [rand(3, 4)])


def test_cross_entropy_with_ignored_positions():
    targets = np.array([0, -100, 1, -100])
    assert_gradients_match(lambda a: F.cross_entropy(a, targets), [rand(4, 5)])


def test_ignored_positions_get_exactly_zero_gradient():
    """An unlabelled MLM position must not push the model in any direction."""
    logits = rand(4, 5)
    F.cross_entropy(logits, np.array([1, -100, 2, -100])).backward()
    assert np.allclose(logits.grad[[1, 3]], 0.0)
    assert np.abs(logits.grad[[0, 2]]).max() > 1e-3


def test_cross_entropy_all_ignored_raises():
    with pytest.raises(ValueError, match="no labelled positions"):
        F.cross_entropy(rand(2, 3), np.array([-100, -100]))


def test_cross_entropy_target_count_mismatch_raises():
    with pytest.raises(ValueError, match="targets"):
        F.cross_entropy(rand(3, 4), np.array([0, 1]))


def test_cross_entropy_matches_hand_computed_value():
    logits = Tensor([[0.0, math.log(3.0)]])
    assert F.cross_entropy(logits, np.array([1])).item() == pytest.approx(math.log(4.0 / 3.0))


def test_embedding():
    ids = np.array([[0, 2], [1, 2]])
    assert_gradients_match(lambda w: weighted_sum(F.embedding(w, ids)), [rand(4, 3)])


def test_dropout_is_identity_in_eval():
    x = rand(4, 8)
    kept = F.dropout(x, 0.5, np.random.default_rng(0), training=False)
    assert kept is x


def test_dropout_scales_survivors_to_preserve_expectation():
    x = Tensor(np.ones((200, 200)))
    kept = F.dropout(x, 0.5, np.random.default_rng(0), training=True)
    assert float(kept.data.mean()) == pytest.approx(1.0, abs=0.02)


# --- graph mechanics --------------------------------------------------------


def test_gradient_accumulates_when_a_tensor_is_used_twice():
    x = rand(3, 4)
    assert_gradients_match(lambda a: weighted_sum(ops.mul(a, a)), [x])


def test_diamond_graph_accumulates_both_paths():
    assert_gradients_match(
        lambda a: weighted_sum(ops.add(ops.exp(a), ops.mul(a, Tensor(3.0)))), [rand(3, 4)]
    )


def test_no_grad_stops_graph_building():
    from selfagent.autograd import no_grad

    x = rand(3, 4)
    with no_grad():
        out = ops.mul(x, x)
    assert not out.requires_grad
    assert out._parents == ()


def test_backward_on_non_scalar_raises():
    with pytest.raises(ValueError, match="scalar loss"):
        ops.mul(rand(3, 4), Tensor(2.0)).backward()


def test_graph_is_freed_without_the_collector():
    # The backward closure used to capture its own output tensor, so every node sat in a reference
    # cycle and a graph holding gigabytes of activations survived until gc ran: training grew by
    # about 1.4 GB per step. The collector is off here, so only refcounting can reclaim the nodes.
    import gc

    def live_tensors():
        return sum(1 for obj in gc.get_objects() if isinstance(obj, Tensor))

    gc.collect()
    gc.disable()
    try:
        before = live_tensors()
        weighted_sum(ops.mul(rand(3, 4), Tensor(2.0))).backward()
        assert live_tensors() == before
    finally:
        gc.enable()


def test_constant_inputs_get_no_gradient():
    trainable = rand(3, 4)
    constant = Tensor(np.ones((3, 4)))
    weighted_sum(ops.mul(trainable, constant)).backward()
    assert constant.grad is None
    assert trainable.grad is not None


# --- decoder ----------------------------------------------------------------


def test_decoder_block_gradients():
    # The memory is checked as well as the input: a decoder that returns no gradient through
    # cross-attention leaves the encoder untrained, and the loss still falls because the decoder
    # learns an unconditional prior instead.
    from selfagent.nn import TransformerDecoderBlock, causal_mask

    block = TransformerDecoderBlock(8, 2, 16, 0.0, np.random.default_rng(0)).eval()
    causal = causal_mask(4)
    assert_gradients_match(
        lambda x, memory: weighted_sum(block(x, memory, causal)),
        [rand(2, 4, 8, seed=1), rand(2, 5, 8, seed=2)],
    )


# --- recurrent --------------------------------------------------------------


def test_gru_gradients_flow_through_every_timestep():
    # Backpropagation through time is where a recurrent layer breaks silently: if the hidden state
    # does not carry gradient backwards, only the last timestep trains and the loss still falls.
    from selfagent.nn import GRU

    gru = GRU(3, 5, np.random.default_rng(0)).eval()
    assert_gradients_match(lambda x: weighted_sum(gru(x)), [rand(2, 4, 3, seed=1)])


def test_gru_first_timestep_reaches_the_last_state():
    # A break in the recurrence leaves the final state independent of the first input, which no
    # gradcheck on a summed output would notice.
    from selfagent.nn import GRU

    gru = GRU(3, 5, np.random.default_rng(0)).eval()
    window = rand(1, 4, 3, seed=2)
    weighted_sum(ops.getitem(gru(window), (slice(None), -1))).backward()
    assert np.abs(window.grad[0, 0]).max() > 1e-8, "the first bar must influence the last state"
