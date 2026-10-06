"""S2: the assembled model can actually learn, and its wiring is what we think it is."""

import json
import math

import numpy as np
import pytest

from backend.models.core import pretrained
from backend.models.data import indicators, prices, qa_pairs, returns
from backend.models.core.autograd import functional as F
from backend.models.core.autograd import no_grad
from backend.models.core.backend import default_rng
from backend.models.core.config import ModelConfig
from backend.models.data.features import to_features
from backend.models.learning.preference import agreement, preference_loss, sequence_logprob
from backend.models.networks import FinanceAgentModel, GroundedGenerator, MaskedLanguageModel, ReturnGenerator, TextTower
from backend.models.networks.garch import Garch
from backend.models.core.autograd.tensor import Tensor
from backend.models.core.nn import TransformerDecoder, padding_mask
from backend.models.core.optim import AdamW, clip_grad_norm
from backend.models.data.masking import mask_tokens
from backend.models.core.tokenizer.vocab import CLS_ID, CONTINUATION, PAD_ID, SEP_ID, SPECIAL_TOKENS, UNK_ID, UNLABELLED
from backend.models.core.tokenizer.wordpiece import WordPiece, pretokenize

TINY = ModelConfig(
    vocab_size=64,
    max_text_length=16,
    dim=32,
    num_heads=4,
    num_layers=2,
    ffn_dim=64,
    dropout=0.0,
    price_window=16,
    patch_length=4,
    patch_stride=4,
    price_layers=1,
    max_answer_length=8,
    decoder_layers=1,
    num_classes=3,
    seed=0,
)


def fake_batch(rng, batch=8):
    token_ids = rng.integers(0, TINY.vocab_size, (batch, TINY.max_text_length))
    bars = 100.0 + rng.standard_normal((batch, TINY.price_window + 1, 5)).cumsum(axis=1)
    bars[..., 4] = rng.integers(1_000, 10_000, (batch, TINY.price_window + 1))
    return token_ids, to_features(bars)


# --- wiring -----------------------------------------------------------------


def test_config_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="not divisible"):
        ModelConfig(dim=32, num_heads=5)


def test_config_rejects_stride_that_skips_timesteps():
    with pytest.raises(ValueError, match="skips timesteps"):
        ModelConfig(patch_length=4, patch_stride=8)


def test_every_parameter_is_registered_once():
    """A parameter reachable by two names would be updated twice per optimizer step."""
    model = FinanceAgentModel(TINY)
    identities = [id(parameter) for parameter in model.parameters()]
    assert len(identities) == len(set(identities))


def test_mlm_output_matrix_is_tied_to_the_input_embedding():
    model = MaskedLanguageModel(TINY)
    names = dict(model.named_parameters())
    assert "text.embedding.tokens.weight" in names
    assert not any(name.startswith("head.output_weight") for name in names)


def test_agent_output_shape():
    rng = np.random.default_rng(0)
    token_ids, window = fake_batch(rng)
    logits = FinanceAgentModel(TINY).eval()(token_ids, window)
    assert logits.shape == (8, TINY.num_classes)


def test_mlm_output_shape():
    rng = np.random.default_rng(0)
    token_ids, _ = fake_batch(rng)
    logits = MaskedLanguageModel(TINY).eval()(token_ids)
    assert logits.shape == (8, TINY.max_text_length, TINY.vocab_size)


def test_text_longer_than_config_raises():
    model = TextTower(TINY, np.random.default_rng(0))
    too_long = np.zeros((2, TINY.max_text_length + 1), dtype=int)
    with pytest.raises(ValueError, match="exceeds max_text_length"):
        model(too_long)


def test_padding_tokens_do_not_change_the_unpadded_output():
    """The whole point of the mask. Without it, padding leaks into every representation."""
    rng = np.random.default_rng(1)
    model = TextTower(TINY, np.random.default_rng(0)).eval()
    real_length = 6
    ids = rng.integers(1, TINY.vocab_size, (1, TINY.max_text_length))
    keep = np.zeros((1, TINY.max_text_length))
    keep[0, :real_length] = 1.0

    with no_grad():
        masked = model(ids * keep.astype(int), keep)
        changed = ids.copy()
        changed[0, real_length:] = rng.integers(1, TINY.vocab_size, TINY.max_text_length - real_length)
        also_masked = model(changed * keep.astype(int), keep)

    difference = np.abs(masked.data[0, :real_length] - also_masked.data[0, :real_length]).max()
    assert difference < 1e-5


def test_decoder_cannot_see_later_positions():
    """The failure that hides: a decoder attending to the future trains to a good loss curve and
    then generates nonsense, because at generation time the future it leaned on is not there."""
    rng = np.random.default_rng(0)
    decoder = TransformerDecoder(2, TINY.dim, TINY.num_heads, TINY.ffn_dim, 0.0, default_rng(0)).eval()
    memory = Tensor(rng.standard_normal((1, 5, TINY.dim)))
    answer = rng.standard_normal((1, 6, TINY.dim))
    rewritten = answer.copy()
    rewritten[0, 3:] = rng.standard_normal((3, TINY.dim))

    with no_grad():
        before = decoder(Tensor(answer), memory).data
        after = decoder(Tensor(rewritten), memory).data

    assert np.abs(before[0, :3] - after[0, :3]).max() < 1e-6, "an earlier position saw a later one"
    assert np.abs(before[0, 3:] - after[0, 3:]).max() > 1e-3, "rewritten positions must change"


def test_decoder_reads_the_retrieved_passages():
    """Grounding is the whole design: if the answer does not depend on the memory, the model is
    reciting its weights and every number it gives is invented."""
    rng = np.random.default_rng(0)
    decoder = TransformerDecoder(2, TINY.dim, TINY.num_heads, TINY.ffn_dim, 0.0, default_rng(0)).eval()
    answer = Tensor(rng.standard_normal((1, 6, TINY.dim)))

    with no_grad():
        one = decoder(answer, Tensor(rng.standard_normal((1, 5, TINY.dim)))).data
        other = decoder(answer, Tensor(rng.standard_normal((1, 5, TINY.dim)))).data

    assert np.abs(one - other).max() > 1e-3


def test_dropout_is_active_in_train_and_off_in_eval():
    config = ModelConfig(**{**TINY.__dict__, "dropout": 0.5})
    rng = np.random.default_rng(0)
    token_ids, window = fake_batch(rng)
    model = FinanceAgentModel(config)

    model.train()
    assert not np.allclose(model(token_ids, window).data, model(token_ids, window).data)
    model.eval()
    assert np.allclose(model(token_ids, window).data, model(token_ids, window).data)


# --- learning ---------------------------------------------------------------


def train_to_convergence(model, loss_fn, steps, lr=3e-3):
    optimizer = AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.0)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        clip_grad_norm(optimizer.parameters, 1.0)
        optimizer.step()
        history.append(loss.item())
    return history


def test_agent_overfits_a_small_batch():
    """S2 gate: if the model cannot memorise 32 examples, something upstream is broken."""
    rng = np.random.default_rng(0)
    token_ids, window = fake_batch(rng, batch=32)
    labels = rng.integers(0, TINY.num_classes, 32)
    model = FinanceAgentModel(TINY)

    history = train_to_convergence(
        model, lambda: F.cross_entropy(model(token_ids, window), labels), steps=200
    )

    assert history[0] > 0.9, "untrained loss should be near ln(3) = 1.10"
    assert history[-1] < 0.02, f"failed to memorise the batch, final loss {history[-1]:.4f}"


def test_mlm_overfits_a_small_batch():
    rng = np.random.default_rng(0)
    token_ids, _ = fake_batch(rng, batch=8)
    targets = token_ids.reshape(-1).copy()
    model = MaskedLanguageModel(TINY)

    def loss_fn():
        logits = model(token_ids)
        flat = logits.reshape((-1, TINY.vocab_size))
        return F.cross_entropy(flat, targets)

    history = train_to_convergence(model, loss_fn, steps=200)

    assert history[0] > np.log(TINY.vocab_size) * 0.8
    assert history[-1] < 0.1, f"failed to memorise the batch, final loss {history[-1]:.4f}"


def test_generator_shares_one_token_matrix_three_ways():
    """Encoder input, answer input and output layer are the same matrix. A second name for it
    would make the optimizer apply every update twice and quietly double its learning rate."""
    model = GroundedGenerator(TINY)
    identities = [id(parameter) for parameter in model.parameters()]
    assert len(identities) == len(set(identities))
    names = dict(model.named_parameters())
    assert "text.embedding.tokens.weight" in names
    assert not any(name.startswith("answer.tokens") for name in names)


def test_generator_overfits_a_small_batch():
    """S2 gate for generation. Teacher forcing, so the targets are the answer shifted one left."""
    rng = np.random.default_rng(0)
    source = rng.integers(1, TINY.vocab_size, (4, TINY.max_text_length))
    answer = np.concatenate(
        [np.full((4, 1), CLS_ID), rng.integers(1, TINY.vocab_size, (4, TINY.max_answer_length - 1))],
        axis=1,
    )
    targets = answer[:, 1:].reshape(-1)
    model = GroundedGenerator(TINY)

    def loss_fn():
        logits = model(source, answer[:, :-1])
        return F.cross_entropy(logits.reshape((-1, TINY.vocab_size)), targets)

    history = train_to_convergence(model, loss_fn, steps=300)

    assert history[0] > np.log(TINY.vocab_size) * 0.8
    assert history[-1] < 0.1, f"failed to memorise the batch, final loss {history[-1]:.4f}"


def _scores(*values):
    """Log-probabilities as the tensors `preference_loss` expects, one per pair."""
    return [Tensor(value, requires_grad=True) for value in values]


def test_preference_costs_ln2_exactly_when_the_model_has_not_left_the_reference():
    """The invariant the training run prints at step 0. If it holds, the reference is wired in; if the
    ranking term starts anywhere else, the model is being anchored to something it never was."""
    for taken, refused in ((-1.0, -2.0), (-4.5, -0.25)):
        loss = preference_loss(_scores(taken), _scores(refused), [taken], [refused], sft=0.0)
        assert loss.item() == pytest.approx(math.log(2), abs=1e-6)


def test_preference_charges_for_losing_ground_the_reference_already_held():
    """What stops the model raising the margin by crushing the rejected answer towards zero: the pair is
    scored against where it began, so being right by less than the reference was still costs."""
    taken, refused = _scores(-1.0), _scores(-2.0)
    behind = preference_loss(taken, refused, [-1.0], [-3.0], sft=0.0).item()
    ahead = preference_loss(taken, refused, [-2.0], [-1.0], sft=0.0).item()
    assert behind > math.log(2) > ahead


def test_preference_charges_for_winning_a_pair_by_making_both_answers_unwritable():
    """The failure that cost a whole training run. Both pairs have an identical margin -- the chosen answer
    beats the rejected one by exactly one nat, and by exactly as much as the reference did -- so the ranking
    term alone scores them the same, and a model is free to drag the whole distribution down. It did: one
    epoch of that objective ended up opening every answer with thirty copies of "you".
    """
    fluent = _scores(-1.0), _scores(-2.0)
    mute = _scores(-5.0), _scores(-6.0)
    assert preference_loss(*fluent, [-1.0], [-2.0], sft=0.0).item() == pytest.approx(
        preference_loss(*mute, [-5.0], [-6.0], sft=0.0).item(), abs=1e-6)
    assert (preference_loss(*mute, [-5.0], [-6.0]).item()
            > preference_loss(*fluent, [-1.0], [-2.0]).item() + 3.0)


def test_preference_scores_an_answer_by_its_mean_token_probability():
    """A summed log-probability grows with length, and the preferred answer is the longer one in 57.6%
    of these pairs, so a sum would be won by writing more rather than by writing better."""
    model = GroundedGenerator(TINY)
    model.eval()
    source = np.zeros((2, TINY.max_text_length), dtype=int)
    answer = np.full((2, TINY.max_answer_length), PAD_ID)
    answer[0, :6] = [CLS_ID, 5, 6, 7, 8, SEP_ID]
    answer[1, :4] = [CLS_ID, 5, 6, SEP_ID]

    with no_grad():
        scored = [one.item() for one in sequence_logprob(model, source, answer)]
        logits = model(source, answer[:, :-1]).data

    for row, length in ((0, 5), (1, 3)):
        step = logits[row, :length]
        step = step - step.max(axis=-1, keepdims=True)
        log_probabilities = step - np.log(np.exp(step).sum(axis=-1, keepdims=True))
        wanted = log_probabilities[np.arange(length), answer[row, 1 : length + 1]].mean()
        assert scored[row] == pytest.approx(wanted, abs=1e-5)
        assert scored[row] != pytest.approx(wanted * length, abs=1e-3)


def test_preference_refuses_a_pair_with_nothing_to_score():
    model = GroundedGenerator(TINY)
    answer = np.full((2, TINY.max_answer_length), PAD_ID)
    answer[0, :3] = [CLS_ID, 5, SEP_ID]
    with pytest.raises(ValueError, match=r"rows \[1\] have no answer token"):
        sequence_logprob(model, np.zeros((2, TINY.max_text_length), dtype=int), answer)


def test_agreement_needs_the_preferred_answer_to_win_outright():
    # A tie is not agreement. Counting it as one would report 100% for a model that scores every answer
    # identically, which is what a collapsed decoder does.
    assert agreement([-1.0, -3.0, -2.0], [-2.0, -1.0, -2.0]) == pytest.approx(1 / 3)


def test_generation_stops_at_the_end_marker_and_strips_it():
    # An untrained model may never emit the marker, so it is made overwhelming. What is under test
    # is that generate() stops there and returns no markers, not what the model would choose.
    model = GroundedGenerator(TINY)
    model.head.output_bias.data[SEP_ID] = 1e3
    assert model.generate(np.zeros((2, TINY.max_text_length), dtype=int)) == [[], []]


def test_generation_caps_length_and_leaves_training_mode_alone():
    model = GroundedGenerator(TINY)
    model.head.output_bias.data[SEP_ID] = -1e9
    ids = model.generate(np.zeros((1, TINY.max_text_length), dtype=int))[0]
    assert len(ids) == TINY.max_answer_length - 1, "generation must stop at max_answer_length"
    assert model.training, "generate() switches to eval and has to switch back"


def test_nucleus_sampling_never_draws_from_outside_the_kept_mass():
    # The failure this pins is top_p being accepted and then ignored: sampling still reaches the
    # tail, which is where the invented non-words come from, and nothing in the output looks wrong.
    model = GroundedGenerator(TINY)
    probabilities = np.array([[0.5, 0.25, 0.15, 0.07, 0.03]])
    kept = model._nucleus(probabilities, 0.8)
    assert kept[0, 3:].tolist() == [0.0, 0.0], "tail beyond top_p must be dropped"
    assert kept.sum() == pytest.approx(1.0), "what survives has to be renormalised"


def test_nucleus_sampling_keeps_the_top_token_when_it_alone_exceeds_top_p():
    # Without subtracting each token's own mass the cumulative sum zeroes every token including the
    # best one, leaving nothing to sample and turning a confident step into a divide by zero.
    model = GroundedGenerator(TINY)
    kept = model._nucleus(np.array([[0.95, 0.03, 0.02]]), 0.9)
    assert kept[0, 0] == pytest.approx(1.0)


def test_generator_starts_from_the_pretrained_encoder(tmp_path):
    path = tmp_path / "pretrained.npz"
    pretrained.save(path, MaskedLanguageModel(TINY), TINY)
    _, weights = pretrained.load(path)
    generator = GroundedGenerator(TINY)
    generator.load_pretrained_encoder(weights)
    np.testing.assert_array_equal(
        generator.text.embedding.tokens.weight.data, weights["text.embedding.tokens.weight"]
    )


def test_generator_refuses_a_checkpoint_holding_no_encoder():
    with pytest.raises(KeyError, match="no encoder to reuse"):
        GroundedGenerator(TINY).load_pretrained_encoder({"head.output_bias": np.zeros(4)})


def test_freezing_the_encoder_leaves_the_shared_token_matrix_trainable():
    """The trap this pins: the token matrix sits under model.text but is also the decoder's output
    layer, so freezing everything named for the encoder would stop the model learning to write."""
    model = GroundedGenerator(TINY).freeze_encoder()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert "text.embedding.tokens.weight" in trainable
    assert not any(name.startswith("text.encoder.") for name in trainable)


def test_a_frozen_encoder_stays_put_while_the_rest_of_the_model_trains():
    """requires_grad is only a promise until an optimizer reads it. Backward still fills .grad on the
    frozen layers, so what holds them still is their absence from trainable_parameters()."""
    rng = np.random.default_rng(0)
    source = rng.integers(1, TINY.vocab_size, (4, TINY.max_text_length))
    answer = np.concatenate(
        [np.full((4, 1), CLS_ID), rng.integers(1, TINY.vocab_size, (4, TINY.max_answer_length - 1))],
        axis=1,
    )
    model = GroundedGenerator(TINY).freeze_encoder()
    before = {name: parameter.data.copy() for name, parameter in model.named_parameters()}

    train_to_convergence(
        model,
        lambda: F.cross_entropy(
            model(source, answer[:, :-1]).reshape((-1, TINY.vocab_size)), answer[:, 1:].reshape(-1)
        ),
        steps=3,
    )

    for name, parameter in model.named_parameters():
        moved = not np.array_equal(parameter.data, before[name])
        assert moved != name.startswith("text.encoder."), name


def test_no_grad_leaves_parameters_untouched():
    rng = np.random.default_rng(0)
    token_ids, window = fake_batch(rng)
    model = FinanceAgentModel(TINY).eval()
    with no_grad():
        model(token_ids, window)
    assert all(parameter.grad is None for parameter in model.parameters())


# --- features ---------------------------------------------------------------


def test_to_features_drops_one_row_and_standardises():
    rng = np.random.default_rng(0)
    bars = np.abs(rng.standard_normal((20, 5))) + 100.0
    features = to_features(bars)
    assert features.shape == (19, 5)
    assert np.allclose(features.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(features.std(axis=0), 1.0, atol=1e-6)


def test_to_features_rejects_non_positive_prices():
    bars = np.ones((5, 5))
    bars[2, 0] = 0.0
    with pytest.raises(ValueError, match="non-positive price"):
        to_features(bars)


def test_to_features_is_scale_invariant():
    """A $5 stock and a $500 stock with the same shape must produce the same features."""
    rng = np.random.default_rng(0)
    bars = np.abs(rng.standard_normal((20, 5))) + 100.0
    scaled = bars.copy()
    scaled[:, :4] *= 100.0
    assert np.allclose(to_features(bars), to_features(scaled))


def test_window_cannot_see_past_its_own_end():
    """The no-look-ahead guarantee: a window must be unchanged by anything after the bar it ends on.

    This is the failure that makes a model look excellent and be worthless, and it survives every
    other test, because a leaked future raises nothing and only improves the score.
    """
    rng = np.random.default_rng(0)
    bars = np.abs(rng.standard_normal((40, 5))) + 100.0
    tampered = bars.copy()
    tampered[25:] *= 3.0
    assert np.allclose(prices.window_at(bars, 24, 16), prices.window_at(tampered, 24, 16))


def test_label_reads_only_forward_of_the_window():
    rng = np.random.default_rng(1)
    bars = np.abs(rng.standard_normal((40, 5))) + 100.0
    tampered = bars.copy()
    tampered[:20] *= 5.0
    assert prices.label_at(bars, 20, 5) == pytest.approx(prices.label_at(tampered, 20, 5))
    expected = math.log(bars[25, 3]) - math.log(bars[20, 3])
    assert prices.label_at(bars, 20, 5) == pytest.approx(expected)


def test_the_window_keeps_the_volatility_that_standardising_removes():
    """The bug this pins cost a training run: per-window z-scoring divides the window by its own
    return spread, so a calm window and a wild one were byte-identical to the model that had to
    predict which was which."""
    rng = np.random.default_rng(4)
    steps = rng.standard_normal(30) * 0.01
    calm = 100.0 * np.exp(np.concatenate([[0.0], steps]).cumsum())
    wild = 100.0 * np.exp(np.concatenate([[0.0], steps * 4.0]).cumsum())
    bars = lambda closes: np.stack(  # noqa: E731
        [closes, closes * 1.01, closes * 0.99, closes, np.full(len(closes), 1e6)], axis=1
    )
    quiet = prices.window_at(bars(calm), 30, 16)
    loud = prices.window_at(bars(wild), 30, 16)
    assert quiet[:, -1].std() == 0.0, "the scale is one number for the whole window"
    assert loud[0, -1] - quiet[0, -1] == pytest.approx(math.log(4.0), abs=0.02)


def test_forward_volatility_reads_only_bars_after_the_window():
    rng = np.random.default_rng(3)
    bars = np.abs(rng.standard_normal((40, 5))) + 100.0
    tampered = bars.copy()
    tampered[:20] *= 5.0
    assert prices.forward_volatility(bars, 20, 5) == pytest.approx(
        prices.forward_volatility(tampered, 20, 5)
    )
    steps = np.diff(np.log(bars[20:26, 3]))
    assert prices.forward_volatility(bars, 20, 5) == pytest.approx(steps.std(ddof=1))


def test_sample_ends_leave_room_for_the_window_and_the_horizon():
    ends = prices.sample_ends(100, window=16, horizon=5)
    assert ends[0] == 16, "a window needs one bar more than its length, for differencing"
    assert ends[-1] + 5 <= 99, "every label must have its full horizon inside the data"


def test_direction_classes_are_balanced_by_construction():
    # Fixed thresholds would make the class mix an accident of the era's volatility, and the
    # majority-class baseline would then flatter the model instead of challenging it.
    returns = np.random.default_rng(2).standard_normal(3000) * 0.02
    labels = prices.to_tertile(returns, prices.tertile_edges(returns))
    counts = np.bincount(labels, minlength=3)
    assert counts.min() / counts.sum() > 0.30, f"classes came out lopsided: {counts.tolist()}"


def test_padding_mask_rejects_wrong_rank():
    with pytest.raises(ValueError, match="token mask"):
        padding_mask(np.ones(8))


def test_wordpiece_keeps_finance_shapes_whole():
    # A number ending on a separator once glued the sentence comma onto "2023,", giving every
    # year two spellings and wasting vocabulary on both.
    assert pretokenize("Revenue rose 12.5% to $1,234 in 2023, up from 2022.") == [
        "revenue", "rose", "12.5%", "to", "$1234", "in", "2023", ",", "up", "from", "2022", ".",
    ]


def test_decode_puts_back_the_spacing_pretokenize_dropped():
    """Served answers read this, and a space before every piece gave users "reliance . ns , - 5.6%".
    The range in the second assert is why a sign needs its left side checked too: binding it to any
    following digit invented a "-2000", 4 of the 15 figures that clause costs on the human Q&A set."""
    line = "it fell -5.6% to 1242.30 . a stop-loss doesn't help from 1993 - 2000 ."
    # Repeated past the word and character frequency floors, or every piece trains away to [UNK].
    tokenizer = WordPiece.train([line] * 12, vocab_size=300, log=lambda *_: None)
    assert tokenizer.decode(tokenizer.encode("it fell -5.6% to 1242.30. a stop-loss doesn't help.")) == (
        "it fell -5.6% to 1242.30. a stop - loss doesn't help.")
    assert tokenizer.decode(tokenizer.encode("from 1993 - 2000")) == "from 1993 - 2000"


def test_normalisation_gives_each_finance_spelling_one_form():
    assert pretokenize("annual report on Form 10-K") == ["annual", "report", "on", "form", "form10k"]
    assert pretokenize("a loss of Rs. 1,250 crore") == ["a", "loss", "of", "rs", "1250", "crore"]
    assert pretokenize("INR 5,000 and US$30") == ["rs", "5000", "and", "$30"]


def test_normalisation_collapses_indian_number_grouping():
    # India writes lakh and crore as 12,50,000. Anchoring the rule on \b matched "50,000" inside
    # that and left the stray comma, turning 664 numbers in the corpus into a third spelling.
    assert pretokenize("Rs. 12,50,000") == ["rs", "1250000"]
    assert pretokenize("in 2009,2010 the rate") == ["in", "2009,2010", "the", "rate"]


def test_normalisation_leaves_a_sum_of_money_alone():
    # Every bare "10k" and "8k" in the corpus sample was money, never a form name, so the rule
    # requires the hyphen. Without that, "$10k in credit card debt" became an SEC filing.
    assert pretokenize("$10k of equity") == ["$10", "k", "of", "equity"]


def test_vocabulary_learned_under_other_rules_is_refused(tmp_path):
    # The vocabulary and the normaliser have to agree or the same text encodes differently, and
    # nothing downstream would notice.
    path = tmp_path / "tokenizer.json"
    WordPiece(list(SPECIAL_TOKENS) + ["a", "b"]).save(path)
    stored = json.loads(path.read_text())
    stored["normalization"] = "0" * 12
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="re-run scripts/prepare_pretrain.py"):
        WordPiece.load(path)
    # And the script that error names has to be able to act on it. Reusing the file whenever it
    # merely exists made that error unfixable: re-running hit load() again and died the same way.
    assert not WordPiece.matches_normalization(path)


def test_vocabulary_learned_under_the_current_rules_is_reused(tmp_path):
    path = tmp_path / "tokenizer.json"
    WordPiece(list(SPECIAL_TOKENS) + ["a", "b"]).save(path)
    assert WordPiece.matches_normalization(path)


def test_wordpiece_keeps_a_character_whose_every_word_is_rare():
    # Each distinct percentage occurs once, so a word-frequency cut alone removed every word
    # holding a '%' and the character vanished, turning every percentage into [UNK].
    corpus = [f"the rate is {value / 100:.2f}% today" for value in range(1, 400)]
    tokenizer = WordPiece.train(corpus, 300, log=lambda message: None)
    assert CONTINUATION + "%" in tokenizer.ids
    assert UNK_ID not in tokenizer.encode("3.25%")


def test_masking_masks_whole_words_and_spares_the_markers():
    tokenizer = WordPiece.train(
        [f"inflation targeting framework number {value}" for value in range(200)],
        200,
        log=lambda message: None,
    )
    continuations = tokenizer.continuation_ids
    body = np.asarray(tokenizer.encode("inflation targeting framework")[:14])
    batch = np.vstack([np.concatenate([[CLS_ID], body, [SEP_ID]])] * 8)

    inputs, labels = mask_tokens(batch, continuations, tokenizer.vocab_size, default_rng(0))
    chosen = labels != UNLABELLED

    assert not chosen[:, 0].any() and not chosen[:, -1].any()
    assert (inputs[~chosen] == batch[~chosen]).all()
    for row, column in zip(*np.nonzero(chosen)):
        following = column + 1
        if following < batch.shape[1] and int(batch[row, following]) in continuations:
            assert chosen[row, following], "a word was masked but its next piece was not"


def test_masking_labels_only_where_it_masked():
    tokenizer = WordPiece.train(
        [f"the reserve bank of india {value}" for value in range(200)], 200, log=lambda m: None
    )
    batch = np.vstack([np.asarray(tokenizer.encode("the reserve bank of india")[:16])] * 4)
    inputs, labels = mask_tokens(batch, tokenizer.continuation_ids, tokenizer.vocab_size, default_rng(1))
    kept = labels != UNLABELLED
    assert kept.any(), "cross_entropy rejects a batch with no labels at all"
    assert (labels[kept] == batch[kept]).all(), "labels must be the original tokens"


def test_saved_model_predicts_the_same_after_reloading(tmp_path):
    token_ids = np.arange(TINY.max_text_length)[None, :] % TINY.vocab_size
    model = MaskedLanguageModel(TINY)
    model.eval()
    with no_grad():
        before = model(token_ids).data

    path = tmp_path / "pretrained.npz"
    pretrained.save(path, model, TINY)
    config, weights = pretrained.load(path)
    reloaded = MaskedLanguageModel(config)
    reloaded.load_state_dict(weights)
    reloaded.eval()
    with no_grad():
        after = reloaded(token_ids).data

    assert config == TINY, "the config has to travel with the weights"
    np.testing.assert_array_equal(before, after)


def test_saved_model_rejects_a_tampered_config(tmp_path):
    # The fingerprint is the guard against loading weights into the wrong shape, which would
    # otherwise surface as a confusing failure deep inside a matmul.
    path = tmp_path / "pretrained.npz"
    pretrained.save(path, MaskedLanguageModel(TINY), TINY)
    stored = dict(np.load(path))
    stored["fingerprint"] = np.asarray("0" * 12)
    np.savez(path, **stored)
    with pytest.raises(ValueError, match="incompatible version"):
        pretrained.load(path)


# --- question-and-answer pairs ----------------------------------------------

def _words(text):
    """Stands in for the tokenizer: one token per word, so budgets are readable in the tests."""
    return text.split()


def test_truncation_stops_at_a_sentence_boundary():
    # A target cut mid-clause teaches the model to stop mid-clause, because the end marker never
    # follows a finished thought. 89% of real answers exceed the budget, so this path is the norm.
    text = "Rates fell sharply. The bond rallied hard. Volume was thin."
    assert qa_pairs.truncate_to_sentences(text, _words, 8) == "Rates fell sharply."
    assert qa_pairs.truncate_to_sentences(text, _words, 9) == "Rates fell sharply. The bond rallied hard."


def test_truncation_drops_a_thread_whose_first_sentence_will_not_fit():
    # Returning a fragment here would be worse than returning nothing: it is a training target that
    # is not a sentence. 0.5% of answers land here at the real budget.
    assert qa_pairs.truncate_to_sentences("One extremely long opening sentence indeed.", _words, 4) == ""


def test_accepted_answer_beats_a_more_popular_one():
    # The asker had the problem and said this solved it. Vote counts come from people who did not.
    question = {"accepted_answer_id": "7"}
    candidates = [{"id": "7", "score": 2}, {"id": "9", "score": 99}]
    assert qa_pairs.best_answer(question, candidates)["id"] == "7"


def test_thread_with_no_accepted_and_no_upvoted_answer_is_dropped():
    # 51% of questions have no accepted answer, so the fallback runs constantly. An answer nobody
    # endorsed is a worse target than no target at all.
    question = {"accepted_answer_id": None}
    assert qa_pairs.best_answer(question, [{"id": "1", "score": 0}, {"id": "2", "score": None}]) is None
    assert qa_pairs.best_answer(question, [{"id": "1", "score": 1}])["id"] == "1"


def test_supervised_pair_carries_its_thread_key():
    # Without the key the retriever cannot exclude this thread, so it hands the model the very answer
    # it is being trained to produce. The model learns to copy one passage and faithfulness lies.
    questions = [{"site": "money", "id": "5", "title": "Why?", "text": "Body.", "accepted_answer_id": "8"}]
    answers = {("money", "5"): [{"id": "8", "score": 3, "text": "Because rates moved."}]}
    (key, asked, answer), = qa_pairs.supervised_pairs(questions, answers, _words, 64)
    assert key == ("money", "5")
    assert asked.startswith("Why?")
    assert answer == "Because rates moved."


def test_preference_pair_is_only_made_when_the_votes_actually_disagree():
    # Equal scores are not a preference. Treating them as one feeds a reward model pure noise.
    questions = [{"site": "quant", "id": "1", "title": "T", "text": "B", "accepted_answer_id": None}]
    tied = {("quant", "1"): [{"id": "1", "score": 4, "text": "a"}, {"id": "2", "score": 4, "text": "b"}]}
    assert list(qa_pairs.preference_pairs(questions, tied)) == []

    split = {("quant", "1"): [{"id": "1", "score": 9, "text": "good"}, {"id": "2", "score": 1, "text": "bad"}]}
    (_, _, chosen, rejected), = qa_pairs.preference_pairs(questions, split)
    assert (chosen, rejected) == ("good", "bad")


# --- retrieval and grounding ------------------------------------------------

def _pool():
    return [
        "A demat account holds shares in electronic form with a depository.",
        "The repo rate is the rate at which the central bank lends to banks.",
        "Dividend income is taxed as ordinary income in most brackets.",
    ]


def test_retrieval_ranks_the_passage_that_shares_rare_terms():
    from backend.models.networks.retriever import BM25
    index = BM25(_pool(), pretokenize)
    assert index.search("what is a demat account", 2)[0] == 0
    assert index.search("how does the repo rate work", 2)[0] == 1


def test_retrieval_returns_nothing_when_no_term_matches():
    # Padding the list out with unmatched passages would ground an answer in irrelevant text. The
    # honest response to a question we hold no evidence for is to abstain, so this must stay empty.
    from backend.models.networks.retriever import BM25
    assert BM25(_pool(), pretokenize).search("zzzz qqqq", 3) == []


def test_retrieval_can_exclude_the_thread_it_is_building_an_example_for():
    # The collected threads are 145 MB of the pretraining corpus. Without this the index returns the
    # exact answer the model is being trained to write, it learns to copy, and faithfulness lies.
    from backend.models.networks.retriever import BM25
    index = BM25(_pool(), pretokenize, owners=["t1", "t2", "t3"])
    assert index.search("demat account shares", 3, exclude_owner="t1") != [0]
    assert 0 not in index.search("demat account shares", 3, exclude_owner="t1")


def test_fusion_in_decoder_encodes_each_passage_separately():
    """k passages become k x length memory positions. Concatenating the text into one sequence
    instead would exceed the window the encoder's position embeddings were trained at."""
    rng = np.random.default_rng(0)
    model = GroundedGenerator(TINY).eval()
    source = rng.integers(1, TINY.vocab_size, (2, 3, TINY.max_text_length))
    keep = np.ones((2, 3, TINY.max_text_length))
    with no_grad():
        memory, mask = model.encode(source, keep)
    assert memory.shape == (2, 3 * TINY.max_text_length, TINY.dim)
    assert mask.shape == (2, 1, 1, 3 * TINY.max_text_length)


def test_a_masked_out_passage_slot_does_not_change_the_answer():
    # Rows that retrieved fewer than k passages keep empty slots. If those leak into the memory the
    # model is grounded partly in padding, and it differs between rows for no reason.
    rng = np.random.default_rng(1)
    model = GroundedGenerator(TINY).eval()
    answer = np.zeros((1, 4), dtype=int)
    real = rng.integers(1, TINY.vocab_size, (1, 1, TINY.max_text_length))
    keep = np.concatenate([np.ones((1, 1, TINY.max_text_length)), np.zeros((1, 1, TINY.max_text_length))], axis=1)

    with no_grad():
        one = model(np.concatenate([real, np.zeros_like(real)], axis=1), answer, keep).data
        other = model(np.concatenate([real, rng.integers(1, TINY.vocab_size, real.shape)], axis=1), answer, keep).data
    assert np.abs(one - other).max() < 1e-6, "a fully masked passage slot still reached the decoder"


def test_generator_rejects_source_ids_of_the_wrong_rank():
    model = GroundedGenerator(TINY).eval()
    with pytest.raises(ValueError, match="expected .* source ids"):
        model.encode(np.zeros((2, 2, 2, TINY.max_text_length), dtype=int))


def test_encoder_fingerprint_ignores_fields_no_encoder_weight_depends_on():
    # This blocked a run once. max_answer_length sizes the decoder's own position table, which the
    # generator builds fresh, so demanding it match would force a 7h retrain that changes nothing.
    assert TINY.encoder_fingerprint == ModelConfig(
        **{**TINY.__dict__, "max_answer_length": TINY.max_answer_length * 2, "decoder_layers": 4}
    ).encoder_fingerprint


def test_encoder_fingerprint_still_catches_a_real_shape_change():
    for field, value in (("dim", 64), ("num_layers", 3), ("vocab_size", 128), ("max_text_length", 32)):
        changed = ModelConfig(**{**TINY.__dict__, field: value})
        assert changed.encoder_fingerprint != TINY.encoder_fingerprint, f"{field} must be caught"


# --- indicators -------------------------------------------------------------


def test_indicators_ignore_bars_after_the_one_they_are_asked_about():
    """Each indicator reads history only, so appending future bars must not change today's value."""
    closes = 100.0 + np.arange(80, dtype=float) + np.sin(np.arange(80)) * 3.0
    highs, lows = closes + 1.0, closes - 1.0
    extended = np.concatenate([closes, closes[-1] * np.ones(10) * 5.0])
    assert indicators.relative_strength_index(closes) == pytest.approx(
        indicators.relative_strength_index(extended[:80])
    )
    assert indicators.realized_volatility(closes) == pytest.approx(
        indicators.realized_volatility(extended[:80])
    )
    assert indicators.average_true_range(highs, lows, closes) > 0.0


def test_rsi_is_100_when_every_bar_rises_and_50_when_flat():
    # The all-gains case divides by a zero average loss; returning nan there would poison every
    # downstream figure with a value that still formats as text.
    assert indicators.relative_strength_index(np.arange(1.0, 40.0)) == 100.0
    assert indicators.relative_strength_index(np.full(40, 7.0)) == 50.0


def test_rsi_matches_a_hand_computed_wilder_value():
    closes = np.array([44.0, 44.5, 44.2, 45.0, 45.5, 45.2, 46.0, 46.5, 46.2, 47.0, 47.5, 47.2, 48.0, 48.5, 48.2])
    change = np.diff(closes)
    expected_gain = np.maximum(change, 0.0)[:14].mean()
    expected_loss = np.maximum(-change, 0.0)[:14].mean()
    expected = 100.0 - 100.0 / (1.0 + expected_gain / expected_loss)
    assert indicators.relative_strength_index(closes, 14) == pytest.approx(expected)


def test_drawdown_is_negative_after_a_fall_and_zero_at_a_peak():
    rising = np.arange(1.0, 70.0)
    assert indicators.max_drawdown(rising, 60) == pytest.approx(0.0)
    fell = np.concatenate([np.arange(1.0, 40.0), np.arange(39.0, 19.0, -1.0)])
    assert indicators.max_drawdown(fell, 50) < -0.4


def test_indicators_refuse_short_history_instead_of_guessing():
    with pytest.raises(ValueError, match="needs 15 bars"):
        indicators.relative_strength_index(np.arange(1.0, 10.0), 14)


def _return_config(**overrides):
    shape = dict(vocab_size=returns.BINS, price_window=16, dim=16, num_heads=2, num_layers=1, ffn_dim=32,
                 dropout=0.0)
    return ModelConfig(**{**shape, **overrides})


def test_a_return_window_is_scaled_by_bars_before_it_and_never_by_its_own():
    """The leak that would pass every loss check. A scale taken over the window divides each input by a
    number that already contains the returns the model is asked to predict."""
    series = np.random.default_rng(0).normal(0, 0.01, 400)
    before = returns.scale_before(series, 300)
    shocked = series.copy()
    shocked[300:] *= 50
    assert returns.scale_before(shocked, 300) == before
    with pytest.raises(ValueError, match="needs 250"):
        returns.scale_before(series, 100)


def test_a_bin_prediction_is_a_density_that_integrates_to_one():
    """What makes the log score comparable with a Gaussian's. If the bin widths were wrong the model would
    be scored on a density that does not integrate to one, and could beat GARCH by that error alone."""
    probabilities = np.random.default_rng(1).dirichlet(np.ones(returns.BINS))
    scale = 0.013
    bounds = returns._BOUNDS * scale
    centres = (bounds[1:] + bounds[:-1]) / 2
    widths = np.diff(bounds)
    assert (returns.density(np.tile(probabilities, (returns.BINS, 1)), centres, scale) * widths).sum() \
        == pytest.approx(1.0)


def test_a_decoded_return_lands_in_the_bin_it_was_drawn_from():
    """A sampler emitting values outside their bin would write paths whose density the model never
    stated, so every synthetic statistic would describe a different model from the one scored."""
    rng = np.random.default_rng(2)
    real = rng.standard_t(3, 5000) * 0.013
    ids = returns.encode(real, 0.013)
    assert (returns.encode(returns.decode(ids, 0.013, rng), 0.013) == ids).all()


def test_a_crash_past_the_last_bin_is_clipped_and_counted():
    assert returns.encode([-0.9], 0.01)[0] == 0
    assert returns.clipped([-0.9, 0.001], 0.01) == 1


def test_the_return_model_cannot_see_the_return_it_predicts():
    """Causality, pinned directly. A missing mask trains to a near-zero loss by reading the next token and
    generates noise, and the training curve looks like the best run this project ever had."""
    model = ReturnGenerator(_return_config())
    model.eval()
    ids = np.random.default_rng(3).integers(0, returns.BINS, (1, 16))
    changed = ids.copy()
    changed[0, 10] = (changed[0, 10] + 100) % returns.BINS
    with no_grad():
        first, second = model(ids).data, model(changed).data
    np.testing.assert_allclose(first[0, :10], second[0, :10], atol=1e-6)
    assert not np.allclose(first[0, 10], second[0, 10])


def test_a_return_path_can_outrun_the_context_it_was_trained_on():
    model = ReturnGenerator(_return_config())
    paths = model.sample_paths(np.zeros((3, 16), dtype=int), steps=40, rng=np.random.default_rng(4))
    assert paths.shape == (3, 40) and paths.min() >= 0 and paths.max() < returns.BINS


def test_a_return_model_built_for_another_bin_count_is_refused():
    with pytest.raises(ValueError, match="bins"):
        ReturnGenerator(_return_config(vocab_size=returns.BINS + 1))


def test_garch_variance_reads_only_the_returns_before_each_bar():
    model = Garch(0.1, 0.85, 5.0)
    series = np.random.default_rng(5).normal(0, 0.01, 50)
    shocked = series.copy()
    shocked[30] = 0.5
    np.testing.assert_allclose(model.variance(series, 1e-4)[:31], model.variance(shocked, 1e-4)[:31])
    assert model.variance(shocked, 1e-4)[31] > model.variance(series, 1e-4)[31]


def test_garch_fit_recovers_the_process_that_generated_the_data():
    """The baseline has to be a real GARCH fit. A broken fit would lose to anything, and the return model
    would be promoted on beating a strawman."""
    truth = Garch(0.08, 0.90, 5.0)
    simulated = truth.simulate(4000, 1e-4, np.random.default_rng(6), paths=4)
    fitted = Garch.fit(list(simulated))
    assert abs(fitted.alpha - 0.08) <= 0.03 and abs(fitted.beta - 0.90) <= 0.04
    assert 3.5 <= fitted.nu <= 8.0


def test_a_realised_return_reads_its_place_in_the_forecast_distribution():
    """The PIT a calibration check rests on. Read off the wrong side of the bin, every forecast looks biased
    by half a bin, and the online temperature would learn to correct an error the model never made."""
    probabilities = np.full((1, returns.BINS), 1.0 / returns.BINS)
    assert returns.cdf(probabilities, [0.0], 0.01)[0] == pytest.approx(0.5)
    assert returns.cdf(probabilities, [1.0], 0.01)[0] == pytest.approx(1.0)
    assert returns.cdf(probabilities, [-1.0], 0.01)[0] == pytest.approx(0.0)


def test_path_gru_reads_a_window_and_its_targets_turn_back_into_prices():
    from backend.models.core.backend import default_rng, xp
    from backend.models.data import returns
    from backend.models.networks import path_gru as G
    closes = 100 * xp.exp(xp.cumsum(default_rng(0).normal(0, 0.01, 400)))
    r = returns.log_returns(closes)
    ranges = xp.full(len(r), 0.01)
    end = len(r) - G.HORIZON - 1
    window, scale = G.window_at(r, r, ranges, end)
    assert window.shape == (G.WINDOW, G.FEATURES)
    target = G.targets_at(r, end, scale)
    # the realised daily returns, compounded, give back the realised closes
    assert xp.allclose(G.median_prices(closes[end + 1], target, scale), closes[end + 2: end + 2 + G.HORIZON])
    mean, log_sd = G.PathGRU(8)(window[None].astype("float32"))
    assert mean.data.shape == log_sd.data.shape == (1, G.HORIZON)
    paths = G.sample_paths(100.0, xp.zeros(G.HORIZON), xp.full(G.HORIZON, -50.0), 1.0, 3)
    assert paths.shape == (3, G.HORIZON) and xp.allclose(paths, 100.0)      # no spread: every path is the median
