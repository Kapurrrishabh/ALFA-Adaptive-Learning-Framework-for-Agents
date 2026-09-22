"""Whole-word MLM masking.

Masking single pieces lets the model rebuild a word from its own fragments: given `fin ##anc`
predicting `##e` is spelling, not finance. So a chosen word has every one of its pieces masked
together, which is what makes the task about meaning.

The 80/10/10 split is BERT's. The 10% left unchanged is the part that matters and looks pointless:
without it the model only ever sees [MASK] at positions it must predict, and learns to trust
unmasked inputs blindly.
"""

from ..backend import xp
from ..tokenizer.vocab import CLS_ID, MASK_ID, PAD_ID, SEP_ID, UNLABELLED

MASK_PROBABILITY = 0.15
REPLACE_WITH_MASK = 0.8
REPLACE_WITH_RANDOM = 0.9

# Never masked: predicting a token the model is told to expect teaches nothing.
_NEVER_MASKED = (PAD_ID, CLS_ID, SEP_ID)


def word_starts(token_ids, continuation_ids):
    """True where a token begins a word, so its pieces can be grouped."""
    is_continuation = xp.isin(token_ids, xp.asarray(sorted(continuation_ids)))
    return ~is_continuation


def _word_index(token_ids, continuation_ids):
    """A group id per position: pieces of one word share a number, rising along the row."""
    return xp.cumsum(word_starts(token_ids, continuation_ids), axis=1)


def mask_tokens(token_ids, continuation_ids, vocab_size, rng):
    """Returns (inputs, labels). Labels are UNLABELLED except where a word was chosen.

    `token_ids` is (batch, length) and is not modified.
    """
    token_ids = xp.asarray(token_ids)
    groups = _word_index(token_ids, continuation_ids)

    # One draw per word rather than per position, then broadcast back over the word's pieces.
    chosen_word = rng.random(groups.shape) < MASK_PROBABILITY
    starts = word_starts(token_ids, continuation_ids)
    # A word's decision is the one taken at its first piece; take_along_axis spreads it rightwards.
    decision_at_start = xp.where(starts, chosen_word, False)
    chosen = _spread_from_starts(decision_at_start, groups)

    for token_id in _NEVER_MASKED:
        chosen &= token_ids != token_id
    if not chosen.any():
        # A short batch can legitimately choose nothing, and cross_entropy rejects an empty one.
        chosen = _force_one_position(token_ids, chosen)

    labels = xp.where(chosen, token_ids, UNLABELLED)

    draw = rng.random(token_ids.shape)
    inputs = token_ids.copy()
    inputs = xp.where(chosen & (draw < REPLACE_WITH_MASK), MASK_ID, inputs)
    random_tokens = rng.integers(len(_NEVER_MASKED) + 1, vocab_size, token_ids.shape)
    replace = chosen & (draw >= REPLACE_WITH_MASK) & (draw < REPLACE_WITH_RANDOM)
    inputs = xp.where(replace, random_tokens, inputs)
    return inputs, labels


def _spread_from_starts(decision_at_start, groups):
    """Carries each word's decision from its first piece across the rest of the word."""
    spread = xp.zeros_like(decision_at_start)
    for row in range(groups.shape[0]):
        # Group ids start at 1 because cumsum counts the first start as one.
        per_word = xp.zeros(int(groups[row].max()) + 2, dtype=bool)
        per_word[groups[row][decision_at_start[row]]] = True
        spread[row] = per_word[groups[row]]
    return spread


def _force_one_position(token_ids, chosen):
    """Masks a single maskable position, so the batch has at least one label."""
    maskable = xp.ones_like(chosen)
    for token_id in _NEVER_MASKED:
        maskable &= token_ids != token_id
    flat = maskable.reshape(-1)
    if not flat.any():
        raise ValueError("batch has no maskable positions; every token is padding or a marker")
    chosen = chosen.copy()
    chosen.reshape(-1)[int(xp.argmax(flat))] = True
    return chosen
