"""Teaching the generator which answer is better, without a reward model and without sampling from it.

This is the generative half of the learning claim. Everything else in `learn/` shapes *whether* the agent
speaks -- the abstention cut, the calibrator, the teacher's verdicts -- and none of it changes a single
weight of the model that writes. This does: the decoder's own probabilities are what move.

**Why preferences and not more supervised answers.** The generator has already been trained to reproduce
endorsed answers and the result is measured: it is fluent and unfaithful, 57.1% of its figures unsupported,
because 30.7% of the answer words in that dataset are absent from the retrieved passages, so ignoring the
evidence was the correct thing to learn from it. Showing it more of the same target cannot fix that. What
fixes it is telling it that *this* answer is better than *that* one for the same question and the same
passages -- a signal supervised training cannot express, because it has only ever seen one right answer.

**Where the comparisons come from, and why they are nearly free.** `data/qa_pairs.preference_pairs` yields
24,639 (question, chosen, rejected) triples wherever a site's voters ranked one answer clearly above
another, widest gap per thread; 24,379 survive the answer budget. They are real human judgements about real
finance answers and nothing in this repository has ever trained on them.

**Why not rank the model's own samples**, which is the other way to do this. Measured in `learn/rank.py`,
eight samples from this generator contain 1.77 distinct answers. There is almost nothing there to prefer
between, so a loop built on its own output would be comparing a string to itself.

**The reference term is necessary and it is not sufficient.** Without it the cheapest way to raise
`logp(chosen) - logp(rejected)` is to drive the rejected answer's probability towards zero, which a language
model can do by collapsing onto a few safe tokens. Anchoring both to the frozen starting checkpoint means a
pair can only be won by moving the two apart *relative to where they began*.

That anchor constrains the two answers being scored. It says nothing about what the model writes when
nothing is being scored, and the first full run proved the difference matters: one epoch on 22,379 pairs took
held-out loss from 0.6931 to 0.6617 and held-out agreement from 58.9% to 60.9%, while free generation
collapsed -- 92.9% of figures unsupported against 59.4% frozen, and 30.8% repeated 4-grams against 1.7%,
with every answer opening on roughly thirty copies of the word "you". The cause is measurable: across the
training pairs `you` is the 4th most over-represented token in the preferred answer of 5,226 frequent tokens
(1.929% against 1.867%), and `your` the 10th. Lifting one frequent unigram raises the mean log-probability
of the chosen side slightly more than the rejected side on nearly every pair, which wins the ranking term
without reading either answer.

**So the ranking term is paired with the preferred answer's own likelihood**, which is the one thing it will
otherwise trade away. `SFT_WEIGHT` sets the price, and at 0.0 the loss is the plain DPO objective that
produced the collapse above.

Length-normalised on purpose, and the reason is measured rather than assumed: across the 24,379 built pairs
the preferred answer runs 143.5 tokens against 128.8 and is the longer of the two in 57.6% of them. A sum of
token log-probabilities is larger in magnitude for a longer answer, so a summed objective would hand the
model "prefer more tokens" as a way to win most pairs without reading either answer.
"""

from backend.models.core.autograd import functional as F
from backend.models.core.autograd import ops
from backend.models.core.backend import xp
from backend.models.core.tokenizer.vocab import PAD_ID

# How much the loss trusts one comparison. 0.1 is the published default and it is not tuned here: it trades
# how far the model may move from the reference against how hard it presses a single pair, and nothing
# measured on this corpus says where that trade should sit. Provisional, and `--beta` is what tests it.
BETA = 0.1

# What the preferred answer's own likelihood is worth against winning the comparison. 1.0 is the published
# default for pairing the two and is provisional here; 0.0 is the plain DPO objective, which on this corpus
# won the pairs and stopped writing English. A sweep on the unsupported-figure rate is what would settle it.
SFT_WEIGHT = 1.0

# Where the preferred answer sits in the pair of margins handed to cross entropy, as in a two-class problem.
CHOSEN = 1

# cross_entropy's own skip value, reused rather than redefined so a padded answer position is ignored by
# the same rule everywhere.
IGNORE = -100


def sequence_logprob(model, source, answer, keep=None):
    """Mean log-probability of each `answer` row under `model`, given `source`. One Tensor per row.

    Per row rather than per batch, because a preference is a comparison between two particular answers and
    a batch mean would blend every pair into one number. The rows are sliced out of a single forward pass,
    so this costs one pass and not one per row.

    The answer is its own input shifted by one: position t predicts token t+1, which is what the decoder was
    trained on. Scoring the unshifted sequence would ask how likely each token is given itself.
    """
    logits = model(source, answer[:, :-1], keep)
    # Widened before the skip value goes in: the datasets store token ids as uint16, where -100 wraps to
    # 65436 and cross entropy reads it as a vocabulary index.
    targets = answer[:, 1:].astype(xp.int64)
    targets[targets == PAD_ID] = IGNORE
    empty = xp.flatnonzero((targets != IGNORE).sum(axis=-1) == 0)
    if len(empty):
        raise ValueError(f"rows {empty.tolist()} have no answer token to score; a preference between an "
                         f"empty answer and anything else is not a judgement about writing")
    # Negated because cross_entropy is the mean *negative* log-probability over the unpadded positions.
    return [ops.neg(F.cross_entropy(ops.getitem(logits, row), targets[row]))
            for row in range(len(answer))]


def margins(chosen, rejected, reference_chosen, reference_rejected, beta=BETA):
    """How much more the model prefers each chosen answer than the frozen reference did.

    Positive means this pair is already learned. The reference log-probabilities are plain numbers because
    that model does not move, which is what lets them be computed once for the whole dataset instead of
    once per step.
    """
    shifted = [ops.sub(ops.sub(taken, refused), float(reference_taken - reference_refused))
               for taken, refused, reference_taken, reference_refused
               in zip(chosen, rejected, reference_chosen, reference_rejected)]
    return ops.mul(ops.concat([ops.reshape(one, (1,)) for one in shifted]), beta)


def preference_loss(chosen, rejected, reference_chosen, reference_rejected, beta=BETA, sft=SFT_WEIGHT):
    """What the judged ordering costs the model, for a batch of pairs. Mean over the batch.

    Two terms. The ranking term is how unlikely the ordering is, routed through cross entropy rather than a
    hand-rolled `-log sigmoid`: they are the same quantity, and that one already subtracts its row maximum,
    so a pair the model has badly wrong costs a large number instead of an infinity that takes every
    gradient in the batch with it.

    The second term is the preferred answer's own negative log-likelihood, so winning a comparison by making
    both answers unwritable costs more than it gains. It is free: `chosen` already holds those numbers.
    """
    margin = margins(chosen, rejected, reference_chosen, reference_rejected, beta)
    rows = margin.shape[0]
    against = ops.concat([xp.zeros((rows, 1)), ops.reshape(margin, (rows, 1))], axis=-1)
    ranked = F.cross_entropy(against, xp.full(rows, CHOSEN))
    if not sft:
        return ranked
    written = ops.neg(ops.mean(ops.concat([ops.reshape(one, (1,)) for one in chosen])))
    return ops.add(ranked, ops.mul(written, sft))


def agreement(chosen, rejected):
    """The share of pairs the model gives the judge's answer the higher probability. 50% is the coin flip.

    Absolute rather than measured against the reference, so the same number can be read off the frozen
    starting checkpoint and off the trained one. That comparison is the result: it says whether preference
    training moved the model's ordering, in a unit that needs no explaining.
    """
    taken, refused = xp.asarray(chosen, dtype=float), xp.asarray(rejected, dtype=float)
    return float((taken > refused).mean())
