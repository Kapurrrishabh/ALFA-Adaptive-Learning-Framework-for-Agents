"""Reward an answer by checking it against its own evidence, so the model can learn from its own writing.

This is the self-learning half of the loop, and it exists because the other two halves are measured dead.

**Why not more supervised answers.** Over 4,000 sampled rows of the supervised set, 2,390 state at least one
figure and only 116 of those -- 4.9% -- have every figure present in their retrieved passages. So 2,274 rows,
56.9% of the whole set, teach the model to state a number its evidence does not contain. The 59.4%
unsupported-figure rate the trained generator produces is not a training failure; it is the corpus, learned
correctly.

**Why not preferences.** The pairs come from the same corpus, so both the chosen and the rejected answer cite
numbers absent from the passages, and no ordering between them expresses grounding. Two full runs confirmed
it: plain DPO won the comparisons and stopped writing English, and pairing it with the preferred answer's own
likelihood kept the writing and moved held-out agreement 58.9% to 59.2%. Neither beat the frozen checkpoint.

**So the signal has to be a check and not a label.** `guardrails.unsupported_figures` reads the answer and
the passages with the same extractor and returns the figures the answer invented. It needs no reward model,
no judge and no second network -- it is the same function the server screens with and the report counts, so
a sample is rewarded in training by the rule that would have refused it in production.

The reward is `rank()`'s serving order expressed as a number, deliberately: a loop trained on a different
ordering than the server serves would optimise for something nobody reads.

**A refusal must not be able to win**, and that is the one trap this design has to dodge. An answer that
states no figure has no unsupported figure, so faithfulness alone is maximised by saying nothing -- the
failure RLFKV hit on financial RAG and had to add a second reward term to stop. Here the model already leans
that way: `rank.py` measured refusals carrying the highest confidence it produces, 0.99538 against 0.98797.
Hence `FIGURE_CREDIT`, which pays a grounded answer for every figure it grounds, so silence scores above
invention and below use.
"""

from ..agent import guardrails
from .rank import GROUNDED, REFUSED, tier

# What a refusal earns, against 1.0 for an answer that states nothing wrong. Between the two on purpose:
# above an invented figure because the project would rather say nothing than say a number that is not
# there, below a grounded answer so a loop cannot improve its score by going quiet. Provisional -- what
# would settle it is the share of kept samples that are refusals over a sweep.
REFUSAL_CREDIT = 0.1

# What one grounded figure adds. Without it every grounded answer scores 1.0 and the cheapest way into the
# kept set is to state no figure at all. Provisional, and large on purpose: the corpus states a figure in
# 59.8% of rows, so a loop that trains the habit away is further from the data than one that keeps it.
FIGURE_CREDIT = 0.5


def reward(answer, evidence, refusal):
    """What one sampled answer earns. Higher is better, and 0.0 means it invented a figure.

    Zero rather than negative because the loop keeps the top of a group and never trains on the rest, so
    all that matters is that nothing outranks a grounded answer.
    """
    kind = tier(answer, evidence, refusal)
    if kind == REFUSED:
        return REFUSAL_CREDIT
    if kind != GROUNDED:
        return 0.0
    # Every figure here is supported, because that is what GROUNDED means.
    return 1.0 + FIGURE_CREDIT * len(guardrails.figures(answer))


def winner(rewards):
    """Which sample of one question's group to train on, or None when the group teaches nothing.

    Two ways a group is dropped, and both matter more on this model than on a large one.

    Every sample invented a figure, so the best of them is still wrong and fine-tuning on it would teach
    the exact failure the reward exists to remove.

    Or every sample earned the same reward, so the pick is not a judgement and the step is the model being
    fine-tuned on its own output, which holds what it already does in place. That case is the common one
    here: `rank.py` measured 1.77 distinct answers out of 8 at the served sampler, and loosening the
    nucleus tenfold moved it by 0.1. DAPO dropped these groups for the neighbouring reason -- under a
    group-normalised advantage they contribute exactly zero gradient -- and reported it was the single
    largest win in their ablation, 42 to 50.
    """
    best = max(range(len(rewards)), key=rewards.__getitem__)
    if not rewards[best] or len(set(rewards)) == 1:
        return None
    return best
