"""Pick the best of k answers with the weights frozen.

The policy: a candidate that states a figure its evidence does not is ranked beneath every candidate that
passes the guardrail, whatever it claims about itself, and confidence only decides among equals. A refusal
sits between the two — worse than a grounded answer, better than an invented one, because the project
would rather say nothing than say a number that is not there.

**Measured, and it does not pay on this model.** On 600 unseen rows a single sample is right 28.0% of the
time; best of 8 by confidence is 27.8%, by rank() 28.5%, and by consensus 29.2%. The cause is not the
ranking — it is that eight samples produce **1.77 distinct answers**, so there is almost nothing to pick
between, and loosening the nucleus tenfold moved distinctness by 0.1 and the ceiling not at all. A model
whose per-token distribution is a near-delta cannot be improved by sampling it more. Consensus rescued 14
rows and spoiled 7, which is p = 0.189 by McNemar — a lead, not a result.

Two findings worth keeping. The confidence ranks *worse* than not ranking, because refusals carry the
highest confidence the model produces (0.99538 against 0.98797 on answers, at 18 words against 32), so
picking the surest candidate picks the refusal — the AUC 0.740 that B3 measured holds across questions
and does not transfer to ranking answers to one. And the tiers do help where they fire, beating
confidence by 1.5 points, but fired on 4% of rows because this checkpoint invents figures on 0.7% of
unseen answers. The policy is kept for C1, where the candidates will come from different retrieved
contexts rather than from resampling one.
"""

from collections import Counter

from ..agent import guardrails

# Smaller is better. The order is the whole policy: grounded, then silent, then invented.
GROUNDED, REFUSED, INVENTED = 0, 1, 2


def tier(answer, evidence, refusal):
    """Which of the three kinds of answer this is, which decides ranking before confidence does."""
    if guardrails.is_refusal(answer, refusal):
        return REFUSED
    return INVENTED if guardrails.unsupported_figures(answer, evidence) else GROUNDED


def rank(candidates, evidence, confidences, refusal):
    """The candidate indices, best first.

    Ties on tier are broken by confidence and then by the order they were generated in, so the same k
    candidates always produce the same pick. A reranker that broke ties by hash would make a learning
    curve irreproducible for no gain.
    """
    if len(candidates) != len(confidences):
        raise ValueError(f"{len(candidates)} candidates against {len(confidences)} confidences")
    if not candidates:
        raise ValueError("cannot rank zero candidates; sample at least one")
    scored = [(tier(answer, evidence, refusal), -confidence, position)
              for position, (answer, confidence) in enumerate(zip(candidates, confidences))]
    return [position for *_, position in sorted(scored)]


def best(candidates, evidence, confidences, refusal):
    """The one answer to serve out of k."""
    return candidates[rank(candidates, evidence, confidences, refusal)[0]]


def consensus(candidates, evidence, refusal):
    """The position of the most repeated answer among those in the best tier any candidate reached.

    Asks a different question from rank(): not which answer the model is surest of, but which one it
    keeps arriving at. That matters because the confidence was measured to rank right above wrong
    across questions and not within one, where it prefers the short confident refusal.
    """
    if not candidates:
        raise ValueError("cannot rank zero candidates; sample at least one")
    tiers = [tier(answer, evidence, refusal) for answer in candidates]
    words = [" ".join(answer.split()) for answer in candidates]
    repeats = Counter(word for word, kind in zip(words, tiers) if kind == min(tiers))
    return max(range(len(candidates)),
               key=lambda position: (-tiers[position], repeats[words[position]], -position))
