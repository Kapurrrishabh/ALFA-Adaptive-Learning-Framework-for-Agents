#!/usr/bin/env python3
"""Generate answers on held-out rows and check them, which is what loss cannot do.

Perplexity on this data is flattering: the answers are templated, so most positions are boilerplate a
model can predict without reading anything, and the handful of digit positions that actually need the
evidence are lost in the average. These are the numbers that decide whether the system works.

  unsupported figures  a figure in the answer that the evidence does not state. The target is near
                       zero, and it is the claim the whole design rests on. On a slot dataset it is
                       zero by construction, and exact match is what decides instead: naming the
                       wrong fact is the only mistake left once the digits are no longer the model's.
  exact match          the produced answer, character for character. Strict, and on templated
                       answers it is the honest number — anything less is a wrong fact or wrong word.
  abstention           on rows whose answer is a refusal, does the model refuse. A model that never
                       refuses will confidently answer questions about earnings it cannot see.
  false refusal        the opposite failure: refusing a question the evidence does answer.
  repeated 4-grams     decoding sanity. Greedy repeated 73% of its own 4-grams where humans repeat
                       1.7%, which nucleus sampling fixed and which can regress silently.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablate_generator import evidence_starts  # noqa: E402
from selfagent import pretrained  # noqa: E402
from selfagent.agent import guardrails  # noqa: E402
from selfagent.autograd import no_grad  # noqa: E402
from selfagent.data import advisory  # noqa: E402
from selfagent.learn.abstain import COVERAGE_FLOOR  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.models.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P  # noqa: E402
from selfagent.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402
from train_generator import load_split  # noqa: E402

def _refuses(text):
    return guardrails.is_refusal(text, advisory.UNSUPPORTED)


def repeated_fraction(text, size=4):
    words = text.split()
    grams = [tuple(words[i : i + size]) for i in range(len(words) - size + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def self_confidence(model, source, keep, produced, budget):
    """The model's own mean token probability on the answer it just wrote, one number per row.

    An abstention threshold has to run on something available at serving time, when the right answer
    is not. This is the cheapest such signal — one extra forward pass and no new parameters — and it is
    only worth building on if it separates the rows the model got right from the ones it got wrong.
    """
    target = np.full((len(produced), budget), PAD_ID, dtype=np.int64)
    for row, ids in enumerate(produced):
        whole = [CLS_ID] + list(ids)[: budget - 2] + [SEP_ID]
        target[row, : len(whole)] = whole
    with no_grad():
        logits = model(source, target[:, :-1], keep).data
    logits = logits - logits.max(axis=-1, keepdims=True)
    log_probabilities = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    wanted = target[:, 1:]
    taken = np.take_along_axis(log_probabilities, wanted[:, :, None], axis=-1)[:, :, 0]
    counted = wanted != PAD_ID
    return np.exp((taken * counted).sum(axis=-1) / np.maximum(counted.sum(axis=-1), 1))


def report_confidence(sure, right, threshold):
    """Does the model know when it is wrong? The question an abstention threshold rests on.

    Reported as accuracy and coverage together, because a gap between two means can be real and still
    useless for deciding one row at a time. Without --threshold the cut is chosen on these same rows,
    which is an upper bound and not a result; fit it on one split and quote it on another.
    """
    if right.all() or not right.any():
        print(f"  self-confidence      {sure.mean():.3f}, nothing to separate")
        return
    print(f"  self-confidence      {sure[right].mean():.3f} when right, "
          f"{sure[~right].mean():.3f} when wrong")
    if threshold:
        answered = sure >= threshold
        accuracy = right[answered].mean() if answered.any() else float("nan")
        print(f"  answer-or-abstain    {accuracy:.1%} correct on the {answered.mean():.1%} it answers, "
              f"at the given {threshold:.4f} (answering everything: {right.mean():.1%})")
        return
    best = max(
        ((right[sure >= cut].mean(), (sure >= cut).mean(), cut)
         for cut in np.unique(sure) if (sure >= cut).mean() >= COVERAGE_FLOOR),
        default=None,
    )
    if best:
        print(f"  answer-or-abstain    {best[0]:.1%} correct on the {best[1]:.1%} it would answer, at "
              f"a cut of {best[2]:.4f} fitted here, so an upper bound "
              f"(answering everything: {right.mean():.1%})")


def load_model(artifacts, dataset, checkpoint=""):
    """The tokenizer and the model for a dataset, in eval mode."""
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    config, weights = pretrained.load(artifacts / (checkpoint or f"{dataset}.npz"))
    model = GroundedGenerator(config)
    model.load_state_dict(weights)
    model.eval()
    return tokenizer, model, config


def answer_rows(model, tokenizer, split, chosen, rng, greedy=False, batch_size=8,
                temperature=ANSWER_TEMPERATURE, top_p=ANSWER_TOP_P):
    """(question, evidence, gold, produced, confidence) for each chosen row.

    Shared with the labelling loop rather than copied: a judge has to see the same answers the scorer
    saw, and regenerating them under a different sampler would have the two sets of numbers describing
    two different models. The sampler defaults to the served one, so only a caller that deliberately
    asks for a different one gets it.
    """
    source, keep, target = split
    starts = evidence_starts(source[chosen])
    if greedy:
        temperature, top_p = 0.0, 1.0

    scored = []
    for start in range(0, len(chosen), batch_size):
        rows = chosen[start : start + batch_size]
        produced = model.generate(source[rows], keep[rows], temperature, top_p, rng)
        sure = self_confidence(model, source[rows], keep[rows], produced, target.shape[1])
        for offset, ids in enumerate(produced):
            row = rows[offset]
            whole = source[row, 0]
            cut = starts[start + offset, 0]
            question = tokenizer.decode([i for i in whole[:cut] if i != PAD_ID])
            evidence = tokenizer.decode([i for i in whole[cut:] if i != PAD_ID])
            # The markers come off so the gold answer is comparable to the produced one, which
            # generate() already strips. They never appear inside an answer, so dropping them is safe.
            wanted = tokenizer.decode([i for i in target[row] if i not in (PAD_ID, CLS_ID, SEP_ID)])
            answer = tokenizer.decode(list(ids))
            scored.append((question, evidence, wanted, answer, float(sure[offset])))
    return scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="advisory")
    parser.add_argument("--checkpoint", default="",
                        help="model to score, when it is not the one named after the dataset")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--greedy", action="store_true", help="decode greedily instead of sampling")
    parser.add_argument("--show", type=int, default=4, help="answers to print in full")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="abstain below this self-confidence; fit it on one split, quote it on another")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer, model, config = load_model(artifacts, args.dataset, args.checkpoint)
    split = load_split(artifacts, args.dataset, args.split)
    rng = np.random.default_rng(config.seed)
    chosen = rng.permutation(len(split[0]))[: args.rows]
    scored = answer_rows(model, tokenizer, split, chosen, rng, args.greedy, args.batch_size)

    refusals = [row for row in scored if _refuses(row[2])]
    answerable = [row for row in scored if not _refuses(row[2])]
    unsupported = [guardrails.unsupported_figures(a, e) for _, e, _, a, _ in answerable]
    stated = sum(len(guardrails.figures(a)) for _, _, _, a, _ in answerable)
    missing = sum(len(u) for u in unsupported)
    refused = sum(_refuses(a) for _, _, _, a, _ in refusals)
    wrongly_refused = sum(_refuses(a) for _, _, _, a, _ in answerable)

    right = [w.split() == a.split() for _, _, w, a, _ in scored]
    matched = sum(right)

    print(f"{len(scored)} rows: {len(answerable)} answerable, {len(refusals)} refusals")
    print(f"  exact match          {matched}/{len(scored)} ({matched / max(len(scored), 1):.1%})")
    print(f"  unsupported figures  {missing}/{stated} "
          f"({missing / max(stated, 1):.1%} of figures, "
          f"{sum(bool(u) for u in unsupported) / max(len(answerable), 1):.1%} of answers)")
    print(f"  abstention           {refused}/{len(refusals)} "
          f"({refused / max(len(refusals), 1):.1%} of rows that should refuse)")
    print(f"  false refusal        {wrongly_refused}/{len(answerable)} "
          f"({wrongly_refused / max(len(answerable), 1):.1%})")
    print(f"  repeated 4-grams     "
          f"{np.mean([repeated_fraction(a) for _, _, _, a, _ in scored]):.1%} "
          f"(reference: the answers themselves are "
          f"{np.mean([repeated_fraction(w) for _, _, w, _, _ in scored]):.1%})")
    report_confidence(np.array([c for *_, c in scored]), np.array(right), args.threshold)

    for question, evidence, wanted, answer, _ in scored[: args.show]:
        print(f"\nQ {question}\nE {evidence}\nwanted   {wanted}\nproduced {answer}")
        bad = guardrails.unsupported_figures(answer, evidence)
        if bad:
            print(f"unsupported {bad}")


if __name__ == "__main__":
    sys.exit(main())
