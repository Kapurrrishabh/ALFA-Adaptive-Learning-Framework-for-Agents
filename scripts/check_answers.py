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
from selfagent.data import advisory  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.models.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P  # noqa: E402
from selfagent.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402
from train_generator import load_split  # noqa: E402

# A refusal is recognised by its opening words rather than exact equality, so a model that refuses and
# then keeps talking still counts as having refused. Words only: decoding respaces the punctuation.
_REFUSAL_OPENING = " ".join(advisory.UNSUPPORTED.split()[:7])


def repeated_fraction(text, size=4):
    words = text.split()
    grams = [tuple(words[i : i + size]) for i in range(len(words) - size + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


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
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    config, weights = pretrained.load(artifacts / (args.checkpoint or f"{args.dataset}.npz"))
    model = GroundedGenerator(config)
    model.load_state_dict(weights)
    model.eval()

    source, keep, target = load_split(artifacts, args.dataset, args.split)
    rng = np.random.default_rng(config.seed)
    chosen = rng.permutation(len(source))[: args.rows]
    starts = evidence_starts(source[chosen])
    temperature = 0.0 if args.greedy else ANSWER_TEMPERATURE
    top_p = 1.0 if args.greedy else ANSWER_TOP_P

    scored = []
    for start in range(0, len(chosen), args.batch_size):
        rows = chosen[start : start + args.batch_size]
        produced = model.generate(source[rows], keep[rows], temperature, top_p, rng)
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
            scored.append((question, evidence, wanted, answer))

    refusals = [row for row in scored if _REFUSAL_OPENING in row[2]]
    answerable = [row for row in scored if _REFUSAL_OPENING not in row[2]]
    unsupported = [guardrails.unsupported_figures(a, e) for _, e, _, a in answerable]
    stated = sum(len(guardrails.figures(a)) for _, _, _, a in answerable)
    missing = sum(len(u) for u in unsupported)
    refused = sum(_REFUSAL_OPENING in a for _, _, _, a in refusals)
    wrongly_refused = sum(_REFUSAL_OPENING in a for _, _, _, a in answerable)

    matched = sum(w.split() == a.split() for _, _, w, a in scored)

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
          f"{np.mean([repeated_fraction(a) for _, _, _, a in scored]):.1%} "
          f"(reference: the answers themselves are "
          f"{np.mean([repeated_fraction(w) for _, _, w, _ in scored]):.1%})")

    for question, evidence, wanted, answer in scored[: args.show]:
        print(f"\nQ {question}\nE {evidence}\nwanted   {wanted}\nproduced {answer}")
        bad = guardrails.unsupported_figures(answer, evidence)
        if bad:
            print(f"unsupported {bad}")


if __name__ == "__main__":
    sys.exit(main())
