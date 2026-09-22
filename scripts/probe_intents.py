#!/usr/bin/env python3
"""Does the encoder know two wordings of the same question mean the same thing?

This is the measurement that decides where the advisory generator's failure lives. Trained on five
phrasings per intent it answered an unseen sixth as if it were a different question -- "does X look
extended to you ?" came back with the buy refusal instead of the rsi reading -- and held-out loss
stopped improving at 0.70 while trained-phrasing loss kept falling to 0.011.

Embed every phrasing and ask whether its nearest trained phrasing shares its intent. Reserved
phrasings cannot be neighbours: left in, one matches another and the whole set scores 90% on an
encoder with no task training, because "i am wondering if X has been torrid" sits one word from
"... has been grim" and both are unseen. Splitting the reserved ones by which axis is novel is what
makes the result readable, and it turns "paraphrases" into three questions of very different difficulty.

On the pretrained encoder, before any task training: a novel word in a familiar frame is placed 15/15,
a novel frame built from familiar words 28/41, and a phrasing novel in both 2/7 against 1/7 by chance.
So the vocabulary is not the problem and the sentence shape is. Fine-tuning on the task then makes the
shape worse, not better: the same frames fall to 13/41 and both-novel to chance, while trained
phrasings rise to 89%. Five phrasings per intent bought memorisation at the cost of the general
language the pretrained encoder already had.

Only the overall figure is comparable across phrasing sets -- a denser set gives every question nearer
twins -- so compare cells within a run and checkpoints against each other, never a number here to one
from a differently sized set. No training, and no labels beyond the intents the dataset declares.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent import pretrained  # noqa: E402
from selfagent.data import advisory  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.models.retriever import embed  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402

# One ticker for every phrasing: the intent has to be read off the wording, not off which stock it is.
TICKER = "AAPL"


def questions():
    """(intent, phrasing, text) for every way every intent can be asked."""
    return [
        (intent, phrasing, advisory.ask(intent, TICKER, phrasing))
        for intent in advisory.INTENTS
        for phrasing in range(advisory.phrasings(intent))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--checkpoint", default="pretrained.npz",
                        help="pretrained.npz for the encoder as the generator got it")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    config, weights = pretrained.load(artifacts / args.checkpoint)
    model = GroundedGenerator(config)
    model.load_pretrained_encoder(weights)
    model.eval()

    asked = questions()
    vectors = embed(model, tokenizer, [text for _, _, text in asked], config.max_text_length)
    # einsum, not the @ operator: numpy 2.0.2 on Apple's Accelerate BLAS raises spurious overflow and
    # divide-by-zero flags for a float64 matmul whose result is correct to 1.3e-15.
    similarity = np.einsum("id,jd->ij", vectors, vectors)
    np.fill_diagonal(similarity, -np.inf)
    # Only trained phrasings can be the neighbour. Left open, a reserved phrasing matches another
    # reserved one -- "i am wondering if X has been torrid" finds "... has been grim", one word apart and
    # both unseen -- and the whole set scores 90% on an encoder that has had no task training at all.
    reserved = [i for i, (intent, phrasing, _) in enumerate(asked)
                if advisory.is_held_out(intent, phrasing)]
    similarity[:, reserved] = -np.inf
    nearest = similarity.argmax(axis=-1)

    right = {intent: 0 for intent in advisory.INTENTS}
    for index, (intent, phrasing, text) in enumerate(asked):
        matched, _, neighbour = asked[nearest[index]]
        right[intent] += matched == intent
        if matched != intent:
            print(f"  MISS {intent:12s} {text}\n       nearest {matched:12s} {neighbour} "
                  f"({similarity[index, nearest[index]]:.3f})")

    total = sum(right.values())
    print(f"\nnearest neighbour shares the intent for {total}/{len(asked)} phrasings "
          f"({total / len(asked):.0%}), against {1 / len(advisory.INTENTS):.0%} by chance")
    for intent in advisory.INTENTS:
        print(f"  {intent:12s} {right[intent]}/{advisory.phrasings(intent)}")

    # A fine-tuned encoder has seen every phrasing but the reserved ones, so only those are a
    # generalisation number for it. Split by which axis is novel: a reserved word sits in a frame the
    # encoder knows and a reserved frame uses words it knows, so only the pair of them is a hard case.
    print()
    for kind in (advisory.TRAINED, "word", "frame", "word and frame"):
        rows = [i for i, (intent, phrasing, _) in enumerate(asked)
                if advisory.novelty(intent, phrasing) == kind]
        matched = sum(asked[nearest[i]][0] == asked[i][0] for i in rows)
        print(f"  {kind:14s} new  {matched}/{len(rows)} ({matched / len(rows):.0%})")


if __name__ == "__main__":
    sys.exit(main())
