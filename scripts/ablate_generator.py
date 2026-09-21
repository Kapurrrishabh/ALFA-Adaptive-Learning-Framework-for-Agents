#!/usr/bin/env python3
"""Ask whether the generator actually reads its evidence, by taking the evidence away.

This is the acceptance test the perplexity number cannot be. A model that ignores its evidence and
recites fluent prose scores a respectable loss, which is exactly what happened on the Stack Exchange
task: hiding every passage cost 0.018 loss, so the evidence was decoration. What that measurement
should look like on a working model is a large, obvious penalty.

Three conditions, same rows, same weights. As given; the evidence blanked but the question left in
place; and the evidence swapped for another row's. The swap is the sharpest of the three, because a
model that has merely learned the shape of an answer still scores well with no evidence, but cannot
score well while staring at the wrong numbers.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfagent import pretrained  # noqa: E402
from selfagent.data import advisory  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.tokenizer.vocab import PAD_ID, SEP_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402
from train_generator import evaluate, load_split  # noqa: E402


def evidence_starts(source):
    """Where the evidence begins in each slot: just past the SEP that closes the question."""
    is_separator = source == SEP_ID
    # argmax takes the first True. A row with no SEP would report 0, which the caller must not meet,
    # so it is an error here rather than a silently blanked question.
    if not is_separator.any(axis=-1).all():
        raise ValueError("a source row has no SEP, so the question and evidence cannot be told apart")
    return is_separator.argmax(axis=-1) + 1


def blanked(split):
    """The same rows with the evidence removed and the question untouched."""
    source, keep, target = (array.copy() for array in split)
    starts = evidence_starts(source)
    columns = np.arange(source.shape[-1])
    after = columns[None, None, :] >= starts[..., None]
    source[after] = PAD_ID
    keep[after] = 0
    return source, keep, target


def swapped(split, rng):
    """The same rows wearing another row's evidence, which no faithful answer can be built from."""
    source, keep, target = (array.copy() for array in split)
    starts = evidence_starts(source)
    order = rng.permutation(len(source))
    for row, other in enumerate(order):
        for slot in range(source.shape[1]):
            mine, theirs = starts[row, slot], starts[other, slot]
            width = min(source.shape[-1] - mine, source.shape[-1] - theirs)
            source[row, slot, mine : mine + width] = split[0][other, slot, theirs : theirs + width]
            keep[row, slot, mine : mine + width] = split[1][other, slot, theirs : theirs + width]
            source[row, slot, mine + width :] = PAD_ID
            keep[row, slot, mine + width :] = 0
    return source, keep, target


def answerable(split, tokenizer):
    """The rows that are not refusals, which are the only ones whose answer needs the evidence."""
    refusal = tokenizer.encode(advisory.UNSUPPORTED)
    source, keep, target = split
    body = target[:, 1 : 1 + len(refusal)]
    is_refusal = (body == np.asarray(refusal, dtype=body.dtype)).all(axis=-1)
    return tuple(array[~is_refusal] for array in split), int(is_refusal.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="advisory")
    parser.add_argument("--checkpoint", default="",
                        help="model to score, when it is not the one named after the dataset")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=200)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    config, weights = pretrained.load(artifacts / (args.checkpoint or f"{args.dataset}.npz"))
    model = GroundedGenerator(config)
    model.load_state_dict(weights)
    model.eval()

    split = load_split(artifacts, args.dataset, args.split)
    split, refusals = answerable(split, tokenizer)
    rows = min(len(split[0]), args.batch_size * args.batches)
    print(f"{args.dataset} {args.split}: {rows:,} answerable rows scored, {refusals:,} refusals set aside")

    rng = np.random.default_rng(config.seed)
    conditions = {
        "as given": split,
        "evidence blanked": blanked(split),
        "evidence swapped": swapped(split, rng),
    }
    baseline = None
    for name, condition in conditions.items():
        loss = evaluate(model, condition, config.vocab_size, args.batch_size, args.batches)
        baseline = loss if baseline is None else baseline
        print(f"  {name:20s} loss {loss:.4f}  perplexity {math.exp(loss):8.1f}  "
              f"{loss - baseline:+.4f} vs as given")


if __name__ == "__main__":
    sys.exit(main())
