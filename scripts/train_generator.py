#!/usr/bin/env python3
"""Train the grounded generator on question-and-answer pairs, starting from the pretrained encoder.

Teacher forcing: the decoder is fed the answer shifted one place right and asked for the next token at
every position. Padding is excluded from the loss — an answer is 143 tokens in a 192 window, so a third
of the positions are padding, and counting them teaches the model mostly to predict padding.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent import pretrained  # noqa: E402
from selfagent.autograd import functional as F  # noqa: E402
from selfagent.autograd import no_grad  # noqa: E402
from selfagent.config import ModelConfig  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.optim import AdamW, clip_grad_norm  # noqa: E402
from selfagent.tokenizer.vocab import PAD_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402

IGNORED = -100


def load_split(artifacts, dataset, name):
    return tuple(
        np.load(artifacts / f"{dataset}_{name}_{part}.npy") for part in ("source", "keep", "target")
    )


def batch_loss(model, source, keep, target, vocab_size):
    """Next-token loss over one batch, padding excluded.

    The decoder reads target[:, :-1] and is scored against target[:, 1:]. Feeding the unshifted answer
    as its own target trains the model to echo its input.
    """
    logits = model(source, target[:, :-1], keep)
    wanted = target[:, 1:].astype(np.int64).copy()
    wanted[wanted == PAD_ID] = IGNORED
    return F.cross_entropy(logits.reshape((-1, vocab_size)), wanted.reshape(-1), ignore_index=IGNORED)


def evaluate(model, split, vocab_size, batch_size, batches):
    source, keep, target = split
    was_training = model.training
    model.eval()
    total = 0.0
    counted = 0
    with no_grad():
        for start in range(0, min(len(source), batches * batch_size), batch_size):
            stop = start + batch_size
            total += batch_loss(model, source[start:stop], keep[start:stop], target[start:stop], vocab_size).item()
            counted += 1
    model.train(was_training)
    return total / max(counted, 1)


def report(model, validation, extra, args, config):
    """Validation, and the extra split beside it. The gap between them is the measurement that matters:
    both hold unseen dates, but only the extra holds question phrasings training never saw.

    Returns the loss worth judging a checkpoint on, which is the extra split's whenever there is one.
    """
    held = evaluate(model, validation, config.vocab_size, args.batch_size, args.eval_batches)
    line = f"  validation loss {held:.4f}  perplexity {math.exp(held):.1f}"
    judged = held
    if extra is not None:
        judged = evaluate(model, extra, config.vocab_size, args.batch_size, args.eval_batches)
        line += f"   {args.extra_split} loss {judged:.4f}  perplexity {math.exp(judged):.1f}"
    print(line)
    return judged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="generator", help="prefix of the .npy split files")
    parser.add_argument("--run-name", default="",
                        help="names the checkpoints, so one arm of an ablation cannot overwrite another")
    parser.add_argument("--extra-split", default="", help="a third split to report separately")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="hold the encoder's layers at their pretrained values")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    train = load_split(artifacts, args.dataset, "train")
    validation = load_split(artifacts, args.dataset, "validation")
    extra = load_split(artifacts, args.dataset, args.extra_split) if args.extra_split else None
    source, keep, target = train

    config = ModelConfig(vocab_size=tokenizer.vocab_size, max_text_length=int(source.shape[-1]))
    model = GroundedGenerator(config)

    # Starting the encoder from noise wastes the pretraining run: reading the passages is the harder
    # half of this task and the encoder is the half that already knows how to read finance text.
    stored_config, weights = pretrained.load(artifacts / "pretrained.npz")
    if stored_config.encoder_fingerprint != config.encoder_fingerprint:
        raise SystemExit(
            f"pretrained.npz holds an encoder of a different shape, so its weights cannot be "
            f"loaded; retrain it or match the config.\n"
            f"  stored:  {stored_config}\n  current: {config}"
        )
    model.load_pretrained_encoder(weights)
    print(f"encoder initialised from {artifacts}/pretrained.npz")
    if args.freeze_encoder:
        model.freeze_encoder()

    checkpoint = f"{args.run_name or args.dataset}.npz"
    best_checkpoint = f"{args.run_name or args.dataset}.best.npz"
    optimizer = AdamW(model.trainable_parameters(), lr=args.learning_rate)
    steps_per_epoch = len(source) // args.batch_size
    print(
        f"{sum(p.size for p in model.trainable_parameters()):,} of "
        f"{sum(p.size for p in model.parameters()):,} parameters training, "
        f"{len(source):,} train rows, {len(validation[0]):,} held out, "
        f"{source.shape[1]} passages x {source.shape[2]} tokens"
    )
    print(f"{args.epochs} epochs x {steps_per_epoch:,} steps at batch {args.batch_size}")
    print(f"uniform baseline loss is ln({config.vocab_size}) = {math.log(config.vocab_size):.3f}")

    rng = np.random.default_rng(config.seed)
    step = 0
    best, best_step = math.inf, 0
    started = time.monotonic()
    model.train()

    def keep_if_best(judged):
        """Validation shares its phrasings with training so it falls forever; only the unseen split
        turns back up, and the turn is where the usable model is. This makes that number a
        model-selection number rather than a clean test one, so quote it as such."""
        nonlocal best, best_step
        if judged < best:
            best, best_step = judged, step
            pretrained.save(artifacts / best_checkpoint, model, config)

    for epoch in range(args.epochs):
        order = rng.permutation(len(source))
        for index in range(steps_per_epoch):
            rows = order[index * args.batch_size : (index + 1) * args.batch_size]
            optimizer.zero_grad()
            loss = batch_loss(model, source[rows], keep[rows], target[rows], config.vocab_size)
            loss.backward()
            clip_grad_norm(optimizer.parameters, 1.0)
            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.monotonic() - started
                left = (args.epochs * steps_per_epoch - step) * elapsed / step / 3600
                print(
                    f"step {step:,}/{args.epochs * steps_per_epoch:,}  loss {loss.item():.4f}  "
                    f"{step / elapsed:.2f} step/s  {left:.1f}h left"
                )
            if step % args.eval_every == 0:
                keep_if_best(report(model, validation, extra, args, config))
            if step % args.checkpoint_every == 0:
                pretrained.save(artifacts / checkpoint, model, config)

    pretrained.save(artifacts / checkpoint, model, config)
    keep_if_best(report(model, validation, extra, args, config))
    print(f"saved {artifacts}/{checkpoint}")
    print(f"best {best:.4f} at step {best_step:,}, saved {artifacts}/{best_checkpoint}")


if __name__ == "__main__":
    sys.exit(main())
