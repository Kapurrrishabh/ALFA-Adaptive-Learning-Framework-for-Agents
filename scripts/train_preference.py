#!/usr/bin/env python3
"""Train the generator on which of two answers is better, starting from the checkpoint that wrote both.

This is the run behind the learning claim's generative half. Supervised training gave the model one right
answer per question and it learned to write fluently and ignore its evidence; this gives it two answers to
the same question with the same passages and tells it which one a reader preferred, which is the only way
to say "less of that" to a model.

The frozen starting checkpoint is scored once, up front, and then never loaded again -- the reference
log-probabilities are just numbers after that, so a step costs two forward and two backward passes rather
than four forwards. They are cached on disk against a fingerprint of both the dataset and the checkpoint,
because a stale cache here would silently anchor training to the wrong model and the loss would still look
healthy.

Dropout is off in both passes. A preference is a comparison between two particular answers, and a dropout
mask drawn separately for each side adds noise to their difference that does not cancel; with it off, the
loss at the first step is exactly ln 2, which is a property a test can check.

Read the writing numbers, not the loss and not the agreement. The first full run settled this: over one epoch
the held-out loss fell from 0.6931 to 0.6617 and held-out agreement rose from 58.9% to 60.9% -- and the model
stopped writing English, opening every answer on about thirty copies of the word "you", with 92.9% of its
figures unsupported against 59.4% frozen and 30.8% of its 4-grams repeated against 1.7%. Loss and agreement
improved monotonically the whole way down. Neither can see the failure, because neither asks the model to
produce anything.

So `writes` generates answers at every check, and a checkpoint is promoted on its unsupported-figure rate
subject to a ceiling on repetition. The ceiling is not optional: a model that has stopped writing states no
figures at all, and would otherwise score a perfect zero.
"""

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_answers import answer_rows, repeated_fraction, tally  # noqa: E402
from selfagent import pretrained  # noqa: E402
from selfagent.autograd import no_grad  # noqa: E402
from selfagent.learn.preference import (BETA, SFT_WEIGHT, agreement, preference_loss,  # noqa: E402
                                        sequence_logprob)
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.optim import AdamW, clip_grad_norm  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402
from train_generator import load_split  # noqa: E402

PARTS = ("source", "keep", "chosen", "rejected")


def fingerprint(*parts):
    """What the reference log-probabilities were computed from, short enough to store and compare."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()[:16]


def score(model, split, batch_size, progress=""):
    """Mean log-probability of both answers of every row. Two arrays of one float per row.

    Batched through the same forward pass the training step uses, so the reference numbers and the policy
    numbers are produced by identical code -- a difference between them then means the weights moved and
    nothing else.
    """
    source, keep, chosen_ids, rejected_ids = split
    chosen, rejected = np.zeros(len(source)), np.zeros(len(source))
    started = time.monotonic()
    with no_grad():
        for start in range(0, len(source), batch_size):
            stop = start + batch_size
            for into, answer in ((chosen, chosen_ids), (rejected, rejected_ids)):
                scored = sequence_logprob(model, source[start:stop], answer[start:stop], keep[start:stop])
                into[start:stop] = [one.item() for one in scored]
            if progress and start and start % (batch_size * 250) == 0:
                done = start / len(source)
                print(f"  {progress} {start:,}/{len(source):,}  "
                      f"{(time.monotonic() - started) / max(done, 1e-9) / 60:.0f} min total")
    return chosen, rejected


def reference_of(model, split, batch_size, cache, checkpoint):
    """The frozen checkpoint's log-probabilities for this split, computed once and reused after that."""
    stamp = fingerprint(*split, checkpoint.read_bytes())
    if cache.exists():
        stored = np.load(cache)
        if str(stored["stamp"]) != stamp:
            raise SystemExit(
                f"{cache} was computed from a different dataset or checkpoint ({stored['stamp']}, now "
                f"{stamp}); delete it to recompute, and do not train against it -- the reference anchor "
                f"would point at a model these weights never were"
            )
        return stored["chosen"], stored["rejected"]
    chosen, rejected = score(model, split, batch_size, progress=cache.stem)
    np.savez(cache, chosen=chosen, rejected=rejected, stamp=stamp)
    return chosen, rejected


def writes(model, tokenizer, split, rows, batch_size, seed):
    """(unsupported figure share, repeated 4-gram share, supported figures per answer) on held-out questions
    the model answers itself.

    The only signal here that catches a model winning every comparison and losing English, so it is what
    selects a checkpoint. Neither of the cheap numbers can do that job: measured on the plain DPO run, loss
    and agreement both improved monotonically while generation collapsed.

    Counted by `check_answers.tally`, the same function the report and the C6 gate read, so a checkpoint is
    not selected on one definition of an unsupported figure and reported on another. One fixed sampling draw
    per call, so two evaluations differ by the weights and not by the dice.
    """
    scored = answer_rows(model, tokenizer, split[:3], np.arange(min(rows, len(split[0]))),
                         np.random.default_rng(seed), batch_size=batch_size)
    counts = tally(scored)
    repeated = float(np.mean([repeated_fraction(answer) for _, _, _, answer, _ in scored]))
    # The third number is what stops a rate being won by silence: fewer figures lowers the share too.
    supported = (counts["figures"] - counts["unsupported_figures"]) / max(len(scored), 1)
    return counts["unsupported_figures"] / max(counts["figures"], 1), repeated, supported


def measure(model, split, reference, batch_size, batches, beta, sft):
    """(agreement, loss, mean margin in nats) on the first `batches` batches of a held-out split."""
    source, keep, chosen_ids, rejected_ids = split
    reference_chosen, reference_rejected = reference
    limit = min(len(source), batches * batch_size)
    was_training = model.training
    model.eval()
    chosen, rejected, total, counted = [], [], 0.0, 0
    with no_grad():
        for start in range(0, limit, batch_size):
            stop = start + batch_size
            taken = sequence_logprob(model, source[start:stop], chosen_ids[start:stop], keep[start:stop])
            refused = sequence_logprob(model, source[start:stop], rejected_ids[start:stop], keep[start:stop])
            total += preference_loss(taken, refused, reference_chosen[start:stop],
                                     reference_rejected[start:stop], beta, sft).item()
            counted += 1
            chosen += [one.item() for one in taken]
            rejected += [one.item() for one in refused]
    model.train(was_training)
    return agreement(chosen, rejected), total / max(counted, 1), float(np.mean(chosen) - np.mean(rejected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="preference", help="prefix of the .npy split files")
    parser.add_argument("--checkpoint", default="generator.npz",
                        help="the trained generator this starts from and is anchored to; not the "
                             "pretrained encoder, which has never written an answer to be preferred")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    # An order of magnitude below the supervised 1e-4: the objective is a shift away from the reference
    # rather than a fit to a target, and a large step spends the model's fluency to win one comparison.
    # Provisional -- what would settle it is held-out agreement across a sweep.
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=BETA,
                        help="how hard one comparison presses against staying near the reference")
    parser.add_argument("--sft-weight", type=float, default=SFT_WEIGHT,
                        help="what keeping the preferred answer writable is worth; 0 is plain DPO, which "
                             "on this corpus won the pairs and stopped writing English")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="change only how the model writes, not how it reads the passages")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=100,
                        help="held-out batches per periodic check; the final number uses all of them")
    parser.add_argument("--eval-rows", type=int, default=64,
                        help="held-out questions the model answers itself at each check, which is what "
                             "selects the checkpoint")
    # Gold answers repeat 1.2% of their own 4-grams and the frozen generator 1.7%; the run that collapsed
    # repeated 30.8%. A ceiling here well above the healthy pair and far below collapse only has to separate
    # the two, and it has to exist: a model that has stopped writing states no figures at all, so the
    # unsupported-figure rate alone would score the collapse perfect.
    parser.add_argument("--max-repeat", type=float, default=0.05,
                        help="a checkpoint repeating more 4-grams than this is not eligible, however good "
                             "its other numbers look")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    train = load_split(artifacts, args.dataset, "train", PARTS)
    validation = load_split(artifacts, args.dataset, "validation", PARTS)

    starting = artifacts / args.checkpoint
    config, weights = pretrained.load(starting)
    model = GroundedGenerator(config)
    model.load_state_dict(weights)
    # Both passes run with dropout off, so `model.train()` is never called: the invariant that the ranking
    # term costs exactly ln 2 at step 0 is worth more here than the regularisation, on one epoch over 22k
    # pairs.
    model.eval()
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    if args.freeze_encoder:
        model.freeze_encoder()

    run = args.run_name or args.dataset
    print(f"starting from {starting}, {sum(p.size for p in model.trainable_parameters()):,} of "
          f"{sum(p.size for p in model.parameters()):,} parameters training")
    print(f"{len(train[0]):,} judged pairs, {len(validation[0]):,} held out, beta {args.beta}, "
          f"sft {args.sft_weight}, lr {args.learning_rate}")

    print("scoring the frozen reference")
    reference = reference_of(model, train, args.batch_size,
                             artifacts / f"{run}_reference_train.npz", starting)
    held_reference = reference_of(model, validation, args.batch_size,
                                 artifacts / f"{run}_reference_validation.npz", starting)
    # The frozen model's own agreement, which is the baseline every later number is read against. It comes
    # free: these are the reference log-probabilities, and the model has not moved yet. Held out, because
    # that is the split the trained number is measured on and the two have to describe the same rows.
    scored = min(len(validation[0]), args.eval_batches * args.batch_size)
    before = agreement(held_reference[0][:scored], held_reference[1][:scored])
    print(f"  frozen agreement {before:.1%} on the {scored:,} held-out pairs the run is judged on, "
          f"{agreement(*reference):.1%} on the training pairs")
    # The frozen model's own writing, so a later checkpoint is compared against what this run started from
    # rather than against a number from some other sample of rows.
    unsupported_before, repeated_before, _ = writes(model, tokenizer, validation, args.eval_rows,
                                                 args.batch_size, config.seed)
    print(f"  frozen writing: {unsupported_before:.1%} of figures unsupported, "
          f"{repeated_before:.1%} repeated 4-grams on {args.eval_rows} answers")

    reference_chosen, reference_rejected = reference
    optimizer = AdamW(model.trainable_parameters(), lr=args.learning_rate)
    steps_per_epoch = len(train[0]) // args.batch_size
    print(f"{args.epochs} epochs x {steps_per_epoch:,} steps at batch {args.batch_size}; at step 0 the "
          f"ranking half of a pair the model already has right costs ln 2 = {math.log(2):.4f}")

    source, keep, chosen_ids, rejected_ids = train
    rng = np.random.default_rng(config.seed)
    step = 0
    best, best_step = math.inf, 0
    started = time.monotonic()

    for epoch in range(args.epochs):
        order = rng.permutation(len(source))
        for index in range(steps_per_epoch):
            rows = order[index * args.batch_size : (index + 1) * args.batch_size]
            optimizer.zero_grad()
            taken = sequence_logprob(model, source[rows], chosen_ids[rows], keep[rows])
            refused = sequence_logprob(model, source[rows], rejected_ids[rows], keep[rows])
            loss = preference_loss(taken, refused, reference_chosen[rows], reference_rejected[rows],
                                   args.beta, args.sft_weight)
            loss.backward()
            clip_grad_norm(optimizer.parameters, 1.0)
            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.monotonic() - started
                left = (args.epochs * steps_per_epoch - step) * elapsed / step / 3600
                won = agreement([one.item() for one in taken], [one.item() for one in refused])
                print(f"step {step:,}/{args.epochs * steps_per_epoch:,}  loss {loss.item():.6f}  "
                      f"batch agreement {won:.0%}  {step / elapsed:.2f} step/s  {left:.1f}h left")
            if step % args.eval_every == 0:
                agreed, held, margin = measure(model, validation, held_reference, args.batch_size,
                                               args.eval_batches, args.beta, args.sft_weight)
                unsupported, repeated, _ = writes(model, tokenizer, validation, args.eval_rows,
                                               args.batch_size, config.seed)
                print(f"  held out: agreement {agreed:.1%} (from {before:.1%})  loss {held:.6f}  "
                      f"margin {margin:+.3f} nats")
                print(f"  writing: {unsupported:.1%} of figures unsupported (from "
                      f"{unsupported_before:.1%})  {repeated:.1%} repeated (from {repeated_before:.1%})")
                if repeated > args.max_repeat:
                    print(f"  not eligible: repeating {repeated:.1%} of its own 4-grams, over the "
                          f"{args.max_repeat:.1%} ceiling")
                elif unsupported < best:
                    best, best_step = unsupported, step
                    pretrained.save(artifacts / f"{run}.best.npz", model, config)
            if step % args.checkpoint_every == 0:
                pretrained.save(artifacts / f"{run}.npz", model, config)

    pretrained.save(artifacts / f"{run}.npz", model, config)
    # Every held-out pair for the number that gets quoted, not the sample the periodic checks use: 800 rows
    # carry a couple of points of noise, which is the size of the effect being claimed.
    whole = len(validation[0])
    agreed, held, margin = measure(model, validation, held_reference, args.batch_size, whole, args.beta,
                                   args.sft_weight)
    frozen = agreement(*held_reference)
    unsupported, repeated, _ = writes(model, tokenizer, validation, args.eval_rows, args.batch_size,
                                   config.seed)
    print(f"\nfinal agreement {agreed:.1%} on all {whole:,} held-out pairs, from {frozen:.1%} frozen; "
          f"loss {held:.6f}, margin {margin:+.3f} nats")
    print(f"final writing: {unsupported:.1%} of figures unsupported (from {unsupported_before:.1%}), "
          f"{repeated:.1%} repeated 4-grams (from {repeated_before:.1%})")
    if best_step:
        print(f"saved {artifacts}/{run}.npz; best unsupported-figure rate {best:.1%} at step {best_step:,} "
              f"in {run}.best.npz")
    else:
        print(f"saved {artifacts}/{run}.npz; no checkpoint was eligible -- every one repeated more than "
              f"{args.max_repeat:.1%} of its 4-grams, so this run has nothing to promote")


if __name__ == "__main__":
    sys.exit(main())
