"""MLM pretraining over the finance corpus.

Reads what scripts/prepare_pretrain.py cached, so this script only trains. Checkpoints hold the
optimizer state as well as the weights, which is what lets an interrupted run resume instead of
restarting: on this machine an epoch is hours, so a crash that costs the whole run is not
acceptable.

    python3 scripts/prepare_pretrain.py --corpus data/corpus
    python3 scripts/train_pretrain.py --epochs 2

S3 in docs/PLAN_OF_ACTION.md asks for MLM loss against the uniform baseline, ln(vocab_size), so
both are logged from the first evaluation onwards.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent.autograd import functional as F  # noqa: E402
from selfagent.backend import default_rng  # noqa: E402
from selfagent.config import ModelConfig  # noqa: E402
from selfagent.data.masking import mask_tokens  # noqa: E402
from selfagent.models import MaskedLanguageModel  # noqa: E402
from selfagent.optim import AdamW, clip_grad_norm, warmup_cosine  # noqa: E402
from selfagent import pretrained  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402

MAX_GRADIENT_NORM = 1.0


def batches(sequences, batch_size, rng):
    """Shuffled batches for one epoch. The last short batch is dropped to keep shapes fixed."""
    order = rng.permutation(len(sequences))
    for start in range(0, len(sequences) - batch_size + 1, batch_size):
        yield sequences[order[start : start + batch_size]].astype(np.int64)


def evaluate(model, sequences, continuation_ids, vocab_size, batch_size, batches_wanted):
    """Mean masked-token loss on held-out sequences, with masking fixed so runs compare."""
    model.eval()
    rng = default_rng(0)
    total, counted = 0.0, 0
    for index, batch in enumerate(batches(sequences, batch_size, rng)):
        if index >= batches_wanted:
            break
        inputs, labels = mask_tokens(batch, continuation_ids, vocab_size, rng)
        logits = model(inputs).reshape((-1, vocab_size))
        total += float(F.cross_entropy(logits, labels.reshape(-1)).data)
        counted += 1
    model.train()
    return total / max(1, counted)


def save_checkpoint(path, model, optimizer, config, step, epoch):
    """Written to a temporary name first, so an interrupt cannot leave a half-written file."""
    temporary = path.with_suffix(".partial")
    np.savez(
        temporary,
        step=step,
        epoch=epoch,
        learning_rate=optimizer.lr,
        adam_steps=optimizer.steps,
        fingerprint=config.fingerprint,
        **{
            f"{pretrained.CHECKPOINT_WEIGHT_PREFIX}{name}": value
            for name, value in model.state_dict().items()
        },
        **{f"moment/{index}": value for index, value in enumerate(optimizer._moment)},
        **{f"velocity/{index}": value for index, value in enumerate(optimizer._velocity)},
    )
    temporary.with_suffix(".partial.npz").replace(path)


def load_checkpoint(path, model, optimizer, config):
    stored = np.load(path)
    if str(stored["fingerprint"]) != config.fingerprint:
        raise SystemExit(
            f"{path} was written by a different model config "
            f"({stored['fingerprint']} vs {config.fingerprint}); delete it or match the config"
        )
    prefix = pretrained.CHECKPOINT_WEIGHT_PREFIX
    model.load_state_dict(
        {key.removeprefix(prefix): stored[key] for key in stored.files if key.startswith(prefix)}
    )
    for index in range(len(optimizer._moment)):
        optimizer._moment[index] = stored[f"moment/{index}"]
        optimizer._velocity[index] = stored[f"velocity/{index}"]
    optimizer.steps = int(stored["adam_steps"])
    return int(stored["step"]), int(stored["epoch"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    training = np.load(artifacts / "pretrain_train.npy")
    validation = np.load(artifacts / "pretrain_validation.npy")

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size, max_text_length=int(training.shape[1])
    )
    config.save(artifacts / "pretrain_config.json")

    model = MaskedLanguageModel(config)
    optimizer = AdamW(model.trainable_parameters(), lr=args.learning_rate)
    continuation_ids = tokenizer.continuation_ids

    steps_per_epoch = len(training) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    checkpoint_path = artifacts / "pretrain.npz"

    step, first_epoch = 0, 0
    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"--resume given but {checkpoint_path} does not exist")
        step, first_epoch = load_checkpoint(checkpoint_path, model, optimizer, config)
        print(f"resumed at step {step:,} of epoch {first_epoch + 1}")

    parameters = sum(p.size for p in model.parameters())
    print(
        f"{parameters:,} parameters, vocab {config.vocab_size:,}, "
        f"{len(training):,} train sequences, {len(validation):,} held out"
    )
    print(
        f"{args.epochs} epochs x {steps_per_epoch:,} steps = {total_steps:,} steps "
        f"at batch {args.batch_size}"
    )
    print(f"uniform baseline loss is ln({config.vocab_size}) = {math.log(config.vocab_size):.3f}")

    history = []
    rng = default_rng(config.seed)
    started = time.monotonic()
    # Rate and ETA count only the steps this run did. Dividing by all of them after a resume
    # reported 56,000 tok/s and one hour left on a six-hour run.
    resumed_at = step
    window_loss, window_steps = 0.0, 0

    for epoch in range(first_epoch, args.epochs):
        for batch in batches(training, args.batch_size, rng):
            if step >= total_steps:
                break
            optimizer.lr = warmup_cosine(step, total_steps, args.learning_rate)
            optimizer.zero_grad()

            inputs, labels = mask_tokens(batch, continuation_ids, config.vocab_size, rng)
            logits = model(inputs).reshape((-1, config.vocab_size))
            loss = F.cross_entropy(logits, labels.reshape(-1))
            loss.backward()
            clip_grad_norm(optimizer.parameters, MAX_GRADIENT_NORM)
            optimizer.step()

            window_loss += float(loss.data)
            window_steps += 1
            step += 1

            if step % args.log_every == 0:
                elapsed = time.monotonic() - started
                done = step - resumed_at
                rate = done * args.batch_size * config.max_text_length / elapsed
                remaining = (total_steps - step) * elapsed / done / 3600
                print(
                    f"step {step:,}/{total_steps:,}  loss {window_loss / window_steps:.4f}  "
                    f"lr {optimizer.lr:.2e}  {rate:,.0f} tok/s  {remaining:.1f}h left",
                    flush=True,
                )
                window_loss, window_steps = 0.0, 0

            if step % args.eval_every == 0:
                held_out = evaluate(
                    model,
                    validation,
                    continuation_ids,
                    config.vocab_size,
                    args.batch_size,
                    args.eval_batches,
                )
                print(
                    f"  validation loss {held_out:.4f}  perplexity {math.exp(held_out):,.1f}",
                    flush=True,
                )
                history.append({"step": step, "validation_loss": held_out})
                (artifacts / "pretrain_history.json").write_text(json.dumps(history, indent=2))

            if step % args.checkpoint_every == 0:
                save_checkpoint(checkpoint_path, model, optimizer, config, step, epoch)

    save_checkpoint(checkpoint_path, model, optimizer, config, step, args.epochs - 1)
    # Exported here rather than by hand afterwards, so a finished run always leaves the artifact
    # the later phases load, not just the checkpoint they cannot use.
    pretrained.save(artifacts / "pretrained.npz", model, config)
    final = evaluate(
        model, validation, continuation_ids, config.vocab_size, args.batch_size, args.eval_batches
    )
    print(f"done in {(time.monotonic() - started) / 3600:.2f}h")
    print(f"final validation loss {final:.4f}  perplexity {math.exp(final):,.1f}")


if __name__ == "__main__":
    main()
