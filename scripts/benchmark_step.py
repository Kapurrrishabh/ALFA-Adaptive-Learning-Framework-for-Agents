"""Measures forward+backward step time so configs/target.json rests on a number.

Run this on the machine that will do the pretraining. The provisional sizes in
docs/PLAN_OF_ACTION.md section 4 are guesses until this has been run.

    python3 scripts/benchmark_step.py --token-budget 15000000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent.autograd import functional as F  # noqa: E402
from selfagent.config import ModelConfig  # noqa: E402
from selfagent.models import MaskedLanguageModel  # noqa: E402
from selfagent.optim import AdamW, clip_grad_norm  # noqa: E402

CANDIDATES = {
    "tiny": dict(num_layers=2, dim=128, num_heads=4, ffn_dim=512),
    "small": dict(num_layers=4, dim=256, num_heads=4, ffn_dim=1024),
    "medium": dict(num_layers=6, dim=384, num_heads=6, ffn_dim=1536),
    "large": dict(num_layers=8, dim=512, num_heads=8, ffn_dim=2048),
}

WARMUP_STEPS = 2
TIMED_STEPS = 5


def time_one_config(name, overrides, batch, sequence_length, vocab_size):
    config = ModelConfig(
        vocab_size=vocab_size, max_text_length=sequence_length, dropout=0.1, **overrides
    )
    rng = np.random.default_rng(0)
    model = MaskedLanguageModel(config)
    optimizer = AdamW(model.trainable_parameters(), lr=3e-4)
    token_ids = rng.integers(0, vocab_size, (batch, sequence_length))
    targets = token_ids.reshape(-1)

    def one_step():
        optimizer.zero_grad()
        logits = model(token_ids).reshape((-1, vocab_size))
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        clip_grad_norm(optimizer.parameters, 1.0)
        optimizer.step()

    for _ in range(WARMUP_STEPS):
        one_step()
    started = time.perf_counter()
    for _ in range(TIMED_STEPS):
        one_step()
    seconds_per_step = (time.perf_counter() - started) / TIMED_STEPS

    return {
        "name": name,
        "parameters": sum(p.size for p in model.parameters()),
        "seconds_per_step": seconds_per_step,
        "tokens_per_second": batch * sequence_length / seconds_per_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=15_000_000,
        help="tokens in the planned pretraining run, used to project total hours",
    )
    parser.add_argument("--only", nargs="*", choices=sorted(CANDIDATES), default=None)
    args = parser.parse_args()

    print(f"batch={args.batch} seq={args.sequence_length} vocab={args.vocab_size}")
    print(f"{'config':8} {'params':>12} {'s/step':>9} {'tok/s':>10} {'budget hours':>13}")

    for name in args.only or sorted(CANDIDATES, key=lambda k: list(CANDIDATES).index(k)):
        measured = time_one_config(
            name, CANDIDATES[name], args.batch, args.sequence_length, args.vocab_size
        )
        hours = args.token_budget / measured["tokens_per_second"] / 3600
        print(
            f"{name:8} {measured['parameters']:12,} {measured['seconds_per_step']:9.3f} "
            f"{measured['tokens_per_second']:10,.0f} {hours:13.1f}"
        )


if __name__ == "__main__":
    main()
