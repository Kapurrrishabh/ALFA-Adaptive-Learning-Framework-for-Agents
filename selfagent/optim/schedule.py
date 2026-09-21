"""Learning-rate schedule: linear warmup then cosine decay."""

import math


def warmup_cosine(step, total_steps, base_lr, warmup_fraction=0.05, final_fraction=0.1):
    """`step` is 0-based. Returns the lr to set on the optimizer before calling step()."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (final_fraction + (1.0 - final_fraction) * cosine)
