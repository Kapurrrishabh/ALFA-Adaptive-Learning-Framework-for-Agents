"""Model configuration. One declared shape for the whole system.

Sizes here are provisional until scripts/benchmark_step.py measures real step time on the
target machine; see docs/PLAN_OF_ACTION.md section 4.
"""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8000
    max_text_length: int = 128
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1

    price_channels: int = 5
    price_window: int = 128
    patch_length: int = 16
    patch_stride: int = 8
    price_layers: int = 2

    # An answer is a few sentences, so it needs far less room than the passages it is grounded in,
    # and the decoder is kept shallower than the encoder: reading is the harder half of the job.
    # 192 because the answers we train on are 167 tokens at the median (wordpiece fertility 1.47).
    # At 64 only 15% of answers survived whole; at 192 it is 56%, and targets end at a real sentence.
    max_answer_length: int = 192
    decoder_layers: int = 2

    num_classes: int = 3
    seed: int = 0

    def __post_init__(self):
        if self.dim % self.num_heads:
            raise ValueError(f"dim {self.dim} is not divisible by num_heads {self.num_heads}")
        if self.patch_stride > self.patch_length:
            raise ValueError(
                f"patch_stride {self.patch_stride} skips timesteps between patches; "
                f"it must be at most patch_length {self.patch_length}"
            )
        if self.price_window < self.patch_length:
            raise ValueError(
                f"price_window {self.price_window} is shorter than one patch of "
                f"{self.patch_length}"
            )

    @property
    def num_patches(self):
        return (self.price_window - self.patch_length) // self.patch_stride + 1

    @property
    def fingerprint(self):
        """Recorded in every checkpoint and results file so numbers stay traceable."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    @property
    def encoder_fingerprint(self):
        """Just the fields that set the shape of an encoder weight.

        Reusing a pretrained encoder only requires these to agree. Comparing the whole config instead
        rejects a perfectly good checkpoint because an unrelated field like max_answer_length moved,
        and the only way to satisfy it would be to repeat a seven-hour run that changes nothing.
        """
        shaping = ("vocab_size", "max_text_length", "dim", "num_heads", "num_layers", "ffn_dim")
        values = {name: getattr(self, name) for name in shaping}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path):
        with open(path) as handle:
            return cls(**json.load(handle))

    def save(self, path):
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)
