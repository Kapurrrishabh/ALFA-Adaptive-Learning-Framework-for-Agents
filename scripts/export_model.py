"""Turns a training checkpoint into the model artifact later phases load.

The checkpoint is written for resuming and is mostly optimizer state. This drops that, keeps the
weights and the config, and then proves the result by reloading it and predicting a masked token,
so a broken export is caught here rather than in Phase 4.

    python3 scripts/export_model.py --artifacts artifacts
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent import pretrained  # noqa: E402
from selfagent.autograd import no_grad  # noqa: E402
from selfagent.config import ModelConfig  # noqa: E402
from selfagent.models import MaskedLanguageModel  # noqa: E402
from selfagent.tokenizer.vocab import CLS_ID, MASK_ID, SEP_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402

# Shown after export so the numbers are eyeballable: the model fills the blank in each of these.
_PROMPTS = (
    "the reserve bank raised the [MASK] rate by 25 basis points",
    "revenue [MASK] 12 percent compared with the prior year",
    "the company reported a net [MASK] for the quarter",
)
_TOP = 5


def weights_from_checkpoint(path):
    stored = np.load(path)
    prefix = pretrained.CHECKPOINT_WEIGHT_PREFIX
    weights = {
        key.removeprefix(prefix): stored[key] for key in stored.files if key.startswith(prefix)
    }
    if not weights:
        raise SystemExit(f"{path} holds no '{prefix}' arrays; it is not a training checkpoint")
    return weights, int(stored["step"])


def fill_blank(model, tokenizer, prompt, length):
    """The model's top guesses for the [MASK] in `prompt`."""
    pieces = [CLS_ID]
    for word in prompt.split():
        pieces.extend([MASK_ID] if word == "[MASK]" else tokenizer.encode_word(word))
    position = pieces.index(MASK_ID)
    pieces = (pieces + [SEP_ID])[:length]
    padded = pieces + [0] * (length - len(pieces))

    with no_grad():
        logits = model(np.asarray([padded]))
    ranked = np.argsort(-logits.data[0, position])[:_TOP]
    return [tokenizer.pieces[index] for index in ranked]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--checkpoint", default=None, help="defaults to <artifacts>/pretrain.npz")
    parser.add_argument("--out", default=None, help="defaults to <artifacts>/pretrained.npz")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    checkpoint = Path(args.checkpoint) if args.checkpoint else artifacts / "pretrain.npz"
    out = Path(args.out) if args.out else artifacts / "pretrained.npz"

    config = ModelConfig.load(artifacts / "pretrain_config.json")
    weights, step = weights_from_checkpoint(checkpoint)
    model = MaskedLanguageModel(config)
    model.load_state_dict(weights)
    pretrained.save(out, model, config)
    print(
        f"exported step {step:,} -> {out} "
        f"({out.stat().st_size / 1e6:.1f} MB, from {checkpoint.stat().st_size / 1e6:.1f} MB)"
    )

    reloaded_config, reloaded_weights = pretrained.load(out)
    reloaded = MaskedLanguageModel(reloaded_config)
    reloaded.load_state_dict(reloaded_weights)
    reloaded.eval()
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    print(f"reloaded {reloaded_config.fingerprint}, filling blanks:")
    for prompt in _PROMPTS:
        guesses = fill_blank(reloaded, tokenizer, prompt, reloaded_config.max_text_length)
        print(f"  {prompt}\n    -> {', '.join(guesses)}")


if __name__ == "__main__":
    main()
