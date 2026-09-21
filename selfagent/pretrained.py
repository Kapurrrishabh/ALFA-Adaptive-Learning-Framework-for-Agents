"""Saving a trained model and loading it back.

Two different artifacts serve two different needs. A training checkpoint carries the optimizer
moments so an interrupted run can resume; this one carries weights and the config that shaped
them, and nothing else. Later phases load this, not the checkpoint, and the optimizer state is
dead weight to them: it is five times the size of the model.

The config travels inside the file. A loader that has to be told the shapes separately is a loader
that will one day be told the wrong ones.
"""

import json
from dataclasses import asdict

from .backend import xp
from .config import ModelConfig

_METADATA = ("config", "fingerprint")

# Training checkpoints prefix their weights, to keep them apart from the optimizer moments stored
# alongside. Declared here because both the training script and the exporter have to agree on it.
CHECKPOINT_WEIGHT_PREFIX = "weight/"


def save(path, model, config):
    weights = model.state_dict()
    clashing = sorted(set(weights) & set(_METADATA))
    if clashing:
        raise ValueError(f"parameter named {clashing} collides with this file's own metadata")
    xp.savez(
        path,
        config=json.dumps(asdict(config), sort_keys=True),
        fingerprint=config.fingerprint,
        **weights,
    )


def load(path):
    """Returns (config, weights) as stored. Rebuilding the model is the caller's job."""
    stored = xp.load(path)
    config = ModelConfig(**json.loads(str(stored["config"])))
    if config.fingerprint != str(stored["fingerprint"]):
        raise ValueError(
            f"{path} stores fingerprint {stored['fingerprint']} but its own config produces "
            f"{config.fingerprint}; the file was written by an incompatible version"
        )
    return config, {key: stored[key] for key in stored.files if key not in _METADATA}
