"""Our NumPy models' weights in a private Hugging Face model repo (ALFA_MODELS_REPO, with HF_TOKEN), so a
host whose disk is wiped on every restart fetches them on start. deploy/publish_to_hf.py uploads them.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from backend.models.serving import alfa_fan, gru_line

PUBLISHED = (alfa_fan.WEIGHTS, gru_line.WEIGHTS)


def repo() -> str:
    return os.environ.get("ALFA_MODELS_REPO", "")


def pull(files: Iterable[Path]) -> None:
    from huggingface_hub import hf_hub_download
    for f in files:
        hf_hub_download(repo(), f.name, local_dir=f.parent)
