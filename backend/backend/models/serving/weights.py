"""Our NumPy models' weights in a private Hugging Face model repo (ALFA_MODELS_REPO, with HF_TOKEN), so a
host whose disk is wiped on every restart fetches them on start. deploy/publish_to_hf.py uploads them.

The self-learning agent's files travel the same way in their own repo (ALFA_AGENT_REPO), fetched once:

    python -m backend.models.serving.weights agent
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from backend.models.serving import alfa_fan, gru_line
from backend.paths import ARTIFACTS, DATA

PUBLISHED = (alfa_fan.WEIGHTS, gru_line.WEIGHTS)

# What the agent server reads on start, traced from models/agent/run/serve.py: its two generators and their
# tokenizer and registry, the routing cut, the week-ahead head, the return model, the retrieval index, and
# the feedback its answer cut is refitted from. Then its price files, SEC's symbol table and the exchange's
# constituent list its Indian company names come from.
AGENT_ARTIFACTS = ("registry.json", "tokenizer.json", "advisory_combined.npz", "generator.npz", "route_gate.json",
                   "price_head.npz", "returns.npz", "reference_index.npz", "reference_sources.json",
                   "reference_vectors_bge.npy",
                   "route_gate_bge.json", "served.sqlite", "served_verdicts.jsonl")
AGENT_DATA = ("prices", "raw/sec_edgar/company_tickers.json", "cache/nifty500.csv")


def repo() -> str:
    return os.environ.get("ALFA_MODELS_REPO", "")


def pull(files: Iterable[Path]) -> None:
    from huggingface_hub import hf_hub_download
    for f in files:
        hf_hub_download(repo(), f.name, local_dir=f.parent)


def agent_files():
    """(path on this machine, path in the agent repo) for everything in AGENT_ARTIFACTS and AGENT_DATA."""
    pairs = [(ARTIFACTS / name, f"artifacts/{name}") for name in AGENT_ARTIFACTS]
    for name in AGENT_DATA:
        local = DATA / name
        pairs += ([(f, f"data/{f.relative_to(DATA)}") for f in sorted(local.glob("*.csv"))] if local.is_dir()
                  else [(local, f"data/{name}")])
    return pairs


def pull_agent(agent_repo: str) -> list:
    """Copy the agent repo's files into place, keeping any already here: the feedback the deployment has
    learned since must not be replaced by the copy it started from. Returns what was copied."""
    from huggingface_hub import snapshot_download
    copied = []
    with tempfile.TemporaryDirectory() as tmp:
        held = Path(snapshot_download(agent_repo, local_dir=tmp))
        for src in sorted(p for p in held.rglob("*") if p.is_file() and not p.relative_to(held).parts[0].startswith(".")):
            top, rest = src.relative_to(held).parts[0], Path(*src.relative_to(held).parts[1:])
            dst = (ARTIFACTS if top == "artifacts" else DATA) / rest
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(str(rest))
    return copied


if __name__ == "__main__":
    if sys.argv[1:] != ["agent"]:
        raise SystemExit("usage: python -m backend.models.serving.weights agent")
    agent_repo = os.environ.get("ALFA_AGENT_REPO", "")
    if not agent_repo:
        raise SystemExit("set ALFA_AGENT_REPO (e.g. you/alfa-agent) and HF_TOKEN; deploy/publish_to_hf.py creates it")
    copied = pull_agent(agent_repo)
    print(f"agent files: {len(copied)} copied from {agent_repo}, the rest already here")
