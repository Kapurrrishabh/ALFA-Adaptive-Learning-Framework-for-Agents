"""Publish the trained models and the model Space to your Hugging Face account.

Everything is created private. Run after `hf auth login` (write token) and after pushing this
repository to GitHub, because the Space installs the backend package from there. The Space gets
its own read-only token, since it only reads the model repos:

    cd backend && HF_SPACE_TOKEN=<read token> .venv/bin/python ../deploy/publish_to_hf.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware

from backend.models.serving import weights
from backend.paths import ARTIFACTS

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", default="stockintel-models")
    args = ap.parse_args()
    space_token = os.environ.get("HF_SPACE_TOKEN")
    if not space_token:
        raise SystemExit("set HF_SPACE_TOKEN to a read-only token from huggingface.co/settings/tokens; "
                         "the Space uses it to read the private model repos")
    api = HfApi()
    user = api.whoami()["name"]
    repos = {"chronos": f"{user}/stockintel-chronos2-nse", "kronos": f"{user}/stockintel-kronos-nse",
             "alfa": f"{user}/alfa-weights"}
    space = f"{user}/{args.space}"
    sources = {"chronos": ARTIFACTS / "external" / "chronos-2-nse", "kronos": ARTIFACTS / "external" / "kronos-nse"}
    missing = [str(p) for p in [*sources.values(), *weights.PUBLISHED] if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)} (train with `stockintel train-chronos` / `train-kronos`, "
                         "models/training/train_return_generator.py and train_path_gru.py)")

    for repo in repos.values():
        api.create_repo(repo, private=True, exist_ok=True)
    for name, folder in sources.items():
        api.upload_folder(repo_id=repos[name], folder_path=folder, commit_message=f"{name} fine-tuned on NSE")
        print("uploaded", repos[name])
    for f in weights.PUBLISHED:
        api.upload_file(path_or_fileobj=f, path_in_repo=f.name, repo_id=repos["alfa"])
    print("uploaded", repos["alfa"])

    # free accounts may host Gradio Spaces only on ZeroGPU, so the hardware is set when the Space is created
    api.create_repo(space, repo_type="space", space_sdk="gradio", space_hardware=SpaceHardware.ZERO_A10G, private=True,
                    exist_ok=True)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=HERE / "hf-models-space", commit_message="model Space")
    for key, value in (("STOCKINTEL_CHRONOS_MODEL", repos["chronos"]), ("STOCKINTEL_KRONOS_MODEL", repos["kronos"]),
                       ("ALFA_MODELS_REPO", repos["alfa"])):
        api.add_space_variable(space, key, value)
    api.add_space_secret(space, "HF_TOKEN", space_token)
    print(f"Space {space} is building on ZeroGPU: https://huggingface.co/spaces/{space}\n"
          f"On Render set STOCKINTEL_MODEL_SPACE={space} and ALFA_MODELS_REPO={repos['alfa']} (and HF_TOKEN).")


if __name__ == "__main__":
    main()
