"""Publish the trained models and the model Space to your Hugging Face account.

Everything is created private. Run after `hf auth login` (write token) and after pushing this
repository to GitHub, because the Space installs the backend package from there. The Space gets
its own read-only token, since it only reads the model repos:

    cd backend && HF_SPACE_TOKEN=<read token> .venv/bin/python ../deploy/publish_to_hf.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, SpaceHardware

from backend.models.serving import weights
from backend.paths import ARTIFACTS

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", default="stockintel-models")
    ap.add_argument("--space-only", action="store_true", help="update the Space and leave the model repos as they are")
    args = ap.parse_args()
    commit = pushed_commit()
    space_token = os.environ.get("HF_SPACE_TOKEN")
    if not space_token:
        raise SystemExit("set HF_SPACE_TOKEN to a read-only token from huggingface.co/settings/tokens; "
                         "the Space uses it to read the private model repos")
    api = HfApi()
    user = api.whoami()["name"]
    repos = {"chronos": f"{user}/stockintel-chronos2-nse", "kronos": f"{user}/stockintel-kronos-nse",
             "alfa": f"{user}/alfa-weights", "agent": f"{user}/alfa-agent"}
    space = f"{user}/{args.space}"
    sources = {"chronos": ARTIFACTS / "external" / "chronos-2-nse", "kronos": ARTIFACTS / "external" / "kronos-nse"}
    missing = [str(p) for p in [*sources.values(), *weights.PUBLISHED, *(f for f, _ in weights.agent_files())]
               if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)} (train with `stockintel train-chronos` / `train-kronos`, "
                         "models/training/train_return_generator.py and train_path_gru.py)")

    if not args.space_only:
        publish_models(api, repos, sources)

    # free accounts may host Gradio Spaces only on ZeroGPU, so the hardware is set when the Space is created
    api.create_repo(space, repo_type="space", space_sdk="gradio", space_hardware=SpaceHardware.ZERO_A10G, private=True,
                    exist_ok=True)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=HERE / "hf-models-space",
                      ignore_patterns=["requirements.txt"], commit_message=f"model Space at {commit[:7]}")
    api.upload_file(path_or_fileobj=pinned_requirements(commit).encode(), path_in_repo="requirements.txt",
                    repo_id=space, repo_type="space", commit_message=f"install the backend at {commit[:7]}")
    for key, value in (("STOCKINTEL_CHRONOS_MODEL", repos["chronos"]), ("STOCKINTEL_KRONOS_MODEL", repos["kronos"]),
                       ("ALFA_MODELS_REPO", repos["alfa"])):
        api.add_space_variable(space, key, value)
    api.add_space_secret(space, "HF_TOKEN", space_token)
    print(f"Space {space} is building on ZeroGPU: https://huggingface.co/spaces/{space}\n"
          f"On the server set STOCKINTEL_MODEL_SPACE={space}, ALFA_MODELS_REPO={repos['alfa']} and "
          f"ALFA_AGENT_REPO={repos['agent']} (and HF_TOKEN).")


def publish_models(api, repos, sources):
    for repo in repos.values():
        api.create_repo(repo, private=True, exist_ok=True)
    for name, folder in sources.items():
        api.upload_folder(repo_id=repos[name], folder_path=folder, commit_message=f"{name} fine-tuned on NSE")
        print("uploaded", repos[name])
    for f in weights.PUBLISHED:
        api.upload_file(path_or_fileobj=f, path_in_repo=f.name, repo_id=repos["alfa"])
    print("uploaded", repos["alfa"])
    api.create_commit(repos["agent"], [CommitOperationAdd(path_in_repo=held, path_or_fileobj=f)
                                       for f, held in weights.agent_files()], commit_message="the self-learning agent")
    print("uploaded", repos["agent"])


def pushed_commit():
    """The commit this checkout is at, refused unless GitHub's main is the same one: the Space installs from there."""
    here = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    remote = subprocess.run(["git", "ls-remote", "origin", "refs/heads/main"], cwd=HERE, capture_output=True, text=True,
                            check=True).stdout.split()
    if not remote or remote[0] != here:
        raise SystemExit(f"GitHub's main is at {remote[0][:7] if remote else 'nothing'} but this checkout is at {here[:7]}; "
                         "push first: git push origin major-project:main")
    return here


def pinned_requirements(commit):
    """The Space's requirements with the backend pinned to `commit`. Against a moving "@main" the Space's build
    cache kept the previous install, so new app code ran against old backend code and crashed on start."""
    text = (HERE / "hf-models-space" / "requirements.txt").read_text()
    if "@main#" not in text:
        raise SystemExit("hf-models-space/requirements.txt no longer installs the backend from @main; update pinned_requirements")
    return text.replace("@main#", f"@{commit}#")


if __name__ == "__main__":
    main()
