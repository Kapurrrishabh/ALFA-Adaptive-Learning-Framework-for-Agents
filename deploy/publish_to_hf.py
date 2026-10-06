"""Publish StockIntel's trained models and the model Space to your Hugging Face account.

Everything is created private. Run after `hf auth login` (write token) and after pushing this
repository to GitHub, because the Space installs StockIntel from there:

    .venv/bin/python deploy/publish_to_hf.py --alfa "/path/to/ALFA-project"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware, get_token

HERE = Path(__file__).resolve().parent
MODELS = Path.home() / ".stockintel" / "models"
# the ALFA files the return generator needs: its loader, the selfagent package, its weights
ALFA_CODE = ("backend/__init__.py", "backend/models/__init__.py", "backend/models/price.py",
             "backend/models/registry.py", "backend/models/scenarios.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alfa", required=True, type=Path, help="the ALFA project folder")
    ap.add_argument("--space", default="stockintel-models")
    args = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    repos = {"chronos": f"{user}/stockintel-chronos2-nse", "kronos": f"{user}/stockintel-kronos-nse",
             "alfa": f"{user}/alfa-return-generator"}
    space = f"{user}/{args.space}"
    sources = {"chronos": MODELS / "chronos-2-nse", "kronos": MODELS / "kronos-nse"}
    missing = [str(p) for p in [*sources.values(), args.alfa / "artifacts/returns.npz", args.alfa / "selfagent"] if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)} (train with `stockintel train-chronos` / `train-kronos`, "
                         "and point --alfa at the ALFA project)")

    for repo in repos.values():
        api.create_repo(repo, private=True, exist_ok=True)
    for name, folder in sources.items():
        api.upload_folder(repo_id=repos[name], folder_path=folder, commit_message=f"{name} fine-tuned on NSE")
        print("uploaded", repos[name])
    api.upload_file(path_or_fileobj=args.alfa / "artifacts/returns.npz", path_in_repo="returns.npz", repo_id=repos["alfa"])
    print("uploaded", repos["alfa"])

    api.create_repo(space, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=HERE / "hf-models-space", commit_message="model Space")
    api.upload_folder(repo_id=space, repo_type="space", folder_path=args.alfa / "selfagent", path_in_repo="alfa/selfagent",
                      ignore_patterns=["**/__pycache__/**"], commit_message="ALFA selfagent package")
    for rel in ALFA_CODE:
        api.upload_file(path_or_fileobj=args.alfa / rel, path_in_repo=f"alfa/{rel}", repo_id=space, repo_type="space")
    for key, value in (("STOCKINTEL_CHRONOS_MODEL", repos["chronos"]), ("STOCKINTEL_KRONOS_MODEL", repos["kronos"]),
                       ("ALFA_RETURNS_REPO", repos["alfa"])):
        api.add_space_variable(space, key, value)
    api.add_space_secret(space, "HF_TOKEN", get_token())   # the Space reads the private model repos with it
    api.request_space_hardware(space, SpaceHardware.ZERO_A10G)
    print(f"Space {space} is building on ZeroGPU: https://huggingface.co/spaces/{space}\n"
          f"Point the app at it with STOCKINTEL_MODEL_SPACE={space} (and HF_TOKEN, since it is private).")


if __name__ == "__main__":
    main()
