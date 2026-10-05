"""Keep the app's state (portfolio database and data caches) in a private Hugging Face
dataset, so a free host whose disk is wiped on every restart keeps your portfolio.

Set STOCKINTEL_STATE_REPO (for example "you/stockintel-state") and HF_TOKEN (a write token).
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
import threading
from contextlib import closing
from pathlib import Path
from typing import List, Tuple

from .data.panel import CACHE_DIR

log = logging.getLogger("stockintel.state")
CACHE_PATTERNS = ("*.json", "*.csv", "*.txt", "panel_*.pkl")
DB_NAME = "stockintel.db"
# provisional: bounds what a crash can lose while staying far below the Hub's commit limits
PUSH_MINUTES = float(os.environ.get("STOCKINTEL_STATE_MINUTES", "10"))


def repo() -> str:
    return os.environ.get("STOCKINTEL_STATE_REPO", "")


def _token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(f"STOCKINTEL_STATE_REPO is {repo()!r} but HF_TOKEN is not set; "
                           "create a write token at huggingface.co/settings/tokens")
    return token


def cache_files(cache_dir: Path = CACHE_DIR) -> List[Path]:
    return sorted({p for pattern in CACHE_PATTERNS for p in cache_dir.glob(pattern)})


def snapshot(db: Path, out: Path, cache_dir: Path = CACHE_DIR) -> None:
    """A consistent copy of the database (SQLite's backup API is safe while the app
    writes, and includes the WAL) plus the cache files."""
    (out / "cache").mkdir(parents=True, exist_ok=True)
    if db.exists():
        with closing(sqlite3.connect(db)) as src, closing(sqlite3.connect(out / DB_NAME)) as dst:
            src.backup(dst)
    for f in cache_files(cache_dir):
        shutil.copy2(f, out / "cache" / f.name)


def push(db: Path, cache_dir: Path = CACHE_DIR) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    api.create_repo(repo(), repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot(db, Path(tmp), cache_dir)
        info = api.upload_folder(repo_id=repo(), repo_type="dataset", folder_path=tmp, commit_message="StockIntel state")
    log.info("saved state to %s (%s)", repo(), info.oid)
    return info.oid


def pull(db: Path, cache_dir: Path = CACHE_DIR, overwrite: bool = False) -> List[str]:
    """Copy the saved state into place and return what was restored. Local files are
    kept unless `overwrite`, so pulling on your own computer never clobbers it by accident."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError
    with tempfile.TemporaryDirectory() as tmp:
        try:
            src = Path(snapshot_download(repo(), repo_type="dataset", token=_token(), local_dir=tmp))
        except RepositoryNotFoundError:
            log.warning("no saved state at %s yet (first run, or HF_TOKEN cannot read it)", repo())
            return []
        pairs: List[Tuple[Path, Path]] = [(src / DB_NAME, db)] + [(f, cache_dir / f.name) for f in (src / "cache").glob("*")]
        restored = []
        for a, b in pairs:
            if a.exists() and (overwrite or not b.exists()):
                b.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(a, b)
                restored.append(b.name)
    log.info("restored %s from %s", ", ".join(restored) or "nothing", repo())
    return restored


class Autosave:
    """Pushes the state every `minutes` when it has changed, and once more on stop."""

    def __init__(self, db: Path, minutes: float, cache_dir: Path = CACHE_DIR):
        self.db, self.cache_dir, self.every = db, cache_dir, minutes * 60
        self._stop = threading.Event()
        self._saved = self._signature()
        self._thread = threading.Thread(target=self._run, daemon=True, name="state-autosave")

    def _signature(self) -> tuple:
        files = [self.db, self.db.with_name(self.db.name + "-wal"), *cache_files(self.cache_dir)]
        return tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files if f.exists())

    def start(self) -> "Autosave":
        self._thread.start()
        return self

    def save_if_changed(self) -> bool:
        sig = self._signature()
        if sig == self._saved:
            return False
        push(self.db, self.cache_dir)
        self._saved = sig
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            try:
                self.save_if_changed()
            except Exception:       # a background thread cannot raise to anyone; retry next tick
                log.exception("saving state to %s failed; will retry in %.0f min", repo(), self.every / 60)

    def stop(self) -> None:
        self._stop.set()
        self.save_if_changed()
