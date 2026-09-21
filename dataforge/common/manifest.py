"""Provenance ledger. Every fetched file records where it came from and under what licence.

Also what makes collection resumable: a file already recorded with a matching size on disk is
not fetched again.
"""

import hashlib
import json
import time
from pathlib import Path


class Manifest:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._by_url = {}
        if self.path.exists():
            with self.path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._by_url[entry["url"]] = entry

    def already_have(self, url, destination):
        """True only if the ledger and the disk agree, so a truncated download is refetched."""
        entry = self._by_url.get(url)
        if entry is None:
            return False
        destination = Path(destination)
        return destination.exists() and destination.stat().st_size == entry["bytes"]

    def record(self, url, destination, source, licence_note):
        destination = Path(destination)
        payload = destination.read_bytes()
        entry = {
            "url": url,
            "path": str(destination),
            "source": source,
            "licence": licence_note,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        self._by_url[url] = entry
        return entry

    def entries_for(self, source):
        return [entry for entry in self._by_url.values() if entry["source"] == source]

    def sources(self):
        return sorted({entry["source"] for entry in self._by_url.values()})

    def all_entries(self):
        return list(self._by_url.values())

    def total_bytes(self):
        return sum(entry["bytes"] for entry in self._by_url.values())

    def __len__(self):
        return len(self._by_url)
