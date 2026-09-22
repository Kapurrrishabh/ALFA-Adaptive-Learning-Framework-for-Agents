"""The documents the retriever may show, each with the date it was published.

Reads `data/manifest.jsonl` as a file format, not as an import: `dataforge/` writes it and must stay
deletable, so nothing here imports from it. The manifest gives the source URL and licence, and the raw
filename gives the publication date.

**A document with no publication date cannot be retrieved under an as-of filter, and is dropped rather
than dated by when we downloaded it.** `fetched_at` is the time of the download, so using it would date a
2015 filing to 2026 -- which either hides the filing from every historical question, or, if undated
documents were let through instead, would let a 2026 press release answer a 2020 one. That second failure
is the look-ahead the whole project forbids, and it would be invisible in the answer.

Two sources survive that rule today: SEC filings, whose name carries the filing date, and Fed press
releases, whose name carries a compact one. Wikipedia, the textbooks and Stack Exchange are undated and
are therefore not in the as-of index at all. RSS items carry a real `published` field per item, which is
C4's to unpack.
"""

import json
import re
from pathlib import Path

# The two naming schemes the dated sources use. A date anywhere in the name, because the surrounding
# parts differ per source: `1770787_2024-02-15_10-K.txt` against `enforcement20260918b.txt`.
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_COMPACT = re.compile(r"(19|20)(\d{2})(\d{2})(\d{2})")

DATED_SOURCES = ("sec_edgar", "fed_press")


class Document:
    """One retrievable document: where it came from, when it was published, and its text on demand.

    The text is read on access rather than up front. The extracted corpus is 2.8 GB and an index run
    touches every document once, so holding them all would cost more than the index itself.
    """

    __slots__ = ("key", "url", "licence", "source", "day", "path")

    def __init__(self, key, url, licence, source, day, path):
        self.key = key
        self.url = url
        self.licence = licence
        self.source = source
        self.day = day
        self.path = path

    @property
    def text(self):
        return self.path.read_text(encoding="utf-8", errors="replace")

    def __repr__(self):
        return f"Document({self.key!r}, {self.source!r}, {self.day!r})"


def published_on(name):
    """The publication date in a raw filename as `YYYY-MM-DD`, or None when it carries none."""
    found = _ISO.search(name)
    if found:
        return "-".join(found.groups())
    found = _COMPACT.search(name)
    if not found:
        return None
    century, year, month, day = found.groups()
    return f"{century}{year}-{month}-{day}"


def documents(manifest, text_dir, sources=DATED_SOURCES):
    """Every dated document with extracted text on disk, in manifest order.

    Skips an entry whose text was never extracted -- an unreadable PDF is logged by the extractor and
    leaves no cache file -- rather than failing the whole index over one of 18,000 documents.
    """
    manifest, text_dir = Path(manifest), Path(text_dir)
    if not manifest.exists():
        raise FileNotFoundError(f"no manifest at {manifest}; run dataforge/collect.py first")
    wanted = set(sources)

    built = []
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["source"] not in wanted:
                continue
            day = published_on(Path(entry["path"]).name)
            if day is None:
                continue
            # The extractor names its cache by the raw file's hash, which is also the document's id:
            # the same filing fetched from two URLs is one document, not two.
            key = entry["sha256"][:16]
            path = text_dir / entry["source"] / f"{key}.txt"
            if path.exists():
                built.append(Document(key, entry["url"], entry["licence"], entry["source"], day, path))
    if not built:
        raise ValueError(f"no dated documents from {sorted(wanted)} under {text_dir}")
    return built
