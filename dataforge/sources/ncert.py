"""NCERT Classes 11-12 textbooks: accountancy, business studies, economics, statistics.

This is how finance is actually taught in India, in teaching language — definitions, worked
examples, rupee amounts, Indian institutions. It is the Indian counterpart to the OpenStax books,
and per token it is the densest explanatory text in the corpus.

Chapters are separate PDFs named `<book code><two-digit chapter>.pdf`, and a chapter that does not
exist answers 404, so the number of chapters per book is discovered by asking.
"""

from .. import config
from ..common.http import BlockedByHost

CHAPTER_URL = "https://ncert.nic.in/textbook/pdf/{code}{chapter:02d}.pdf"
LICENCE = "NCERT, free for educational use"
SOURCE = "ncert"

# Books number their chapters from 1 with no gaps, so the first miss is the end of the book.
_STOP_AFTER_MISSES = 1


def collect(session, manifest, settings, log):
    destination_root = config.RAW_DIR / SOURCE
    destination_root.mkdir(parents=True, exist_ok=True)
    fetched = 0

    for code in config.NCERT_BOOK_CODES:
        misses, chapters = 0, 0
        for chapter in range(1, config.NCERT_MAX_CHAPTERS + 1):
            url = CHAPTER_URL.format(code=code, chapter=chapter)
            destination = destination_root / f"{code}{chapter:02d}.pdf"
            if manifest.already_have(url, destination):
                chapters += 1
                continue

            try:
                destination.write_bytes(session.get(url).content)
            except BlockedByHost:
                raise
            except RuntimeError:
                misses += 1
                if misses >= _STOP_AFTER_MISSES:
                    break
                continue

            manifest.record(url, destination, SOURCE, LICENCE)
            fetched += 1
            chapters += 1
        log(f"  {code}: {chapters} chapters")

    return fetched
