"""Public-domain economics: Smith, Ricardo, Mill, Marx, Keynes, Bagehot.

These supply the theoretical vocabulary the modern textbooks are built on — value, rent, capital,
the price mechanism — in long sustained argument rather than the short definitions a textbook uses.

Books are listed by id in config rather than searched, because /ebooks/search is the one path
gutenberg.org's robots.txt disallows.
"""

from .. import config
from ..common.http import BlockedByHost

TEXT_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
LICENCE = "Public domain (Project Gutenberg)"
SOURCE = "gutenberg"

# Gutenberg wraps every book in a licence header and footer. The markers are stable and have
# bounded the actual text for years.
_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def strip_boilerplate(text):
    """The book between Gutenberg's own markers, or the whole text if they are absent."""
    start = text.find(_START)
    if start != -1:
        start = text.find("\n", start) + 1
    end = text.find(_END)
    return text[max(start, 0) : end if end != -1 else len(text)].strip()


def collect(session, manifest, settings, log):
    destination_root = config.RAW_DIR / SOURCE
    destination_root.mkdir(parents=True, exist_ok=True)

    book_ids = config.GUTENBERG_BOOK_IDS
    limit = settings["gutenberg_books"]
    if limit is not None:
        book_ids = book_ids[:limit]

    fetched = 0
    for book_id in book_ids:
        url = TEXT_URL.format(book_id=book_id)
        destination = destination_root / f"pg{book_id}.txt"
        if manifest.already_have(url, destination):
            continue

        try:
            text = strip_boilerplate(session.get_text(url))
        except BlockedByHost:
            raise
        except RuntimeError as error:
            log(f"  {book_id}: {error}")
            continue

        destination.write_text(text, encoding="utf-8")
        manifest.record(url, destination, SOURCE, LICENCE)
        fetched += 1
        log(f"  {book_id}: {len(text) // 1024} KB")

    return fetched
