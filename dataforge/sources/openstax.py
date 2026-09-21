"""OpenStax textbooks as PDF. Small but the highest-quality prose in the corpus.

Textbooks give the model explanatory finance language — definitions, worked reasoning — which
filings and news do not contain.
"""

from .. import config
from ..common.http import BlockedByHost

BOOK_LIST = "https://openstax.org/apps/cms/api/v2/pages/"
SOURCE = "openstax"

# Titles are matched against these. Finance proper is only a few books, so the neighbouring
# subjects are included: they share vocabulary and roughly triple the usable text.
SUBJECT_KEYWORDS = (
    "financ",
    "econom",
    "account",
    "business",
    "statistic",
    "entrepreneur",
    "marketing",
    "management",
)


def _relevant_books(session):
    listing = session.get_json(
        BOOK_LIST, params={"type": "books.Book", "fields": "title,slug", "limit": 200}
    )
    books = []
    for item in listing["items"]:
        title = item["title"]
        if any(keyword in title.lower() for keyword in SUBJECT_KEYWORDS):
            books.append((item["id"], item["meta"]["slug"], title))
    return books


def collect(session, manifest, settings, log):
    destination_root = config.RAW_DIR / SOURCE
    destination_root.mkdir(parents=True, exist_ok=True)

    books = _relevant_books(session)
    limit = settings["openstax_books"]
    if limit is not None:
        books = books[:limit]
    log(f"  {len(books)} relevant books")

    fetched = 0
    for page_id, slug, title in books:
        detail = session.get_json(f"{BOOK_LIST}{page_id}/")
        url = detail.get("pdf_url")
        if not url:
            log(f"  {slug}: no PDF offered")
            continue

        destination = destination_root / f"{slug}.pdf"
        if manifest.already_have(url, destination):
            continue
        try:
            destination.write_bytes(session.get(url).content)
        except BlockedByHost:
            raise
        except RuntimeError as error:
            log(f"  {slug}: {error}")
            continue
        # Licences differ per book — some are CC BY, some CC BY-NC-SA. Record what this one says.
        manifest.record(url, destination, SOURCE, detail.get("license_name", "see openstax.org"))
        fetched += 1
        log(f"  {title} ({destination.stat().st_size // 1024} KB)")

    return fetched
