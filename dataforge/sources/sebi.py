"""SEBI circulars, regulations and acts. The Indian counterpart to the SEC filings.

The listing pages are HTML and page in 25s through an ordinary `nextValue` GET parameter — the
"next page" links look like JavaScript, but the same page answers a plain GET, so no form posting
or session token is needed. The circular itself is not in the page: the body is a PDF referenced
from an iframe as `../../../web/?file=<absolute url>`, and the page text alone is only about 500
characters of title and breadcrumb. So this follows the iframe and stores the PDF.
"""

from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from .. import config
from ..common.http import BlockedByHost

LISTING = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do"
LICENCE = "Government of India work, freely reproducible with attribution"
SOURCE = "sebi"

_PER_PAGE = 25


def _document_links(html):
    """Absolute links to the circular or regulation pages listed on one page."""
    soup = BeautifulSoup(html, "lxml")
    seen = {}
    for link in soup.select('a[href*="/legal/"]'):
        href = link["href"]
        if href.lower().endswith(".html"):
            seen[href] = link.get_text(strip=True)
    return list(seen.items())


def _attachment_url(html):
    """The PDF behind a circular page, taken from the iframe that displays it."""
    soup = BeautifulSoup(html, "lxml")
    for frame in soup.select("iframe[src]"):
        query = parse_qs(urlparse(frame["src"]).query).get("file")
        if query and query[0].lower().endswith(".pdf"):
            return query[0]
    return None


def _listing_pages(session, section_id, wanted, log):
    """Walks the listing until it has `wanted` documents or the pages stop changing."""
    found, seen_urls = [], set()
    pages = config.SEBI_MAX_LISTING_PAGES
    if wanted:
        pages = min(pages, -(-wanted // _PER_PAGE))

    for page in range(1, pages + 1):
        parameters = {"doListing": "yes", "sid": 1, "ssid": section_id, "smid": 0}
        if page > 1:
            parameters["nextValue"] = page
        try:
            html = session.get_text(LISTING, params=parameters)
        except BlockedByHost:
            raise
        except RuntimeError as error:
            log(f"  listing page {page}: {error}")
            break

        new = [(url, title) for url, title in _document_links(html) if url not in seen_urls]
        if not new:
            break
        seen_urls.update(url for url, _ in new)
        found.extend(new)
        if wanted and len(found) >= wanted:
            break
    return found[:wanted] if wanted else found


def collect(session, manifest, settings, log):
    destination_root = config.RAW_DIR / SOURCE
    destination_root.mkdir(parents=True, exist_ok=True)
    wanted = settings["sebi_circulars"]
    fetched = 0

    for section, section_id in config.SEBI_SECTIONS.items():
        documents = _listing_pages(session, section_id, wanted, log)
        log(f"  {section}: {len(documents)} documents listed")
        section_dir = destination_root / section
        section_dir.mkdir(parents=True, exist_ok=True)

        for url, title in documents:
            # The trailing _NNNNN in the slug is SEBI's own document id, so it names the file.
            stem = url.rsplit("/", 1)[-1].removesuffix(".html")[:100]
            destination = section_dir / f"{stem}.pdf"
            if manifest.already_have(url, destination):
                continue

            try:
                attachment = _attachment_url(session.get_text(url))
                if attachment is None:
                    log(f"    {title[:60]}: no attachment on the page")
                    continue
                destination.write_bytes(session.get(attachment).content)
            except BlockedByHost:
                raise
            except RuntimeError as error:
                log(f"    {title[:60]}: {error}")
                continue

            # Recorded against the page, not the PDF: the page URL is the citable one.
            manifest.record(url, destination, SOURCE, LICENCE)
            fetched += 1
            if fetched % 100 == 0:
                log(f"  {fetched} documents saved")

    return fetched
