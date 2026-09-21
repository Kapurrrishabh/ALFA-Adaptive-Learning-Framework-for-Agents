"""Federal Reserve press releases — the whole archive, not the last few days.

The RSS feeds only ever carry headlines, which is why the corpus held 24 KB of news against 2.3 GB of
filings. This is the fix. The Board publishes its complete press-release index as a single JSON file:
4,632 releases back to 2006, of which 1,073 are monetary-policy statements. That is the register an
agent needs in order to say what a central bank just did and why, and nothing else in the corpus
carries it — filings, circulars and textbooks are all written to a different purpose.

Two things measured before writing this. A release page is about 9.6 KB of text, of which only 400
to 2,500 characters are the release; the rest is navigation that would otherwise repeat in several
thousand files, so the content container is required rather than nice to have. And some index entries
point at pages that no longer exist: the site answers them with HTTP 200 and its own homepage, which
has no container, so those are skipped instead of being stored as thousands of identical files.
"""

import json

from bs4 import BeautifulSoup

from .. import config
from ..common.http import BlockedByHost

LICENCE = "Public domain (US federal government work)"
SOURCE = "fed_press"

# Where the release itself lives on the page. Its absence means the page is not a release.
_CONTENT = "#article"
MINIMUM_CHARACTERS = 200


def _index_entries(session):
    """(title, kind, url) for every release, newest first.

    The index is served with a UTF-8 byte-order mark, which json.loads rejects as a stray token.
    Its last record is a bare update stamp rather than a release, so a record with no date is not
    one and is dropped here.
    """
    raw = session.get(config.FED_PRESS_INDEX).content.decode("utf-8-sig")
    return [
        (entry["t"], entry.get("pt", ""), config.FED_PRESS_BASE + entry["l"])
        for entry in json.loads(raw)
        if entry.get("d") and entry.get("l")
    ]


def _release_text(html):
    """The release body, or None if this page is not a release."""
    container = BeautifulSoup(html, "lxml").select_one(_CONTENT)
    return container.get_text("\n", strip=True) if container else None


def collect(session, manifest, settings, log):
    destination_root = config.RAW_DIR / SOURCE
    destination_root.mkdir(parents=True, exist_ok=True)

    entries = _index_entries(session)
    wanted = settings["fed_releases"]
    if wanted:
        entries = entries[:wanted]
    log(f"  {len(entries)} releases in the index")

    fetched, not_a_release, skipped_short = 0, 0, 0
    for title, kind, url in entries:
        destination = destination_root / (url.rsplit("/", 1)[-1].replace(".htm", "") + ".txt")
        if manifest.already_have(url, destination):
            continue

        try:
            text = _release_text(session.get_text(url))
        except BlockedByHost:
            raise
        except RuntimeError as error:
            log(f"    {title[:60]}: {error}")
            continue

        if text is None:
            not_a_release += 1
            continue
        if len(text) < MINIMUM_CHARACTERS:
            skipped_short += 1
            continue

        # The body already opens with the date and the title, so only the kind is prepended. It is
        # worth keeping: "Monetary Policy" is what separates an FOMC decision from a staff notice.
        destination.write_text(f"{kind}\n\n{text}", encoding="utf-8")
        manifest.record(url, destination, SOURCE, LICENCE)
        fetched += 1
        if fetched % 250 == 0:
            log(f"    {fetched} stored")

    log(f"  {not_a_release} index entries no longer exist, {skipped_short} too short")
    return fetched
