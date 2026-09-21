"""Reserve Bank of India speeches, press releases, notifications and bulletin articles.

The speeches are the most valuable text here — 20-30 KB each of reasoned monetary-policy argument,
which no filing and no textbook contains.

Two things to know. rbi.org.in answers /robots.txt with 418, so there is no directive to read;
this source is therefore throttled harder than any other rather than treated as permitted. And the
RSS feeds carry only the last 10 items each, so the archive grows by re-running this over time —
except the bulletin index, which is the one archive page that answers a plain GET.
"""

from urllib.parse import urljoin
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from .. import config
from ..common.http import BlockedByHost
from ..extract import html_text
from . import news_rss

LICENCE = "Reserve Bank of India, reproducible with attribution"
SOURCE = "rbi"

MINIMUM_CHARACTERS = 1500


def _feed_documents(session, log):
    """(url, title) for every item across the RSS feeds."""
    found = []
    for feed_url in config.RBI_FEEDS:
        try:
            articles = news_rss.items(session.get_text(feed_url))
        except BlockedByHost:
            raise
        except (RuntimeError, ElementTree.ParseError) as error:
            log(f"  {feed_url}: {error}")
            continue
        found.extend((article["link"], article["title"]) for article in articles if article["link"])
    return found


def _bulletin_documents(session, log):
    try:
        html = session.get_text(config.RBI_BULLETIN_INDEX)
    except BlockedByHost:
        raise
    except RuntimeError as error:
        log(f"  bulletin index: {error}")
        return []

    soup = BeautifulSoup(html, "lxml")
    return [
        (urljoin(config.RBI_BULLETIN_INDEX, link["href"]), link.get_text(strip=True))
        for link in soup.select('a[href*="Id="]')
    ]


def collect(session, manifest, settings, log):
    destination_root = config.RAW_DIR / SOURCE
    destination_root.mkdir(parents=True, exist_ok=True)

    documents = _feed_documents(session, log) + _bulletin_documents(session, log)
    # A speech appears in both its feed and the bulletin, so the same link arrives twice.
    documents = list(dict.fromkeys(documents))
    wanted = settings["rbi_documents"]
    if wanted:
        documents = documents[:wanted]
    log(f"  {len(documents)} documents listed")

    fetched, skipped_short = 0, 0
    for url, title in documents:
        stem = "".join(character if character.isalnum() else "_" for character in title)[:100]
        destination = destination_root / f"{stem or 'untitled'}.txt"
        if manifest.already_have(url, destination):
            continue

        try:
            text = html_text.extract(session.get_text(url))
        except BlockedByHost:
            raise
        except RuntimeError as error:
            log(f"    {title[:60]}: {error}")
            continue

        if len(text) < MINIMUM_CHARACTERS:
            skipped_short += 1
            continue
        destination.write_text(f"{title}\n\n{text}", encoding="utf-8")
        manifest.record(url, destination, SOURCE, LICENCE)
        fetched += 1

    log(f"  {skipped_short} skipped as too short (under {MINIMUM_CHARACTERS} characters)")
    return fetched
