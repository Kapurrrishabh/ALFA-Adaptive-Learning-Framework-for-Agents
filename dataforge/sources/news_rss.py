"""Finance news and regulator press releases from public RSS feeds.

Only the headline and the summary the publisher puts in the feed are kept — no article scraping.
That keeps this inside what the feeds are published for, and headlines are what sentiment work
uses anyway.

A feed is a snapshot of the last few days, so re-runs merge new items into the existing file
instead of replacing it. Running this daily is how the news corpus grows.
"""

import hashlib
import json
import xml.etree.ElementTree as ElementTree
from urllib.parse import urlparse

from .. import config
from ..common.http import BlockedByHost

LICENCE = "Publisher terms; headline and feed summary only"
SOURCE = "news_rss"

# Atom uses a namespace, RSS does not, so both spellings of each field are tried.
_TITLE = ("title", "{http://www.w3.org/2005/Atom}title")
_SUMMARY = ("description", "{http://www.w3.org/2005/Atom}summary")
_LINK = ("link", "{http://www.w3.org/2005/Atom}id")
_PUBLISHED = ("pubDate", "{http://www.w3.org/2005/Atom}published", "{http://purl.org/dc/elements/1.1/}date")


def _first_text(element, names):
    for name in names:
        found = element.find(name)
        if found is not None and (found.text or "").strip():
            return found.text.strip()
    return ""


def _article(entry):
    return {
        "title": _first_text(entry, _TITLE),
        "summary": _first_text(entry, _SUMMARY),
        "link": _first_text(entry, _LINK),
        "published": _first_text(entry, _PUBLISHED),
    }


def items(feed_xml):
    # Several feeds are served with a UTF-8 BOM, which the XML parser rejects as a stray token.
    root = ElementTree.fromstring(feed_xml.lstrip("﻿").strip())
    entries = list(root.iter("item")) or list(root.iter("{http://www.w3.org/2005/Atom}entry"))
    articles = [_article(entry) for entry in entries]
    return [article for article in articles if article["title"]]


def _merge(destination, articles):
    """Adds articles not already stored. Returns how many were new."""
    known = set()
    if destination.exists():
        with destination.open() as handle:
            for line in handle:
                if line.strip():
                    known.add(json.loads(line)["link"])

    new = [article for article in articles if article["link"] not in known]
    with destination.open("a") as handle:
        for article in new:
            handle.write(json.dumps(article) + "\n")
    return len(new)


def collect(session, manifest, settings, log):
    destination_root = config.RAW_DIR / SOURCE
    destination_root.mkdir(parents=True, exist_ok=True)
    total_new = 0

    for feed_url in config.RSS_FEEDS:
        try:
            feed_xml = session.get_text(feed_url)
        except BlockedByHost as error:
            # One publisher refusing us says nothing about the others, so this one is skipped.
            log(f"  {feed_url}: {error}")
            continue
        except RuntimeError as error:
            log(f"  {feed_url}: {error}")
            continue

        try:
            articles = items(feed_xml)
        except ElementTree.ParseError as error:
            log(f"  {feed_url}: not parseable as RSS or Atom ({error})")
            continue

        host = urlparse(feed_url).netloc.replace(".", "_")
        # Python's str hash is salted per process, so the digest keeps the filename stable across
        # runs and the merge finds the file it wrote last time.
        digest = hashlib.sha1(feed_url.encode()).hexdigest()[:8]
        destination = destination_root / f"{host}_{digest}.jsonl"
        new_count = _merge(destination, articles)
        manifest.record(feed_url, destination, SOURCE, LICENCE)
        total_new += new_count
        log(f"  {host}: {len(articles)} in feed, {new_count} new")

    return total_new
