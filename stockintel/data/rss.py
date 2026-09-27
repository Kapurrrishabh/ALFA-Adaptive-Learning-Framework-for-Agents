"""RSS news source. Google News RSS gives broad Indian-press coverage
(ET, Mint, Business Standard, Moneycontrol...) for a company query; any other
RSS/Atom feed (exchange announcement feeds, publisher feeds) works the same.
"""
from __future__ import annotations

import calendar
import re
from datetime import datetime, timezone
from typing import List
from urllib.parse import quote_plus

from .provider import DataUnavailable, NewsItem
from ..evidence import Provenance

GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def _iso(entry) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc).isoformat(
            timespec="seconds")
    return ""


def _clean(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "").replace("&nbsp;", " ").strip()


def fetch_feed(url: str, symbol: str, limit: int = 30) -> List[NewsItem]:
    try:
        import feedparser
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailable("feedparser is not installed; `pip install feedparser`") from exc
    feed = feedparser.parse(url)
    if feed.get("bozo") and not feed.get("entries"):
        raise DataUnavailable(f"RSS fetch failed for {url}: {feed.get('bozo_exception')}")
    items: List[NewsItem] = []
    for e in feed.get("entries", [])[:limit]:
        title = e.get("title", "").strip()
        if not title:
            continue
        source = (e.get("source") or {}).get("title") if isinstance(e.get("source"), dict) else None
        if " - " in title and "news.google.com" in url:
            # Google News appends " - Publisher" to titles.
            head, tail = title.rsplit(" - ", 1)
            if not source or tail.strip() == source.strip():
                title, source = head, source or tail
        published = _iso(e)
        items.append(NewsItem(headline=title, summary=_clean(e.get("summary", ""))[:500],
                              published_at=published, source=source or "rss",
                              url=e.get("link", ""), symbol=symbol,
                              provenance=Provenance(source=f"rss:{url[:80]}",
                                                    as_of=published or "unknown")))
    return items


def google_news(query: str, symbol: str, limit: int = 30) -> List[NewsItem]:
    return fetch_feed(GOOGLE_NEWS.format(q=quote_plus(query)), symbol, limit)
