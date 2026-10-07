"""Recent Indian market news, kept on disk so a question never waits on a newswire.

A feed lists only its newest items -- 4 to 60 each, measured on 2026-10-07 -- so a question asked on Friday
about Monday's results would find nothing in the feed itself. `refresh` saves every item it has not seen,
and the archive grows into weeks of dated headlines that `sources.NewsWire` searches with no network call.

Only feeds whose robots.txt allows a reader, checked on 2026-10-07 with this project's user agent. Google
News forbids it; Business Standard, PIB and Moneycontrol answer that user agent with 403; RBI serves no
robots.txt we could read, and no rules is not the same as permission.

    python -m backend.database.live.news --every 30
"""
import argparse
import calendar
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

from backend.paths import DATA

log = logging.getLogger("alfa.news")

FEEDS = (
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Economic Times", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("Mint", "https://www.livemint.com/rss/markets"),
    ("Mint", "https://www.livemint.com/rss/companies"),
    ("Hindu BusinessLine", "https://www.thehindubusinessline.com/markets/feeder/default.rss"),
    ("NDTV Profit", "https://feeds.feedburner.com/ndtvprofit-latest"),
    ("SEBI", "https://www.sebi.gov.in/sebirss.xml"),
)
ARCHIVE = DATA / "news" / "wire.jsonl"
# provisional: long enough for "this quarter's results", short enough to stay one small file
KEEP_DAYS = 90

# Economic Times publishes a "Share Price Live Updates: <name> Stock Details" page per stock per day: 22 of
# its 59 items on 2026-10-07, none of them reporting anything, and each one matched any question naming it
FILLER = re.compile(r"share price live updates", re.I)

_READ = {}


def parse(text, source):
    """The dated items of one feed, as archive rows. An undated item is dropped: it could be years old."""
    import feedparser
    from backend.database.live.sources import as_text
    rows = []
    for entry in feedparser.parse(text).entries:
        when = entry.get("published_parsed")
        day = (datetime.fromtimestamp(calendar.timegm(when), timezone.utc).date().isoformat() if when
               else _sebi_day(entry.get("published", "")))
        if not day or not entry.get("link") or not entry.get("title") or FILLER.search(entry.title):
            continue
        rows.append({"link": entry.link, "day": day, "source": source, "title": as_text(entry.title),
                     "summary": as_text(entry.get("summary", ""))})
    return rows


def _sebi_day(stamp):
    """SEBI writes "05 Oct, 2026 +0530", which the feed parser does not read; None for anything else."""
    try:
        return datetime.strptime(stamp, "%d %b, %Y %z").date().isoformat()
    except ValueError:
        return None


def refresh(session, archive=ARCHIVE, feeds=FEEDS, today=None):
    """Fetch every feed, add the items not yet held, drop those older than KEEP_DAYS. Returns how many were new.

    A feed that refuses us costs its own items and is logged; all of them refusing is an error, because an
    archive that stopped growing would go on answering "lately" with last month's news.
    """
    from backend.database.live.session import BlockedByHost
    held = read(archive)
    seen = {row["link"] for row in held}
    fresh, failed = [], []
    for source, url in feeds:
        try:
            rows = parse(session.get_text(url), source)
        except (BlockedByHost, RuntimeError) as error:
            failed.append(f"{source} ({url}): {error}")
            continue
        for row in rows:
            if row["link"] not in seen:
                seen.add(row["link"])
                fresh.append(row)
    if failed and len(failed) == len(feeds):
        raise RuntimeError("no news feed could be read: " + "; ".join(failed))
    for message in failed:
        log.warning("feed skipped: %s", message)
    oldest = ((today or date.today()) - timedelta(days=KEEP_DAYS)).isoformat()
    kept = sorted((row for row in fresh + held if row["day"] >= oldest and not FILLER.search(row["title"])),
                  key=lambda row: row["day"], reverse=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".partial")
    partial.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    os.replace(partial, archive)          # so a reader never sees a half-written archive
    return len(fresh)


def read(archive=ARCHIVE):
    """Every held item, newest first. Re-read only when the file changed, so a question costs no parse."""
    if not archive.exists():
        return []
    stamp = archive.stat().st_mtime_ns
    if _READ.get(archive, (None,))[0] != stamp:
        _READ[archive] = (stamp, [json.loads(line) for line in archive.read_text().splitlines() if line])
    return _READ[archive][1]


def main():
    from backend.database.live.session import PoliteSession
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--every", type=float, default=0, help="minutes between refreshes; 0 refreshes once")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    session = PoliteSession(timeout=20, attempts=2)
    while True:
        try:
            log.info("%d new items; %d held", refresh(session), len(read()))
        except RuntimeError:
            if not args.every:
                raise
            log.exception("news refresh failed; trying again in %.0f minutes", args.every)
        if not args.every:
            return
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
