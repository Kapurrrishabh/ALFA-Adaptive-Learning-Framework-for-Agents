"""Collected headlines, read off disk with the URL and licence they arrived under.

`dataforge` fetches; this reads. Nothing here touches the network, so the agent can answer with no
feed reachable, and the collector stays deletable.

Provenance travels *on* the article rather than beside it. C4's gate is that a scored article traces
to its source URL and licence, and the way to hold that is to make an article that cannot name its
source unconstructible -- a field that has to be joined back later is a field that gets dropped. So
`read` raises on a file the manifest does not describe rather than serving it unattributed.
"""

import json
import re
from collections import namedtuple
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

SOURCE = "news_rss"

Article = namedtuple("Article", "title summary link published feed licence tickers")

# Legal-form words carry no identity: "Apple Inc." and "Apple" are the same company to a headline,
# and every registrant ends in one of these.
_LEGAL_FORMS = re.compile(
    r"\b(inc|corp|corporation|company|co|plc|ltd|limited|holdings|group|class|and|the)\b\.?", re.I)
_PUNCTUATION = re.compile(r"[^a-z0-9& ]+")
_SHORTEST_NAME = 4


def provenance(manifest_path, source=SOURCE):
    """path -> (feed url, licence) for one source.

    The last row for a path wins: collection appends a row per fetch, so a file that has been
    re-fetched has several, and the newest describes what is on disk now.
    """
    found = {}
    with Path(manifest_path).open() as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["source"] == source:
                found[Path(entry["path"]).name] = (entry["url"], entry["licence"])
    return found


def company_names(path, tickers):
    """ticker -> the names it is called by, for the tickers asked for.

    Reads SEC's own symbol table. Restricted to the tickers passed in rather than reading the price
    directory itself, so this file learns nothing about which instruments the agent serves.
    """
    wanted = set(tickers)
    return names_for({row["ticker"]: row["title"]
                      for row in json.loads(Path(path).read_text()).values()
                      if row["ticker"] in wanted})


def names_for(titles):
    """ticker -> the names it is called by, from a ticker-to-registered-name mapping."""
    return {ticker: _variants(title) for ticker, title in titles.items()}


def _variants(title):
    plain = " ".join(_PUNCTUATION.sub(" ", title.lower()).split())
    short = " ".join(_LEGAL_FORMS.sub(" ", plain).split())
    return tuple(sorted({name for name in (plain, short) if len(name) >= _SHORTEST_NAME}))


def tag(text, names):
    """The tickers a headline is about: its symbol as a word, or one of the company's names.

    Whole words only, and that is the whole difference between a tagger and a random number: matched
    as substrings, "Intel" tags every headline about artificial intelligence and "V" tags Vestas,
    Vistiq and "virtual" -- 6 of 147 collected headlines, every one of them wrong.
    """
    lowered = text.lower()
    found = set()
    for ticker, variants in names.items():
        if _says(text, ticker) or any(_says(lowered, name) for name in variants):
            found.add(ticker)
    return frozenset(found)


def _says(text, term):
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text) is not None


def read(directory, manifest_path, names=()):
    """Every collected article, deduplicated, each carrying its feed URL, licence and tickers."""
    sources = provenance(manifest_path)
    names = dict(names)
    articles, at = [], {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        if path.name not in sources:
            raise ValueError(
                f"{path} has no {SOURCE} row in {manifest_path}, so nothing scored from it could "
                f"name its source or licence; re-run the collector or delete the file")
        feed, licence = sources[path.name]
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                stored = json.loads(line)
                # A per-symbol feed states its instrument; a topic feed does not, and either way the
                # text can name others. Both are kept, so the tag is never narrower than what is known.
                about = tag(f"{stored['title']} {stored['summary']}", names)
                if stored.get("ticker"):
                    about |= {stored["ticker"]}
                article = Article(stored["title"], stored["summary"], stored["link"],
                                  stored["published"], feed, licence, frozenset(about))

                # Identity is the headline and the day, not the link. One publisher gives every item
                # in a symbol's feed the same URL -- 26 separate stories under one link, measured --
                # so deduplicating on the link throws away almost a whole feed. The day is part of it
                # because a recurring headline is a different story each week.
                key = (_headline_key(article.title), as_of(article))
                if key in at:
                    # One story in two symbols' feeds is about both instruments. Dropping the second
                    # copy outright would silently drop the second instrument with it.
                    kept = articles[at[key]]
                    articles[at[key]] = kept._replace(tickers=kept.tickers | article.tickers)
                    continue
                at[key] = len(articles)
                articles.append(article)
    return articles


def as_of(article):
    """The publication day as YYYY-MM-DD, or None when the feed's stamp cannot be read.

    None rather than today: a headline dated by when it was downloaded would be joined to the wrong
    bar, and an article with no date is one that must be left out of any measurement.
    """
    stamp = article.published.strip()
    if not stamp:
        return None
    try:
        return parsedate_to_datetime(stamp).date().isoformat()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(stamp).date().isoformat()
    except ValueError:
        return None


def _headline_key(title):
    return " ".join(_PUNCTUATION.sub(" ", title.lower()).split())
