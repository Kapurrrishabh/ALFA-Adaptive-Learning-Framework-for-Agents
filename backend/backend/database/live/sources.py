"""Passages fetched while the question is being answered, for a question the corpus on disk cannot cover.

The index holds 519,135 passages and no news at all, so a question about what a company announced last
month matched on the word "news" and was served a passage about credit-card utilisation. That is the gap
this closes, and what comes back goes through the same reader. A company with an SEC number is searched for
before the corpus is read; anything else is searched for only once the stored passages have failed the gate.

**Two sources, and the list is short because robots.txt decided it rather than ranking.** Every candidate
was checked rather than assumed. Google News and Yahoo Finance answer `Disallow: /`. NSE's API refuses a
scripted client outright. `en.wikipedia.org` disallows `/w/` and `/api/` for everyone, so the MediaWiki
search API there is not ours to call -- but the same content is served by `api.wikimedia.org`, which
publishes no robots.txt at all and is the host the corpus collector already reads. So what is left is SEC
EDGAR's keyless full-text search, and Wikipedia through the host whose rules allow it.

That matters most for the half of our instruments EDGAR has never heard of: **the 51 Indian listings file
with no-one here**, and Wikipedia is the only live source they have. It is an encyclopedia and not a
newswire, so what it can answer about them is what the company is and has done, not what it announced this
week. Anything narrower than that comes back refused, with the reason named.

**Dated, so a live passage cannot become look-ahead.** A filing carries its filing date and the search is
asked for nothing later than the as-of date; an article is read at the revision that was current on that
date rather than at today's.

Off unless a caller builds one of these and hands it to `Reference`. With it left out the agent makes no
network call at all, which is how the gate "answers a free-text question end to end, no network" is still
met by the same code.
"""

import html
import re
from collections import namedtuple
from urllib.parse import quote

from backend.knowledge_base.retrieval.chunk import chunks as into_chunks
from backend.knowledge_base.retrieval.chunk import deduplicate
from backend.database.live.session import BlockedByHost

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

WIKIPEDIA = "https://api.wikimedia.org/core/v1/wikipedia/en"

# Where a reader goes to check a passage. Not the URL we fetch: `/wiki/` is the allowed path on the
# encyclopedia's own host, and pinning the revision is what makes the citation match what was read.
WIKIPEDIA_CITATION = "https://en.wikipedia.org/wiki/{key}?oldid={revision}"

# EDGAR's full-text index starts in 2001, so an as-of search needs a lower bound and this is the only
# honest one: asking for earlier returns nothing rather than erroring.
EDGAR_EPOCH = "2001-01-01"

# The SEC's current-report form. It is the one form worth reading at question time because its whole text
# is the announcement -- a 10-K cut to a readable length is its cover page and table of contents.
CURRENT_REPORT = "8-K"

FILINGS = 2
ARTICLES = 2

# provisional: how many of the newest matching headlines one question reads
NEWS_ITEMS = 6

# The size the reference index was chunked at, so a fetched passage is laid out like a stored one.
CHUNK_TOKENS = 192

# Enough for an exhibit, which is a press release. A cap rather than a stream because the form chosen is
# what keeps the fetch small, and this only has to stop one mis-targeted filing from filling the turn.
KEEP_CHARACTERS = 20000

# Elements whose text is never prose, so a tag strip would keep it. Scripts and stylesheets because their
# bodies are not markup at all; `sup` because a stripped footnote leaves "[ 83 ]" mid-sentence; `ix:header`
# because it is a filing's whole XBRL taxonomy, which is what served "false 0001018724 AMAZON COM INC
# 2026-07-09" against "what does AMZN do" -- the corpus extractor needed this same rule.
_STRIPPED = "script|style|sup|ix:header"

# One element holds a Wikipedia article's whole reference list, and it has to go for the reason the rest
# does: it is a page dense with the company's own name, so BM25 ranked it first and a question about TCS
# was answered out of its own footnotes.
_REFERENCES = r'<ol\b[^>]*mw-references[^>]*>.*?</ol>'

#
# A tag is quoted strings plus anything that is neither a bracket nor a quote, not "up to the next `>`":
# Wikipedia carries a template's whole wikitext in a `data-mw` attribute, `>` and all, so the cheaper
# pattern ended the tag early and leaked `{{infobox ...}}` into a passage we were about to quote. The three
# alternatives share no character, which is what stops a stray `<` in prose from backtracking exponentially.
_TAG = r'(?:"[^"]*"|\'[^\']*\'|[^<>"\'])*'
_MARKUP = re.compile(rf"(?is)<({_STRIPPED})\b{_TAG}>.*?</\1>|{_REFERENCES}|<{_TAG}>")

# The three fields the corpus chunker reads off a document, so the same chunker cuts a fetched page
# exactly as it cut the corpus. `key` is the URL here rather than a manifest hash: it is the provenance,
# and a reader can follow it.
Fetched = namedtuple("Fetched", "key day text")


class Filings:
    """Exhibits to the current reports whose text matches the question's rare terms.

    Full-text search returns metadata only, so the document itself is fetched and tag-stripped. That is a
    thinner extractor than the corpus one in `dataforge`, deliberately: this reads one HTML exhibit, not a
    nested submission full of PDFs, and the corpus extractor is deleted with the folder it lives in.
    """

    name = "sec_edgar"

    def __init__(self, session, filings=FILINGS, form=CURRENT_REPORT, keep=KEEP_CHARACTERS):
        self.session = session
        self.filings = filings
        self.form = form
        self.keep = keep

    def documents(self, query, terms, as_of=None, subject=None):
        """The newest filings about `subject`, or matching `terms` when the question named no instrument.

        The date bound is pushed into the search rather than filtered after it: filtering a top-two list
        down to what existed on the day returns nothing and calls it two.

        A subject is searched by its SEC number, which is what makes this a search for a company rather
        than for a word: "reliance" alone returns filings by whoever used the word, and the subject gate
        would pass them because they do say it. An instrument with no number files nothing here, so this
        returns nothing for it rather than something about someone else.

        Newest first, which the search itself is not: `q=earnings` for Apple returns 2006, 2004 and 2015 in
        that order, so taking the search's own top two serves a twenty-year-old filing to a question about
        now. Ordered here rather than by a parameter because EDGAR's has none.
        """
        words = [subject.named] if subject else terms
        if not words or (subject is not None and not subject.cik):
            return []
        asked = {"q": " ".join(words), "forms": self.form}
        if subject is not None:
            asked["ciks"] = subject.cik
        if as_of:
            asked.update(dateRange="custom", startdt=EDGAR_EPOCH, enddt=str(as_of))
        hits = self.session.get_json(EDGAR_SEARCH, asked)["hits"]["hits"]
        newest = sorted(hits, key=lambda hit: hit["_source"]["file_date"], reverse=True)
        return [Fetched(url, hit["_source"]["file_date"], self._text(url))
                for url, hit in ((_archive(hit), hit) for hit in newest[: self.filings]) if url]

    def _text(self, url):
        return as_text(self.session.get_text(url))[: self.keep]


class Encyclopedia:
    """Wikipedia articles about the question's subject, read at the revision current on the as-of date.

    Searched by the subject's name and not by the question, which is measured: "Reliance Industries"
    returns that article at rank 1, while "Reliance Industries recent news" returns Jio Platforms, a
    green-energy complex and a businessman. A question naming no instrument is passed whole instead,
    because this search ranks a sentence rather than intersecting terms the way EDGAR's does.
    """

    name = "wikipedia"

    def __init__(self, session, articles=ARTICLES, keep=KEEP_CHARACTERS):
        self.session = session
        self.articles = articles
        self.keep = keep

    def documents(self, query, terms, as_of=None, subject=None):
        asked = subject.named if subject else query
        found = self.session.get_json(f"{WIKIPEDIA}/search/page",
                                      {"q": asked, "limit": self.articles})["pages"]
        return [self._article(page["key"], as_of) for page in found]

    def _article(self, key, as_of):
        """One article as it stood on `as_of`, so a dated question cannot be answered from a later edit.

        One page of history, which is 20 revisions, and that is as far back as this reaches: measured, it
        covers two months on a page edited as often as Apple's and sixteen on Reliance's. Beyond it the
        source says so rather than serving today's text against a date it does not cover.
        """
        revisions = self.session.get_json(f"{WIKIPEDIA}/page/{quote(key, safe='')}/history")["revisions"]
        current = next((edit for edit in revisions if _day(edit) <= str(as_of or _day(revisions[0]))), None)
        if current is None:
            raise RuntimeError(f"the last 20 revisions of {key} all postdate {as_of}, the oldest being "
                               f"{_day(revisions[-1])}; reading today's article would be look-ahead")
        text = as_text(self.session.get_text(f"{WIKIPEDIA}/revision/{current['id']}/html"))
        return Fetched(WIKIPEDIA_CITATION.format(key=key, revision=current["id"]),
                       _day(current), text[: self.keep])


class Live:
    """The sources asked in turn, their pages cut into passages by the corpus chunker.

    Pooled rather than ranked here. The caller already has an index and a relevance gate, so ranking twice
    by two rules would decide the answer somewhere no measurement can see it.
    """

    def __init__(self, sources, tokenizer, chunk_tokens=CHUNK_TOKENS):
        if not sources:
            raise ValueError("a live searcher with no sources would return nothing and look like a miss")
        self.sources = list(sources)
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens

    def passages(self, query, terms, as_of=None, subject=None):
        """`Chunk`s from every source that answered, or [] when they all answered with nothing.

        A source that refuses us costs its own passages and not the turn, as long as another one found
        something. **A refusal that produced nothing travels**, because a source that could not be reached
        is not a source that had nothing, and returning [] there would blame the corpus for the network.
        """
        found, refused = [], []
        for source in self.sources:
            try:
                pages = source.documents(query, terms, as_of, subject)
            except (BlockedByHost, RuntimeError) as error:
                refused.append(f"{source.name}: {error}")
                continue
            for page in pages:
                found += into_chunks(page, self.tokenizer, self.chunk_tokens)
        if refused and not found:
            raise BlockedByHost("no source could be searched -- " + "; ".join(refused))
        return deduplicate(found)


def _archive(hit):
    """Where the matched document sits under EDGAR's archive, or empty when the hit names no CIK.

    The id is `<accession>:<file>` and the path wants the accession without its dashes and the CIK without
    its leading zeros, which is EDGAR's own layout rather than a choice.
    """
    accession, _, name = hit["_id"].partition(":")
    ciks = hit["_source"].get("ciks") or []
    if not ciks or not name:
        return ""
    return f"{EDGAR_ARCHIVE}/{int(ciks[0])}/{accession.replace('-', '')}/{name}"


def _day(revision):
    """The calendar day of a revision. Its timestamp is to the second, and every date here is a day."""
    return revision["timestamp"][:10]


def as_text(markup):
    """Tag-stripped, entity-decoded, whitespace-collapsed text."""
    return " ".join(html.unescape(_MARKUP.sub(" ", markup)).split())


class NewsWire:
    """Recent Indian market headlines from the archive `news.refresh` keeps, read with no network call.

    About the subject when there is one, by the same name gate the stored passages pass; otherwise sharing
    one of the question's rare words. Newest first, because "why is it falling" means this week.
    """

    name = "news"

    def __init__(self, archive=None, items=NEWS_ITEMS):
        from backend.database.live import news
        self.news = news
        self.archive = archive or news.ARCHIVE
        self.items = items

    def documents(self, query, terms, as_of=None, subject=None):
        found = []
        for row in self.news.read(self.archive):
            if as_of and row["day"] > str(as_of):
                continue
            text = f"{row['title']}. {row['summary']}" if row["summary"] else row["title"]
            lowered = text.lower()
            if subject.mentions(text) if subject is not None else any(_says(lowered, term) for term in terms):
                found.append(Fetched(row["link"], row["day"], f"{row['source']}, {_long_day(row['day'])}: {text}"))
                if len(found) == self.items:
                    break
        return found


def _says(text, term):
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _long_day(day):
    """"2026-10-06" as "6 Oct 2026", the form a reader writes, so a rewrite naming the date uses its figures."""
    year, month, dom = day.split("-")
    return f"{int(dom)} {('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')[int(month) - 1]} {year}"
