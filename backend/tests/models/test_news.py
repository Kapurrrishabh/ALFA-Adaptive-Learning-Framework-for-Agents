"""C4: a scored headline names its source, its licence and its instrument, or it is not served.

Each test here pins a failure that was measured on the real feeds, not one that was imagined. Three
of them are bugs this module shipped and had fixed: one publisher gives every item in a symbol's feed
the same URL, so link identity collapsed 26 stories into one; a story carried by two symbols' feeds
was deduplicated down to one instrument; and matching company names as substrings tagged every
headline about artificial intelligence to Intel.
"""

import json

import pytest

from backend.advisory.sentiment.feeds import feed, sentiment

LICENCE = "Publisher terms; headline and feed summary only"
FEED = "https://example.test/rss/AAPL.xml"


def _collected(tmp_path, articles, name="example_test_AAPL.jsonl", recorded=True):
    """A collected feed file and the manifest row describing it, as the collector writes them."""
    directory = tmp_path / "news_rss"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text("".join(json.dumps(article) + "\n" for article in articles))
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"url": FEED, "path": str(directory / name), "source": "news_rss", "licence": LICENCE}]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows) if recorded else "")
    return directory, manifest


def _article(title, link="https://example.test/one", published="Tue, 22 Sep 2026 14:35:37 -0400",
             summary="", ticker=None):
    stored = {"title": title, "summary": summary, "link": link, "published": published}
    if ticker:
        stored["ticker"] = ticker
    return stored


def test_a_file_the_manifest_does_not_describe_is_refused_rather_than_read(tmp_path):
    """The C4 gate. An article read from an unrecorded file could not name its URL or licence, and
    the only place that can be enforced is where the article is built."""
    directory, manifest = _collected(tmp_path, [_article("Apple rises")], recorded=False)
    with pytest.raises(ValueError, match="no news_rss row"):
        feed.read(directory, manifest)


def test_every_article_carries_the_url_and_licence_it_arrived_under(tmp_path):
    directory, manifest = _collected(tmp_path, [_article("Apple rises")])
    (article,) = feed.read(directory, manifest)
    assert (article.feed, article.licence) == (FEED, LICENCE)


def test_stories_sharing_one_feed_url_are_kept_apart(tmp_path):
    """The measured bug: a symbol feed puts its own page URL on all 30 items, so deduplicating by
    link threw away 26 of them."""
    directory, manifest = _collected(tmp_path, [
        _article("Apple unveils a screenless tracker", link=FEED),
        _article("Beats unveils new headphones", link=FEED),
    ])
    assert len(feed.read(directory, manifest)) == 2


def test_the_same_headline_on_another_day_is_another_story(tmp_path):
    directory, manifest = _collected(tmp_path, [
        _article("Jobless claims stay low", published="Tue, 22 Sep 2026 14:35:37 -0400"),
        _article("Jobless claims stay low", published="Tue, 15 Sep 2026 14:35:37 -0400"),
    ])
    assert len(feed.read(directory, manifest)) == 2


def test_one_story_in_two_symbol_feeds_is_about_both_instruments(tmp_path):
    """Deduplicating the second copy away used to drop the second instrument with it, which makes an
    article about two tickers into an article about whichever feed was read first."""
    directory, manifest = _collected(tmp_path, [_article("Chip deal agreed", ticker="AMD")])
    second = directory / "example_test_NVDA.jsonl"
    second.write_text(json.dumps(_article("Chip deal agreed", ticker="NVDA")) + "\n")
    with manifest.open("a") as handle:
        handle.write(json.dumps({"url": "https://example.test/rss/NVDA.xml", "path": str(second),
                                 "source": "news_rss", "licence": LICENCE}) + "\n")

    (article,) = feed.read(directory, manifest)
    assert article.tickers == frozenset({"AMD", "NVDA"})


def test_a_company_name_is_matched_as_a_word_and_not_as_a_substring():
    """'Intel' inside 'intelligence' tagged 6 of 147 real headlines, every one of them wrongly."""
    names = feed.names_for({"INTC": "INTEL CORP", "V": "VISA INC."})
    assert feed.tag("Opening remarks on artificial intelligence", names) == frozenset()
    assert feed.tag("Vestas stock slumps as margins disappoint", names) == frozenset()
    assert feed.tag("Intel's stock rises on memory chips", names) == frozenset({"INTC"})
    assert feed.tag("INTC leads the chip rally", names) == frozenset({"INTC"})


def test_a_negated_word_counts_for_the_other_side():
    assert sentiment.polarity("Apple beats expectations") == pytest.approx(1.0)
    assert sentiment.polarity("Apple not expected to beat expectations") == pytest.approx(-1.0)
    assert sentiment.polarity("Apple holds its annual meeting") == 0.0


def test_the_scorer_version_follows_the_lexicon_it_scores_with(monkeypatch):
    """A stored polarity has to name what produced it, so editing one word has to change the name."""
    before = sentiment.VERSION
    monkeypatch.setattr(sentiment, "POSITIVE", sentiment.POSITIVE | {"moonshot"})
    assert sentiment._digest() != before.removeprefix("lexicon@")


# --- the newswire archive the agent reads recent news from ------------------

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Titan Company shares fall 4% after Q2 business update</title><link>https://n.test/titan</link>
<pubDate>Wed, 07 Oct 2026 04:00:00 GMT</pubDate><description>Titan reported 25% growth.</description></item>
<item><title>Titan Company Share Price Live Updates: Titan Company Stock Details</title><link>https://n.test/live</link>
<pubDate>Wed, 07 Oct 2026 05:00:00 GMT</pubDate></item>
<item><title>An undated story</title><link>https://n.test/undated</link></item>
</channel></rss>"""

SEBI = """<?xml version="1.0"?><rss><channel><item><title>SEBI order on X</title><link>https://s.test/1</link>
<pubDate>05 Oct, 2026 +0530</pubDate></item></channel></rss>"""


class FeedSession:
    """Serves fixed feed bodies by URL; a URL missing from the table is refused the way a host refuses us."""

    def __init__(self, bodies):
        self.bodies = bodies

    def get_text(self, url):
        if url not in self.bodies:
            raise RuntimeError(f"403 from {url}")
        return self.bodies[url]


def test_an_item_is_kept_only_when_it_is_dated_and_reports_something():
    """An undated item could be years old, and the per-stock "Live Updates" pages were 22 of Economic Times'
    59 items on the day measured, each matching any question that named the stock and saying nothing."""
    from backend.database.live import news
    assert [row["link"] for row in news.parse(RSS, "ET")] == ["https://n.test/titan"]


def test_sebis_own_date_format_is_read():
    from backend.database.live import news
    assert news.parse(SEBI, "SEBI")[0]["day"] == "2026-10-05"


def test_refreshing_adds_only_unseen_items_and_drops_those_past_the_window(tmp_path):
    from datetime import date
    from backend.database.live import news
    archive = tmp_path / "wire.jsonl"
    feeds = (("ET", "https://e.test/rss"), ("SEBI", "https://s.test/rss"))
    session = FeedSession({"https://e.test/rss": RSS, "https://s.test/rss": SEBI})
    assert news.refresh(session, archive, feeds, today=date(2026, 10, 7)) == 2
    assert news.refresh(session, archive, feeds, today=date(2026, 10, 7)) == 0
    assert [row["day"] for row in news.read(archive)] == ["2026-10-07", "2026-10-05"]
    news.refresh(session, archive, feeds, today=date(2027, 1, 10))     # 95 days on: both are past the window
    assert news.read(archive) == []


def test_one_feed_refusing_costs_its_items_and_all_refusing_is_an_error(tmp_path):
    """An archive that silently stopped growing would go on answering "lately" with last month's news."""
    from backend.database.live import news
    archive = tmp_path / "wire.jsonl"
    feeds = (("ET", "https://e.test/rss"), ("Moneycontrol", "https://m.test/rss"))
    assert news.refresh(FeedSession({"https://e.test/rss": RSS}), archive, feeds) == 1
    with pytest.raises(RuntimeError, match="no news feed could be read"):
        news.refresh(FeedSession({}), archive, feeds)


def test_the_newswire_reads_the_newest_items_about_the_subject_up_to_the_as_of_date(tmp_path):
    from backend.database.live import news
    from backend.database.live.sources import NewsWire
    from backend.models.agent.finance import Subject
    archive = tmp_path / "wire.jsonl"
    rows = [{"link": f"https://n.test/{day}", "day": day, "source": "Mint", "title": f"Infosys update {day}", "summary": ""}
            for day in ("2026-10-07", "2026-10-06", "2026-10-05")]
    rows.append({"link": "https://n.test/tcs", "day": "2026-10-07", "source": "Mint", "title": "TCS update", "summary": ""})
    archive.write_text("".join(json.dumps(row) + "\n" for row in rows))
    wire = NewsWire(archive, items=2)
    found = wire.documents("why is infosys falling", [], as_of="2026-10-06", subject=Subject("INFY.NS", ("infosys",), None))
    assert [page.day for page in found] == ["2026-10-06", "2026-10-05"]
    assert found[0].text.startswith("Mint, 6 Oct 2026: Infosys update")
    assert [page.key for page in wire.documents("tcs news", ["tcs"])] == ["https://n.test/tcs"]
