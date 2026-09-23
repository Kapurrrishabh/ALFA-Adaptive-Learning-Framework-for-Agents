"""C4: a scored headline names its source, its licence and its instrument, or it is not served.

Each test here pins a failure that was measured on the real feeds, not one that was imagined. Three
of them are bugs this module shipped and had fixed: one publisher gives every item in a symbol's feed
the same URL, so link identity collapsed 26 stories into one; a story carried by two symbols' feeds
was deduplicated down to one instrument; and matching company names as substrings tagged every
headline about artificial intelligence to Intel.
"""

import json

import pytest

from backend.news import feed, sentiment

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
