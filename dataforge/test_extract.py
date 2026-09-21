"""Pins the extraction chain. Deleted along with the rest of dataforge/.

The SEC branch is a fallback: nothing the collectors write today reaches it, because sec_edgar
stores already-extracted text. These tests are what keep it from rotting while unused.
"""

from dataforge.extract import html_text, sec_text
from dataforge.process import clean, pipeline
from dataforge.sources import gutenberg, sebi

_PROSE = (
    "The Company competes in a market that is highly seasonal and subject to rapid change, "
    "and its results in any one period are not a reliable indicator of future performance."
)

_SUBMISSION = f"""<SEC-DOCUMENT>0000000000-24-000001.txt
<DOCUMENT>
<TYPE>10-K
<TEXT>
<XBRL>
<html><body>
<div style="display:none"><ix:header>us-gaap:RevenueMember us-gaap:AssetsMember</ix:header></div>
<p>{_PROSE}</p>
</body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-10.1
<TEXT>
<html><body><p>This exhibit is an employment agreement and must not reach the corpus.</p></body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def test_primary_document_is_extracted_and_exhibits_are_not():
    text = sec_text.extract(_SUBMISSION)
    assert _PROSE in " ".join(text.split())
    assert "employment agreement" not in text


def test_inline_xbrl_wrapper_does_not_hide_the_filing():
    # <XBRL> wraps the primary document in modern filings. Treating it as a skip marker once cost
    # us every filing after 2019.
    assert sec_text.extract(_SUBMISSION) != ""


def test_hidden_taxonomy_header_is_dropped():
    assert "us-gaap:RevenueMember" not in html_text.extract(_SUBMISSION)


def test_pipeline_routes_a_submission_to_the_sec_extractor(tmp_path):
    path = tmp_path / "filing.txt"
    path.write_text(_SUBMISSION)
    assert _PROSE in " ".join(pipeline._extract(path).split())


def test_pipeline_returns_plain_text_unchanged(tmp_path):
    path = tmp_path / "article.txt"
    path.write_text(_PROSE)
    assert pipeline._extract(path) == _PROSE


def test_cleaner_keeps_prose_and_drops_a_numeric_table_row():
    text = f"{_PROSE}\n$1,234 $5,678 (912) 4.5% $9,101\n{_PROSE}"
    cleaned = clean.clean(text)
    assert _PROSE in cleaned
    assert "5,678" not in cleaned


def test_cleaner_keeps_the_english_half_of_a_bilingual_circular():
    # SEBI circulars carry Hindi and English on the same page, and the Hindi extracts as mangled
    # glyphs. Dropping the whole document would lose the English alongside it.
    hindi = "महोदय महोदया ͪवषय साइबर हमलɉ कȧ घटनाओं कȧ जानकारȣ देने के ͧलए बनाए गए पोट[ल"
    cleaned = clean.clean(f"{hindi}\n{_PROSE}")
    assert _PROSE in cleaned
    assert "महोदय" not in cleaned


def test_cleaner_keeps_a_line_with_a_few_accents():
    accented = (
        "The café levy and the naïve résumé of costs are restated in the note below, and the "
        "Société Générale exposure is disclosed separately in the schedule that follows it."
    )
    assert accented in clean.clean(accented)


def test_gutenberg_licence_wrapper_is_removed():
    book = (
        "Gutenberg header junk\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK THE WEALTH OF NATIONS ***\n"
        f"{_PROSE}\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK THE WEALTH OF NATIONS ***\n"
        "Licence terms follow and must not reach the corpus.\n"
    )
    stripped = gutenberg.strip_boilerplate(book)
    assert stripped == _PROSE
    assert "Licence terms" not in stripped


def test_budget_sample_measures_cleaned_size_not_extracted_size(monkeypatch):
    # The cap is spent in cleaned characters. Measuring extracted text put sec_edgar at 24% of the
    # corpus instead of the configured 50%, because cleaning drops most of a filing.
    extracted = f"$1,234 $5,678 (912) 4.5% $9,101\n{_PROSE}\n" * 10
    monkeypatch.setattr(pipeline, "_cached_text", lambda entry, log: extracted)
    average = pipeline._average_cleaned_characters([{}] * 10, clean.clean, None)
    assert 0 < average < len(extracted)


def test_sebi_attachment_is_read_from_the_iframe():
    # The circular body is a PDF behind an iframe. There is no <a href> to it, which is why an
    # anchor-only search found nothing and the page text alone is just a title.
    page = (
        '<html><body><a href="/legal/circulars/sep-2026/other_1.html">Related</a>'
        '<iframe src="../../../web/?file=https://www.sebi.gov.in/sebi_data/attachdocs/x/1.PDF">'
        "</iframe></body></html>"
    )
    assert sebi._attachment_url(page) == "https://www.sebi.gov.in/sebi_data/attachdocs/x/1.PDF"


def test_sebi_page_without_an_attachment_returns_none():
    assert sebi._attachment_url("<html><body><iframe src='/web/?file=x.html'></iframe></body></html>") is None
