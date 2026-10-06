"""HTML to text. Shared by the SEC extractor and anything else that fetches a web page."""

from bs4 import BeautifulSoup

# Tags whose text is never prose.
_DROP_TAGS = ("script", "style", "noscript", "head", "meta", "link")


def extract(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    # Hidden text is never prose. In inline-XBRL filings this one rule removes the whole taxonomy
    # header, which is otherwise tens of thousands of characters of us-gaap: tag names.
    for tag in soup.select('[style*="display:none"], [style*="display: none"]'):
        tag.decompose()
    # Tables in filings are dense numeric grids. Newline-separating cells keeps them from
    # collapsing into one unreadable line, which the cleaner can then drop by line length.
    return soup.get_text("\n")
