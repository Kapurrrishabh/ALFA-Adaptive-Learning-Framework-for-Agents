"""Finance and economics articles from Wikipedia.

Two hosts, for one reason: en.wikipedia.org/robots.txt disallows `/w/`, which is where the
MediaWiki action API lives, but allows the ordinary `/wiki/Category:...` pages. So discovery reads
category pages, and content comes from api.wikimedia.org, the REST host meant for programmatic
access.
"""

from collections import deque
from urllib.parse import parse_qs, quote, urlparse

from bs4 import BeautifulSoup

from .. import config
from ..common.http import BlockedByHost
from ..extract import html_text

ARTICLE_URL = "https://en.wikipedia.org/wiki/{title}"
CATEGORY_URL = "https://en.wikipedia.org/wiki/{category}"
CONTENT_URL = "https://api.wikimedia.org/core/v1/wikipedia/en/page/{title}/html"
LICENCE = "CC BY-SA 4.0"
SOURCE = "wikipedia"

# Stubs are not worth a training example and they dominate the long tail of these categories.
MINIMUM_CHARACTERS = 1200
# Categories with thousands of members page in 200s. This bounds one category's cost.
MAX_CATEGORY_PAGES = 10

# Links to other kinds of page, which are navigation rather than articles.
_NAMESPACES = ("Category:", "Special:", "Help:", "Portal:", "Template:", "Wikipedia:", "File:")


def _members_on_page(html):
    """Returns (article titles, subcategory titles, next-page url) for one category page."""
    soup = BeautifulSoup(html, "lxml")

    def titles_under(element_id, keep_categories):
        container = soup.find(id=element_id)
        if container is None:
            return []
        found = []
        for link in container.select('a[href^="/wiki/"]'):
            title = link.get("title") or ""
            if not title:
                continue
            if title.startswith("Category:") == keep_categories:
                found.append(title)
        return found

    articles = [
        title
        for title in titles_under("mw-pages", keep_categories=False)
        if not title.startswith(_NAMESPACES)
    ]
    subcategories = titles_under("mw-subcategories", keep_categories=True)

    # The rendered "next page" link points at /w/index.php, which robots.txt disallows. Only the
    # pagefrom value is taken, and the caller rebuilds it as a /wiki/ URL, which is allowed.
    page_from = None
    for link in soup.select('a[href*="pagefrom="]'):
        if "next page" in link.get_text().lower():
            page_from = parse_qs(urlparse(link["href"]).query).get("pagefrom", [None])[0]
            break
    return articles, subcategories, page_from


def _category_members(session, category):
    articles, subcategories = [], []
    base = CATEGORY_URL.format(category=category.replace(" ", "_"))
    url = base

    for _ in range(MAX_CATEGORY_PAGES):
        page_articles, page_subcategories, page_from = _members_on_page(session.get_text(url))
        articles.extend(page_articles)
        subcategories.extend(page_subcategories)
        if not page_from:
            break
        url = f"{base}?pagefrom={quote(page_from)}"
    return articles, subcategories


def _discover_titles(session, wanted, log):
    """Breadth-first walk of the category roots, stopping once `wanted` titles are found."""
    titles, seen_categories = [], set()
    queue = deque((root, 0) for root in config.WIKIPEDIA_CATEGORIES)

    while queue and len(titles) < wanted:
        category, depth = queue.popleft()
        if category in seen_categories:
            continue
        seen_categories.add(category)

        try:
            articles, subcategories = _category_members(session, category)
        except BlockedByHost:
            raise
        except RuntimeError as error:
            log(f"  {category}: {error}")
            continue

        titles.extend(articles)
        if depth < config.WIKIPEDIA_CATEGORY_DEPTH:
            queue.extend((subcategory, depth + 1) for subcategory in subcategories)
        # Discovery runs before any article is fetched and can take many minutes, so it reports.
        if len(seen_categories) % 20 == 0:
            log(f"  discovering: {len(seen_categories)} categories read, {len(titles)} titles")

    # A page sits in several categories, so duplicates are expected here.
    unique = list(dict.fromkeys(titles))
    log(f"  {len(unique)} distinct articles across {len(seen_categories)} categories")
    return unique[:wanted]


def _safe_filename(title):
    return "".join(character if character.isalnum() else "_" for character in title)[:120] + ".txt"


def collect(session, manifest, settings, log):
    destination_root = config.RAW_DIR / SOURCE
    destination_root.mkdir(parents=True, exist_ok=True)

    titles = _discover_titles(session, settings["wikipedia_articles"], log)
    fetched, skipped_short = 0, 0

    for title in titles:
        url = ARTICLE_URL.format(title=title.replace(" ", "_"))
        destination = destination_root / _safe_filename(title)
        if manifest.already_have(url, destination):
            continue

        # A title such as "EV/EBITDA" would otherwise split the REST path and 404.
        encoded = quote(title.replace(" ", "_"), safe="")
        try:
            html = session.get_text(CONTENT_URL.format(title=encoded))
        except BlockedByHost:
            raise
        except RuntimeError as error:
            log(f"  {title}: {error}")
            continue

        text = html_text.extract(html)
        if len(text) < MINIMUM_CHARACTERS:
            skipped_short += 1
            continue
        destination.write_text(f"{title}\n\n{text}", encoding="utf-8")
        manifest.record(url, destination, SOURCE, LICENCE)
        fetched += 1
        if fetched % 200 == 0:
            log(f"  {fetched} articles saved")

    log(f"  {skipped_short} skipped as too short (under {MINIMUM_CHARACTERS} characters)")
    return fetched
