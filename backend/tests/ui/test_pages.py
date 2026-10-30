"""The web app in a real browser: the navs at three widths, every page's basics, leaving a page mid-load,
the search box and the theme toggle."""
import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed: pip install -e '.[dev]'")

from .conftest import KEY  # noqa: E402

pytestmark = pytest.mark.ui
WIDTHS = [390, 800, 1366]   # provisional: a phone, the nav breakpoint itself, a laptop
SECTIONS = ["Home", "Markets", "Portfolio", "Ask", "Learn"]
PAGES = ["home", "explore", "stock/TEST", "search?q=power", "portfolio", "signals", "builder", "chat", "learn",
         "performance", "reports", "nowhere"]
# Chrome logs every failed fetch as "Failed to load resource" without its address, so failures are read from the
# responses instead. An API 404 is how the empty portfolio, the unbuilt track record and a missing model show up.
UNNAMED_FETCH_ERROR = "Failed to load resource"


class Tab:
    """One signed-in browser tab on the app, counting its requests in flight and the errors it logs."""

    def __init__(self, browser, base_url, width, **context_options):
        self.context = browser.new_context(viewport={"width": width, "height": 900}, **context_options)
        self.page = self.context.new_page()
        self.inflight, self.requests, self.errors = 0, [], []
        self.page.on("request", self._started)
        self.page.on("requestfinished", self._ended)
        self.page.on("requestfailed", self._ended)
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("console", lambda m: self.errors.append(m.text) if m.type == "error" and not m.text.startswith(UNNAMED_FETCH_ERROR) else None)
        self.page.on("response", lambda r: self.errors.append(f"{r.status} {r.url}") if r.status >= 500 or (r.status >= 400 and "/static/" in r.url) else None)
        self.page.goto(f"{base_url}/#key={KEY}")
        self.settle()

    def _started(self, request):
        self.inflight += 1
        self.requests.append(request.url)

    def _ended(self, _request):
        self.inflight -= 1

    def go(self, hash_):
        self.page.evaluate("(h) => { location.hash = h; }", hash_)
        self.settle()

    def settle(self, quiet_ms=300, limit_ms=30000):
        """Wait until no request has been in flight for quiet_ms."""
        quiet = waited = 0
        while quiet < quiet_ms:
            self.page.wait_for_timeout(50)
            waited += 50
            quiet = quiet + 50 if self.inflight == 0 else 0
            assert waited < limit_ms, f"requests still in flight after {limit_ms} ms"


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture(scope="module")
def visits(browser, base_url):
    """What each page showed at each width: its h1 texts, the errors it logged, and whether it scrolled sideways."""
    out = {}
    for width in WIDTHS:
        tab = Tab(browser, base_url, width)
        out[(width, "navs")] = tab.page.locator("#nav a:visible, #bnav a:visible").all_inner_texts()
        for name in PAGES:
            seen = len(tab.errors)
            tab.go(f"#/{name}")
            out[(width, name)] = {"h1": tab.page.locator("h1").all_inner_texts(), "errors": tab.errors[seen:],
                                  "sideways": tab.page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")}
        tab.context.close()
    return out


@pytest.mark.parametrize("width", WIDTHS)
def test_every_section_has_one_visible_nav_link(visits, width):
    assert sorted(t.strip() for t in visits[(width, "navs")]) == sorted(SECTIONS)


@pytest.mark.parametrize("width", WIDTHS)
def test_every_page_has_exactly_one_h1(visits, width):
    assert {name: len(visits[(width, name)]["h1"]) for name in PAGES} == {name: 1 for name in PAGES}


@pytest.mark.parametrize("width", WIDTHS)
def test_no_page_logs_an_error(visits, width):
    assert {name: visits[(width, name)]["errors"] for name in PAGES if visits[(width, name)]["errors"]} == {}


@pytest.mark.parametrize("width", WIDTHS)
def test_no_page_scrolls_sideways(visits, width):
    assert [name for name in PAGES if visits[(width, name)]["sideways"]] == []


def test_an_unknown_page_says_so(visits):
    assert visits[(1366, "nowhere")]["h1"] == ["Page not found"]


def test_leaving_explore_mid_load_keeps_the_new_page(browser, base_url):
    # page.route cannot hold a request that a service worker sees, so this tab has none
    tab = Tab(browser, base_url, 1366, service_workers="block")
    held = []
    tab.page.route("**/ui/explore", lambda route: held.append(route))
    tab.page.evaluate("location.hash = '#/explore'")
    tab.page.wait_for_function("document.querySelector('.welcome') !== null")
    tab.page.evaluate("location.hash = '#/learn'")
    tab.page.wait_for_selector("input[placeholder^='Search lessons']")
    for route in held:
        route.fulfill(status=500, json={"detail": "late answer for the page that was left"})
    tab.settle()
    assert tab.page.locator("#view .welcome, #view .card.empty").count() == 0


def test_hopping_between_pages_mid_load_logs_no_error(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    for name in ["stock/TEST", "explore", "builder", "portfolio", "chat", "learn"]:
        tab.page.evaluate("(h) => { location.hash = h; }", f"#/{name}")
        tab.page.wait_for_timeout(30)
    tab.settle()
    assert tab.errors == []


def test_enter_on_a_name_no_stock_has_opens_search_not_a_stock_page(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.page.fill("#q", "Parag Parikh Flexi Cap")
    tab.settle()
    tab.page.press("#q", "Enter")
    tab.settle()
    assert tab.page.evaluate("location.hash") == "#/search?q=Parag%20Parikh%20Flexi%20Cap"


def test_theme_toggle_sends_no_request(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/stock/TEST")
    sent = len(tab.requests)
    tab.page.click("#theme")
    tab.settle()
    assert tab.requests[sent:] == []


def test_theme_toggle_recolours_the_price_chart(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/stock/TEST")
    grid = tab.page.locator("#view .chart svg line").first
    before = grid.get_attribute("stroke")
    tab.page.click("#theme")
    assert grid.get_attribute("stroke") != before
