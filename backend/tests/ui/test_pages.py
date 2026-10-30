"""The web app in a real browser: the navs at three widths, every page's basics, leaving a page mid-load,
the search box and the theme toggle."""
import re

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed: pip install -e '.[dev]'")

from .conftest import KEY  # noqa: E402

pytestmark = pytest.mark.ui
WIDTHS = [390, 800, 1366]   # provisional: a phone, the nav breakpoint itself, a laptop
SECTIONS = ["Home", "Markets", "Portfolio", "Ask", "Learn"]
NO_PAGE = ["nowhere", "stock", "stock/%E0%A4"]
PAGES = ["home", "explore", "stock/TEST", "search?q=power", "portfolio", "signals", "builder", "chat", "learn",
         "performance", "reports", *NO_PAGE]
# Chrome logs a failed fetch without its address, so failures are read from the responses; an API 404 is how a missing model shows up
UNNAMED_FETCH_ERROR = "Failed to load resource"
LONG_NAME = "Parag Parikh Flexi Cap Fund Direct Plan Growth"


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

    def wait_until(self, condition, limit_ms=10000):
        waited = 0
        while not condition():
            self.page.wait_for_timeout(50)
            waited += 50
            assert waited < limit_ms, f"still waiting after {limit_ms} ms"


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


@pytest.mark.parametrize("name", NO_PAGE)
def test_an_address_with_no_page_says_so(visits, name):
    assert visits[(1366, name)]["h1"] == ["Page not found"]


def test_leaving_explore_mid_load_keeps_the_new_page(browser, base_url):
    # page.route cannot hold a request that a service worker sees, so this tab has none
    tab = Tab(browser, base_url, 1366, service_workers="block")
    held = []
    tab.page.route("**/ui/explore", lambda route: held.append(route))
    tab.page.evaluate("location.hash = '#/explore'")
    tab.wait_until(lambda: held)
    tab.page.evaluate("location.hash = '#/learn'")
    tab.page.wait_for_selector("input[placeholder^='Search lessons']")
    for route in held:
        route.fulfill(status=500, json={"detail": "late answer for the page that was left"})
    tab.settle()
    assert tab.page.locator("#view .welcome, #view .card.empty").count() == 0


def test_hopping_between_pages_mid_load_leaves_no_cancelled_request_unhandled(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    for name in ["stock/TEST", "explore", "builder", "portfolio", "chat", "learn"]:
        tab.page.evaluate("(h) => { location.hash = h; }", f"#/{name}")
        tab.page.wait_for_timeout(30)
    tab.settle()
    assert tab.errors == []


def test_leaving_a_stock_page_before_its_chart_options_load_shows_no_toast_later(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    held = []
    tab.page.route("**/ui/features", lambda route: held.append(route))
    tab.page.evaluate("location.hash = '#/stock/TEST'")
    tab.wait_until(lambda: held)
    tab.page.evaluate("location.hash = '#/learn'")
    tab.page.wait_for_selector("input[placeholder^='Search lessons']")
    for route in held:
        route.fulfill(status=500, json={"detail": "late answer for the page that was left"})
    tab.settle()
    assert tab.page.locator(".toast").all_inner_texts() == []


def test_tab_from_the_page_title_reaches_the_sub_nav(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/portfolio")
    tab.page.keyboard.press("Tab")
    assert tab.page.evaluate("document.activeElement.closest('.subnav') !== null")


def search_and_press_enter(tab, text):
    tab.page.fill("#q", text)
    tab.settle()
    tab.page.press("#q", "Enter")
    tab.settle()


def test_enter_on_a_matching_name_opens_its_stock_page(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    search_and_press_enter(tab, "Test")
    assert tab.page.evaluate("location.hash") == "#/stock/TEST"


def test_enter_on_a_name_no_stock_has_opens_search_not_a_stock_page(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    search_and_press_enter(tab, "Parag Parikh Flexi Cap")
    assert tab.page.evaluate("location.hash") == "#/search?q=Parag%20Parikh%20Flexi%20Cap"


def test_a_name_too_long_to_search_says_so_on_the_search_page(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    search_and_press_enter(tab, LONG_NAME)
    assert "at most 40 characters" in tab.page.inner_text("#view")


def test_typing_a_name_too_long_to_search_says_why(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.page.fill("#q", LONG_NAME)
    assert "at most 40 characters" in tab.page.locator(".toast").first.inner_text()


def test_a_late_answer_to_older_search_text_is_dropped(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    held = []
    tab.page.route(lambda url: url.endswith("/ui/search?q=Pe"), lambda route: held.append(route))
    tab.page.fill("#q", "Pe")
    tab.wait_until(lambda: held)
    tab.page.fill("#q", "Test")
    tab.page.wait_for_selector("#suggest.open")
    for route in held:
        route.continue_()
    tab.settle()
    assert tab.page.locator("#suggest .sym").all_inner_texts() == ["TEST"]


def test_arrow_up_with_no_suggestions_points_at_no_option(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.page.press("#q", "ArrowUp")
    assert tab.page.get_attribute("#q", "aria-activedescendant") is None


def test_theme_toggle_sends_no_request(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/stock/TEST")
    sent = len(tab.requests)
    tab.page.click("#theme")
    tab.settle()
    assert tab.requests[sent:] == []


THEME_COLOURS = """() => { const names = new Set();
  for (const sheet of document.styleSheets) if (sheet.href.startsWith(location.origin))
    for (const rule of sheet.cssRules) for (const p of rule.style || []) if (p.startsWith("--")) names.add(p);
  return [...names].map((n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim().toLowerCase()); }"""
ATTRIBUTES_WITH = """([root, colours]) => [...document.querySelectorAll(`${root} *`)].flatMap((el) => [...el.attributes]
  .filter((a) => colours.some((c) => a.value.toLowerCase().includes(c))).map((a) => `<${el.tagName} ${a.name}="${a.value}">`))"""


def toggle_theme(tab):
    """Switch the theme, and return the colours the old theme used that the new one never does."""
    before = set(tab.page.evaluate(THEME_COLOURS))
    tab.page.click("#theme")
    stale = sorted(c for c in before - set(tab.page.evaluate(THEME_COLOURS)) if re.fullmatch(r"#[0-9a-f]{3,8}", c))
    assert stale, "the two themes share every colour, so no check on them can fail"
    return stale


def show_chart_tab(tab, name):
    tab.page.click(f".mode-tabs button:text-is('{name}')")
    tab.settle()


@pytest.mark.parametrize("name, chart_tab", [("home", None), ("explore", None), ("stock/TEST", "Technical"), ("stock/TEST", "Intraday"),
                                             ("stock/TEST", "Price & projection"), ("portfolio", None), ("builder", None), ("performance", None)])
def test_theme_toggle_leaves_no_old_theme_colour(browser, base_url, name, chart_tab):
    tab = Tab(browser, base_url, 1366)
    tab.go(f"#/{name}")
    if chart_tab:
        show_chart_tab(tab, chart_tab)
    stale = toggle_theme(tab)
    assert tab.page.evaluate(ATTRIBUTES_WITH, ["#view", stale]) == []


def hide_a_chart_tab_during_a_theme_change(browser, base_url):
    tab = Tab(browser, base_url, 390)
    tab.go("#/stock/TEST")
    show_chart_tab(tab, "Price & projection")
    show_chart_tab(tab, "Technical")
    stale = toggle_theme(tab)
    show_chart_tab(tab, "Price & projection")
    return tab, stale


def test_a_chart_tab_hidden_during_a_theme_change_fits_its_width_when_shown(browser, base_url):
    tab, _ = hide_a_chart_tab_during_a_theme_change(browser, base_url)
    plot = tab.page.locator(".card:has(> .mode-tabs) .chart")
    assert plot.locator("svg").get_attribute("viewBox").split()[2] == str(plot.evaluate("el => el.clientWidth"))


def test_a_chart_tab_hidden_during_a_theme_change_takes_the_new_colours_when_shown(browser, base_url):
    tab, stale = hide_a_chart_tab_during_a_theme_change(browser, base_url)
    assert tab.page.evaluate(ATTRIBUTES_WITH, [".card:has(> .mode-tabs)", stale]) == []
