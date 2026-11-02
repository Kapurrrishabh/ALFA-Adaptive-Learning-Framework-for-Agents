"""The web app in a real browser: the navs at four widths, every page's basics, leaving a page mid-load,
the search box, the theme toggle, the shared dialog, toasts, and error and empty states, and what a keyboard
or screen reader user gets."""
import re
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed: pip install -e '.[dev]'")

from backend.paths import FRONTEND  # noqa: E402
from .conftest import EMAIL, FUND, PASSWORD, RUN_REQUESTS, sign_in  # noqa: E402

pytestmark = pytest.mark.ui
WIDTHS = [360, 390, 800, 1366]   # provisional: a small phone, a phone, the nav breakpoint itself, a laptop
AXE_WIDTHS = [390, 1366]   # every element shown at 800 px is also shown at one of these
THEMES = ["dark", "light"]
SECTIONS = ["Home", "Markets", "Portfolio", "Ask", "Learn"]
NO_PAGE = ["nowhere", "stock", "stock/%E0%A4"]
PAGES = ["home", "explore", "stock/TEST", "search?q=power", "portfolio", "signals", "builder", "chat", "learn",
         "performance", "reports", "guide", f"fund/{FUND}", *NO_PAGE]
SIGNED_OUT_PAGES = ["signin", "signup", "verify?token=old", "reset", "reset?token=old"]
# Chrome logs a failed fetch without its address, so failures are read from the responses; an API 404 is how a missing model shows up,
# and a 429 means the run outgrew the test key's rate limit, which otherwise shows only as a click that times out
UNNAMED_FETCH_ERROR = "Failed to load resource"
LONG_NAME = "Parag Parikh Flexi Cap Fund Direct Plan Growth"
EMOJI = re.compile("[\u2600-\u27bf\U0001f300-\U0001faff]")
CUT_LABELS = """(root) => [...document.querySelectorAll(`${root} svg[viewBox] text`)].filter((t) => t.getClientRects().length).flatMap((t) => {
  const v = t.ownerSVGElement.viewBox.baseVal, b = t.getBBox(), slack = 0.5;
  return b.x < v.x - slack || b.y < v.y - slack || b.x + b.width > v.x + v.width + slack || b.y + b.height > v.y + v.height + slack ? [t.textContent] : []; })"""
# a line chart's transparent hover area is its plot
LINE_PLOT_SHARES = """() => [...document.querySelectorAll("#view svg[aria-label*='. Latest: ']")].map((svg) =>
  svg.querySelector(":scope > rect[fill=transparent]").width.baseVal.value / svg.viewBox.baseVal.width)"""
CURRENT_TAB_CUT = """() => { const tab = document.querySelector("#view .subnav [aria-current]"); if (!tab) return false;
  const t = tab.getBoundingClientRect(), nav = tab.parentElement.getBoundingClientRect(); return t.left < nav.left || t.right > nav.right; }"""
CUT_CHART_CONTROLS = """() => { const card = document.querySelector(".card:has(> .mode-tabs)").getBoundingClientRect();
  return [...document.querySelectorAll(".card:has(> .mode-tabs) :is(.mode-tabs, .seg) button")].filter((b) => {
    const r = b.getBoundingClientRect(), box = b.parentElement.getBoundingClientRect();
    return r.left < box.left || r.right > box.right || box.left < card.left || box.right > card.right; }).map((b) => b.textContent); }"""


class Tab:
    """One browser tab on the app, signed in unless told otherwise, counting its requests in flight and the errors it logs."""

    def __init__(self, browser, base_url, width, signed_in=True, **context_options):
        self.context = browser.new_context(viewport={"width": width, "height": 900}, **context_options)
        self.page = self.context.new_page()
        self.inflight, self.requests, self.errors = 0, [], []
        self.page.on("request", self._started)
        self.page.on("requestfinished", self._ended)
        self.page.on("requestfailed", self._ended)
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("console", lambda m: self.errors.append(m.text) if m.type == "error" and not m.text.startswith(UNNAMED_FETCH_ERROR) else None)
        self.page.on("response", lambda r: self.errors.append(f"{r.status} {r.url}") if r.status >= 500 or r.status == 429 or (r.status >= 400 and "/static/" in r.url) else None)
        if signed_in:
            sign_in(self.context, base_url)
        self.page.goto(base_url)
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

    def open_every_disclosure(self):
        """Open each <details> on the page, so what waits behind "Show details" or "Chart options" is checked too."""
        self.page.evaluate("document.querySelectorAll('#view details').forEach((d) => { d.open = true; })")
        self.settle()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture(autouse=True)
def close_tabs(browser):
    """Close the tabs each test opened: left open, they keep animating and slow every test after them."""
    yield
    for context in browser.contexts:
        context.close()


@pytest.fixture(scope="module")
def visits(browser, base_url):
    """What each page showed at each width, every disclosure open: its h1 texts, the errors it logged, whether it scrolled sideways,
    the text of its empty states and banners, the names of its charts, the chart labels cut off at a chart's edge,
    the share of each line chart's width its plot had, and whether its current sub-nav tab was cut off."""
    out = {}
    for width in WIDTHS:
        tab = Tab(browser, base_url, width)
        out[(width, "navs")] = tab.page.locator("#nav a:visible, #bnav a:visible").all_inner_texts()
        for name in PAGES:
            seen = len(tab.errors)
            tab.go(f"#/{name}")
            tab.open_every_disclosure()
            out[(width, name)] = {"h1": tab.page.locator("h1").all_inner_texts(), "errors": tab.errors[seen:],
                                  "sideways": tab.page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"),
                                  "states": tab.page.locator("#view .empty, #view .banner").all_inner_texts(),
                                  "charts": tab.page.locator("#view svg[role=img]").evaluate_all("els => els.map((el) => el.getAttribute('aria-label'))"),
                                  "cut": tab.page.evaluate(CUT_LABELS, "#view"), "plots": tab.page.evaluate(LINE_PLOT_SHARES),
                                  "tab_cut": tab.page.evaluate(CURRENT_TAB_CUT)}
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


@pytest.mark.parametrize("width", WIDTHS)
def test_no_empty_state_or_banner_shows_an_emoji(visits, width):
    assert {name: visits[(width, name)]["states"] for name in PAGES if EMOJI.search("".join(visits[(width, name)]["states"]))} == {}


@pytest.mark.parametrize("name", NO_PAGE)
def test_an_address_with_no_page_says_so(visits, name):
    assert visits[(1366, name)]["h1"] == ["Page not found"]


def test_every_chart_is_named_from_its_data(visits):
    unnamed = {name: [c for c in visits[(1366, name)]["charts"] if not c or re.search(r"undefined|NaN|\[object", c)] for name in PAGES}
    assert {name: charts for name, charts in unnamed.items() if charts} == {}


@pytest.mark.parametrize("width", WIDTHS)
def test_no_chart_label_is_cut_off_at_the_chart_edge(visits, width):
    assert {name: visits[(width, name)]["cut"] for name in PAGES if visits[(width, name)]["cut"]} == {}


@pytest.mark.parametrize("width", WIDTHS)
def test_every_line_chart_keeps_half_its_width_for_the_plot(visits, width):
    # PROVISIONAL: half; the names at the line ends once left a 390 px phone's plot about a third of its chart
    shares = {name: visits[(width, name)]["plots"] for name in PAGES if visits[(width, name)]["plots"]}
    assert shares, "no page drew a line chart, so this check cannot fail"
    assert {name: s for name, s in shares.items() if min(s) < 0.5} == {}


@pytest.mark.parametrize("width", WIDTHS)
def test_the_current_sub_nav_tab_is_in_view(visits, width):
    assert [name for name in PAGES if visits[(width, name)]["tab_cut"]] == []


@pytest.mark.parametrize("chart_tab", ["Technical", "Intraday", "Price & projection"])
def test_no_stock_chart_label_is_cut_off_on_a_phone(browser, base_url, chart_tab):
    tab = Tab(browser, base_url, 390)
    tab.go("#/stock/TEST")
    show_chart_tab(tab, chart_tab)
    assert tab.page.evaluate(CUT_LABELS, "#view") == []


def test_no_label_in_show_details_is_cut_off_on_a_phone_even_with_a_one_digit_rsi(browser, base_url):
    tab = Tab(browser, base_url, 360, service_workers="block")

    def one_digit_rsi(route):
        indicators = route.fetch().json()
        route.fulfill(json={**indicators, "rsi": [*indicators["rsi"][:-1], 5.0]})
    tab.page.route("**/indicators?*", one_digit_rsi)
    tab.go("#/stock/TEST")
    tab.open_every_disclosure()
    assert tab.page.evaluate(CUT_LABELS, ".stock-details") == []


@pytest.mark.parametrize("chart_tab", ["Technical", "Intraday", "Price & projection"])
def test_every_chart_control_fits_on_a_small_phone(browser, base_url, chart_tab):
    tab = Tab(browser, base_url, 360)
    tab.go("#/stock/TEST")
    show_chart_tab(tab, chart_tab)
    assert tab.page.evaluate(CUT_CHART_CONTROLS) == []


def test_on_a_phone_the_verdict_comes_before_the_chart(browser, base_url):
    tab = Tab(browser, base_url, 390)
    tab.go("#/stock/TEST")
    verdict, chart = (tab.page.locator(card).bounding_box() for card in (".verdict", ".card:has(> .mode-tabs)"))
    assert verdict["y"] + verdict["height"] <= chart["y"]


def test_a_stock_page_asks_for_no_analyst_note_on_its_own(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/stock/TEST")
    assert [url for url in tab.requests if "/analyst" in url] == []


def test_explain_in_plain_words_asks_for_the_analyst_note(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/stock/TEST")
    tab.page.click("button:text-is('Explain in plain words')")
    tab.settle()
    assert [url for url in tab.requests if "/analyst" in url] == [f"{base_url}/ui/stock/TEST/analyst"]


def paths_asked_with_every_stock_panel_open(browser, base_url):
    """The paths a stock page asked for once Show details and the Research tab were opened too."""
    tab = Tab(browser, base_url, 1366)
    sent = len(tab.requests)
    tab.go("#/stock/TEST")
    tab.page.click("summary:text-is('Show details')")
    tab.page.get_by_role("tab", name="Research").click()
    tab.settle()
    return [urlsplit(url).path for url in tab.requests[sent:]]


@pytest.mark.parametrize("path", ["/ui/stock/TEST/technical", "/stocks/TEST/analysis", "/ui/stock/TEST/verdict"])
def test_a_stock_page_asks_for_each_of_its_data_once(browser, base_url, path):
    assert paths_asked_with_every_stock_panel_open(browser, base_url).count(path) == 1


def test_on_a_phone_the_chat_send_box_is_above_the_bottom_nav(browser, base_url):
    tab = Tab(browser, base_url, 360)
    tab.go("#/chat")
    composer, nav = tab.page.locator(".composer").bounding_box(), tab.page.locator("#bnav").bounding_box()
    assert composer["y"] + composer["height"] <= nav["y"], (composer, nav)


def test_each_pattern_chart_names_its_proven_edges(visits):
    names = [c for c in visits[(1366, "performance")]["charts"] if c.startswith("Excess return")]
    assert [c.rsplit(". ", 1)[1] for c in names] == ["Proven edge: none", "Proven edge: double top"]


@pytest.fixture(scope="module")
def axe_findings(browser, base_url):
    """axe's serious and critical findings on every page, every disclosure open, in each theme, at a phone's and a laptop's width:
    each rule with the elements it flagged."""
    axe = pytest.importorskip("axe_playwright_python.sync_playwright", reason="axe is not installed: pip install -e '.[dev]'").Axe()
    out = {}
    for theme in THEMES:
        for width in AXE_WIDTHS:
            for signed_in, names in [(True, PAGES), (False, SIGNED_OUT_PAGES)]:
                tab = Tab(browser, base_url, width, signed_in=signed_in)
                if tab.page.evaluate("document.documentElement.dataset.theme") != theme:
                    tab.page.click("#theme")
                for name in names:
                    tab.go(f"#/{name}")
                    tab.open_every_disclosure()
                    out[(theme, width, name)] = {v["id"]: [n["target"] for n in v["nodes"]] for v in axe.run(tab.page).response["violations"]
                                                 if v["impact"] in ("serious", "critical")}
                tab.context.close()
    return out


@pytest.mark.parametrize("width", AXE_WIDTHS)
@pytest.mark.parametrize("theme", THEMES)
def test_no_page_has_a_serious_accessibility_finding(axe_findings, theme, width):
    assert {name: axe_findings[(theme, width, name)] for name in PAGES if axe_findings[(theme, width, name)]} == {}


@pytest.mark.parametrize("width", AXE_WIDTHS)
@pytest.mark.parametrize("theme", THEMES)
def test_no_sign_in_page_has_a_serious_accessibility_finding(axe_findings, theme, width):
    assert {name: axe_findings[(theme, width, name)] for name in SIGNED_OUT_PAGES if axe_findings[(theme, width, name)]} == {}


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


def test_cleanup_a_page_registers_after_it_was_left_still_runs(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.page.evaluate("""() => ROUTES.push({ path: "late", title: "Late", section: "Home", render: async (view, ctx) => {
      await new Promise((resolve) => { window.finishLate = resolve; }); ctx.onLeave(() => { window.cleanedUp = true; }); } })""")
    tab.go("#/late")
    tab.go("#/learn")
    assert tab.page.evaluate("async () => { finishLate(); await new Promise((r) => setTimeout(r)); return window.cleanedUp === true; }")


def test_the_guide_shows_its_sections_to_someone_not_signed_in(browser, base_url):
    tab = Tab(browser, base_url, 1366, signed_in=False)
    tab.go("#/guide")
    sections = [line.removeprefix("# ") for line in (FRONTEND / "guide.md").read_text().splitlines() if line.startswith("# ")]
    assert tab.page.locator("#view h2").all_inner_texts() == sections


def test_a_page_link_in_the_guide_opens_that_page_in_the_same_tab(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/guide")
    tab.page.locator("#view .md").get_by_role("link", name="Holdings", exact=True).click()
    assert tab.page.evaluate("location.hash") == "#/portfolio"


PRIVATE_PAGES = [p for p in PAGES if p != "guide"]
NEW_PASSWORD = "a kettle sings on the stove"


def signed_out_at(browser, base_url, hash_):
    tab = Tab(browser, base_url, 1366, signed_in=False)
    tab.go(hash_)
    return tab


def sign_in_with_the_form(tab, email=EMAIL, password=PASSWORD):
    tab.page.get_by_label("Email").fill(email)
    tab.page.get_by_label("Password").fill(password)
    tab.page.click("#view button[type=submit]")
    tab.settle()


def token_sent(outbox):
    return next(word for word in outbox.sent[-1].get_content().split() if "token=" in word).split("token=", 1)[1]


def new_account(base_url, outbox, email, verified=True):
    """An account of its own, made through the sign-up routes, so a test can change it without touching the shared one."""
    with httpx.Client(base_url=base_url) as client:
        client.post("/auth/register", json={"email": email, "password": PASSWORD}).raise_for_status()
        if verified:
            client.post("/auth/verify", json={"token": token_sent(outbox)}).raise_for_status()


def test_signed_out_every_private_page_lands_on_sign_in(browser, base_url):
    tab = Tab(browser, base_url, 1366, signed_in=False)
    landed = {}
    for name in PRIVATE_PAGES:
        tab.go(f"#/{name}")
        landed[name] = tab.page.evaluate("location.hash").split("?")[0]
    assert landed == {name: "#/signin" for name in PRIVATE_PAGES}


def test_signing_in_with_email_reaches_home(browser, base_url):
    tab = Tab(browser, base_url, 1366, signed_in=False)
    sign_in_with_the_form(tab)
    assert tab.page.locator("h1").all_inner_texts() == ["Home"]


def test_signing_in_returns_to_the_page_that_asked_for_it(browser, base_url):
    tab = signed_out_at(browser, base_url, "#/portfolio")
    sign_in_with_the_form(tab)
    assert tab.page.evaluate("location.hash") == "#/portfolio"


def test_a_session_that_ends_mid_visit_sends_the_next_request_to_sign_in(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.context.clear_cookies()
    tab.go("#/explore")
    assert tab.page.evaluate("location.hash") == "#/signin?next=%23%2Fexplore"


def test_a_wrong_password_is_refused_in_plain_words(browser, base_url):
    tab = signed_out_at(browser, base_url, "#/signin")
    sign_in_with_the_form(tab, password="not the password at all")
    assert tab.page.inner_text("#view .dialog-error") == "That email and password do not match an account."


def test_a_next_page_off_the_site_is_never_followed(browser, base_url):
    tab = signed_out_at(browser, base_url, "#/signin?next=%2F%2Fevil.example")
    sign_in_with_the_form(tab)
    assert tab.page.evaluate("location.href") == f"{base_url}/#/home"


def signed_out_by_the_menu(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/chat")
    tab.page.evaluate("sessionStorage.setItem('si-chat-log', '[]')")
    tab.page.click("#account summary")
    tab.page.click("#account button:text-is('Sign out')")
    tab.settle()
    return tab


def test_sign_out_returns_to_sign_in(browser, base_url):
    assert signed_out_by_the_menu(browser, base_url).page.evaluate("location.hash") == "#/signin"


def test_sign_out_clears_the_chat_log(browser, base_url):
    assert signed_out_by_the_menu(browser, base_url).page.evaluate("sessionStorage.getItem('si-chat-log')") is None


def test_sign_out_ends_the_session_on_the_server(browser, base_url):
    tab = signed_out_by_the_menu(browser, base_url)
    assert tab.context.request.get(f"{base_url}/users/me").status == 401


def test_the_account_menu_holds_what_the_guide_says(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.page.click("#account summary")
    assert tab.page.locator("#account .menu > *").all_inner_texts() == [EMAIL, "Guide", "Privacy", "Delete my account", "Sign out"]


def test_the_account_menu_closes_on_escape(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.page.click("#account summary")
    tab.page.keyboard.press("Escape")
    assert tab.page.locator("#account details[open]").count() == 0


def test_the_sign_in_page_links_to_the_guide(browser, base_url):
    tab = signed_out_at(browser, base_url, "#/signin")
    tab.page.get_by_role("link", name="guide", exact=True).click()
    tab.settle()
    assert tab.page.locator("h1").all_inner_texts() == ["How to use StockIntel"]


def test_deleting_the_account_says_what_goes(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.page.click("#account summary")
    tab.page.click("#account button:text-is('Delete my account')")
    text = tab.page.inner_text("[role=dialog]")
    assert [word for word in [EMAIL, "portfolio", "trades", "plans", "reports", "cannot be undone"] if word not in text] == []


def test_a_deleted_account_can_no_longer_sign_in(browser, base_url, outbox):
    new_account(base_url, outbox, "leaving@example.com")
    tab = signed_out_at(browser, base_url, "#/signin")
    sign_in_with_the_form(tab, "leaving@example.com")
    tab.page.click("#account summary")
    tab.page.click("#account button:text-is('Delete my account')")
    tab.page.click("[role=dialog] button[type=submit]")
    tab.settle()
    sign_in_with_the_form(tab, "leaving@example.com")
    assert tab.page.inner_text("#view .dialog-error") == "That email and password do not match an account."


def test_a_bad_google_sign_in_link_shows_a_plain_error(browser, base_url):
    tab = Tab(browser, base_url, 1366, signed_in=False)
    tab.page.goto(f"{base_url}/login/google?code=bad&state=bad")
    tab.settle()
    assert tab.page.inner_text("#view .dialog-error") == "Signing in with Google did not finish. This sign-in link is broken. Start again."


def test_a_bad_google_sign_in_link_leaves_no_code_in_the_address(browser, base_url):
    tab = Tab(browser, base_url, 1366, signed_in=False)
    tab.page.goto(f"{base_url}/login/google?code=bad&state=bad")
    tab.settle()
    assert "code=" not in tab.page.evaluate("location.href")


def test_continue_with_google_comes_back_signed_in_to_the_page_asked_for(browser, base_url):
    tab = signed_out_at(browser, base_url, "#/portfolio")
    # Google's page answers at once, sending the browser back with a code and the state it was given
    tab.page.route("https://accounts.example/**", lambda route: route.fulfill(status=302, headers={
        "Location": f"{base_url}/login/google?code=ok&state={parse_qs(urlsplit(route.request.url).query)['state'][0]}"}))
    tab.page.click("#view button:text-is('Continue with Google')")
    tab.page.wait_for_url(f"{base_url}/#/portfolio")
    tab.settle()
    assert tab.page.locator("h1").all_inner_texts() == ["Holdings"]


def test_a_new_account_confirmed_by_its_emailed_link_can_sign_in(browser, base_url, outbox):
    tab = signed_out_at(browser, base_url, "#/signup")
    tab.page.get_by_label("Email").fill("newcomer@example.com")
    tab.page.get_by_label("Password").fill(PASSWORD)
    tab.page.click("#view button[type=submit]")
    tab.settle()
    tab.go(f"#/verify?token={token_sent(outbox)}")
    tab.go("#/signin")
    sign_in_with_the_form(tab, "newcomer@example.com")
    assert tab.page.locator("h1").all_inner_texts() == ["Home"]


def test_a_short_password_is_refused_with_the_servers_rule(browser, base_url):
    tab = signed_out_at(browser, base_url, "#/signup")
    tab.page.get_by_label("Email").fill("brief@example.com")
    tab.page.get_by_label("Password").fill("short words")
    tab.page.click("#view button[type=submit]")
    tab.settle()
    assert tab.page.inner_text("#view .dialog-error").startswith("Use ")


def resent_from(tab, outbox):
    sent = len(outbox.sent)
    tab.page.click("#view button:text-is('Send the link again')")
    tab.settle()
    return outbox.sent[sent:]


def test_an_unconfirmed_sign_in_can_send_the_link_again(browser, base_url, outbox):
    new_account(base_url, outbox, "unsure@example.com", verified=False)
    tab = signed_out_at(browser, base_url, "#/signin")
    sign_in_with_the_form(tab, "unsure@example.com")
    assert [m["To"] for m in resent_from(tab, outbox)] == ["unsure@example.com"]


def test_signing_up_again_with_an_unconfirmed_email_can_send_the_link_again(browser, base_url, outbox):
    new_account(base_url, outbox, "again@example.com", verified=False)
    tab = signed_out_at(browser, base_url, "#/signup")
    tab.page.get_by_label("Email").fill("again@example.com")
    tab.page.get_by_label("Password").fill(PASSWORD)
    tab.page.click("#view button[type=submit]")
    tab.settle()
    assert [m["To"] for m in resent_from(tab, outbox)] == ["again@example.com"]


def test_a_broken_confirmation_link_can_send_the_link_again(browser, base_url, outbox):
    new_account(base_url, outbox, "expired@example.com", verified=False)
    tab = signed_out_at(browser, base_url, "#/verify?token=old")
    tab.page.get_by_label("Email").fill("expired@example.com")
    assert [m["To"] for m in resent_from(tab, outbox)] == ["expired@example.com"]


def test_a_forgotten_password_is_replaced_through_the_emailed_link(browser, base_url, outbox):
    new_account(base_url, outbox, "forgetful@example.com")
    tab = signed_out_at(browser, base_url, "#/reset")
    tab.page.get_by_label("Email").fill("forgetful@example.com")
    tab.page.click("#view button[type=submit]")
    tab.settle()
    tab.go(f"#/reset?token={token_sent(outbox)}")
    tab.page.get_by_label("New password").fill(NEW_PASSWORD)
    tab.page.click("#view button[type=submit]")
    tab.settle()
    tab.go("#/signin")
    sign_in_with_the_form(tab, "forgetful@example.com", NEW_PASSWORD)
    assert tab.page.locator("h1").all_inner_texts() == ["Home"]


def test_tab_from_the_page_title_reaches_the_sub_nav(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/portfolio")
    tab.page.keyboard.press("Tab")
    assert tab.page.evaluate("document.activeElement.closest('.subnav') !== null")


def tab_to(tab, selector, presses=40):
    """Press Tab, as a keyboard user would, until focus is on an element matching selector."""
    tab.page.locator(selector).first.wait_for()   # settle() can end before a starved page has drawn it
    for _ in range(presses):
        tab.page.keyboard.press("Tab")
        if tab.page.evaluate("(sel) => document.activeElement.matches(sel)", selector):
            return
    raise AssertionError(f"{presses} presses of Tab never reached {selector}")


@pytest.mark.parametrize("name, opens", [("portfolio", "#/stock/TEST"), ("builder", "#/stock/(TEST|PEER)")])
def test_tab_then_enter_opens_a_stock_from_its_table(browser, base_url, name, opens):
    tab = Tab(browser, base_url, 1366)
    tab.go(f"#/{name}")
    tab_to(tab, "table a")
    tab.page.keyboard.press("Enter")
    assert re.fullmatch(opens, tab.page.evaluate("location.hash"))


def press_enter_on_the_pattern_card(browser, base_url):
    """The stock page after Enter on the card of its one pattern, which the page highlights first."""
    tab = Tab(browser, base_url, 1366)
    tab.go("#/stock/TEST")
    tab_to(tab, ".pattern-card")
    tab.page.keyboard.press("Enter")
    return tab


def test_enter_on_the_highlighted_pattern_card_shows_all_patterns(browser, base_url):
    tab = press_enter_on_the_pattern_card(browser, base_url)
    assert tab.page.locator(".pattern-card").get_attribute("aria-pressed") == "false"


def test_enter_on_a_pattern_card_keeps_the_focus_on_it(browser, base_url):
    tab = press_enter_on_the_pattern_card(browser, base_url)
    assert tab.page.evaluate("document.activeElement.matches('.pattern-card')")


def test_the_range_pressed_last_is_the_only_one_announced_as_pressed(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/stock/TEST")
    ranges = tab.page.locator(".card:has(> .mode-tabs) .seg")
    ranges.get_by_role("button", name="1M", exact=True).click()
    assert ranges.get_by_role("button", pressed=True).all_inner_texts() == ["1M"]


def test_the_chart_tab_shown_is_the_one_announced_as_selected(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/stock/TEST")
    tabs = tab.page.get_by_role("tablist", name="Chart")
    tabs.get_by_role("tab", name="Intraday").click()
    assert tabs.get_by_role("tab", selected=True).all_inner_texts() == ["Intraday"]


def press_on_the_chart_tabs(browser, base_url, *keys):
    """The stock page after Tab reaches its chart tabs and keys are pressed there."""
    tab = Tab(browser, base_url, 1366)
    tab.go("#/stock/TEST")
    tab_to(tab, "[role=tab]")
    for key in keys:
        tab.page.keyboard.press(key)
        tab.settle()
    return tab


def test_enter_on_a_chart_tab_keeps_the_focus_on_it(browser, base_url):
    tab = press_on_the_chart_tabs(browser, base_url, "Enter")
    assert tab.page.evaluate("document.activeElement.textContent") == "Technical"


def test_arrow_right_on_a_chart_tab_shows_and_focuses_the_next_one(browser, base_url):
    tab = press_on_the_chart_tabs(browser, base_url, "ArrowRight")
    assert tab.page.evaluate("[document.activeElement.textContent, document.activeElement.ariaSelected]") == ["Intraday", "true"]


def test_tab_from_the_chosen_chart_tab_goes_into_its_chart(browser, base_url):
    tab = press_on_the_chart_tabs(browser, base_url, "ArrowRight", "Tab")
    assert tab.page.evaluate("document.activeElement.closest('[role=tabpanel]') !== null")


def test_a_late_answer_for_a_details_tab_arrowed_past_does_not_replace_the_one_shown(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    held = []
    tab.go("#/stock/TEST")
    tab.page.route("**/stocks/TEST/analysis", lambda route: held.append(route))
    tab.page.get_by_role("tab", name="Overview").focus()
    tab.page.keyboard.press("ArrowRight")
    tab.wait_until(lambda: held)
    tab.page.keyboard.press("ArrowRight")
    news = tab.page.get_by_text("No recent news found.").or_(tab.page.get_by_text("News unavailable"))
    news.wait_for()
    for route in held:
        route.continue_()
    tab.settle()
    assert tab.page.get_by_role("tab", selected=True).all_inner_texts()[-1] == "News" and news.count() == 1
    assert tab.page.get_by_text("Every engine's evidence").count() == 0


def test_a_sector_left_out_of_the_basket_is_announced_as_pressed(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/builder")
    sectors = tab.page.get_by_role("group", name="Sectors to leave out")
    sectors.get_by_role("button", name="Energy", exact=True).click()
    assert sectors.get_by_role("button", pressed=True).all_inner_texts() == ["Energy"]


def test_tapping_a_chart_shows_its_tooltip(browser, base_url):
    tab = Tab(browser, base_url, 390, has_touch=True)
    tab.go("#/portfolio")
    chart = tab.page.locator("#view .chart")
    chart.locator("svg > rect[fill=transparent]").tap()
    assert chart.locator(".tip").is_visible()


def test_a_pattern_with_a_proven_edge_says_so_in_words(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/performance")
    assert [t for t in tab.page.locator("#view svg text").all_text_contents() if t.endswith("(edge)")] == ["double top (edge)"]


def calibration_plot(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/performance")
    return tab.page.get_by_role("img", name=re.compile("^How often stocks went up"))


def test_each_calibration_horizon_has_its_own_dash(browser, base_url):
    dashes = calibration_plot(browser, base_url).locator("path").evaluate_all("els => els.map((el) => el.getAttribute('stroke-dasharray'))")
    assert len(set(dashes)) == len(dashes) == 2


def test_the_calibration_plot_marks_its_reference_line_in_words(browser, base_url):
    assert calibration_plot(browser, base_url).locator("text").all_text_contents().count("perfect") == 1


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


def test_a_name_no_stock_has_offers_no_stock_page(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    search_and_press_enter(tab, "Parag Parikh Flexi Cap")
    assert tab.page.locator("#view a[href^='#/stock/']").count() == 0


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


def test_a_late_search_failure_for_older_text_shows_no_toast(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    held = []
    tab.page.route(lambda url: "/ui/search?q=Parag" in url, lambda route: held.append(route))
    tab.page.fill("#q", LONG_NAME)
    tab.wait_until(lambda: held)
    tab.page.fill("#q", "Test")
    tab.page.wait_for_selector("#suggest.open")
    for route in held:
        route.continue_()
    tab.settle()
    assert tab.page.locator(".toast").all_inner_texts() == []


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


def before_the_app_script(tab, expression):
    """What a JavaScript expression gives just before app.js runs, on a reload."""
    def note_it_first(route):
        script = route.fetch()
        route.fulfill(response=script, body=f"window.atStart = {expression};\n" + script.text())
    tab.page.route("**/static/app.js", note_it_first)
    tab.page.reload()
    tab.settle()
    return tab.page.evaluate("window.atStart")


@pytest.mark.parametrize("scheme, chosen", [("light", None), ("dark", None), ("light", "dark"), ("dark", "light")])
def test_the_chosen_or_else_the_system_theme_is_in_place_before_the_app_script_runs(browser, base_url, scheme, chosen):
    tab = Tab(browser, base_url, 1366, color_scheme=scheme, service_workers="block")
    if chosen:
        tab.page.click("#theme")
    assert before_the_app_script(tab, "document.documentElement.dataset.theme") == (chosen or scheme)


def test_the_theme_button_has_its_final_size_before_its_icon_arrives(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    size = "(({ width, height }) => [width, height])(document.querySelector('#theme').getBoundingClientRect())"
    assert before_the_app_script(tab, size) == tab.page.evaluate(size)


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


DIALOGS = {"transaction": {"page": "#/stock/TEST", "opener": "+ Add to portfolio", "action": "Save", "writes_to": "**/portfolio/transactions"},
           "delete": {"page": "#/portfolio", "opener": "Delete", "action": "Delete", "writes_to": "**/portfolio/transactions/0"},
           "builder": {"page": "#/builder", "opener": "Add all to my portfolio", "action": "Add to portfolio", "writes_to": "**/ui/builder/apply"}}


def open_dialog(tab, which):
    """One of DIALOGS, opened with a click as a person would."""
    tab.go(DIALOGS[which]["page"])
    tab.page.click(f"button:text-is('{DIALOGS[which]['opener']}')")
    dialog = tab.page.get_by_role("dialog")
    dialog.wait_for(timeout=5000)
    return dialog


def writes_from(tab, which, **response):
    """The methods of the writes the dialog sends, each answered with response, so the server never sees them."""
    sent = []
    tab.page.route(DIALOGS[which]["writes_to"], lambda route: (sent.append(route.request.method), route.fulfill(**response)))
    return sent


def click_main_button(dialog, which):
    dialog.get_by_role("button", name=DIALOGS[which]["action"], exact=True).click()


def focus_stays_in_a_dialog(tab, key):
    inside = []
    for _ in range(10):   # more than the dialog's eight controls, so focus has to wrap
        tab.page.keyboard.press(key)
        inside.append(tab.page.evaluate("document.activeElement.closest('[role=dialog]') !== null"))
    return all(inside)


def save_with_enter(tab):
    """Press Enter where the dialog left focus, and return the transaction it sent; the server never sees it."""
    writes_from(tab, "transaction", json={"status": "saved", "transactions": 2})
    with tab.page.expect_request(DIALOGS["transaction"]["writes_to"]) as sent:
        tab.page.keyboard.press("Enter")
    return sent.value.post_data_json


@pytest.mark.parametrize("key", ["Tab", "Shift+Tab"])
def test_tabbing_never_leaves_the_dialog(browser, base_url, key):
    tab = Tab(browser, base_url, 1366)
    open_dialog(tab, "transaction")
    assert focus_stays_in_a_dialog(tab, key)


def test_escape_closes_the_dialog(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    dialog = open_dialog(tab, "transaction")
    tab.page.keyboard.press("Escape")
    assert dialog.count() == 0


def test_closing_the_dialog_gives_focus_back_to_the_button_that_opened_it(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    open_dialog(tab, "transaction")
    tab.page.keyboard.press("Escape")
    assert tab.page.evaluate("document.activeElement.textContent") == "+ Add to portfolio"


def test_enter_in_the_transaction_dialog_saves_it(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    open_dialog(tab, "transaction")
    assert save_with_enter(tab)["symbol"] == "TEST"


def test_the_transaction_dialog_sends_the_fees(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    open_dialog(tab, "transaction").get_by_label("Fees (₹)").fill("20")
    assert save_with_enter(tab)["fees"] == 20


def test_a_quantity_the_server_refuses_is_explained_at_the_quantity(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    quantity = open_dialog(tab, "transaction").get_by_label("Quantity", exact=True)
    quantity.fill("0")
    quantity.press("Enter")
    sync_api.expect(quantity).to_have_accessible_description(re.compile("greater than 0"))


def test_the_transaction_dialog_dates_a_trade_by_the_day_in_india(browser, base_url):
    tab = Tab(browser, base_url, 1366, timezone_id="UTC")
    two_am_on_the_30th_in_india = "2026-10-29T20:30:00Z"
    tab.page.clock.set_fixed_time(two_am_on_the_30th_in_india)
    assert open_dialog(tab, "transaction").get_by_label("Date", exact=True).input_value() == "2026-10-30"


def test_deleting_a_transaction_asks_in_the_apps_own_dialog(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    open_dialog(tab, "delete")
    assert tab.page.get_by_role("dialog", name="Delete this transaction?").is_visible()


@pytest.mark.parametrize("which, says", [("delete", r"^BUY 10 TEST at .+ on \d{4}-\d{2}-\d{2}\.$"), ("builder", r"^This adds \d+ BUY transactions")])
def test_a_confirmation_dialog_is_described_by_what_it_will_write(browser, base_url, which, says):
    tab = Tab(browser, base_url, 1366)
    sync_api.expect(open_dialog(tab, which)).to_have_accessible_description(re.compile(says))


@pytest.mark.parametrize("which", ["delete", "builder"])
@pytest.mark.parametrize("key", ["Escape", "Enter"])
def test_the_first_key_pressed_in_a_confirmation_writes_nothing(browser, base_url, which, key):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    sent = writes_from(tab, which, json={})
    open_dialog(tab, which)
    tab.page.keyboard.press(key)
    tab.settle()
    assert sent == []


@pytest.mark.parametrize("which, method", [("delete", "DELETE"), ("builder", "POST")])
def test_confirming_sends_the_write(browser, base_url, which, method):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    sent = writes_from(tab, which, json={})
    click_main_button(open_dialog(tab, which), which)
    tab.settle()
    assert sent == [method]


def a_dialog_whose_write_failed(browser, base_url, which):
    """The dialog after its main button was clicked while the server answered 500."""
    tab = Tab(browser, base_url, 1366, service_workers="block")
    writes_from(tab, which, **SERVER_ERROR)
    dialog = open_dialog(tab, which)
    click_main_button(dialog, which)
    sync_api.expect(dialog.locator(".dialog-error")).to_have_text("Something went wrong on our side.")
    return tab, dialog


@pytest.mark.parametrize("which", ["transaction", "delete"])
def test_escape_closes_a_dialog_whose_write_failed(browser, base_url, which):
    tab, dialog = a_dialog_whose_write_failed(browser, base_url, which)
    tab.page.keyboard.press("Escape")
    assert dialog.count() == 0


@pytest.mark.parametrize("which", ["transaction", "delete"])
def test_tabbing_never_leaves_a_dialog_whose_write_failed(browser, base_url, which):
    tab, _ = a_dialog_whose_write_failed(browser, base_url, which)
    assert focus_stays_in_a_dialog(tab, "Tab")


@pytest.mark.parametrize("refusal", [{"loc": ["body"], "msg": "That trade is dated after today"}, {"msg": "That trade is dated after today"}],
                         ids=["whole-form", "no-place"])
def test_a_refusal_no_field_can_show_is_shown_in_the_dialog(browser, base_url, refusal):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    writes_from(tab, "transaction", status=422, json={"detail": [{"loc": ["body", "quantity"], "msg": "Input should be greater than 0"}, refusal]})
    dialog = open_dialog(tab, "transaction")
    tab.page.keyboard.press("Enter")
    sync_api.expect(dialog.locator(".dialog-error")).to_have_text("That trade is dated after today")


def a_dialog_whose_write_is_running(browser, base_url, which):
    """The dialog once its main button was clicked and the browser moved focus off that disabled button,
    and its write, held unanswered until the test answers it."""
    tab = Tab(browser, base_url, 1366, service_workers="block")
    held = []
    tab.page.route(DIALOGS[which]["writes_to"], lambda route: held.append(route))
    dialog = open_dialog(tab, which)
    click_main_button(dialog, which)
    tab.wait_until(lambda: held and not tab.page.evaluate("document.activeElement.disabled"))
    return tab, dialog, held[0]


@pytest.mark.parametrize("which", ["transaction", "delete"])
def test_escape_closes_a_dialog_whose_write_is_running(browser, base_url, which):
    tab, dialog, _ = a_dialog_whose_write_is_running(browser, base_url, which)
    tab.page.keyboard.press("Escape")
    assert dialog.count() == 0


@pytest.mark.parametrize("which", ["transaction", "delete"])
def test_tabbing_never_leaves_a_dialog_whose_write_is_running(browser, base_url, which):
    tab, _, _ = a_dialog_whose_write_is_running(browser, base_url, which)
    assert focus_stays_in_a_dialog(tab, "Tab")


def test_a_write_that_ends_after_escape_leaves_focus_where_it_was_moved(browser, base_url):
    tab, _, write = a_dialog_whose_write_is_running(browser, base_url, "transaction")
    tab.page.keyboard.press("Escape")
    tab.page.focus("#q")
    write.fulfill(json={"status": "saved", "transactions": 2})
    tab.settle()
    assert tab.page.evaluate("document.activeElement.id") == "q"


@pytest.mark.parametrize("closing", ["Escape", "leaving the page"])
def test_a_write_that_fails_after_its_dialog_closed_says_so(browser, base_url, closing):
    tab, _, write = a_dialog_whose_write_is_running(browser, base_url, "transaction")
    if closing == "Escape":
        tab.page.keyboard.press("Escape")
    else:
        tab.page.evaluate("location.hash = '#/explore'")
    write.fulfill(**SERVER_ERROR)
    sync_api.expect(tab.page.locator(".toasts")).to_contain_text("Save failed: Something went wrong on our side.")


def test_leaving_the_page_closes_its_dialog(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    dialog = open_dialog(tab, "transaction")
    tab.go("#/explore")
    assert dialog.count() == 0


SERVER_ERROR = {"status": 500, "body": "Internal Server Error"}


def explore_answered_with(browser, base_url, **response):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    tab.page.route("**/ui/explore", lambda route: route.fulfill(**response))
    tab.go("#/explore")
    return tab


def test_a_server_error_says_it_is_on_our_side(browser, base_url):
    assert "Something went wrong on our side." in explore_answered_with(browser, base_url, **SERVER_ERROR).page.inner_text("#view")


def test_a_server_error_never_shows_http_500(browser, base_url):
    assert "HTTP 500" not in explore_answered_with(browser, base_url, **SERVER_ERROR).page.inner_text("#view")


def test_a_server_error_shows_no_emoji(browser, base_url):
    assert not EMOJI.search(explore_answered_with(browser, base_url, **SERVER_ERROR).page.inner_text("#view"))


def test_try_again_after_a_server_error_loads_the_page(browser, base_url):
    tab = explore_answered_with(browser, base_url, **SERVER_ERROR)
    tab.page.unroute("**/ui/explore")
    tab.page.click("#view button:text-is('Try again')")
    tab.settle()
    assert tab.page.locator("#view .welcome").count() == 1


def test_a_rate_limit_shows_the_servers_words_with_the_limit(browser, base_url):
    limit = "rate limit of 120 requests/minute exceeded"
    assert limit in explore_answered_with(browser, base_url, status=429, json={"detail": limit}).page.inner_text("#view")


def test_the_signed_in_test_user_is_never_refused_in_a_whole_runs_requests(base_url):
    with httpx.Client(base_url=base_url) as client:
        assert client.post("/auth/cookie/login", data={"username": EMAIL, "password": PASSWORD}).is_success
        codes = {client.get("/search", params={"q": "TE"}).status_code for _ in range(RUN_REQUESTS)}
    assert codes == {200}


def test_two_toasts_do_not_overlap(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.page.evaluate("() => { toast('Saved to portfolio'); toast('Added to your portfolio'); }")
    first, second = tab.page.locator(".toast").evaluate_all("els => els.map((el) => el.getBoundingClientRect().toJSON())")
    assert first["bottom"] <= second["top"] or second["bottom"] <= first["top"]


def test_toasts_on_a_phone_leave_the_bottom_nav_uncovered(browser, base_url):
    tab = Tab(browser, base_url, 390)
    tab.page.evaluate("() => { toast('Saved to portfolio'); toast('Added to your portfolio'); }")
    toasts, nav = tab.page.locator(".toasts").bounding_box(), tab.page.locator("#bnav").bounding_box()
    assert toasts["y"] + toasts["height"] <= nav["y"], (toasts, nav)


def test_typing_a_funds_name_lists_it_under_mutual_funds(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.page.fill("#q", "hdfc mid")
    funds = tab.page.locator("#suggest [role=group][aria-label='Mutual funds'] [role=option]")
    funds.first.wait_for()
    assert funds.first.inner_text().startswith("HDFC Mid Cap Fund")


def test_the_search_page_lists_stocks_and_funds_apart(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go("#/search?q=hdfc%20mid")
    assert tab.page.locator("#view h3").all_inner_texts() == ["Stocks", "Mutual funds"]
    assert tab.page.locator("#view a[href^='#/fund/']").first.inner_text().startswith("HDFC Mid Cap Fund")


def test_the_fund_page_shows_the_nav_with_its_date_and_source(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go(f"#/fund/{FUND}")
    text = tab.page.inner_text("#view")
    assert "₹210.5120" in text and "NAV on 2026-10-29" in text and "Source: AMFI" in text


def test_a_fund_added_from_its_page_shows_in_the_portfolio_and_its_total(browser, base_url):
    tab = Tab(browser, base_url, 1366)
    tab.go(f"#/fund/{FUND}")
    tab.page.click("text=+ Add to portfolio")
    tab.page.fill("input[name=units]", "10")
    tab.page.click(".modal button[type=submit]")
    tab.settle()
    try:
        tab.go("#/portfolio")
        view = tab.page.inner_text("#view")
        total = tab.context.request.get(f"{base_url}/portfolio/analysis").json()["total_value"] + 2105.12
        assert "Mutual funds (1)" in view and f"With your mutual funds: {tab.page.evaluate('(v) => inr(v)', total)}" in view
    finally:       # the other tests share this user's portfolio
        for tx in tab.context.request.get(f"{base_url}/portfolio/funds").json()["transactions"]:
            tab.context.request.delete(f"{base_url}/portfolio/funds/transactions/{tx['id']}")


def test_the_stocks_show_while_the_funds_are_still_loading(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    held = []
    tab.page.route("**/portfolio/funds", lambda route: held.append(route))
    tab.page.evaluate("location.hash = '#/portfolio'")
    tab.page.wait_for_selector("#view table a[href='#/stock/TEST']")
    assert len(held) == 1
    held[0].continue_()


def test_a_funds_failure_shows_in_the_funds_card_and_leaves_the_stocks(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    tab.page.route("**/portfolio/funds", lambda route: route.fulfill(status=503, json={"detail": "AMFI is down"}))
    tab.go("#/portfolio")
    card = tab.page.locator("#view .card", has=tab.page.locator("h3", has_text="Mutual funds"))
    assert "Something went wrong on our side." in card.inner_text()
    assert tab.page.locator("#view table a[href='#/stock/TEST']").count() == 1


def test_a_fund_amfi_no_longer_lists_stays_deletable_beside_the_stocks(browser, base_url):
    tab = Tab(browser, base_url, 1366, service_workers="block")
    gone = {"valuation_error": "AMFI's NAV file has no NAV in the last 7 days for the schemes [999999] you hold",
            "transactions": [{"id": 1, "scheme_code": 999999, "side": "BUY", "units": 10, "nav": 200.0, "fees": 0.0,
                              "trade_date": "2025-01-02", "scheme": None}], "source": "AMFI"}
    tab.page.route("**/portfolio/funds", lambda route: route.fulfill(json=gone))   # the shape GET /portfolio/funds gives
    tab.go("#/portfolio")
    card = tab.page.locator("#view .card", has=tab.page.locator("h3", has_text="Mutual funds"))
    assert ("[999999]" in card.inner_text(), card.locator("button", has_text="Delete").count()) == (True, 1)
    assert tab.page.locator("#view table a[href='#/stock/TEST']").count() == 1


def test_no_earlier_test_leaves_a_tab_open(browser):
    # last in the file, so every other test has opened its tabs and ended before this looks
    assert browser.contexts == []
