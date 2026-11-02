"""Static checks on the web app's files, for what pytest can see without a browser."""
import json
import re

from backend.models.external.eval_user_questions import grade, plain
from backend.paths import FRONTEND

GUIDE = (FRONTEND / "guide.md").read_text()
GUIDE_PAGE_LINKS = re.findall(r"\]\(#/([^)]*)\)", GUIDE)


def test_the_guide_types_in_no_figures():
    assert re.findall(r"\d", re.sub(r"\]\(#/[^)]*\)", "]", GUIDE)) == []


def test_the_guide_reads_at_plain_language_grade():
    # PROVISIONAL: grade 9, the plain-language target the user-questions eval also reports against
    assert grade(plain(GUIDE)) <= 9


def test_every_page_the_guide_links_to_is_a_route():
    routes = set(re.findall(r'\{ path: "([^"]+)"', (FRONTEND / "app.js").read_text()))
    assert GUIDE_PAGE_LINKS, "the guide links to no page, so this check cannot fail"
    assert sorted({re.split(r"[/?]", link)[0] for link in GUIDE_PAGE_LINKS} - routes) == []


def test_service_worker_caches_exactly_the_files_index_html_loads():
    loaded = set(re.findall(r'(?:src|href)="(/static/[^"]+)"', (FRONTEND / "index.html").read_text()))
    cached = json.loads(re.search(r"const FILES = (\[.*?\]);", (FRONTEND / "sw.js").read_text()).group(1))
    assert sorted(cached) == sorted({"/"} | loaded)


def test_index_html_loads_the_setup_wizard_after_app_js():
    """welcome.js pushes its route onto app.js's ROUTES, so it must load after app.js."""
    loaded = re.findall(r'<script src="([^"]+)"', (FRONTEND / "index.html").read_text())
    assert loaded.index("/static/pages/welcome.js") > loaded.index("/static/app.js")


def test_the_import_template_is_read_as_our_csv_template():
    from backend.user.portfolio import importers
    assert importers.parse((FRONTEND / "import-template.csv").read_bytes()).source == "CSV template"


def test_index_html_runs_no_inline_script():
    bodies = re.findall(r"<script\b[^>]*>(.*?)</script>", (FRONTEND / "index.html").read_text(), re.S)
    assert bodies and not [b for b in bodies if b.strip()]


def test_index_html_loads_nothing_from_another_origin():
    addresses = re.findall(r'(?:src|href)="([^"]+)"', (FRONTEND / "index.html").read_text())
    assert [a for a in addresses if re.match(r"[a-z][a-z0-9+.-]*:|//", a, re.I)] == []


def tokens_set_by(selector):
    blocks = re.findall(re.escape(selector) + r"\s*\{([^}]*)\}", (FRONTEND / "app.css").read_text())
    return set(re.findall(r"(--[\w-]+)\s*:", "".join(blocks)))


def test_both_themes_set_the_same_tokens():
    light, dark = tokens_set_by(':root:not([data-theme="dark"])'), tokens_set_by(':root[data-theme="dark"]')
    assert light, "no light theme block was found, so this check cannot fail"
    assert sorted(light ^ dark) == []


def test_the_app_sends_no_shared_key_header():
    assert "X-API-Key" not in (FRONTEND / "app.js").read_text()


def test_the_app_never_reads_the_old_saved_key():
    assert re.findall(r"""getItem\(\s*["']si-key""", (FRONTEND / "app.js").read_text()) == []


def test_only_the_sign_in_pages_and_the_guide_open_signed_out():
    public = re.findall(r'\{ path: "([^"]+)"[^\n]*public: true', (FRONTEND / "app.js").read_text())
    assert sorted(public) == ["guide", "reset", "signin", "signup", "verify"]
