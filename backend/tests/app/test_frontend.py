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


def test_index_html_runs_no_inline_script():
    bodies = re.findall(r"<script\b[^>]*>(.*?)</script>", (FRONTEND / "index.html").read_text(), re.S)
    assert bodies and not [b for b in bodies if b.strip()]
