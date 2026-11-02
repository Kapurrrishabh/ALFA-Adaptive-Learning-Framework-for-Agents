"""Static checks on the web app's files, for what pytest can see without a browser."""
import json
import re

from backend.paths import FRONTEND


def test_service_worker_caches_exactly_the_files_index_html_loads():
    loaded = set(re.findall(r'(?:src|href)="(/static/[^"]+)"', (FRONTEND / "index.html").read_text()))
    cached = json.loads(re.search(r"const FILES = (\[.*?\]);", (FRONTEND / "sw.js").read_text()).group(1))
    assert sorted(cached) == sorted({"/"} | loaded)


def test_index_html_loads_the_setup_wizard_after_app_js():
    """welcome.js pushes its route onto app.js's ROUTES, so it must load after app.js."""
    loaded = re.findall(r'<script src="([^"]+)"', (FRONTEND / "index.html").read_text())
    assert loaded.index("/static/pages/welcome.js") > loaded.index("/static/app.js")


def test_index_html_runs_no_inline_script():
    bodies = re.findall(r"<script\b[^>]*>(.*?)</script>", (FRONTEND / "index.html").read_text(), re.S)
    assert bodies and not [b for b in bodies if b.strip()]
