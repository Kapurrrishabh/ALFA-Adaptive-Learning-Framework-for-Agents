"""S11: the no-deep-learning-library constraint is machine-checked, not just claimed.

It covers ALFA's from-scratch code, listed in FROM_SCRATCH. Open-source models (Chronos, Kronos, the
language model) live in models/external/, and the advisory app's statistics are outside the claim.
numpy, pandas and matplotlib are allowed; scipy and sklearn only under models/eval/, where they
compute metrics and baselines the model itself never touches.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "backend"
EVALUATION_ONLY_PACKAGE = PACKAGE_ROOT / "models" / "eval"
FROM_SCRATCH = ("models/core", "models/networks", "models/data", "models/learning", "models/guardrails",
                "models/agent", "models/serving", "models/training", "knowledge_base/retrieval",
                "advisory/sentiment/feeds", "database/live", "database/dataforge", "database/store.py",
                "api/agent_server.py")
XP_MODULE = Path("backend/models/core/backend.py")

BANNED_EVERYWHERE = {"torch", "tensorflow", "keras", "transformers", "jax", "flax", "theano"}
BANNED_OUTSIDE_EVAL = {"sklearn", "scipy", "statsmodels", "xgboost", "lightgbm"}


def imported_root_modules(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def python_files():
    out = []
    for part in FROM_SCRATCH:
        target = PACKAGE_ROOT / part
        out += [target] if target.suffix == ".py" else sorted(target.glob("**/*.py"))
    return out


def test_there_are_source_files_to_check():
    """Guards against the guard silently passing because the glob matched nothing."""
    assert len(python_files()) > 10


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_no_deep_learning_library_is_imported(path):
    found = imported_root_modules(path) & BANNED_EVERYWHERE
    assert not found, f"{path.relative_to(PROJECT_ROOT)} imports {sorted(found)}"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_evaluation_only_libraries_stay_in_eval(path):
    if EVALUATION_ONLY_PACKAGE in path.parents:
        return
    found = imported_root_modules(path) & BANNED_OUTSIDE_EVAL
    assert not found, (
        f"{path.relative_to(PROJECT_ROOT)} imports {sorted(found)}, which is allowed only "
        "under models/eval/"
    )


def test_numpy_is_imported_in_one_place_only():
    """Every module goes through backend.xp, so a later cupy swap is a one-line change."""
    # scripts and the corpus collector were never under this rule
    scripts = (PACKAGE_ROOT / "models" / "training", PACKAGE_ROOT / "models" / "agent" / "run", PACKAGE_ROOT / "database" / "dataforge")
    importers = [path.relative_to(PROJECT_ROOT) for path in python_files()
                 if not any(s in path.parents for s in scripts) and "numpy" in imported_root_modules(path)]
    assert importers == [XP_MODULE], f"numpy imported directly in {importers}"
