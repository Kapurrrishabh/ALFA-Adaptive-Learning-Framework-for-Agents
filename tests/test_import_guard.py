"""S11: the no-deep-learning-library constraint is machine-checked, not just claimed.

numpy, pandas and matplotlib are allowed anywhere. scipy and sklearn are allowed only under
selfagent/eval/, where they compute metrics and baselines the model itself never touches.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "selfagent"
EVALUATION_ONLY_PACKAGE = PACKAGE_ROOT / "eval"

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
    return sorted(PROJECT_ROOT.glob("selfagent/**/*.py")) + sorted(
        PROJECT_ROOT.glob("scripts/**/*.py")
    )


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
        "under selfagent/eval/"
    )


def test_numpy_is_imported_in_one_place_only():
    """Every module goes through backend.xp, so a later cupy swap is a one-line change."""
    importers = [
        path.relative_to(PROJECT_ROOT)
        for path in PROJECT_ROOT.glob("selfagent/**/*.py")
        if "numpy" in imported_root_modules(path)
    ]
    assert importers == [Path("selfagent/backend.py")], f"numpy imported directly in {importers}"
