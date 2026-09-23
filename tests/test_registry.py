"""C6: a deliberately bad model is refused promotion by the gate.

The exact-match numbers here are the measured ones -- A2's final checkpoint matches 36.0% of the reworded
split and A1's loss-picked checkpoint 32.5% -- so the comparison these tests pin is the one that actually
decided against the loss-picked model. The figure counts are chosen to sit either side of S14's bar rather
than copied from a run, because what is being pinned is which side of the bar gets served.
"""

import pytest

from backend.models import registry

SAMPLER = "nucleus temperature 0.9, top_p 0.95, seed 0"


def _tally(matched=72, figures=400, unsupported_figures=2, dataset="advisory_casual", rows=200):
    return {"dataset": dataset, "rows": rows, "matched": matched, "answerable": 193,
            "figures": figures, "unsupported_figures": unsupported_figures,
            "unsupported_answers": unsupported_figures, "should_refuse": 7, "refused": 7,
            "wrongly_refused": 7}


def _artifact(tmp_path, name="advisory_combined.npz", body=b"weights"):
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _promote(tmp_path, artifact, measured, role=registry.GENERATOR):
    return registry.promote(role, artifact, measured, SAMPLER, tmp_path / "registry.json")


def test_a_deliberately_bad_model_is_refused_and_the_promoted_one_keeps_serving(tmp_path):
    """The loss-picked checkpoint: better on held-out loss, worse on every generation number there is."""
    served = _artifact(tmp_path)
    _promote(tmp_path, served, {"unseen": _tally(), "casual": _tally(matched=189)})

    bad = _artifact(tmp_path, "advisory_casual.best.npz", b"other weights")
    with pytest.raises(registry.Refused) as refused:
        _promote(tmp_path, bad, {"unseen": _tally(matched=65, unsupported_figures=40)})
    assert "10.0% of the 400 figures" in str(refused.value)
    assert "32.5% of unseen against" in str(refused.value)
    assert registry.serving(registry.GENERATOR, tmp_path / "registry.json") == served


def test_a_candidate_is_refused_when_only_its_loss_would_have_promoted_it(tmp_path):
    """Faithful enough to pass S14 and still 3.5 points of exact match behind what already serves."""
    _promote(tmp_path, _artifact(tmp_path), {"unseen": _tally()})
    with pytest.raises(registry.Refused, match="not more than the 200 rows can distinguish"):
        _promote(tmp_path, _artifact(tmp_path, "candidate.npz", b"quieter"), {"unseen": _tally(65)})


def test_the_trained_split_alone_cannot_promote_anything(tmp_path):
    with pytest.raises(registry.Refused, match="nothing was measured on the unseen split"):
        _promote(tmp_path, _artifact(tmp_path), {"casual": _tally(matched=189)})


def test_a_record_that_found_no_figures_to_check_is_refused(tmp_path):
    with pytest.raises(registry.Refused, match="state no figures on unseen"):
        _promote(tmp_path, _artifact(tmp_path), {"unseen": _tally(figures=0, unsupported_figures=0)})


def test_a_clean_run_on_too_few_figures_cannot_clear_the_bar(tmp_path):
    """Nothing unsupported in 40 figures is 0%, and a measurement at 2% carries more noise than that."""
    with pytest.raises(registry.Refused, match="0.0% of the 40 figures"):
        _promote(tmp_path, _artifact(tmp_path), {"unseen": _tally(figures=40, unsupported_figures=0)})


def test_a_candidate_measured_on_other_rows_is_not_compared_against_the_promoted_one(tmp_path):
    """A2's gate compared two datasets' unseen splits and could not be settled either way."""
    _promote(tmp_path, _artifact(tmp_path), {"unseen": _tally()})
    with pytest.raises(registry.Refused, match="which are not the same rows"):
        _promote(tmp_path, _artifact(tmp_path, "other.npz", b"elsewhere"),
                 {"unseen": _tally(matched=120, dataset="advisory")})


def test_an_artifact_overwritten_after_its_promotion_is_not_served(tmp_path):
    artifact = _artifact(tmp_path)
    entry = _promote(tmp_path, artifact, {"unseen": _tally()})
    artifact.write_bytes(b"retrained in place")
    with pytest.raises(ValueError, match=f"record says {entry['digest']}"):
        registry.serving(registry.GENERATOR, tmp_path / "registry.json")


def test_a_role_with_nothing_promoted_serves_nothing(tmp_path):
    with pytest.raises(LookupError, match="scripts/promote.py"):
        registry.serving(registry.GENERATOR, tmp_path / "registry.json")
