from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    REFERENCE_FALLBACK_MODE,
    TARGETED_HANDOVER_MODE,
    certified_reference_dominance,
)
from hypothesis_mvp.pcpi.reference.decision_fixture import (
    exact_discrete_class_eig,
    exact_discrete_entropy_reduction,
    reference_dominance_fixture,
    zero_capacity_fixture,
)
from scripts.run_pcpi_p3d1_reference_dominance_diagnostic import (
    FIXTURE_ROLE,
    STAGE,
    _evaluate,
    _load_config,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3d_1_reference_dominance_diagnostic.json"


def _config() -> dict[str, object]:
    return _load_config(CONFIG, ROOT)


def test_exact_discrete_eig_equals_expected_entropy_reduction() -> None:
    classes, likelihoods, _, _ = reference_dominance_fixture()
    eig = exact_discrete_class_eig(classes, likelihoods)
    reduction = exact_discrete_entropy_reduction(classes, likelihoods)
    np.testing.assert_allclose(eig, reduction, atol=1e-14, rtol=0.0)
    entropy = -float(np.sum(classes * np.log(classes)))
    assert np.all(eig >= 0.0)
    assert np.all(eig <= entropy + 1e-14)


def test_separated_action_certifiably_dominates_uniform_reference() -> None:
    classes, likelihoods, identifiers, reference = reference_dominance_fixture()
    scores = exact_discrete_class_eig(classes, likelihoods)
    radius = 1e-10
    decision = certified_reference_dominance(
        scores, scores - radius, scores + radius, reference, identifiers,
        reference_seed=20260814,
    )
    assert decision.targeted_handover
    assert decision.utility_mode == TARGETED_HANDOVER_MODE
    assert decision.selected_candidate_id == identifiers[int(np.argmax(scores))]
    assert scores[decision.selected_position] > float(reference @ scores)


def test_overlapping_intervals_return_to_registered_reference() -> None:
    classes, likelihoods, identifiers, reference = reference_dominance_fixture()
    scores = exact_discrete_class_eig(classes, likelihoods)
    decision = certified_reference_dominance(
        scores, scores - 0.4, scores + 0.4, reference, identifiers,
        reference_seed=20260814,
    )
    repeated = certified_reference_dominance(
        scores, scores - 0.4, scores + 0.4, reference, identifiers,
        reference_seed=20260814,
    )
    assert not decision.targeted_handover
    assert decision.utility_mode == REFERENCE_FALLBACK_MODE
    assert decision.selected_candidate_id == repeated.selected_candidate_id
    assert decision.selected_candidate_id == decision.reference_sample_candidate_id


def test_alternate_registered_seed_changes_only_reference_draw() -> None:
    classes, likelihoods, identifiers, reference = reference_dominance_fixture()
    scores = exact_discrete_class_eig(classes, likelihoods)
    first = certified_reference_dominance(
        scores, scores - 0.4, scores + 0.4, reference, identifiers,
        reference_seed=20260814,
    )
    alternate = certified_reference_dominance(
        scores, scores - 0.4, scores + 0.4, reference, identifiers,
        reference_seed=20260815,
    )
    assert not first.targeted_handover
    assert not alternate.targeted_handover
    assert first.leader_candidate_id == alternate.leader_candidate_id
    assert first.selected_candidate_id != alternate.selected_candidate_id


def test_zero_class_capacity_returns_to_reference_without_threshold() -> None:
    classes, likelihoods, identifiers, reference = zero_capacity_fixture()
    scores = exact_discrete_class_eig(classes, likelihoods)
    np.testing.assert_array_equal(scores, np.zeros_like(scores))
    decision = certified_reference_dominance(
        scores, scores, scores, reference, identifiers, reference_seed=20260814
    )
    assert not decision.targeted_handover
    assert decision.utility_mode == REFERENCE_FALLBACK_MODE


def test_candidate_permutation_preserves_selected_and_reference_identity() -> None:
    classes, likelihoods, identifiers, reference = reference_dominance_fixture()
    scores = exact_discrete_class_eig(classes, likelihoods)
    radius = 1e-10
    first = certified_reference_dominance(
        scores, scores - radius, scores + radius, reference, identifiers,
        reference_seed=20260814,
    )
    order = np.asarray([2, 0, 3, 1])
    second = certified_reference_dominance(
        scores[order], scores[order] - radius, scores[order] + radius,
        reference[order], identifiers[order], reference_seed=20260814,
    )
    assert first.selected_candidate_id == second.selected_candidate_id
    assert first.reference_sample_candidate_id == second.reference_sample_candidate_id


@pytest.mark.parametrize(
    "mutation",
    ("duplicate_ids", "bad_probability_sum", "nan", "reversed", "outside"),
)
def test_reference_dominance_rejects_malformed_inputs(mutation: str) -> None:
    classes, likelihoods, identifiers, reference = reference_dominance_fixture()
    scores = exact_discrete_class_eig(classes, likelihoods)
    lower, upper = scores - 1e-10, scores + 1e-10
    if mutation == "duplicate_ids":
        identifiers = np.zeros_like(identifiers)
    elif mutation == "bad_probability_sum":
        reference = 0.9 * reference
    elif mutation == "nan":
        scores = scores.copy()
        scores[0] = np.nan
    elif mutation == "reversed":
        lower, upper = upper, lower
    else:
        scores = scores + 1.0
    with pytest.raises(ValueError):
        certified_reference_dominance(
            scores, lower, upper, reference, identifiers,
            reference_seed=20260814,
        )


def test_p3d1_diagnostic_passes_all_correctness_decisions() -> None:
    rows, diagnostics, summary = _evaluate(_config())
    assert len(rows) == 4
    assert summary["fixture_role"] == FIXTURE_ROLE
    assert summary["gate_decision_count"] == 14
    assert summary["gate_passed"]
    assert all(summary["gate_decisions"].values())
    assert not summary["formal_efficacy_evidence"]
    assert not summary["heldout_opened"]
    assert diagnostics["tight_decision"]["targeted_handover"]
    assert not diagnostics["unresolved_decision"]["targeted_handover"]


def test_p3d1_cli_is_correctness_only() -> None:
    parser = build_parser()
    options = {action.dest for action in parser._actions}
    assert options == {
        "help", "output_dir", "source_artifact", "config", "phase",
        "heldout_state",
    }
    args = parser.parse_args([
        "--output-dir", "diagnostic",
        "--config", str(CONFIG),
    ])
    assert args.phase == STAGE
    assert args.heldout_state == "not-applicable"
    assert args.source_artifact is None


def test_p3d1_is_not_imported_by_real_acquisition_runtime() -> None:
    source = (ROOT / "hypothesis_mvp" / "pcpi" / "real_acquisition.py").read_text()
    assert "certified_reference_dominance" not in source
    assert "P3D.1" not in source


def test_p3d1_config_has_no_real_data_surface() -> None:
    config = json.loads(CONFIG.read_text())
    forbidden = {"datasets", "data_root", "validation_budget", "policies"}
    assert not forbidden & set(config)
    assert config["heldout_state"] == "not-applicable"
    assert config["fixture_role"] == FIXTURE_ROLE
