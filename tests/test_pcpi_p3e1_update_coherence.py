from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.pcpi import REFERENCE_FALLBACK_MODE
from hypothesis_mvp.pcpi.reference.update_coherence import (
    UPDATE_COHERENT_REFERENCE_DOMINANCE_METHOD,
    UPDATE_COHERENT_TARGETED_HANDOVER_MODE,
    certified_update_coherent_reference_dominance,
    exact_update_coherent_utility,
    generalized_update_ranking_reversal_fixture,
)
from scripts.run_pcpi_p3e1_update_coherence_diagnostic import (
    FIXTURE_ROLE,
    STAGE,
    _evaluate,
    _load_config,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3e_1_update_coherence_diagnostic.json"


def _fixture(power: float = 0.25):
    structures, likelihoods, mapping, identifiers, reference = (
        generalized_update_ranking_reversal_fixture()
    )
    result = exact_update_coherent_utility(
        structures, likelihoods, mapping, likelihood_power=power
    )
    return structures, likelihoods, mapping, identifiers, reference, result


def test_eta_one_recovers_ordinary_class_mutual_information() -> None:
    *_, result = _fixture(power=1.0)
    np.testing.assert_allclose(
        result.update_coherent_entropy_reduction,
        result.ordinary_class_mi,
        atol=1e-14,
        rtol=0.0,
    )


def test_generalized_update_can_reverse_ordinary_mi_ranking() -> None:
    *_, result = _fixture()
    assert int(np.argmax(result.ordinary_class_mi)) == 1
    assert int(np.argmax(result.update_coherent_entropy_reduction)) == 0
    assert not np.allclose(
        result.ordinary_class_mi,
        result.update_coherent_entropy_reduction,
        atol=1e-4,
        rtol=0.0,
    )


def test_expected_utility_is_marginal_weighted_realized_entropy_change() -> None:
    *_, result = _fixture()
    direct = np.sum(
        result.outcome_probabilities * result.realized_entropy_reduction, axis=1
    )
    np.testing.assert_allclose(
        result.update_coherent_entropy_reduction, direct, atol=1e-15, rtol=0.0
    )
    np.testing.assert_allclose(
        np.sum(result.outcome_probabilities, axis=1), 1.0, atol=1e-14, rtol=0.0
    )
    np.testing.assert_allclose(
        np.sum(result.updated_class_probabilities, axis=2),
        1.0,
        atol=1e-14,
        rtol=0.0,
    )


def test_state_outcome_and_action_permutations_preserve_identity() -> None:
    structures, likelihoods, mapping, *_rest, result = _fixture()
    state_order = np.asarray([1, 0])
    action_order = np.asarray([1, 0])
    permuted = exact_update_coherent_utility(
        structures[state_order],
        likelihoods[action_order][:, state_order][:, :, ::-1],
        np.asarray([0, 1], dtype=np.int64),
        likelihood_power=0.25,
    )
    np.testing.assert_allclose(
        permuted.ordinary_class_mi[action_order], result.ordinary_class_mi,
        atol=1e-14, rtol=0.0,
    )
    np.testing.assert_allclose(
        permuted.update_coherent_entropy_reduction[action_order],
        result.update_coherent_entropy_reduction,
        atol=1e-14, rtol=0.0,
    )


def test_one_class_has_zero_utility_for_every_power() -> None:
    structures = np.asarray([0.3, 0.7])
    likelihoods = np.asarray([[[0.9, 0.1], [0.2, 0.8]]])
    mapping = np.asarray([0, 0])
    result = exact_update_coherent_utility(
        structures, likelihoods, mapping, likelihood_power=0.25
    )
    np.testing.assert_allclose(result.ordinary_class_mi, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        result.update_coherent_entropy_reduction, 0.0, atol=1e-15
    )


def test_update_coherent_decision_uses_aligned_ranking_and_positive_floor() -> None:
    *_, identifiers, reference, result = _fixture()
    scores = result.update_coherent_entropy_reduction
    radius = 1e-10
    decision = certified_update_coherent_reference_dominance(
        scores,
        scores - radius,
        scores + radius,
        reference,
        identifiers,
        reference_seed=20260814,
    )
    assert decision.method == UPDATE_COHERENT_REFERENCE_DOMINANCE_METHOD
    assert decision.utility_mode == UPDATE_COHERENT_TARGETED_HANDOVER_MODE
    assert decision.targeted_handover
    assert decision.selected_candidate_id == 101

    negative = np.asarray([-0.10, -0.20])
    fallback = certified_update_coherent_reference_dominance(
        negative,
        negative - radius,
        negative + radius,
        reference,
        identifiers,
        reference_seed=20260814,
    )
    assert not fallback.targeted_handover
    assert fallback.utility_mode == REFERENCE_FALLBACK_MODE
    assert fallback.selected_candidate_id == fallback.reference_sample_candidate_id
    assert fallback.dominance_gap < 0.0


@pytest.mark.parametrize(
    "mutation", ["bad_sum", "negative", "nan", "mapping", "power", "shape"]
)
def test_update_coherent_utility_fails_closed(mutation: str) -> None:
    structures, likelihoods, mapping, *_ = generalized_update_ranking_reversal_fixture()
    power = 0.25
    if mutation == "bad_sum":
        structures = 0.9 * structures
    elif mutation == "negative":
        likelihoods = likelihoods.copy()
        likelihoods[0, 0, 0] = -1.0
    elif mutation == "nan":
        structures = structures.copy()
        structures[0] = np.nan
    elif mutation == "mapping":
        mapping = np.asarray([0, 2])
    elif mutation == "power":
        power = 0.0
    else:
        likelihoods = likelihoods[:, :1]
    with pytest.raises(ValueError):
        exact_update_coherent_utility(
            structures, likelihoods, mapping, likelihood_power=power
        )


def test_p3e1_config_is_correctness_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config == {
        "schema": "pcpi-p3e1-update-coherence-diagnostic-config-v1",
        "stage": "P3E.1",
        "fixture_role": "inference_correctness_diagnostic_fixture",
        "likelihood_power": 0.25,
        "reference_seed": 20260814,
        "interval_radius": 1e-10,
        "identity_tolerance": 1e-13,
        "heldout_state": "not-applicable",
    }
    forbidden = {"datasets", "data_root", "validation_budget", "policies"}
    assert not forbidden & set(config)


def test_p3e1_diagnostic_passes_correctness_gate() -> None:
    config = _load_config(CONFIG, ROOT)
    rows, summary = _evaluate(config)
    assert len(rows) == 2
    assert summary["fixture_role"] == FIXTURE_ROLE
    assert summary["gate_decision_count"] == 10
    assert summary["gate_passed"]
    assert all(summary["gate_decisions"].values())
    assert not summary["formal_efficacy_evidence"]
    assert not summary["heldout_opened"]


def test_p3e1_cli_has_no_real_data_surface() -> None:
    parser = build_parser()
    assert {action.dest for action in parser._actions} == {
        "help", "output_dir", "config", "phase", "heldout_state"
    }
    args = parser.parse_args(["--output-dir", "diagnostic"])
    assert args.phase == STAGE
    assert args.heldout_state == "not-applicable"


def test_p3e1_is_not_imported_by_real_acquisition_runtime() -> None:
    source = (ROOT / "hypothesis_mvp" / "pcpi" / "real_acquisition.py").read_text()
    assert "update_coherent" not in source
    runner = (ROOT / "scripts" / "run_pcpi_p3d_real.py").read_text()
    assert "P3E.1" not in runner
