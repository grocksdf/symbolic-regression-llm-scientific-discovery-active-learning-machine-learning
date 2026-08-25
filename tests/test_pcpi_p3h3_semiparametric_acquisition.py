"""P3H.3 transformed-law acquisition algebra and isolation boundaries."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3H_CLASS_COUPLING,
    P3H_CLASS_EIG_METHOD,
    ClassPartition,
    PredictiveComponents,
    estimate_semiparametric_class_eig,
    estimate_semiparametric_class_eig_until_ranked,
    exact_class_eig,
    semiparametric_class_coupling,
)
from hypothesis_mvp.pcpi.reference import DyadicPolyaTreePredictiveLaw
import hypothesis_mvp.pcpi.semiparametric_acquisition as implementation
import hypothesis_mvp.pcpi.real_acquisition as real_acquisition
from scripts import run_pcpi_p3h3_semiparametric_acquisition_correctness as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3h_3_semiparametric_acquisition_correctness.json"


def _components() -> PredictiveComponents:
    return PredictiveComponents(
        structure_probabilities=np.asarray([0.30, 0.20, 0.50]),
        degrees_freedom=np.asarray([8.0, 12.0, 20.0]),
        locations=np.asarray([[-2.0, 0.0], [-1.0, 0.2], [2.0, 0.1]]),
        scales=np.asarray([[0.7, 1.0], [0.8, 0.9], [0.6, 1.1]]),
        partition=ClassPartition(
            class_ids=("left", "right"),
            member_indices=((0, 1), (2,)),
            class_probabilities=(0.5, 0.5),
            structure_to_class=(0, 0, 1),
        ),
    )


def _residual_law() -> DyadicPolyaTreePredictiveLaw:
    return DyadicPolyaTreePredictiveLaw(
        depth=2,
        leaf_probabilities=np.asarray([0.50, 0.10, 0.15, 0.25]),
        history_count=16,
    )


def test_coupling_preserves_transformed_outcome_and_frozen_class_marginals() -> None:
    coupling = semiparametric_class_coupling(
        _components(), _residual_law(), action_index=0, nodes_per_leaf=24
    )
    np.testing.assert_allclose(
        np.sum(coupling.joint_probabilities, axis=1),
        coupling.outcome_probabilities,
        rtol=0.0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        np.sum(coupling.joint_probabilities, axis=0),
        coupling.class_probabilities,
        rtol=0.0,
        atol=2e-13,
    )
    leaf_count = len(_residual_law().leaf_probabilities)
    grid_leaf_mass = np.asarray([
        np.sum(
            coupling.outcome_probabilities[
                np.floor(coupling.raw_pit_nodes * leaf_count).astype(int) == index
            ]
        )
        for index in range(leaf_count)
    ])
    np.testing.assert_allclose(
        grid_leaf_mass, _residual_law().leaf_probabilities, atol=2e-15
    )
    assert coupling.maximum_marginal_error <= 2e-13


def test_transformed_eig_is_true_mutual_information_not_student_t_reuse() -> None:
    components = _components()
    transformed = estimate_semiparametric_class_eig(
        components, _residual_law(), 32
    )
    old_student_t = exact_class_eig(components, epsabs=1e-11, epsrel=1e-10)
    assert transformed.integration_method == P3H_CLASS_EIG_METHOD
    assert transformed.coupling_method == P3H_CLASS_COUPLING
    assert np.max(np.abs(transformed.scores - old_student_t.scores)) > 1e-3
    assert np.all(transformed.scores >= 0.0)
    assert np.all(transformed.scores <= components.partition.entropy + 2e-13)


def test_identity_residual_converges_to_original_student_t_information() -> None:
    components = _components()
    identity = DyadicPolyaTreePredictiveLaw(0, np.asarray([1.0]), 0)
    transformed = estimate_semiparametric_class_eig(components, identity, 128)
    exact = exact_class_eig(components, epsabs=1e-11, epsrel=1e-10)
    assert np.all(
        np.abs(transformed.scores - exact.scores)
        <= transformed.error_bounds + exact.quadrature_errors
    )


def test_nested_quadrature_is_deterministic_and_refines() -> None:
    components = _components()
    residual = _residual_law()
    coarse = estimate_semiparametric_class_eig(components, residual, 16)
    first = estimate_semiparametric_class_eig(components, residual, 32)
    second = estimate_semiparametric_class_eig(components, residual, 32)
    fine = estimate_semiparametric_class_eig(components, residual, 64)
    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.error_bounds, second.error_bounds)
    assert np.max(np.abs(fine.scores - first.scores)) < np.max(
        np.abs(first.scores - coarse.scores)
    )
    assert first.sample_count == 32 * 4
    assert first.coarse_sample_count == 16 * 4


def test_transformed_utility_is_positive_affine_response_invariant() -> None:
    components = _components()
    transformed_components = PredictiveComponents(
        components.structure_probabilities,
        components.degrees_freedom,
        7.0 + 3.5 * components.locations,
        3.5 * components.scales,
        components.partition,
    )
    first = estimate_semiparametric_class_eig(components, _residual_law(), 32)
    second = estimate_semiparametric_class_eig(
        transformed_components, _residual_law(), 32
    )
    np.testing.assert_allclose(first.scores, second.scores, atol=2e-14)


def test_adaptive_utility_selects_only_after_interval_dominance() -> None:
    resolved = estimate_semiparametric_class_eig_until_ranked(
        _components(), _residual_law(), 8, 32
    )
    assert resolved.ranking_resolved
    assert resolved.selected_action_index == 0
    assert resolved.interval_gap > 0.0

    components = _components()
    tied = PredictiveComponents(
        components.structure_probabilities,
        components.degrees_freedom,
        np.repeat(components.locations[:, :1], 2, axis=1),
        np.repeat(components.scales[:, :1], 2, axis=1),
        components.partition,
    )
    unresolved = estimate_semiparametric_class_eig_until_ranked(
        tied, _residual_law(), 8, 16
    )
    assert not unresolved.ranking_resolved
    assert unresolved.selected_action_index is None
    assert unresolved.interval_gap <= 0.0


def test_maximin_production_scoring_chain_dispatches_p3h_per_model() -> None:
    components = _components()
    residual = _residual_law()
    result = real_acquisition._estimate_maximin_joint_until_ranked(
        (components,),
        np.zeros((1, 2)),
        (0.5,),
        np.ones(2, dtype=bool),
        8,
        16,
        4.0,
        2,
        (residual,),
    )
    assert result.estimates[0].integration_method == P3H_CLASS_EIG_METHOD
    with pytest.raises(ValueError, match="one-to-one"):
        real_acquisition._estimate_maximin_joint_until_ranked(
            (components,),
            np.zeros((1, 2)),
            (0.5,),
            np.ones(2, dtype=bool),
            8,
            16,
            4.0,
            2,
            (),
        )


def test_production_route_aborts_instead_of_using_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _components()
    tied = PredictiveComponents(
        components.structure_probabilities,
        components.degrees_freedom,
        np.repeat(components.locations[:, :1], 2, axis=1),
        np.repeat(components.scales[:, :1], 2, axis=1),
        components.partition,
    )
    representative = real_acquisition.RepresentativeSafeSet(
        current_mmd_squared=0.0,
        augmented_mmd_squared=np.zeros(2),
        safe_mask=np.ones(2, dtype=bool),
        tolerance=1e-12,
        kernel_bandwidth_squared=1.0,
        method="correctness-fixture",
    )
    monkeypatch.setattr(
        real_acquisition, "representative_mmd_safe_set", lambda *args: representative
    )
    monkeypatch.setattr(
        real_acquisition,
        "_validated_posterior_models",
        lambda *args: (SimpleNamespace(likelihood_power=0.5),),
    )
    monkeypatch.setattr(
        real_acquisition,
        "_model_components_and_offsets",
        lambda *args: ((tied,), np.zeros((1, 2))),
    )
    with pytest.raises(FloatingPointError, match="selection is forbidden"):
        real_acquisition._score_pcpi_discriminative(
            object(),
            object(),
            np.asarray([[0.0], [1.0]]),
            tied,
            np.asarray([[0.0], [1.0]]),
            np.asarray([[0.0]]),
            None,
            (_residual_law(),),
            minimum_samples=8,
            maximum_samples=8,
            error_safety_factor=4.0,
            growth_factor=2,
        )


def test_one_class_has_zero_transformed_information() -> None:
    components = _components()
    one_class = PredictiveComponents(
        components.structure_probabilities,
        components.degrees_freedom,
        components.locations,
        components.scales,
        ClassPartition(
            class_ids=("all",),
            member_indices=((0, 1, 2),),
            class_probabilities=(1.0,),
            structure_to_class=(0, 0, 0),
        ),
    )
    estimate = estimate_semiparametric_class_eig(one_class, _residual_law(), 16)
    np.testing.assert_allclose(estimate.scores, 0.0, atol=2e-15)


def test_invalid_partition_fails_closed_before_projection() -> None:
    components = _components()
    invalid = PredictiveComponents(
        components.structure_probabilities,
        components.degrees_freedom,
        components.locations,
        components.scales,
        ClassPartition(
            class_ids=("left", "right"),
            member_indices=((0,), (2,)),
            class_probabilities=(0.5, 0.5),
            structure_to_class=(0, 0, 1),
        ),
    )
    with pytest.raises(ValueError, match="partition every structure once"):
        estimate_semiparametric_class_eig(invalid, _residual_law(), 16)


def test_source_has_no_response_rng_or_old_eig_dispatch_surface() -> None:
    source = inspect.getsource(implementation)
    assert "estimate_class_eig(" not in source
    assert "exact_class_eig(" not in source
    assert "np.random" not in source
    assert "candidate_targets" not in source
    parameters = set(inspect.signature(estimate_semiparametric_class_eig).parameters)
    assert "responses" not in parameters
    assert "targets" not in parameters


def test_frozen_p3h3_config_and_runner_evaluation_pass() -> None:
    config = runner._load_config(CONFIG)
    result = runner._evaluate(config)
    assert result["schema"] == runner.RESULT_SCHEMA
    assert result["status"] == "passed"
    assert all(result["checks"].values())
    assert not result["real_data_access"]
    assert not result["candidate_selection_executed"]
    assert not result["operational_execution_authorized"]


def test_p3h3_config_tampering_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["transformed_utility"]["old_student_t_eig_reuse"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="contract was modified"):
        runner._load_config(path)
