"""P3I decision-target alignment and marginal-transport invariance."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi import (
    DECISION_REGRET_DISTANCE_METRIC,
    DECISION_TARGETED_POLICY,
    P3I_COPULA_TRANSPORT_METHOD,
    ClassPartition,
    PosteriorModel,
    PredictiveComponents,
    SequentialReferencePosterior,
    aggregate_decision_equivalent_classes,
    budget_resolved_distance_threshold,
    class_partition,
    copula_transport_class_coupling,
    exact_class_eig,
    reconstruct_conditioned_likelihood_power_residual_family,
    score_decision_targeted_actions,
    semiparametric_class_coupling,
)
from hypothesis_mvp.pcpi.reference import DyadicPolyaTreePredictiveLaw
from hypothesis_mvp.pcpi.reference.classes import _decision_regret_distances
from tests._pcpi_fixtures import unit_bank
from scripts import run_pcpi_p3i1_decision_alignment_correctness as gate


ROOT = Path(__file__).resolve().parents[1]


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


def _family_case():
    x = np.linspace(-1.5, 1.5, 12)[:, None]
    y = 0.75 - 1.2 * x[:, 0] + 0.55 * np.square(x[:, 0])
    engines = tuple(
        SequentialReferencePosterior(unit_bank(), power)
        for power in (0.5, 1.0)
    )
    family = reconstruct_conditioned_likelihood_power_residual_family(
        engines, x[:4], y[:4], x[4:], y[4:]
    )
    models = tuple(
        PosteriorModel(state.likelihood_power, state.engine, state.posterior)
        for state in family.model_states
    )
    nominal = family.model_states[-1]
    return x, nominal.engine, nominal.posterior, models, family


def test_copula_transport_preserves_the_base_information_target() -> None:
    components, residual = _components(), _residual_law()
    transported = copula_transport_class_coupling(
        components, residual, action_index=0, nodes_per_leaf=64
    )
    base = exact_class_eig(components, epsabs=1e-11, epsrel=1e-10)
    assert transported.method == P3I_COPULA_TRANSPORT_METHOD
    assert transported.invariance_error <= 2e-15
    assert transported.maximum_marginal_error < 2e-6
    assert abs(transported.mutual_information - base.scores[0]) < 1e-4
    assert np.max(np.abs(
        transported.output_response_nodes - transported.base_response_nodes
    )) > 1.0


def test_p3h_projection_and_p3i_transport_are_distinct_model_choices() -> None:
    old = semiparametric_class_coupling(
        _components(), _residual_law(), action_index=0, nodes_per_leaf=64
    )
    new = copula_transport_class_coupling(
        _components(), _residual_law(), action_index=0, nodes_per_leaf=64
    )
    assert abs(old.mutual_information - new.mutual_information) > 0.05
    np.testing.assert_allclose(
        np.sum(new.joint_probabilities, axis=1),
        new.outcome_probabilities,
        rtol=0.0,
        atol=2e-15,
    )


def test_decision_regret_distance_targets_means_not_nuisance_quantiles() -> None:
    means = np.asarray([[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    distances = _decision_regret_distances(means, np.asarray([2.0, 4.0]))
    assert distances[0, 1] == 0.0
    assert distances[0, 2] == distances[1, 2] > 0.0


def test_decision_equivalent_classes_freeze_one_common_scale() -> None:
    x, engine, posterior, _, _ = _family_case()
    threshold = budget_resolved_distance_threshold(32)
    forward = aggregate_decision_equivalent_classes(
        engine, posterior, x, distance_threshold=threshold
    )
    reverse = aggregate_decision_equivalent_classes(
        engine, posterior, x[::-1], distance_threshold=threshold
    )
    assert forward.metric == DECISION_REGRET_DISTANCE_METRIC
    assert forward.quantile_levels == ()
    assert forward.scale_hash != "pairwise-pooled-predictive-scale"
    assert forward == reverse


def test_p3i_score_excludes_predictive_nuisance_from_primary_target() -> None:
    x, engine, posterior, models, family = _family_case()
    target = np.linspace(-1.5, 1.5, 24)[:, None]
    classes = aggregate_decision_equivalent_classes(
        engine,
        posterior,
        target,
        distance_threshold=budget_resolved_distance_threshold(32),
    )
    candidates = np.asarray([[-1.25], [-0.50], [0.25], [1.00]])
    result = score_decision_targeted_actions(
        engine,
        posterior,
        candidates,
        target_partition=class_partition(posterior, classes),
        predictive_target_actions=target,
        representative_observed_actions=x,
        posterior_models=models,
        semiparametric_residual_family=family,
        minimum_samples=32,
        maximum_samples=64,
        error_safety_factor=4.0,
        growth_factor=2,
    )
    assert result.policy == DECISION_TARGETED_POLICY
    assert result.semiparametric_information_invariance_applied
    assert result.semiparametric_transport_method == P3I_COPULA_TRANSPORT_METHOD
    np.testing.assert_array_equal(
        result.conditional_predictive_eig_scores, np.zeros(len(candidates))
    )
    np.testing.assert_allclose(
        result.joint_class_predictive_scores,
        result.class_eig_scores,
        rtol=0.0,
        atol=2e-14,
    )


def test_p3i_source_has_no_old_projection_or_conditional_epig_dispatch() -> None:
    source = inspect.getsource(score_decision_targeted_actions)
    forbidden = (
        "semiparametric_class_coupling",
        "estimate_semiparametric_class_eig",
        "class_conditional_predictive_eig",
        "validation",
        "heldout",
        "candidate_targets",
    )
    assert not any(token in source for token in forbidden)
    assert "_bound_semiparametric_residual_laws" in source
    assert "zero_conditional" in source


def test_p3i_correctness_gate_is_response_free_and_blocks_real_execution() -> None:
    config = gate._load_config(
        ROOT / "configs" / "p3i_1_decision_alignment_correctness.json"
    )
    result = gate._evaluate(config)
    assert result["status"] == "passed-correctness-no-real-experiment-authorized"
    assert all(result["decisions"].values())
    assert result["real_data_access"] is False
    assert result["candidate_response_access"] is False
    assert result["heldout_access"] is False
    assert result["operational_execution_authorized"] is False
