from __future__ import annotations

from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.data import PoolOracle
from hypothesis_mvp.pcpi import (
    ANALYTIC_CLASS_EIG_BOUNDS_METHOD,
    ClassPartition,
    PredictiveComponents,
    REFERENCE_DOMINANCE_POLICY,
    REFERENCE_FALLBACK_MODE,
    SequentialReferencePosterior,
    aggregate_operational_classes,
    analytic_class_eig_bounds,
    class_partition,
    exact_class_eig,
    score_reference_dominance_actions,
    stable_reference_policy_seed,
)
from hypothesis_mvp.pcpi.reference import (
    DevelopmentStandardizer,
    fit_bank_preconditioner,
    generic_real_bank,
)
from hypothesis_mvp.pcpi.reference.inference_fixture import (
    correctness_diagnostic_bank,
    correctness_diagnostic_observations,
)
from scripts.progress import ProgressReporter
from scripts.run_pcpi_p3b_real import _load_config, _run_policy
from scripts.run_pcpi_p3d_real import P3D2_PROTOCOL, build_parser


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3d_2_reference_dominance_real_acquisition.json"


def _continuous_fixture() -> PredictiveComponents:
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


def test_analytic_class_eig_bounds_contain_independent_exact_quadrature() -> None:
    components = _continuous_fixture()
    bounds = analytic_class_eig_bounds(components)
    exact = exact_class_eig(components, epsabs=1e-11, epsrel=1e-10)
    assert bounds.method == ANALYTIC_CLASS_EIG_BOUNDS_METHOD
    assert np.all(bounds.lower_bounds <= exact.scores + exact.quadrature_errors)
    assert np.all(exact.scores - exact.quadrature_errors <= bounds.upper_bounds)
    assert np.all(bounds.lower_bounds >= 0.0)
    assert np.all(bounds.upper_bounds <= components.partition.entropy)


def test_analytic_bounds_are_positive_affine_response_invariant() -> None:
    components = _continuous_fixture()
    transformed = PredictiveComponents(
        components.structure_probabilities,
        components.degrees_freedom,
        7.0 + 3.5 * components.locations,
        3.5 * components.scales,
        components.partition,
    )
    first = analytic_class_eig_bounds(components)
    second = analytic_class_eig_bounds(transformed)
    np.testing.assert_allclose(first.lower_bounds, second.lower_bounds, atol=3e-14)
    np.testing.assert_allclose(first.upper_bounds, second.upper_bounds, atol=3e-14)


def test_one_class_has_exactly_zero_information_capacity() -> None:
    components = _continuous_fixture()
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
    bounds = analytic_class_eig_bounds(one_class)
    np.testing.assert_array_equal(bounds.lower_bounds, np.zeros(2))
    np.testing.assert_array_equal(bounds.upper_bounds, np.zeros(2))


@pytest.mark.parametrize(
    "levels",
    ((), (0.2, 0.2), (-0.1, 0.5), (0.5, 1.0), (0.7, 0.3)),
)
def test_analytic_bounds_reject_unregistered_quantizers(
    levels: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        analytic_class_eig_bounds(
            _continuous_fixture(), quantization_probability_levels=levels
        )


def test_reference_acquisition_is_response_free_and_permutation_invariant() -> None:
    assert set(signature(score_reference_dominance_actions).parameters).isdisjoint(
        {"targets", "y", "validation_y", "pool_y"}
    )
    bank = correctness_diagnostic_bank()
    X, y = correctness_diagnostic_observations(5, 12)
    engine = SequentialReferencePosterior(bank)
    posterior = engine.fit_batch(X[:6], y[:6])
    actions = np.linspace(-3.0, 3.0, 21)
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=0.2
    )
    partition = class_partition(posterior, classes)
    identifiers = np.arange(100, 121, dtype=np.int64)
    seed = stable_reference_policy_seed(20260814, 0)
    first = score_reference_dominance_actions(
        engine,
        posterior,
        actions,
        identifiers,
        target_partition=partition,
        reference_seed=seed,
    )
    order = np.asarray([7, 2, 18, 0, 14, 4, 10, 20, 1, 6, 9, 12, 3, 5, 8, 11, 13, 15, 16, 17, 19])
    second = score_reference_dominance_actions(
        engine,
        posterior,
        actions[order],
        identifiers[order],
        target_partition=partition,
        reference_seed=seed,
    )
    assert first.policy == REFERENCE_DOMINANCE_POLICY
    assert first.decision.targeted_handover
    assert first.decision.selected_candidate_id == second.decision.selected_candidate_id
    assert first.decision.reference_sample_candidate_id == second.decision.reference_sample_candidate_id
    assert first.target_partition_hash == partition.stable_hash


def test_unresolved_class_eig_intervals_return_exactly_to_reference() -> None:
    bank = correctness_diagnostic_bank()
    X, y = correctness_diagnostic_observations(5, 12)
    engine = SequentialReferencePosterior(bank)
    posterior = engine.fit_batch(X[:6], y[:6])
    actions = np.zeros(7)
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=1e6
    )
    partition = class_partition(posterior, classes)
    result = score_reference_dominance_actions(
        engine,
        posterior,
        actions,
        np.asarray([70, 10, 60, 20, 50, 30, 40]),
        target_partition=partition,
        reference_seed=stable_reference_policy_seed(20260814, 1),
    )
    assert not result.decision.targeted_handover
    assert result.decision.utility_mode == REFERENCE_FALLBACK_MODE
    assert result.decision.selected_candidate_id == (
        result.decision.reference_sample_candidate_id
    )
    np.testing.assert_array_equal(
        result.utility_bounds.lower_bounds, np.zeros(len(actions))
    )
    np.testing.assert_array_equal(
        result.utility_bounds.upper_bounds, np.zeros(len(actions))
    )


def test_p3d2_config_is_real_only_and_excludes_p3b_p3c_module_contracts() -> None:
    config = _load_config(CONFIG, ROOT, P3D2_PROTOCOL)
    assert config["heldout_state"] == "closed"
    assert tuple(config["policies"])[-1] == REFERENCE_DOMINANCE_POLICY
    forbidden = {
        "pcpi_joint_target",
        "conditional_predictive_information_method",
        "representative_guard",
        "representative_discrepancy",
        "pcpi_ambiguity_set",
        "pcpi_robust_utility",
        "pcpi_discrepancy_profile",
        "pcpi_uncertified_eig_action",
    }
    assert not forbidden & set(config)


def test_p3d2_cli_requires_local_real_data_root() -> None:
    parser = build_parser(P3D2_PROTOCOL)
    options = {action.dest for action in parser._actions}
    assert options == {
        "help", "data_root", "output_dir", "source_artifact", "config",
        "phase", "heldout_state",
    }
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_p3d2_policy_loop_passes_correctness_fixture_only(tmp_path: Path) -> None:
    raw_X, raw_y = correctness_diagnostic_observations(20260814, 64)
    raw_X = raw_X[:, None]
    initial_raw_X, initial_raw_y = raw_X[:20], raw_y[:20]
    validation_raw_X, validation_raw_y = raw_X[20:32], raw_y[20:32]
    pool_X, pool_y = raw_X[32:], raw_y[32:]
    standardizer = DevelopmentStandardizer.fit(initial_raw_X, initial_raw_y)
    initial_X = standardizer.transform_X(initial_raw_X)
    initial_y = standardizer.transform_y(initial_raw_y)
    design_preconditioner = fit_bank_preconditioner(
        generic_real_bank(initial_X.shape[1]), initial_X
    )
    config = dict(_load_config(CONFIG, ROOT, P3D2_PROTOCOL))
    config["acquisition_observation_budget"] = 3
    summary, curves, queries = _run_policy(
        dataset_id="correctness_fixture_not_efficacy",
        seed=20260814,
        policy=REFERENCE_DOMINANCE_POLICY,
        initial_X=initial_X,
        initial_y=initial_y,
        validation_X=standardizer.transform_X(validation_raw_X),
        validation_y=standardizer.transform_y(validation_raw_y),
        fixed_domain_X=standardizer.transform_X(pool_X),
        candidate_indices=np.arange(len(pool_X)),
        pool_X=pool_X,
        pool_row_ids=np.asarray([
            f"correctness:{index}" for index in range(len(pool_X))
        ]),
        oracle=PoolOracle(pool_X, pool_y),
        standardizer=standardizer,
        subset_commitments={
            "initial": "a" * 64,
            "validation": "b" * 64,
            "candidate": "c" * 64,
        },
        config=config,
        reporter=ProgressReporter(tmp_path / "progress.jsonl"),
        design_preconditioner=design_preconditioner,
        protocol=P3D2_PROTOCOL,
    )
    assert len(curves) == 4
    assert len(queries) == 3
    assert summary["pcpi_decision_rule_valid_rate"] == 1.0
    assert summary["pcpi_class_eig_used_rate"] == 1.0
    assert summary["pcpi_maximin_joint_eig_used_rate"] == 0.0
    assert summary["pcpi_targeted_handover_rate"] + summary[
        "pcpi_reference_fallback_rate"
    ] == pytest.approx(1.0)
    assert all(row["reference_dominance_applied"] for row in queries)
    assert all(not row["representative_guard_applied"] for row in queries)
    assert all(row["robust_model_count"] == 0 for row in queries)
    assert all(row["discrepancy_method"] == "not-applied" for row in queries)
    assert all(
        row["acquisition_target_partition_hash"]
        == row["initial_frozen_class_partition_hash"]
        for row in queries
    )
