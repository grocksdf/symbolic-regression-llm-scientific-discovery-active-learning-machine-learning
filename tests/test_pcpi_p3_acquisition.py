from __future__ import annotations

import json
from inspect import signature
import math
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.data import PoolOracle
from hypothesis_mvp.pcpi import (
    ACQUISITION_POLICIES,
    ASYMPTOTIC_RANK_CERTIFICATE,
    DEFAULT_QUADRATURE_SAFETY_FACTOR,
    GAUSS_JACOBI_INTEGRATION,
    GAUSSIAN_CLASS_CONDITIONAL_EPIG,
    MAXIMIN_RANK_CERTIFICATE,
    PosteriorModel,
    REPRESENTATIVE_MMD_METHOD,
    SequentialReferencePosterior,
    aggregate_operational_classes,
    budget_resolved_distance_threshold,
    class_conditional_predictive_eig,
    estimate_class_eig_until_ranked,
    exact_class_eig,
    fixed_class_entropy,
    least_favorable_model_indices,
    normalized_area_under_learning_curve,
    posterior_epistemic_variance,
    predictive_components,
    predictive_components_for_partition,
    predictive_variance,
    realized_fixed_class_entropy_gain,
    representative_mmd_safe_set,
    score_acquisition_actions,
    select_stable_argmax,
    stable_derived_seed,
)
from hypothesis_mvp.pcpi.acquisition import class_partition
from hypothesis_mvp.pcpi.real_acquisition import _validated_posterior_models
from hypothesis_mvp.pcpi.reference.classes import _complete_link_clusters
from hypothesis_mvp.pcpi.reference import (
    DevelopmentStandardizer,
    fit_bank_preconditioner,
    generic_real_bank,
)
from hypothesis_mvp.pcpi.reference import calibrate_likelihood_power
from hypothesis_mvp.pcpi.reference.basis import design_matrix
from scripts.plot_pcpi_p2b_diagnostic import _mean_ci
from scripts.run_pcpi_p3a_eig import CLAIM_BOUNDARY as P3A_CLAIM_BOUNDARY
from scripts.run_pcpi_p3a_eig import _export_evidence
from scripts.run_pcpi_p3a_eig import _load_config as load_p3a_config
from scripts.run_pcpi_p3a_eig import _record_evidence
from scripts.run_pcpi_p3b_real import CLAIM_BOUNDARY as P3B_CLAIM_BOUNDARY
from scripts.run_pcpi_p3b_real import _expected_candidate_evaluations
from scripts.run_pcpi_p3b_real import _export_evidence as export_p3b_evidence
from scripts.run_pcpi_p3b_real import _family_seed_rows, _paired_effects
from scripts.run_pcpi_p3b_real import _evidence_flags
from scripts.run_pcpi_p3b_real import _load_config as load_p3b_config
from scripts.run_pcpi_p3b_real import _record_evidence as record_p3b_evidence
from scripts.run_pcpi_p3b_real import _run_policy, build_parser
from scripts.run_pcpi_p3b3_diagnostic import (
    _load_config as load_p3b3_diagnostic_config,
)
from scripts.run_pcpi_p3b3_diagnostic import (
    build_parser as build_p3b3_diagnostic_parser,
)
from scripts.run_pcpi_p3b6_predictive_consistency_diagnostic import (
    _load_config as load_p3b6_diagnostic_config,
)
from scripts.run_pcpi_p3b6_predictive_consistency_diagnostic import (
    build_parser as build_p3b6_diagnostic_parser,
)
from scripts.run_pcpi_p3b7_budget_resolved_classes_diagnostic import (
    _load_config as load_p3b7_diagnostic_config,
)
from scripts.run_pcpi_p3b7_budget_resolved_classes_diagnostic import (
    _refines as budget_partition_refines,
)
from scripts.run_pcpi_p3b7_budget_resolved_classes_diagnostic import (
    build_parser as build_p3b7_diagnostic_parser,
)
from scripts.run_pcpi_p3b8_joint_eig_diagnostic import (
    _evaluate as evaluate_p3b8_diagnostic,
)
from scripts.run_pcpi_p3b8_joint_eig_diagnostic import (
    _load_config as load_p3b8_diagnostic_config,
)
from scripts.run_pcpi_p3b8_joint_eig_diagnostic import (
    build_parser as build_p3b8_diagnostic_parser,
)
from scripts.run_pcpi_p3b9_representative_safe_diagnostic import (
    _evaluate as evaluate_p3b9_diagnostic,
)
from scripts.run_pcpi_p3b9_representative_safe_diagnostic import (
    _load_config as load_p3b9_diagnostic_config,
)
from scripts.run_pcpi_p3b9_representative_safe_diagnostic import (
    build_parser as build_p3b9_diagnostic_parser,
)
from scripts.run_pcpi_p3b10_maximin_joint_eig_diagnostic import (
    _evaluate as evaluate_p3b10_diagnostic,
)
from scripts.run_pcpi_p3b10_maximin_joint_eig_diagnostic import (
    _load_config as load_p3b10_diagnostic_config,
)
from scripts.run_pcpi_p3b10_maximin_joint_eig_diagnostic import (
    build_parser as build_p3b10_diagnostic_parser,
)
from scripts.progress import ProgressReporter
from tests._pcpi_fixtures import unit_bank, unit_observations


ROOT = Path(__file__).resolve().parents[1]


def test_real_evidence_flags_separate_protocol_from_efficacy() -> None:
    negative = _evidence_flags(
        True,
        {"strong_evidence": False, "strong_structural_evidence": False},
    )
    assert negative == {
        "formal_protocol_evidence": True,
        "formal_efficacy_evidence": False,
    }
    positive = _evidence_flags(
        True,
        {"strong_evidence": True, "strong_structural_evidence": False},
    )
    assert positive["formal_protocol_evidence"]
    assert positive["formal_efficacy_evidence"]


def test_p3_configs_and_claim_boundaries_are_frozen_by_evidence_role() -> None:
    p3a = load_p3a_config(ROOT / "configs" / "p3a_exact_class_eig.json", ROOT)
    p3b = load_p3b_config(ROOT / "configs" / "p3b_real_acquisition.json", ROOT)
    p3b3 = load_p3b3_diagnostic_config(
        ROOT / "configs" / "p3b_3_decision_rule_diagnostic.json", ROOT
    )
    p3b6 = load_p3b6_diagnostic_config(
        ROOT / "configs" / "p3b_6_predictive_consistency_diagnostic.json", ROOT
    )
    p3b7 = load_p3b7_diagnostic_config(
        ROOT / "configs" / "p3b_7_budget_resolved_classes_diagnostic.json", ROOT
    )
    p3b8 = load_p3b8_diagnostic_config(
        ROOT / "configs" / "p3b_8_joint_class_predictive_eig_diagnostic.json",
        ROOT,
    )
    p3b9 = load_p3b9_diagnostic_config(
        ROOT / "configs" / "p3b_9_representative_safe_joint_diagnostic.json",
        ROOT,
    )
    p3b10 = load_p3b10_diagnostic_config(
        ROOT / "configs" / "p3b_10_maximin_joint_eig_diagnostic.json",
        ROOT,
    )
    assert p3a["fixture_role"] == "inference_correctness_diagnostic_fixture"
    assert p3a["stage"] == "P3A.2"
    assert len(p3a["scenarios"]) == 8
    assert p3a["evaluation_counts"] == [32, 64, 128, 256, 512]
    assert p3a["operational_class_distance_threshold"] == 1.0
    assert p3a["error_safety_factor"] == DEFAULT_QUADRATURE_SAFETY_FACTOR
    assert p3a["estimator_integration_method"] == GAUSS_JACOBI_INTEGRATION
    assert p3b["heldout_state"] == "closed"
    assert p3b["stage"] == "P3B.10"
    assert p3b["posterior_type"] == "power-likelihood-generalized-bayes"
    assert p3b["likelihood_power_calibration_role"] == "initial-development-only"
    assert p3b["operational_class_resolution_method"] == (
        "one-unit-aggregate-predictive-separation"
    )
    assert p3b["operational_class_aggregate_separation"] == 1.0
    assert p3b["class_evaluation_partition"] == "initial-frozen"
    assert p3b["pcpi_class_target_partition"] == "initial-frozen"
    assert p3b["pcpi_uncertified_eig_action"] == "posterior-epistemic-variance"
    assert p3b["pcpi_joint_target"] == (
        "initial-frozen-class-and-target-prediction"
    )
    assert p3b["predictive_target_distribution"] == (
        "registered-action-domain-uniform"
    )
    assert p3b["conditional_predictive_information_method"] == (
        GAUSSIAN_CLASS_CONDITIONAL_EPIG
    )
    assert p3b["representative_discrepancy"] == REPRESENTATIVE_MMD_METHOD
    assert p3b["representative_empty_safe_set_action"] == (
        "minimum-augmented-mmd"
    )
    assert p3b["predictive_design_transform"] == "posterior-target-frozen"
    assert p3b["eig_rank_certificate_method"] == MAXIMIN_RANK_CERTIFICATE
    assert p3b["likelihood_power_candidates"] == [0.125, 0.25, 0.5, 1.0]
    assert p3b["pcpi_ambiguity_set"] == "frozen-likelihood-power-candidates"
    assert p3b["pcpi_robust_utility"] == (
        "maximin-joint-class-predictive-information"
    )
    assert p3b["datasets"] == [
        "uci_ccpp", "uci_gas_turbine_co", "uci_gas_turbine_nox"
    ]
    assert "not real-data acquisition efficacy evidence" in P3A_CLAIM_BOUNDARY
    assert "provenance-verified CCPP" in P3B_CLAIM_BOUNDARY
    assert p3b3["stage"] == "P3B.3"
    assert p3b3["fixture_role"] == "inference_correctness_diagnostic_fixture"
    assert p3b3["heldout_state"] == "not-applicable"
    assert p3b6["stage"] == "P3B.6"
    assert p3b6["fixture_role"] == "inference_correctness_diagnostic_fixture"
    assert p3b6["heldout_state"] == "not-applicable"
    assert p3b7["stage"] == "P3B.7"
    assert p3b7["fixture_role"] == "inference_correctness_diagnostic_fixture"
    assert p3b7["measurement_budgets"] == [1, 4, 16, 32, 64, 256]
    assert p3b8["stage"] == "P3B.8"
    assert p3b8["fixture_role"] == "inference_correctness_diagnostic_fixture"
    assert p3b8["predictive_target_distribution"] == (
        "registered-action-domain-uniform"
    )
    assert p3b9["stage"] == "P3B.9"
    assert p3b9["fixture_role"] == "inference_correctness_diagnostic_fixture"
    assert p3b9["representative_discrepancy"] == REPRESENTATIVE_MMD_METHOD
    assert p3b10["stage"] == "P3B.10"
    assert p3b10["likelihood_power_candidates"] == [0.125, 0.25, 0.5, 1.0]


def test_p3a_tables_are_read_only_exports_of_the_evidence_registry(tmp_path: Path) -> None:
    for name in ("hypotheses", "diagnostics", "tables"):
        (tmp_path / name).mkdir()
    exact_rows = [
        {"action_index": 0, "action": 0.0, "exact_eig": 0.1, "quadrature_error": 1e-12}
    ]
    metric = {
        "scenario_id": "unit",
        "quadrature_evaluations": 128,
        "spearman": 1.0,
        "failure_status": "",
    }
    scores = [
        {
            "scenario_id": "unit",
            "quadrature_evaluations": 128,
            "action_index": 0,
            "action": 0.0,
            "exact_eig": 0.1,
            "estimated_eig": 0.11,
            "standard_error": 0.01,
        }
    ]
    summary = {
        "failures": [],
        "scenario_aggregates": [
            {
                "scenario_id": "unit",
                "quadrature_evaluations": 128,
                "mean_spearman": 1.0,
            }
        ],
        "overall_aggregates": [
            {"quadrature_evaluations": 128, "mean_spearman": 1.0}
        ],
        "evaluation_counts": [128],
        "gate_passed": True,
    }
    registry, evidence = _record_evidence(
        tmp_path,
        exact_rows,
        [metric],
        scores,
        [],
        summary,
        {"unit": {"partition_hash": "a" * 64, "classes": []}},
        {},
    )
    exported = _export_evidence(tmp_path, registry)
    assert evidence["event_count"] == 3
    assert exported["summary"] == summary
    assert exported["rows"] == [metric]
    export_manifest = json.loads(exported["export_path"].read_text())
    assert export_manifest["registry_head_hash"] == evidence["head_hash"]
    assert "tables/estimator_scores.csv" in export_manifest["files"]
    registry.lock_path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("seeds", list(range(1, 9))),
        ("candidate_pool_budget", 64),
        ("eig_quadrature_error_safety_factor", 2.0),
        ("failure_policy", "replace_failed_seed"),
        ("pcpi_class_target_partition", "dynamic"),
        ("pcpi_uncertified_eig_action", "first-index"),
        ("operational_class_aggregate_separation", 2.0),
        ("likelihood_power_candidates", [0.5, 1.0]),
        ("predictive_design_transform", "raw-basis"),
        ("pcpi_joint_target", "class-only"),
        ("predictive_target_distribution", "selected-actions"),
        ("conditional_predictive_information_method", "predictive-variance"),
        ("representative_discrepancy", "linear-mmd"),
        ("representative_safe_set_rule", "top-half"),
        ("representative_empty_safe_set_action", "first-index"),
    ),
)
def test_p3b_frozen_protocol_rejects_mutated_values(
    tmp_path: Path, key: str, value: object
) -> None:
    config = json.loads((ROOT / "configs" / "p3b_real_acquisition.json").read_text())
    config[key] = value
    candidate = tmp_path / "mutated.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_p3b_config(candidate, tmp_path)


def test_operational_class_eig_matches_exact_ranking_and_is_repeatable() -> None:
    bank = unit_bank()
    engine = SequentialReferencePosterior(bank)
    X, y = unit_observations(20260807, 8)
    posterior = engine.fit_batch(X, y)
    actions = np.linspace(-1.5, 1.5, 7)
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=1.0
    )
    components = predictive_components(engine, posterior, classes, actions)
    exact = exact_class_eig(components, epsabs=1e-9, epsrel=1e-8)
    first = estimate_class_eig_until_ranked(components, 32, 512)
    second = estimate_class_eig_until_ranked(components, 32, 512)
    assert len(classes.classes) < len(bank.structures)
    assert np.all(exact.scores >= 0.0)
    assert first.error_safety_factor == DEFAULT_QUADRATURE_SAFETY_FACTOR
    assert first.planned_looks == 5
    assert first.looks_used <= first.planned_looks
    assert first.certificate_method == ASYMPTOTIC_RANK_CERTIFICATE
    assert first.ranking_certified == (first.certificate_gap > 0.0)
    if not first.ranking_certified:
        assert first.estimate.sample_count == 512
    best = int(np.argmax(first.estimate.scores))
    competitors = np.asarray([index for index in range(len(actions)) if index != best])
    differences = first.estimate.scores[best] - first.estimate.scores[competitors]
    assert first.ranking_certified == bool(
        np.all(
            differences
            > first.estimate.error_bounds[best]
            + first.estimate.error_bounds[competitors]
        )
    )
    assert np.argmax(first.estimate.scores) == np.argmax(exact.scores)
    np.testing.assert_array_equal(first.estimate.scores, second.estimate.scores)
    np.testing.assert_array_equal(
        first.estimate.error_bounds, second.estimate.error_bounds
    )
    assert first.estimate.integration_method == GAUSS_JACOBI_INTEGRATION
    assert sum(first.estimate.structure_allocations) == first.estimate.sample_count


def test_budget_resolved_threshold_has_root_budget_identity() -> None:
    for budget in (1, 4, 16, 32, 64, 256):
        threshold = budget_resolved_distance_threshold(budget)
        assert threshold * math.sqrt(budget) == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError):
        budget_resolved_distance_threshold(0)
    with pytest.raises(TypeError):
        budget_resolved_distance_threshold(True)


def test_joint_class_predictive_diagnostic_passes_all_correctness_gates() -> None:
    config = load_p3b8_diagnostic_config(
        ROOT / "configs" / "p3b_8_joint_class_predictive_eig_diagnostic.json",
        ROOT,
    )
    rows, diagnostics, summary = evaluate_p3b8_diagnostic(config)
    assert rows
    assert summary["gate_passed"]
    assert all(summary["gate_decisions"].values())
    assert diagnostics["conditional_predictive_eig_min"] >= 0.0
    assert diagnostics["singleton_gaussian_identity_max_abs_error"] <= 2e-12
    assert diagnostics["exact_joint_inside_adaptive_error_envelope"]


def test_joint_diagnostic_cli_exposes_only_controlled_roles() -> None:
    parser = build_p3b8_diagnostic_parser()
    choices = parser._option_string_actions["--heldout-state"].choices
    assert tuple(choices) == ("not-applicable",)
    phase_choices = parser._option_string_actions["--phase"].choices
    assert tuple(phase_choices) == ("P3B.8",)


def test_representative_safe_diagnostic_passes_seventeen_decisions() -> None:
    config = load_p3b9_diagnostic_config(
        ROOT / "configs" / "p3b_9_representative_safe_joint_diagnostic.json",
        ROOT,
    )
    rows, diagnostics, summary = evaluate_p3b9_diagnostic(config)
    assert rows
    assert summary["gate_passed"]
    assert summary["gate_decision_count"] == 17
    assert all(summary["gate_decisions"].values())
    representative = diagnostics["representative"]
    assert representative["selected_mmd_squared"] <= (
        representative["current_mmd_squared"] + 2e-13
    )
    assert representative["fallback_used"]
    assert not representative["fallback_safe_set_nonempty"]


def test_representative_safe_diagnostic_cli_is_correctness_only() -> None:
    parser = build_p3b9_diagnostic_parser()
    assert tuple(
        parser._option_string_actions["--heldout-state"].choices
    ) == ("not-applicable",)
    assert tuple(parser._option_string_actions["--phase"].choices) == ("P3B.9",)


def test_maximin_joint_diagnostic_passes_twenty_seven_decisions() -> None:
    config = load_p3b10_diagnostic_config(
        ROOT / "configs" / "p3b_10_maximin_joint_eig_diagnostic.json",
        ROOT,
    )
    rows, diagnostics, summary = evaluate_p3b10_diagnostic(config)
    assert rows
    assert summary["gate_passed"]
    assert summary["gate_decision_count"] == 27
    assert all(summary["gate_decisions"].values())
    assert diagnostics["maximin"]["selected_action_index"] == (
        diagnostics["maximin"]["exact_selected_action_index"]
    )


def test_maximin_joint_diagnostic_cli_is_correctness_only() -> None:
    parser = build_p3b10_diagnostic_parser()
    assert tuple(
        parser._option_string_actions["--heldout-state"].choices
    ) == ("not-applicable",)
    assert tuple(parser._option_string_actions["--phase"].choices) == ("P3B.10",)
    assert "--data-root" not in parser._option_string_actions
    assert not parser._option_string_actions["--source-artifact"].required


def test_least_favorable_ties_choose_the_smallest_power() -> None:
    indices = least_favorable_model_indices(
        np.asarray([[0.2, 0.1], [0.2, 0.3]]), (0.125, 1.0)
    )
    np.testing.assert_array_equal(indices, np.asarray([0, 0]))


def test_posterior_model_rejects_a_mismatched_likelihood_power() -> None:
    bank = unit_bank()
    engine = SequentialReferencePosterior(bank, 1.0)
    X, y = unit_observations(20260807, 8)
    with pytest.raises(ValueError):
        PosteriorModel(0.5, engine, engine.fit_batch(X, y))


def test_posterior_model_rejects_a_posterior_from_another_power() -> None:
    bank = unit_bank()
    X, y = unit_observations(20260807, 8)
    ordinary = SequentialReferencePosterior(bank, 1.0)
    tempered = SequentialReferencePosterior(bank, 0.5)
    with pytest.raises(ValueError, match="does not match its posterior"):
        PosteriorModel(1.0, ordinary, tempered.fit_batch(X, y))


def test_maximin_family_rejects_a_different_observed_history() -> None:
    bank = unit_bank()
    X, y = unit_observations(20260807, 8)
    nominal_engine = SequentialReferencePosterior(bank, 0.5)
    nominal = nominal_engine.fit_batch(X, y)
    ordinary_engine = SequentialReferencePosterior(bank, 1.0)
    changed_y = y.copy()
    changed_y[0] += 0.25
    models = (
        PosteriorModel(0.5, nominal_engine, nominal),
        PosteriorModel(1.0, ordinary_engine, ordinary_engine.fit_batch(X, changed_y)),
    )
    with pytest.raises(ValueError, match="share one bank, design transform"):
        _validated_posterior_models(nominal_engine, nominal, models)


def test_maximin_family_rejects_a_different_design_preconditioner() -> None:
    bank = unit_bank()
    X, y = unit_observations(20260807, 12)
    nominal_preconditioner = fit_bank_preconditioner(bank, X)
    other_preconditioner = fit_bank_preconditioner(bank, X + 0.4)
    nominal_engine = SequentialReferencePosterior(
        bank, 0.5, nominal_preconditioner
    )
    other_engine = SequentialReferencePosterior(bank, 1.0, other_preconditioner)
    nominal = nominal_engine.fit_batch(X, y)
    models = (
        PosteriorModel(0.5, nominal_engine, nominal),
        PosteriorModel(1.0, other_engine, other_engine.fit_batch(X, y)),
    )
    with pytest.raises(ValueError, match="share one bank, design transform"):
        _validated_posterior_models(nominal_engine, nominal, models)


def test_maximin_family_must_include_the_nominal_model() -> None:
    bank = unit_bank()
    X, y = unit_observations(20260807, 8)
    nominal_engine = SequentialReferencePosterior(bank, 0.5)
    nominal = nominal_engine.fit_batch(X, y)
    models = tuple(
        PosteriorModel(power, engine, engine.fit_batch(X, y))
        for power in (0.25, 1.0)
        for engine in (SequentialReferencePosterior(bank, power),)
    )
    with pytest.raises(ValueError, match="include the nominal model"):
        _validated_posterior_models(nominal_engine, nominal, models)


def test_maximin_family_rejects_an_explicit_empty_family() -> None:
    bank = unit_bank()
    X, y = unit_observations(20260807, 8)
    nominal_engine = SequentialReferencePosterior(bank, 0.5)
    nominal = nominal_engine.fit_batch(X, y)
    with pytest.raises(ValueError, match="include the nominal model"):
        _validated_posterior_models(nominal_engine, nominal, ())


def test_maximin_family_accepts_one_history_and_design_preconditioner() -> None:
    bank = unit_bank()
    X, y = unit_observations(20260807, 12)
    preconditioner = fit_bank_preconditioner(bank, X)
    engines = tuple(
        SequentialReferencePosterior(bank, power, preconditioner)
        for power in (0.25, 0.5, 1.0)
    )
    models = tuple(
        PosteriorModel(engine.likelihood_power, engine, engine.fit_batch(X, y))
        for engine in engines
    )
    nominal = models[1]
    assert _validated_posterior_models(
        nominal.engine, nominal.posterior, models
    ) == models


def test_budget_resolved_partitions_refine_and_are_unit_invariant() -> None:
    rng = np.random.default_rng(20260811)
    X = rng.normal(size=(24, 2))
    y = 0.3 + X[:, 0] + 0.4 * X[:, 1] ** 2 + rng.normal(scale=0.8, size=24)
    actions = rng.normal(size=(64, 2))
    standardizer = DevelopmentStandardizer.fit(X, y)
    standardized_X = standardizer.transform_X(X)
    standardized_y = standardizer.transform_y(y)
    standardized_actions = standardizer.transform_X(actions)
    bank = generic_real_bank(2)
    engine = SequentialReferencePosterior(
        bank, 0.5, fit_bank_preconditioner(bank, standardized_X)
    )
    posterior = engine.fit_batch(standardized_X, standardized_y)
    classes = [
        aggregate_operational_classes(
            engine, posterior, standardized_actions,
            distance_threshold=budget_resolved_distance_threshold(budget),
        )
        for budget in (1, 4, 16, 32, 64, 256)
    ]
    assert all(
        budget_partition_refines(right, left)
        for left, right in zip(classes[:-1], classes[1:], strict=True)
    )
    assert 1 < len(classes[3].classes) < len(bank.structures)

    transformed_X = 11.0 + 3.5 * X
    transformed_y = -4.0 + 2.0 * y
    transformed_standardizer = DevelopmentStandardizer.fit(
        transformed_X, transformed_y
    )
    transformed_standardized_X = transformed_standardizer.transform_X(
        transformed_X
    )
    transformed_engine = SequentialReferencePosterior(
        bank, 0.5, fit_bank_preconditioner(bank, transformed_standardized_X)
    )
    transformed = aggregate_operational_classes(
        transformed_engine,
        transformed_engine.fit_batch(
            transformed_standardized_X,
            transformed_standardizer.transform_y(transformed_y),
        ),
        transformed_standardizer.transform_X(11.0 + 3.5 * actions),
        distance_threshold=budget_resolved_distance_threshold(32),
    )
    assert {
        frozenset(item.structure_ids) for item in transformed.classes
    } == {
        frozenset(item.structure_ids) for item in classes[3].classes
    }


def test_all_real_policies_share_one_target_free_scoring_interface() -> None:
    parameters = set(signature(score_acquisition_actions).parameters)
    assert "targets" not in parameters
    assert "y" not in parameters
    bank = generic_real_bank(2)
    engine = SequentialReferencePosterior(bank)
    X = np.column_stack((np.linspace(-1.0, 1.0, 20), np.linspace(1.0, -1.0, 20)))
    y = 0.2 + X[:, 0] - 0.5 * X[:, 1]
    posterior = engine.fit_batch(X, y)
    actions = X[:9]
    classes = aggregate_operational_classes(engine, posterior, actions, resolution=0.05)
    for policy in ACQUISITION_POLICIES:
        representative_observed = (
            X if policy == "pcpi_representative_safe_maximin_joint_eig" else None
        )
        result = score_acquisition_actions(
            engine,
            posterior,
            classes,
            actions,
            policy=policy,
            seed=stable_derived_seed(44, policy, 0),
            eig_min_samples=128,
            eig_max_samples=256,
            eig_error_safety_factor=DEFAULT_QUADRATURE_SAFETY_FACTOR,
            eig_growth_factor=2,
            qbc_committee_size=16,
            representative_observed_actions=representative_observed,
        )
        assert result.scores.shape == (len(actions),)
        assert np.all(np.isfinite(result.scores))


def test_representative_mmd_safe_set_is_label_free_and_unit_invariant() -> None:
    rng = np.random.default_rng(20260812)
    observed = rng.normal(loc=0.8, size=(12, 2))
    candidates = rng.normal(size=(9, 2))
    targets = rng.normal(size=(23, 2))
    first = representative_mmd_safe_set(observed, candidates, targets)
    permutation = np.arange(len(candidates))[::-1]
    scale = np.asarray([3.5, -2.0])
    offset = np.asarray([11.0, -7.0])
    second = representative_mmd_safe_set(
        offset + observed[::-1] * scale,
        offset + candidates[permutation] * scale,
        (offset + targets * scale)[::-1],
    )
    assert first.method == REPRESENTATIVE_MMD_METHOD
    assert first.safe_set_nonempty
    assert first.current_mmd_squared == pytest.approx(
        second.current_mmd_squared, abs=3e-13
    )
    np.testing.assert_allclose(
        first.augmented_mmd_squared,
        second.augmented_mmd_squared[::-1],
        atol=3e-13,
    )
    np.testing.assert_array_equal(first.safe_mask, second.safe_mask[::-1])


def test_power_posterior_batch_sequential_and_quadrature_agree() -> None:
    bank = unit_bank()
    X, y = unit_observations(20260807, 10)
    for likelihood_power in (0.125, 0.5, 1.0):
        engine = SequentialReferencePosterior(bank, likelihood_power)
        batch = engine.fit_batch(X, y)
        sequential = engine.fit_sequential(X, y)
        np.testing.assert_allclose(
            [member.probability for member in batch.members],
            [member.probability for member in sequential.members],
            atol=2e-13,
        )
        for member in batch.members:
            assert member.log_marginal_likelihood == pytest.approx(
                engine.log_marginal_quadrature(member.structure, X, y), abs=2e-12
            )


def test_likelihood_power_calibration_is_deterministic_and_task_agnostic() -> None:
    assert set(signature(calibrate_likelihood_power).parameters) == {
        "bank", "x", "y", "candidates", "design_preconditioner"
    }
    rng = np.random.default_rng(20260811)
    X = np.zeros((48, 1), dtype=float)
    informative = np.arange(len(X)) % 2 == 0
    X[informative, 0] = rng.uniform(-1.0, 1.0, int(np.sum(informative)))
    y = np.zeros(len(X), dtype=float)
    y[informative] = rng.normal(size=int(np.sum(informative)))
    candidates = (0.125, 0.25, 0.5, 1.0)
    bank = generic_real_bank(1)
    preconditioner = fit_bank_preconditioner(bank, X)
    first = calibrate_likelihood_power(bank, X, y, candidates, preconditioner)
    second = calibrate_likelihood_power(bank, X, y, candidates, preconditioner)
    assert first == second
    assert first.stable_hash == second.stable_hash
    assert first.selected_likelihood_power < 1.0
    assert first.role == "initial-development-only"


def test_basis_preconditioning_is_x_only_and_unit_invariant() -> None:
    rng = np.random.default_rng(9)
    raw_X = rng.normal(size=(32, 3))
    raw_y = 2.0 + raw_X[:, 0] - 0.4 * raw_X[:, 1] + rng.normal(size=32)
    first = DevelopmentStandardizer.fit(raw_X, raw_y)
    second = DevelopmentStandardizer.fit(11.0 + 3.5 * raw_X, -4.0 + 2.0 * raw_y)
    bank = generic_real_bank(3)
    first_X, second_X = first.transform_X(raw_X), second.transform_X(11.0 + 3.5 * raw_X)
    first_y, second_y = first.transform_y(raw_y), second.transform_y(-4.0 + 2.0 * raw_y)
    first_pre = fit_bank_preconditioner(bank, first_X)
    second_pre = fit_bank_preconditioner(bank, second_X)
    left = SequentialReferencePosterior(bank, 0.5, first_pre).fit_batch(first_X, first_y)
    right = SequentialReferencePosterior(bank, 0.5, second_pre).fit_batch(second_X, second_y)
    np.testing.assert_allclose(
        [member.probability for member in left.members],
        [member.probability for member in right.members],
        atol=2e-12,
    )


def test_preconditioned_predictive_components_use_posterior_design_coordinates() -> None:
    rng = np.random.default_rng(20260812)
    X = rng.normal(size=(36, 3))
    y = 0.2 + X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2] ** 2
    actions = rng.normal(size=(11, 3))
    bank = generic_real_bank(3)
    preconditioner = fit_bank_preconditioner(bank, X)
    engine = SequentialReferencePosterior(bank, 0.5, preconditioner)
    posterior = engine.fit_batch(X, y)
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=1.0
    )
    components = predictive_components(engine, posterior, classes, actions)

    raw_difference_observed = False
    for index, member in enumerate(posterior.members):
        mean, variance = engine.predictive_moments(member, actions)
        degrees = components.degrees_freedom[index]
        component_variance = (
            np.square(components.scales[index]) * degrees / (degrees - 2.0)
        )
        np.testing.assert_allclose(components.locations[index], mean, atol=2e-13)
        np.testing.assert_allclose(component_variance, variance, atol=2e-13)
        raw_rows = design_matrix(actions, member.structure.basis_terms)
        target_rows = engine.design_rows(actions, member.structure)
        raw_difference_observed |= not np.allclose(raw_rows, target_rows)
    assert raw_difference_observed


def test_preconditioned_epistemic_variance_uses_posterior_design_coordinates() -> None:
    rng = np.random.default_rng(73)
    X = rng.normal(size=(40, 2))
    y = -0.3 + 0.8 * X[:, 0] + 0.2 * X[:, 0] * X[:, 1]
    actions = rng.normal(size=(13, 2))
    bank = generic_real_bank(2)
    engine = SequentialReferencePosterior(
        bank, 0.25, fit_bank_preconditioner(bank, X)
    )
    posterior = engine.fit_batch(X, y)

    weights = np.asarray([member.probability for member in posterior.members])[:, None]
    means, within = [], []
    for member in posterior.members:
        rows = engine.design_rows(actions, member.structure)
        parameters = engine.conditional_parameters(member)
        noise = parameters.noise_scale / (parameters.noise_shape - 1.0)
        means.append(rows @ parameters.mean)
        within.append(
            noise * np.einsum(
                "ij,jk,ik->i", rows, parameters.covariance_factor, rows
            )
        )
    means_array = np.vstack(means)
    expected = (
        np.sum(weights * (np.vstack(within) + np.square(means_array)), axis=0)
        - np.square(np.sum(weights * means_array, axis=0))
    )
    np.testing.assert_allclose(
        posterior_epistemic_variance(engine, posterior, actions),
        np.maximum(0.0, expected),
        atol=2e-13,
    )


def test_acquisition_module_has_one_posterior_design_route() -> None:
    source = (ROOT / "hypothesis_mvp" / "pcpi" / "acquisition.py").read_text()
    assert "design_matrix(" not in source
    assert source.count("engine.design_rows(") >= 2


def test_posterior_randomized_log_loss_matches_direct_posterior_sampling() -> None:
    bank = unit_bank()
    X, y = unit_observations(20260807, 10)
    engine = SequentialReferencePosterior(bank, 0.5)
    posterior = engine.fit_batch(X, y)
    action = np.asarray([[0.35]])
    target = np.asarray([0.2])
    analytic = float(
        engine.posterior_randomized_log_loss(posterior, action, target)[0]
    )
    rng = np.random.default_rng(91)
    probabilities = np.asarray([member.probability for member in posterior.members])
    losses = []
    for member_index in rng.choice(len(posterior.members), size=40000, p=probabilities):
        member = posterior.members[int(member_index)]
        coefficients, noise_variance = engine.sample_conditional(member, rng)
        row = design_matrix(action, member.structure.basis_terms)[0]
        residual = float(target[0] - row @ coefficients)
        losses.append(
            0.5 * (
                math.log(2.0 * math.pi * noise_variance)
                + residual * residual / noise_variance
            )
        )
    assert float(np.mean(losses)) == pytest.approx(analytic, abs=0.025)


def test_generic_real_bank_uses_all_pairs_and_no_column_adjacency_rule() -> None:
    bank = generic_real_bank(4)
    structures = {item.structure_id: item for item in bank.structures}
    full = structures["full_quadratic"]
    interactions = {
        term for term in full.basis_terms
        if term.count("_x") == 1 and not term.endswith("_sq")
    }
    assert interactions == {
        "x0_x1", "x0_x2", "x0_x3", "x1_x2", "x1_x3", "x2_x3"
    }
    assert all("adjacent" not in item.structure_id for item in bank.structures)


def test_operational_classes_are_order_invariant_and_uncertainty_scaled() -> None:
    bank = unit_bank()
    engine = SequentialReferencePosterior(bank)
    X, y = unit_observations(20260807, 8)
    posterior = engine.fit_batch(X, y)
    actions = np.linspace(-1.5, 1.5, 7)
    forward = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=1.0
    )
    reverse = aggregate_operational_classes(
        engine, posterior, actions[::-1], distance_threshold=1.0
    )
    assert len(forward.classes) < len(bank.structures)
    assert forward.action_hash == reverse.action_hash
    assert forward.classes == reverse.classes


def test_complete_link_partition_prevents_epsilon_chaining() -> None:
    distances = np.asarray([
        [0.0, 0.6, 1.2],
        [0.6, 0.0, 0.6],
        [1.2, 0.6, 0.0],
    ])
    assert _complete_link_clusters(distances, ("a", "b", "c"), 1.0) == (
        (0, 1), (2,)
    )


def test_initial_frozen_entropy_gain_uses_one_random_variable() -> None:
    bank = unit_bank()
    engine = SequentialReferencePosterior(bank)
    X, y = unit_observations(20260807, 9)
    initial = engine.fit_batch(X[:5], y[:5])
    later = engine.fit_batch(X, y)
    classes = aggregate_operational_classes(
        engine, initial, np.linspace(-1.5, 1.5, 9), distance_threshold=1.0
    )
    partition = class_partition(initial, classes)
    assert realized_fixed_class_entropy_gain(partition, later) == pytest.approx(
        partition.entropy - fixed_class_entropy(partition, later)
    )


def test_frozen_partition_components_preserve_the_class_random_variable() -> None:
    bank = unit_bank()
    engine = SequentialReferencePosterior(bank)
    X, y = unit_observations(20260807, 10)
    initial = engine.fit_batch(X[:5], y[:5])
    updated = engine.fit_batch(X, y)
    actions = np.linspace(-1.5, 1.5, 9)
    classes = aggregate_operational_classes(
        engine, initial, actions, distance_threshold=1.0
    )
    frozen = class_partition(initial, classes)
    components = predictive_components_for_partition(
        engine, updated, frozen, actions
    )
    assert components.partition.stable_hash == frozen.stable_hash
    assert components.partition.member_indices == frozen.member_indices
    assert sum(components.partition.class_probabilities) == pytest.approx(1.0)


def test_uncertified_joint_eig_uses_target_free_epistemic_fallback() -> None:
    bank = unit_bank()
    engine = SequentialReferencePosterior(bank)
    X, y = unit_observations(20260807, 8)
    posterior = engine.fit_batch(X, y)
    actions = np.zeros(9)
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=0.05
    )
    frozen = class_partition(posterior, classes)
    assert len(frozen.class_ids) > 1
    result = score_acquisition_actions(
        engine, posterior, classes, actions,
        policy="pcpi_representative_safe_maximin_joint_eig", seed=3,
        eig_min_samples=32, eig_max_samples=128,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
        qbc_committee_size=16, target_partition=frozen,
        representative_observed_actions=X[:4],
    )
    assert not result.ranking_certified
    assert result.utility_mode == (
        "representative-safe-posterior-epistemic-variance-uncertified-maximin-joint-eig"
    )
    np.testing.assert_allclose(
        result.scores,
        posterior_epistemic_variance(engine, posterior, actions),
    )
    assert result.target_partition_hash == frozen.stable_hash


def test_certified_joint_rule_matches_class_plus_predictive_top_action() -> None:
    bank = unit_bank()
    engine = SequentialReferencePosterior(bank)
    X, y = unit_observations(20260807, 8)
    posterior = engine.fit_batch(X, y)
    actions = np.linspace(-1.5, 1.5, 7)
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=1.0
    )
    frozen = class_partition(posterior, classes)
    components = predictive_components_for_partition(
        engine, posterior, frozen, actions
    )
    exact = exact_class_eig(components, epsabs=1e-9, epsrel=1e-8)
    conditional = class_conditional_predictive_eig(
        engine, posterior, frozen, actions, actions
    )
    result = score_acquisition_actions(
        engine, posterior, classes, actions,
        policy="pcpi_representative_safe_maximin_joint_eig", seed=3,
        eig_min_samples=32, eig_max_samples=512,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
        qbc_committee_size=16, target_partition=frozen,
        representative_observed_actions=X[:4],
    )
    assert result.ranking_certified
    assert result.utility_mode == (
        "representative-safe-maximin-joint-eig-surrogate"
    )
    safe = result.representative_safe_mask
    safe_indices = np.flatnonzero(safe)
    expected = int(safe_indices[np.argmax((exact.scores + conditional)[safe])])
    assert np.argmax(result.scores) == expected
    np.testing.assert_allclose(
        result.conditional_predictive_eig_scores, conditional, atol=2e-13
    )
    assert result.target_partition_hash == frozen.stable_hash


def test_single_class_joint_target_retains_predictive_information() -> None:
    bank = unit_bank()
    engine = SequentialReferencePosterior(bank)
    X, y = unit_observations(20260807, 8)
    posterior = engine.fit_batch(X, y)
    actions = np.linspace(-1.0, 1.0, 7)
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=1e6
    )
    frozen = class_partition(posterior, classes)
    result = score_acquisition_actions(
        engine, posterior, classes, actions,
        policy="pcpi_representative_safe_maximin_joint_eig", seed=3,
        eig_min_samples=32, eig_max_samples=128,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
        qbc_committee_size=16, target_partition=frozen,
        representative_observed_actions=X,
    )
    assert len(frozen.class_ids) == 1
    np.testing.assert_allclose(result.class_eig_scores, 0.0, atol=2e-13)
    assert np.max(result.conditional_predictive_eig_scores) > 0.0
    assert np.max(result.scores) > 0.0


def test_epistemic_variance_excludes_only_expected_observation_noise() -> None:
    bank = unit_bank()
    engine = SequentialReferencePosterior(bank)
    X, y = unit_observations(20260807, 8)
    posterior = engine.fit_batch(X, y)
    actions = np.linspace(-1.0, 1.0, 7)
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=1.0
    )
    components = predictive_components(engine, posterior, classes, actions)
    expected_noise = 0.0
    for member in posterior.members:
        parameters = engine.conditional_parameters(member)
        expected_noise += member.probability * (
            parameters.noise_scale / (parameters.noise_shape - 1.0)
        )
        assert design_matrix(actions[:, None], member.structure.basis_terms).shape[0] == len(actions)
    np.testing.assert_allclose(
        posterior_epistemic_variance(engine, posterior, actions),
        predictive_variance(components) - expected_noise,
        rtol=1e-11,
        atol=1e-12,
    )


def test_stable_argmax_uses_original_pool_index_for_ties() -> None:
    assert select_stable_argmax(np.asarray([1.0, 2.0, 2.0]), np.asarray([9, 7, 4])) == 4


def test_normalized_aulc_uses_no_learning_baseline() -> None:
    assert normalized_area_under_learning_curve(np.asarray([2.0, 2.0, 2.0])) == pytest.approx(1.0)
    assert normalized_area_under_learning_curve(np.asarray([2.0, 1.0, 0.5])) < 1.0


@pytest.mark.parametrize("policy", ACQUISITION_POLICIES)
def test_real_policy_loop_reveals_only_selected_measurements(
    tmp_path: Path, policy: str
) -> None:
    rng = np.random.default_rng(18)
    development_X = rng.normal(size=(20, 2))
    development_y = 0.4 + development_X[:, 0] - 0.3 * development_X[:, 1]
    standardizer = DevelopmentStandardizer.fit(development_X, development_y)
    pool_X = rng.normal(size=(12, 2))
    pool_y = 0.4 + pool_X[:, 0] - 0.3 * pool_X[:, 1]
    validation_X = rng.normal(size=(10, 2))
    validation_y = 0.4 + validation_X[:, 0] - 0.3 * validation_X[:, 1]
    oracle = PoolOracle(pool_X, pool_y)
    config = {
        "acquisition_observation_budget": 3,
        "operational_class_aggregate_separation": 1.0,
        "operational_class_quantile_levels": [0.1, 0.5, 0.9],
        "eig_quadrature_min_evaluations": 32,
        "eig_quadrature_max_evaluations": 128,
        "eig_quadrature_growth_factor": 2,
        "eig_quadrature_error_safety_factor": 4.0,
        "qbc_committee_size": 16,
    }
    initial_X = standardizer.transform_X(development_X)
    design_preconditioner = fit_bank_preconditioner(
        generic_real_bank(initial_X.shape[1]), initial_X
    )
    summary, curves, queries = _run_policy(
        dataset_id="unit_real_protocol",
        seed=7,
        policy=policy,
        initial_X=initial_X,
        initial_y=standardizer.transform_y(development_y),
        validation_X=standardizer.transform_X(validation_X),
        validation_y=standardizer.transform_y(validation_y),
        fixed_domain_X=standardizer.transform_X(pool_X),
        candidate_indices=np.arange(len(pool_X)),
        pool_X=pool_X,
        pool_row_ids=np.asarray([f"pool:{index}" for index in range(len(pool_X))]),
        oracle=oracle,
        standardizer=standardizer,
        subset_commitments={
            "initial": "a" * 64,
            "validation": "b" * 64,
            "candidate": "c" * 64,
        },
        config=config,
        reporter=ProgressReporter(tmp_path / "progress.jsonl"),
        design_preconditioner=design_preconditioner,
    )
    assert summary["acquired_observations"] == 3
    assert len(curves) == 4
    assert len(queries) == 3
    assert len({row["selected_pool_index"] for row in queries}) == 3
    assert summary["initial_frozen_class_partition_hash"]
    assert "frozen_class_entropy_gain" in summary
    assert summary["candidate_evaluations"] == _expected_candidate_evaluations(12, 3)
    assert summary["candidate_subset_hash"] == "c" * 64
    assert summary["pcpi_decision_rule_valid_rate"] == 1.0
    assert sum(row["realized_query_local_class_entropy_gain"] for row in queries) == pytest.approx(
        summary["frozen_class_entropy_gain"]
    )
    if policy == "pcpi_representative_safe_maximin_joint_eig":
        assert all(
            row["acquisition_target_partition_hash"]
            == summary["initial_frozen_class_partition_hash"]
            for row in queries
        )
        assert all(
            row["utility_mode"] in {
                "representative-safe-maximin-joint-eig-surrogate",
                "representative-safe-posterior-epistemic-variance-uncertified-maximin-joint-eig",
                "representative-minimum-mmd-no-nonincreasing-action",
            }
            for row in queries
        )
        assert summary["pcpi_class_eig_used_rate"] + summary[
            "pcpi_epistemic_fallback_rate"
        ] + summary["pcpi_representative_fallback_rate"] == pytest.approx(1.0)
        assert all(
            row["predictive_target_distribution"]
            == "registered-action-domain-uniform"
            for row in queries
        )
        assert all(row["representative_guard_applied"] for row in queries)
        assert all(
            row["representative_selected_in_safe_set"]
            or row["representative_fallback_used"]
            for row in queries
        )


def test_p3b_tables_are_read_only_registry_exports(tmp_path: Path) -> None:
    for name in ("hypotheses", "diagnostics", "tables"):
        (tmp_path / name).mkdir()
    run_row = {
        "dataset_id": "uci_ccpp", "dataset_family": "uci_ccpp",
        "seed": 11, "policy": "random", "candidate_evaluations": 9,
        "normalized_aulc_validation_rmse": 1.0,
    }
    curve = [{
        "dataset_id": "uci_ccpp", "dataset_family": "uci_ccpp",
        "seed": 11, "policy": "random", "acquired_observations": 0,
        "validation_rmse": 1.0,
    }]
    query = [{
        "dataset_id": "uci_ccpp", "dataset_family": "uci_ccpp",
        "seed": 11, "policy": "random", "acquisition_round": 1,
        "selected_pool_index": 3,
    }]
    assessment = {"status": "REAL_ADVANTAGE_NOT_DEMONSTRATED"}
    summary = {
        "protocol_gate_passed": True, "failures": [],
        "effectiveness_assessment": assessment,
    }
    context = {
        "canonical_ast_hash": "a" * 64, "dataset_id": "uci_ccpp",
        "dataset_family": "uci_ccpp", "raw_data_hash": "b" * 64,
        "split_hash": "c" * 64, "role": "development",
        "code_hash": "d" * 64, "config_hash": "e" * 64,
        "engine": "unit", "provider": "none", "observation_budget": 2,
        "heldout_opened": False, "selection_used_heldout": False,
        "parent_lineage": [], "claim_boundary": P3B_CLAIM_BOUNDARY,
    }
    registry, evidence = record_p3b_evidence(
        tmp_path, [run_row], curve, query, [], {"uci_ccpp": context},
        summary, [{"dataset_id": "uci_ccpp"}], [],
        {"uci_ccpp": {"combined_source_hash": "b" * 64}}, context,
    )
    exported = export_p3b_evidence(tmp_path, registry)
    assert evidence["event_count"] == 2
    assert exported["summary"] == summary
    assert exported["run_rows"] == [run_row]
    assert json.loads((tmp_path / "diagnostics" / "failure_runs.json").read_text()) == []
    export_manifest = json.loads(
        (tmp_path / "diagnostics" / "evidence_export_manifest.json").read_text()
    )
    assert export_manifest["registry_head_hash"] == evidence["head_hash"]
    assert "tables/learning_curves.csv" in export_manifest["files"]
    registry.lock_path.unlink(missing_ok=True)


def test_gas_targets_are_averaged_within_seed_before_family_pairing() -> None:
    rows = []
    for seed in (1, 2):
        for target, offset in (("co", 0.0), ("nox", 0.2)):
            for policy, value, gain in (
                ("random", 1.0, 0.1), ("uncertainty", 0.95, 0.15),
                    (
                        "qbc", 0.9, 0.2
                    ), (
                        "pcpi_representative_safe_maximin_joint_eig", 0.8, 0.3
                    ),
            ):
                rows.append({
                    "dataset_id": f"uci_gas_turbine_{target}",
                    "dataset_family": "uci_gas_turbine", "seed": seed,
                    "policy": policy,
                    "normalized_aulc_validation_rmse": value + offset,
                    "frozen_class_entropy_gain": gain,
                    "initial_class_aggregation_fraction": 0.2,
                })
    family = _family_seed_rows(rows)
    assert len(family) == 8
    effect = next(
        item for item in _paired_effects(rows)
        if item["scope_type"] == "dataset_family"
        and item["scope_id"] == "uci_gas_turbine"
        and item["baseline"] == "random"
    )
    assert effect["paired_seeds"] == 2
    assert effect["mean_delta_normalized_aulc_rmse"] == pytest.approx(-0.2)
    assert effect["mean_delta_frozen_class_entropy_gain"] == pytest.approx(0.2)


def test_p2b_plot_error_bar_is_95_percent_t_interval() -> None:
    grouped = {128: [{"metric": value} for value in (1.0, 2.0, 3.0, 4.0)]}
    mean, interval = _mean_ci(grouped, np.asarray([128]), "metric")
    assert mean[0] == pytest.approx(2.5)
    assert interval[0] > np.std([1.0, 2.0, 3.0, 4.0], ddof=1) / 2.0


def test_p3b_cli_requires_real_data_root_and_has_no_hash_bypass() -> None:
    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert "--data-root" in options
    assert "--skip-hash" not in options
    assert "--no-verify-hash" not in options
    assert not build_parser()._option_string_actions["--source-artifact"].required
    diagnostic_options = {
        option
        for action in build_p3b3_diagnostic_parser()._actions
        for option in action.option_strings
    }
    assert "--data-root" not in diagnostic_options
    assert "--source-artifact" in diagnostic_options
    p3b6_options = {
        option
        for action in build_p3b6_diagnostic_parser()._actions
        for option in action.option_strings
    }
    assert "--data-root" not in p3b6_options
    assert "--source-artifact" in p3b6_options
    p3b7_options = {
        option
        for action in build_p3b7_diagnostic_parser()._actions
        for option in action.option_strings
    }
    assert "--data-root" not in p3b7_options
    assert "--source-artifact" in p3b7_options
