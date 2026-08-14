from __future__ import annotations

import json
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.data import PoolOracle
from hypothesis_mvp.pcpi import (
    DISCREPANCY_AWARE_POLICY,
    DISCREPANCY_PROFILE_METHOD,
    PosteriorModel,
    SequentialReferencePosterior,
    aggregate_operational_classes,
    class_conditional_predictive_eig,
    class_conditional_predictive_eig_with_discrepancy,
    class_partition,
    discrepancy_predictive_profile,
    exact_class_eig,
    estimate_class_eig,
    fit_bank_preconditioner,
    inflate_predictive_components,
    predictive_components_for_partition,
    score_discrepancy_aware_actions,
    stable_derived_seed,
)
from hypothesis_mvp.pcpi.reference import DevelopmentStandardizer, generic_real_bank
from scripts.progress import ProgressReporter
from scripts.run_pcpi_p3b_real import (
    _export_evidence,
    _load_config,
    _record_evidence,
    _run_policy,
)
from scripts.run_pcpi_p3c_real import P3C1_PROTOCOL, build_parser


ROOT = Path(__file__).resolve().parents[1]


def _case() -> tuple[
    SequentialReferencePosterior,
    object,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(20260813)
    x = rng.normal(size=(36, 2))
    y = (
        0.2
        + x[:, 0]
        - 0.4 * x[:, 1]
        + 1.5 * np.sin(2.0 * x[:, 0])
        + rng.normal(scale=0.1, size=len(x))
    )
    candidates = rng.normal(size=(11, 2))
    targets = rng.normal(size=(19, 2))
    bank = generic_real_bank(2)
    engine = SequentialReferencePosterior(bank, 0.5, fit_bank_preconditioner(bank, x))
    posterior = engine.fit_batch(x, y)
    classes = aggregate_operational_classes(engine, posterior, targets, distance_threshold=1.0)
    return engine, posterior, x, candidates, targets, np.asarray(y)


def test_discrepancy_profile_is_finite_and_permutation_equivariant() -> None:
    engine, posterior, observed, candidates, targets, _ = _case()
    profile = discrepancy_predictive_profile(
        engine, posterior, observed, candidates, targets
    )
    assert profile.method == DISCREPANCY_PROFILE_METHOD
    assert profile.candidate_variance.shape == (len(candidates),)
    assert profile.target_variance.shape == (len(targets),)
    assert np.all(np.isfinite(profile.candidate_variance))
    assert profile.residual_excess_variance > 0.0
    permuted = discrepancy_predictive_profile(
        engine,
        posterior,
        observed[::-1],
        candidates[::-1],
        targets[::-1],
    )
    np.testing.assert_allclose(
        profile.candidate_variance, permuted.candidate_variance[::-1], atol=2e-13
    )
    np.testing.assert_allclose(
        profile.target_variance, permuted.target_variance[::-1], atol=2e-13
    )
    assert profile.residual_excess_variance == permuted.residual_excess_variance
    assert profile.support_bandwidth_squared == pytest.approx(
        permuted.support_bandwidth_squared, abs=2e-15
    )


def test_zero_discrepancy_is_an_exact_noop() -> None:
    engine, posterior, observed, candidates, targets, _ = _case()
    classes = aggregate_operational_classes(engine, posterior, targets, distance_threshold=1.0)
    partition = class_partition(posterior, classes)
    components = predictive_components_for_partition(engine, posterior, partition, candidates)
    zeros_candidate = np.zeros(len(candidates))
    zeros_target = np.zeros(len(targets))
    inflated = inflate_predictive_components(components, zeros_candidate)
    np.testing.assert_allclose(inflated.locations, components.locations, atol=0.0)
    np.testing.assert_allclose(inflated.scales, components.scales, atol=2e-15)
    baseline = class_conditional_predictive_eig(
        engine, posterior, partition, candidates, targets
    )
    repaired = class_conditional_predictive_eig_with_discrepancy(
        engine, posterior, partition, candidates, targets,
        zeros_candidate, zeros_target,
    )
    np.testing.assert_allclose(repaired, baseline, atol=2e-13)


def test_discrepancy_component_is_still_checked_by_exact_reference() -> None:
    engine, posterior, observed, candidates, targets, _ = _case()
    classes = aggregate_operational_classes(engine, posterior, targets, distance_threshold=1.0)
    partition = class_partition(posterior, classes)
    components = predictive_components_for_partition(engine, posterior, partition, candidates)
    profile = discrepancy_predictive_profile(engine, posterior, observed, candidates, targets)
    inflated = inflate_predictive_components(components, profile.candidate_variance)
    exact = exact_class_eig(inflated, epsabs=1e-9, epsrel=1e-8)
    estimate = estimate_class_eig(
        inflated, 512, error_safety_factor=4.0
    )
    np.testing.assert_allclose(estimate.scores, exact.scores, atol=3e-6)
    assert np.all(np.isfinite(exact.quadrature_errors))


def test_discrepancy_aware_score_reports_its_audit_profile() -> None:
    engine, posterior, observed, candidates, targets, _ = _case()
    classes = aggregate_operational_classes(engine, posterior, targets, distance_threshold=1.0)
    partition = class_partition(posterior, classes)
    models = (PosteriorModel(engine.likelihood_power, engine, posterior),)
    result = score_discrepancy_aware_actions(
        engine,
        posterior,
        classes,
        candidates,
        seed=7,
        eig_min_samples=32,
        eig_max_samples=128,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
        qbc_committee_size=8,
        predictive_target_actions=targets,
        representative_observed_actions=observed,
        target_partition=partition,
        posterior_models=models,
    )
    assert result.policy == DISCREPANCY_AWARE_POLICY
    assert result.discrepancy_method == DISCREPANCY_PROFILE_METHOD
    assert result.discrepancy_candidate_variance is not None
    assert result.discrepancy_target_variance is not None
    assert len(result.discrepancy_candidate_variance) == len(candidates)
    assert len(result.discrepancy_target_variance) == len(targets)
    assert np.all(np.isfinite(result.scores))
    assert result.target_partition_hash == partition.stable_hash
    parameters = set(signature(score_discrepancy_aware_actions).parameters)
    assert "y" not in parameters
    assert "targets" not in parameters


def test_p3c_config_and_cli_are_frozen() -> None:
    path = ROOT / "configs" / "p3c_1_discrepancy_real_acquisition.json"
    config = _load_config(path, ROOT, P3C1_PROTOCOL)
    assert tuple(config["policies"]) == P3C1_PROTOCOL.policies
    assert config["pcpi_discrepancy_profile"] == DISCREPANCY_PROFILE_METHOD
    parser = build_parser(P3C1_PROTOCOL)
    assert parser._option_string_actions["--phase"].choices == ("P3C.1",)
    assert parser._option_string_actions["--heldout-state"].choices == ("closed",)
    assert stable_derived_seed(4, DISCREPANCY_AWARE_POLICY, 0) >= 0


def test_p3c_config_rejects_discrepancy_contract_drift(tmp_path) -> None:
    source = ROOT / "configs" / "p3c_1_discrepancy_real_acquisition.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["pcpi_discrepancy_support_rule"] = "result-tuned-distance"
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="discrepancy contract"):
        _load_config(path, tmp_path, P3C1_PROTOCOL)


def test_p3c_policy_loop_records_discrepancy_without_pool_label_access(
    tmp_path,
) -> None:
    rng = np.random.default_rng(20260814)
    development_x = rng.normal(size=(20, 2))
    development_y = (
        development_x[:, 0]
        + 1.5 * np.sin(2.0 * development_x[:, 1])
    )
    pool_x = rng.normal(size=(10, 2))
    pool_y = pool_x[:, 0] + 1.5 * np.sin(2.0 * pool_x[:, 1])
    validation_x = rng.normal(size=(8, 2))
    validation_y = validation_x[:, 0] + 1.5 * np.sin(2.0 * validation_x[:, 1])
    standardizer = DevelopmentStandardizer.fit(development_x, development_y)
    initial_x = standardizer.transform_X(development_x)
    bank = generic_real_bank(2)
    preconditioner = fit_bank_preconditioner(bank, initial_x)
    config = {
        "acquisition_observation_budget": 2,
        "operational_class_aggregate_separation": 1.0,
        "operational_class_quantile_levels": [0.1, 0.5, 0.9],
        "eig_quadrature_min_evaluations": 32,
        "eig_quadrature_max_evaluations": 128,
        "eig_quadrature_growth_factor": 2,
        "eig_quadrature_error_safety_factor": 4.0,
        "qbc_committee_size": 8,
        "likelihood_power_candidates": [0.5],
        "pcpi_discrepancy_profile": DISCREPANCY_PROFILE_METHOD,
        "pcpi_robust_utility": (
            "discrepancy-aware-maximin-joint-class-predictive-information"
        ),
    }
    summary, curves, queries = _run_policy(
        dataset_id="controlled_p3c_protocol",
        seed=9,
        policy=DISCREPANCY_AWARE_POLICY,
        initial_X=initial_x,
        initial_y=standardizer.transform_y(development_y),
        validation_X=standardizer.transform_X(validation_x),
        validation_y=standardizer.transform_y(validation_y),
        fixed_domain_X=standardizer.transform_X(pool_x),
        candidate_indices=np.arange(len(pool_x)),
        pool_X=pool_x,
        pool_row_ids=np.asarray([f"pool:{index}" for index in range(len(pool_x))]),
        oracle=PoolOracle(pool_x, pool_y),
        standardizer=standardizer,
        subset_commitments={
            "initial": "a" * 64,
            "validation": "b" * 64,
            "candidate": "c" * 64,
        },
        config=config,
        reporter=ProgressReporter(tmp_path / "progress.jsonl"),
        design_preconditioner=preconditioner,
        likelihood_power=0.5,
        protocol=P3C1_PROTOCOL,
    )
    assert len(curves) == 3
    assert len(queries) == 2
    assert summary["pcpi_discrepancy_profile"] == DISCREPANCY_PROFILE_METHOD
    assert summary["pcpi_decision_rule_valid_rate"] == 1.0
    assert all(
        row["discrepancy_method"] == DISCREPANCY_PROFILE_METHOD
        for row in queries
    )
    assert all(row["selected_discrepancy_candidate_variance"] >= 0.0 for row in queries)


def test_p3c_evidence_registry_uses_its_own_hypothesis_identity(tmp_path) -> None:
    for name in ("hypotheses", "diagnostics", "tables"):
        (tmp_path / name).mkdir()
    row = {
        "dataset_id": "uci_ccpp",
        "dataset_family": "uci_ccpp",
        "seed": 1,
        "policy": DISCREPANCY_AWARE_POLICY,
        "candidate_evaluations": 2,
    }
    curve = [{
        "dataset_id": "uci_ccpp",
        "dataset_family": "uci_ccpp",
        "seed": 1,
        "policy": DISCREPANCY_AWARE_POLICY,
        "acquired_observations": 0,
    }]
    query = [{
        "dataset_id": "uci_ccpp",
        "dataset_family": "uci_ccpp",
        "seed": 1,
        "policy": DISCREPANCY_AWARE_POLICY,
        "acquisition_round": 1,
    }]
    summary = {
        "protocol_gate_passed": True,
        "failures": [],
        "effectiveness_assessment": {
            "status": "REAL_ADVANTAGE_NOT_DEMONSTRATED"
        },
    }
    context = {
        "canonical_ast_hash": "a" * 64,
        "dataset_id": "uci_ccpp",
        "dataset_family": "uci_ccpp",
        "raw_data_hash": "b" * 64,
        "split_hash": "c" * 64,
        "role": "development",
        "code_hash": "d" * 64,
        "config_hash": "e" * 64,
        "engine": "unit",
        "provider": "none",
        "observation_budget": 2,
        "heldout_opened": False,
        "selection_used_heldout": False,
        "parent_lineage": list(P3C1_PROTOCOL.parent_lineage),
        "claim_boundary": P3C1_PROTOCOL.claim_boundary,
    }
    registry, evidence = _record_evidence(
        tmp_path,
        [row],
        curve,
        query,
        [],
        {"uci_ccpp": context},
        summary,
        [{"dataset_id": "uci_ccpp"}],
        [],
        {"uci_ccpp": {"combined_source_hash": "b" * 64}},
        context,
        P3C1_PROTOCOL,
    )
    exported = _export_evidence(tmp_path, registry, P3C1_PROTOCOL)
    assert evidence["event_count"] == 2
    assert exported["summary"] == summary
    assert len(tuple(registry.events(hypothesis_id=P3C1_PROTOCOL.hypothesis_id))) == 2
    registry.lock_path.unlink(missing_ok=True)
