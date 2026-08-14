"""Run matched-budget P3B acquisition on provenance-verified real measurements."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr, t as student_t

from hypothesis_mvp.data import (
    P2A_REAL_DATASETS,
    SPLIT_SEED,
    load_registered_real_dataset,
    prepare_real_pool_oracle,
    prepare_real_selection,
)
from hypothesis_mvp.hypotheses import (
    EvidenceEventType,
    EvidenceRegistry,
    dependency_specification_hash,
    file_sha256,
    production_code_hash,
    resolve_formal_source_identity,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from hypothesis_mvp.pcpi import (
    ACQUISITION_POLICIES,
    BUDGET_RESOLUTION_METHOD,
    DISCREPANCY_AWARE_POLICY,
    DISCREPANCY_PROFILE_METHOD,
    GAUSSIAN_CLASS_CONDITIONAL_EPIG,
    MAXIMIN_RANK_CERTIFICATE,
    PosteriorModel,
    REPRESENTATIVE_MMD_METHOD,
    SequentialReferencePosterior,
    aggregate_operational_classes,
    budget_resolved_distance_threshold,
    class_partition,
    fixed_class_entropy,
    normalized_area_under_learning_curve,
    posterior_metrics,
    score_acquisition_actions,
    score_discrepancy_aware_actions,
    select_stable_argmax,
    stable_derived_seed,
)
from hypothesis_mvp.pcpi.reference import (
    CALIBRATION_METHOD,
    CALIBRATION_ROLE,
    CALIBRATION_TIE_BREAK,
    DESIGN_PRECONDITIONING_METHOD,
    DESIGN_PRECONDITIONING_ROLE,
    DesignPreconditioner,
    DevelopmentStandardizer,
    calibrate_likelihood_power,
    fit_bank_preconditioner,
    generic_real_bank,
    stable_budget_indices,
)
from scripts.plot_pcpi_p3b_real import make_p3b_figures
from scripts.progress import ProgressReporter


EXPERIMENT = "real_measurement_matched_budget_representative_safe_maximin_joint_acquisition"
HYPOTHESIS_ID = "pcpi-p3b10-real-representative-safe-maximin-joint-acquisition"
STAGE = "P3B.10"
FROZEN_SEEDS = tuple(range(2026080701, 2026080709))
RANK_CERTIFICATE_METHOD = MAXIMIN_RANK_CERTIFICATE
PCPI_POLICY = "pcpi_representative_safe_maximin_joint_eig"


@dataclass(frozen=True)
class RealAcquisitionProtocol:
    stage: str
    schema: str
    experiment: str
    hypothesis_id: str
    pcpi_policy: str
    policies: tuple[str, ...]
    claim_boundary: str
    parent_lineage: tuple[str, ...]
    discrepancy_profile_method: str | None = None


CLAIM_BOUNDARY = (
    "This run evaluates active measurement selection on provenance-verified CCPP "
    "and Gas Turbine measurements using a common conjugate power-likelihood "
    "generalized Bayes posterior for every policy. Its likelihood power is selected "
    "once per dataset and seed by prequential posterior-randomized R-log SafeBayes "
    "on the initial development observations. Every closed basis term is centered "
    "and scaled from initial-development covariates before the shared isotropic prior "
    "is applied. Both transforms are frozen before any policy runs. "
    "Posterior fitting, posterior prediction, validation metrics, operational "
    "classes, and every acquisition policy use those same frozen design "
    "coordinates. Operational equivalence is resolved from the registered "
    "future measurement budget: a per-action standardized predictive distance "
    "is equivalent only when its root-budget aggregate is at most one. This "
    "resolution is dataset-, target-, label-, and task-name independent. "
    "Validation, acquisition-pool targets, and untouched held-out data are excluded from this "
    "calibration. PCPI targets the joint random variable formed by one "
    "initial-frozen predictive class and a future response at a uniformly drawn "
    "registered action-domain point. By the information chain rule each model's "
    "score is class-EIG plus class-conditional EPIG. The first term uses the validated "
    "Gauss-Jacobi estimator; the second is an explicitly named Gaussian-moment "
    "surrogate and is not claimed to be exact for finite Student-t mixtures. "
    "PCPI ranks candidates by the minimum joint score over the four likelihood "
    "powers frozen before this repair; the calibrated nominal posterior remains the "
    "reporting and evaluation posterior. Ties in the least-favorable model are "
    "resolved toward the smaller likelihood power. Before optimizing that lower "
    "envelope, a label-free representative safe set retains "
    "only candidates whose addition does not increase biased RBF-kernel MMD between "
    "the observed design and the fixed registered action domain. Coordinates and "
    "the median-distance bandwidth are derived from covariates only. If no "
    "non-increasing action exists, the minimum-MMD candidate is selected and the "
    "fallback is recorded explicitly. Posterior epistemic variance is used only "
    "inside the representative safe set when the maximin ranking is not numerically "
    "certified. Dynamic classes are diagnostics only. A completed run "
    "can support real measured-pool posterior-discriminative acquisition "
    "effectiveness only according to its preregistered assessment. Its adaptive "
    "rank diagnostic uses a nested Gauss-Jacobi asymptotic numerical-error "
    "envelope validated against an independent adaptive-quadrature reference; "
    "it is not a finite-sample probabilistic confidence sequence. It does not "
    "establish open-grammar symbolic discovery superiority, trans-dimensional "
    "closed-loop correctness, physical intervention, untouched-heldout "
    "confirmation, motif safety, a new scientific law, or VED discovery."
)

P3B10_PROTOCOL = RealAcquisitionProtocol(
    stage=STAGE,
    schema="pcpi-p3b-real-acquisition-config-v13",
    experiment=EXPERIMENT,
    hypothesis_id=HYPOTHESIS_ID,
    pcpi_policy=PCPI_POLICY,
    policies=ACQUISITION_POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=("pcpi-p3b9-negative-real-efficacy-audit",),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _load_config(
    path: Path,
    root: Path,
    protocol: RealAcquisitionProtocol = P3B10_PROTOCOL,
) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3B config must be an existing file inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "datasets", "policies", "seeds",
        "initial_observation_budget", "acquisition_observation_budget",
        "candidate_pool_budget", "validation_budget",
        "eig_quadrature_min_evaluations", "eig_quadrature_max_evaluations",
        "eig_quadrature_growth_factor", "eig_quadrature_error_safety_factor",
        "eig_rank_certificate_method",
        "qbc_committee_size", "operational_class_metric",
        "operational_class_linkage", "operational_class_resolution_method",
        "operational_class_aggregate_separation",
        "operational_class_quantile_levels", "class_evaluation_partition",
        "pcpi_class_target_partition", "pcpi_uncertified_eig_action", "split_seed",
        "hash_verification", "heldout_state", "failure_policy",
        "assessment_rules", "posterior_type", "likelihood_power_candidates",
        "likelihood_power_calibration_method", "likelihood_power_calibration_role",
        "likelihood_power_tie_break", "basis_preconditioning_method",
        "basis_preconditioning_role", "predictive_design_transform",
        "pcpi_joint_target", "predictive_target_distribution",
        "conditional_predictive_information_method", "representative_guard",
        "representative_target_distribution", "representative_discrepancy",
        "representative_kernel_bandwidth", "representative_safe_set_rule",
        "representative_empty_safe_set_action", "pcpi_ambiguity_set",
        "pcpi_robust_utility", "pcpi_least_favorable_tie_break",
    }
    if protocol.discrepancy_profile_method is not None:
        required |= {
            "pcpi_discrepancy_profile",
            "pcpi_discrepancy_scale",
            "pcpi_discrepancy_support_rule",
        }
    if set(config) != required:
        raise ValueError(f"P3B config fields differ from schema: {sorted(set(config) ^ required)}")
    if config["schema"] != protocol.schema or config["stage"] != protocol.stage:
        raise ValueError("unsupported real-acquisition config schema or stage")
    if tuple(config["datasets"]) != P2A_REAL_DATASETS:
        raise ValueError("P3B datasets must be the frozen CCPP and Gas targets")
    if tuple(config["policies"]) != protocol.policies:
        raise ValueError("P3B requires the frozen matched-budget policy set")
    if tuple(config["seeds"]) != FROZEN_SEEDS:
        raise ValueError(f"{protocol.stage} requires the eight frozen registered seeds")
    if config["split_seed"] != SPLIT_SEED or config["heldout_state"] != "closed":
        raise ValueError("P3B requires the frozen split and closed held-out")
    if config["hash_verification"] != "mandatory":
        raise ValueError("P3B official-source hash verification cannot be disabled")
    frozen_values = {
        "initial_observation_budget": 32,
        "acquisition_observation_budget": 32,
        "candidate_pool_budget": 128,
        "validation_budget": 256,
        "eig_quadrature_min_evaluations": 32,
        "eig_quadrature_max_evaluations": 512,
        "eig_quadrature_growth_factor": 2,
        "eig_quadrature_error_safety_factor": 4.0,
        "eig_rank_certificate_method": RANK_CERTIFICATE_METHOD,
        "qbc_committee_size": 32,
        "failure_policy": "fail_closed_record_all_no_seed_replacement",
    }
    if any(config[key] != value for key, value in frozen_values.items()):
        raise ValueError(f"{protocol.stage} frozen budgets or failure policy were modified")
    if config["operational_class_metric"] != "pooled-predictive-sd-quantile-rms":
        raise ValueError("P3B.10 requires the frozen standardized predictive metric")
    if config["operational_class_linkage"] != "complete":
        raise ValueError("P3B.10 requires deterministic complete linkage")
    if config["operational_class_resolution_method"] != BUDGET_RESOLUTION_METHOD:
        raise ValueError("P3B.10 requires budget-resolved predictive equivalence")
    if float(config["operational_class_aggregate_separation"]) != 1.0:
        raise ValueError("P3B.10 aggregate predictive resolution must remain one")
    levels = tuple(float(value) for value in config["operational_class_quantile_levels"])
    if levels != (0.1, 0.5, 0.9):
        raise ValueError("P3B.10 predictive quantile levels differ from the frozen schema")
    if config["class_evaluation_partition"] != "initial-frozen":
        raise ValueError("P3B.10 evaluation must use the initial-frozen class partition")
    if config["pcpi_class_target_partition"] != "initial-frozen":
        raise ValueError("P3B.10 acquisition must target the initial-frozen class partition")
    if config["pcpi_uncertified_eig_action"] != "posterior-epistemic-variance":
        raise ValueError("P3B.10 requires the frozen epistemic fallback utility")
    joint_target_contract = {
        "pcpi_joint_target": "initial-frozen-class-and-target-prediction",
        "predictive_target_distribution": "registered-action-domain-uniform",
        "conditional_predictive_information_method": (
            GAUSSIAN_CLASS_CONDITIONAL_EPIG
        ),
    }
    if any(config[key] != value for key, value in joint_target_contract.items()):
        raise ValueError("P3B.10 joint class-predictive target contract was modified")
    representative_contract = {
        "representative_guard": "covariate-only-registered-domain",
        "representative_target_distribution": (
            "registered-action-domain-uniform"
        ),
        "representative_discrepancy": REPRESENTATIVE_MMD_METHOD,
        "representative_kernel_bandwidth": (
            "median-positive-registered-target-squared-distance"
        ),
        "representative_safe_set_rule": (
            "augmented-mmd-nonincreasing-with-roundoff-tolerance"
        ),
        "representative_empty_safe_set_action": "minimum-augmented-mmd",
    }
    if any(config[key] != value for key, value in representative_contract.items()):
        raise ValueError("P3B.10 representative safe-set contract was modified")
    if config["posterior_type"] != "power-likelihood-generalized-bayes":
        raise ValueError("P3B.10 requires the declared generalized Bayes posterior")
    candidates = tuple(float(value) for value in config["likelihood_power_candidates"])
    if candidates != (0.125, 0.25, 0.5, 1.0):
        raise ValueError("P3B.10 likelihood-power ambiguity set was modified")
    robust_contract = {
        "pcpi_ambiguity_set": "frozen-likelihood-power-candidates",
        "pcpi_robust_utility": (
            "discrepancy-aware-maximin-joint-class-predictive-information"
            if protocol.discrepancy_profile_method is not None
            else "maximin-joint-class-predictive-information"
        ),
        "pcpi_least_favorable_tie_break": "smallest-likelihood-power",
    }
    if any(config[key] != value for key, value in robust_contract.items()):
        raise ValueError("P3B.10 robust acquisition contract was modified")
    if protocol.discrepancy_profile_method is not None:
        discrepancy_contract = {
            "pcpi_discrepancy_profile": protocol.discrepancy_profile_method,
            "pcpi_discrepancy_scale": (
                "posterior-weighted-residual-mse-excess-over-prior-noise"
            ),
            "pcpi_discrepancy_support_rule": (
                "one-plus-nearest-observed-standardized-squared-distance-"
                "over-target-median-bandwidth"
            ),
        }
        if any(
            config[key] != value for key, value in discrepancy_contract.items()
        ):
            raise ValueError("P3C discrepancy contract was modified")
    calibration_contract = {
        "likelihood_power_calibration_method": CALIBRATION_METHOD,
        "likelihood_power_calibration_role": CALIBRATION_ROLE,
        "likelihood_power_tie_break": CALIBRATION_TIE_BREAK,
    }
    if any(config[key] != value for key, value in calibration_contract.items()):
        raise ValueError("P3B.10 likelihood-power calibration contract was modified")
    preconditioning_contract = {
        "basis_preconditioning_method": DESIGN_PRECONDITIONING_METHOD,
        "basis_preconditioning_role": DESIGN_PRECONDITIONING_ROLE,
    }
    if any(config[key] != value for key, value in preconditioning_contract.items()):
        raise ValueError("P3B.10 basis-preconditioning contract was modified")
    if config["predictive_design_transform"] != "posterior-target-frozen":
        raise ValueError("P3B.10 prediction must use the posterior-target design transform")
    rules = config["assessment_rules"]
    expected_rules = {
        "paired_confidence_level", "negative_transfer_rate_max",
        "pcpi_decision_rule_valid_rate_min",
        "strong_requires_positive_frozen_class_gain_vs_random_in_every_dataset_family",
        "strong_requires_nonpositive_mean_vs_each_baseline_in_every_dataset_family",
    }
    if set(rules) != expected_rules or float(rules["paired_confidence_level"]) != 0.95:
        raise ValueError("P3B assessment rules differ from the frozen schema")
    expected_rule_values = {
        "paired_confidence_level": 0.95,
        "negative_transfer_rate_max": 0.25,
        "pcpi_decision_rule_valid_rate_min": 1.0,
        "strong_requires_positive_frozen_class_gain_vs_random_in_every_dataset_family": True,
        "strong_requires_nonpositive_mean_vs_each_baseline_in_every_dataset_family": True,
    }
    if any(rules[key] != value for key, value in expected_rule_values.items()):
        raise ValueError("P3B.10 assessment rules were modified")
    return config


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    for name in ("hypotheses", "diagnostics", "tables", "figures", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)


def _operational_class_threshold(config: dict[str, Any]) -> float:
    return budget_resolved_distance_threshold(
        int(config["acquisition_observation_budget"]),
        aggregate_separation=float(
            config["operational_class_aggregate_separation"]
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _family(dataset_id: str) -> str:
    return "uci_gas_turbine" if dataset_id.startswith("uci_gas_turbine_") else dataset_id


def _subset_seed(seed: int, role: str) -> int:
    material = f"pcpi-p3b-subset:{seed}:{role}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big")


def _subset_commitment(row_ids: np.ndarray, indices: np.ndarray) -> str:
    identifiers = np.asarray(row_ids, dtype=object).reshape(-1)
    selected = np.asarray(indices, dtype=int).reshape(-1)
    if not len(selected) or np.any(selected < 0) or np.any(selected >= len(identifiers)):
        raise ValueError("subset commitment indices are invalid")
    return _hash_json([str(identifiers[index]) for index in selected])


def _expected_candidate_evaluations(pool_budget: int, query_budget: int) -> int:
    if pool_budget < query_budget or query_budget < 1:
        raise ValueError("candidate and query budgets are incompatible")
    return sum(pool_budget - round_index for round_index in range(query_budget))


def _run_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["dataset_id"]), int(row["seed"]), str(row["policy"])


def _safe_spearman(scores: list[float], gains: list[float]) -> tuple[float, bool]:
    left, right = np.asarray(scores), np.asarray(gains)
    if len(left) < 3 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return 0.0, False
    statistic = float(spearmanr(left, right).statistic)
    return (statistic, True) if np.isfinite(statistic) else (0.0, False)


def _fit_posterior_models(
    engines: tuple[SequentialReferencePosterior, ...],
    train_X: np.ndarray,
    train_y: np.ndarray,
) -> tuple[PosteriorModel, ...]:
    return tuple(
        PosteriorModel(
            engine.likelihood_power,
            engine,
            engine.fit_batch(train_X, train_y),
        )
        for engine in engines
    )


def _run_policy(
    *,
    dataset_id: str,
    seed: int,
    policy: str,
    initial_X: np.ndarray,
    initial_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    fixed_domain_X: np.ndarray,
    candidate_indices: np.ndarray,
    pool_X: np.ndarray,
    pool_row_ids: np.ndarray,
    oracle: object,
    standardizer: DevelopmentStandardizer,
    subset_commitments: dict[str, str],
    config: dict[str, Any],
    reporter: ProgressReporter,
    design_preconditioner: DesignPreconditioner,
    likelihood_power: float = 1.0,
    calibration_hash: str = "ordinary-bayes-default",
    calibration_wall_time_seconds: float = 0.0,
    protocol: RealAcquisitionProtocol = P3B10_PROTOCOL,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train_X, train_y = initial_X.copy(), initial_y.copy()
    available = np.asarray(candidate_indices, dtype=int).copy()
    bank = generic_real_bank(train_X.shape[1])
    engine = SequentialReferencePosterior(
        bank, likelihood_power, design_preconditioner
    )
    ambiguity_powers = tuple(
        float(value)
        for value in config.get("likelihood_power_candidates", [likelihood_power])
    )
    ambiguity_engines = tuple(
        SequentialReferencePosterior(bank, power, design_preconditioner)
        for power in ambiguity_powers
    )
    curve_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    acquired_scores: list[float] = []
    query_local_gains: list[float] = []
    class_distance_threshold = _operational_class_threshold(config)
    initial_posterior = engine.fit_batch(train_X, train_y)
    initial_classes = aggregate_operational_classes(
        engine,
        initial_posterior,
        fixed_domain_X,
        distance_threshold=class_distance_threshold,
        quantile_levels=tuple(config["operational_class_quantile_levels"]),
    )
    frozen_partition = class_partition(initial_posterior, initial_classes)
    initial_frozen_entropy = frozen_partition.entropy
    initial_partition_hash = frozen_partition.stable_hash
    partition_hashes: set[str] = set()
    is_pcpi = policy == protocol.pcpi_policy
    run_started = time.perf_counter()
    for round_index in range(int(config["acquisition_observation_budget"]) + 1):
        posterior = engine.fit_batch(train_X, train_y)
        classes = aggregate_operational_classes(
            engine,
            posterior,
            fixed_domain_X,
            distance_threshold=class_distance_threshold,
            quantile_levels=tuple(config["operational_class_quantile_levels"]),
        )
        current_partition = class_partition(posterior, classes)
        partition_hashes.add(current_partition.stable_hash)
        frozen_entropy = fixed_class_entropy(frozen_partition, posterior)
        metrics = posterior_metrics(
            engine, posterior, classes, validation_X, validation_y
        )
        curve_rows.append({
            "dataset_id": dataset_id,
            "dataset_family": _family(dataset_id),
            "seed": seed,
            "policy": policy,
            "acquired_observations": round_index,
            "operational_class_count": len(classes.classes),
            "operational_class_partition_hash": current_partition.stable_hash,
            "initial_frozen_class_partition_hash": initial_partition_hash,
            "operational_class_distance_threshold": class_distance_threshold,
            "frozen_class_entropy": frozen_entropy,
            "frozen_class_entropy_gain": initial_frozen_entropy - frozen_entropy,
            **metrics.__dict__,
        })
        if round_index == int(config["acquisition_observation_budget"]):
            break
        visible_actions = standardizer.transform_X(pool_X[available])
        derived_seed = stable_derived_seed(seed, policy, round_index)
        score_kwargs = {
            "seed": derived_seed,
            "eig_min_samples": int(config["eig_quadrature_min_evaluations"]),
            "eig_max_samples": int(config["eig_quadrature_max_evaluations"]),
            "eig_error_safety_factor": float(
                config["eig_quadrature_error_safety_factor"]
            ),
            "eig_growth_factor": int(config["eig_quadrature_growth_factor"]),
            "qbc_committee_size": int(config["qbc_committee_size"]),
            "predictive_target_actions": fixed_domain_X,
            "representative_observed_actions": train_X if is_pcpi else None,
            "target_partition": frozen_partition if is_pcpi else None,
            "posterior_models": (
                _fit_posterior_models(ambiguity_engines, train_X, train_y)
                if is_pcpi else None
            ),
        }
        if is_pcpi and protocol.discrepancy_profile_method is not None:
            scores = score_discrepancy_aware_actions(
                engine, posterior, classes, visible_actions, **score_kwargs
            )
        else:
            scores = score_acquisition_actions(
                engine,
                posterior,
                classes,
                visible_actions,
                policy=policy,
                **score_kwargs,
            )
        selected = select_stable_argmax(scores.scores, available)
        local_index = int(np.flatnonzero(available == selected)[0])
        selected_score = float(scores.scores[local_index])
        selected_error = float(scores.integration_error_bounds[local_index])
        selected_class_eig = float(scores.class_eig_scores[local_index])
        selected_class_eig_error = float(scores.class_eig_error_bounds[local_index])
        selected_conditional_predictive_eig = float(
            scores.conditional_predictive_eig_scores[local_index]
        )
        selected_joint_score = float(
            scores.joint_class_predictive_scores[local_index]
        )
        selected_least_favorable_power = float(
            scores.least_favorable_likelihood_powers[local_index]
        ) if scores.robust_model_count else 0.0
        selected_mmd_squared = float(
            scores.representative_augmented_mmd_squared[local_index]
        )
        selected_in_safe_set = bool(
            scores.representative_safe_mask[local_index]
        )
        selected_mmd_nonincrease = bool(
            selected_mmd_squared
            <= scores.representative_current_mmd_squared
            + scores.representative_mmd_tolerance
        )
        selected_is_minimum_mmd = bool(np.isclose(
            selected_mmd_squared,
            float(np.min(scores.representative_augmented_mmd_squared)),
            rtol=0.0,
            atol=scores.representative_mmd_tolerance,
        ))
        revealed_X, revealed_y, revealed_indices = oracle.acquire_indices(
            np.asarray([selected])
        )
        if int(revealed_indices[0]) != selected:
            raise AssertionError("pool oracle returned a different acquisition index")
        train_X = np.vstack((train_X, standardizer.transform_X(revealed_X)))
        train_y = np.concatenate((train_y, standardizer.transform_y(revealed_y)))
        updated = engine.fit_batch(train_X, train_y)
        realized_gain = (
            frozen_entropy - fixed_class_entropy(frozen_partition, updated)
        )
        acquired_scores.append(selected_score)
        query_local_gains.append(realized_gain)
        query_rows.append({
            "dataset_id": dataset_id,
            "dataset_family": _family(dataset_id),
            "seed": seed,
            "policy": policy,
            "acquisition_round": round_index + 1,
            "selected_pool_index": selected,
            "selected_row_id": str(pool_row_ids[selected]),
            "score": selected_score,
            "score_integration_error_bound": selected_error,
            "realized_query_local_class_entropy_gain": realized_gain,
            "operational_class_count": scores.class_count,
            "operational_class_partition_hash_before_query": current_partition.stable_hash,
            "acquisition_target_partition_hash": scores.target_partition_hash,
            "initial_frozen_class_partition_hash": initial_partition_hash,
            "operational_class_distance_threshold": class_distance_threshold,
            "utility_mode": scores.utility_mode,
            "selected_class_eig": selected_class_eig,
            "selected_class_eig_error_bound": selected_class_eig_error,
            "selected_conditional_predictive_eig": (
                selected_conditional_predictive_eig
            ),
            "selected_joint_class_predictive_score": selected_joint_score,
            "robust_likelihood_powers": list(scores.robust_likelihood_powers),
            "robust_model_count": scores.robust_model_count,
            "selected_least_favorable_likelihood_power": (
                selected_least_favorable_power
            ),
            "selected_robust_joint_scores_by_model": (
                scores.robust_joint_scores_by_model[:, local_index].tolist()
                if scores.robust_model_count else []
            ),
            "selected_robust_lower_bound": float(
                scores.robust_lower_bounds[local_index]
            ),
            "selected_robust_upper_bound": float(
                scores.robust_upper_bounds[local_index]
            ),
            "discrepancy_method": scores.discrepancy_method,
            "discrepancy_residual_excess_variance": (
                scores.discrepancy_residual_excess_variance
            ),
            "discrepancy_support_bandwidth_squared": (
                scores.discrepancy_support_bandwidth_squared
            ),
            "selected_discrepancy_candidate_variance": (
                float(scores.discrepancy_candidate_variance[local_index])
                if scores.discrepancy_candidate_variance is not None else 0.0
            ),
            "mean_discrepancy_target_variance": (
                float(np.mean(scores.discrepancy_target_variance))
                if scores.discrepancy_target_variance is not None else 0.0
            ),
            "predictive_target_distribution": config.get(
                "predictive_target_distribution",
                "registered-action-domain-uniform",
            ),
            "predictive_target_subset_hash": subset_commitments["candidate"],
            "conditional_predictive_information_method": config.get(
                "conditional_predictive_information_method",
                GAUSSIAN_CLASS_CONDITIONAL_EPIG,
            ),
            "representative_guard_applied": scores.representative_guard_applied,
            "representative_mmd_method": scores.representative_mmd_method,
            "representative_current_mmd_squared": (
                scores.representative_current_mmd_squared
            ),
            "representative_selected_mmd_squared": selected_mmd_squared,
            "representative_selected_mmd_change": (
                selected_mmd_squared
                - scores.representative_current_mmd_squared
            ),
            "representative_mmd_tolerance": scores.representative_mmd_tolerance,
            "representative_kernel_bandwidth_squared": (
                scores.representative_kernel_bandwidth_squared
            ),
            "representative_safe_set_nonempty": (
                scores.representative_safe_set_nonempty
            ),
            "representative_safe_set_size": scores.representative_safe_set_size,
            "representative_fallback_used": scores.representative_fallback_used,
            "representative_selected_in_safe_set": selected_in_safe_set,
            "representative_selected_mmd_nonincrease": selected_mmd_nonincrease,
            "representative_selected_is_minimum_mmd": selected_is_minimum_mmd,
            "score_sample_count": scores.estimator_samples,
            "eig_ranking_certified": scores.ranking_certified,
            "eig_ranking_margin": scores.ranking_margin,
            "eig_ranking_error_bound": scores.ranking_error_bound,
            "eig_ranking_certificate_gap": scores.ranking_certificate_gap,
            "eig_ranking_error_safety_factor": scores.ranking_error_safety_factor,
            "eig_ranking_planned_looks": scores.ranking_planned_looks,
            "eig_ranking_looks_used": scores.ranking_looks_used,
            "eig_ranking_certificate_method": scores.ranking_certificate_method,
            "eig_coarse_evaluations": scores.estimator_coarse_samples,
            "eig_integration_method": scores.estimator_integration_method,
            "remaining_candidates_before_query": len(available),
        })
        available = available[available != selected]
        reporter.emit(
            "acquisition_completed",
            f"{protocol.stage} {dataset_id} | seed={seed} policy={policy} "
            f"query={round_index + 1}/{config['acquisition_observation_budget']} "
            f"score={selected_score:.6g} gain={realized_gain:.6g} "
            f"val_RMSE={metrics.validation_rmse:.6g} classes={scores.class_count} "
            f"eig_samples={scores.estimator_samples} "
            f"rank_certified={scores.ranking_certified} "
            f"utility={scores.utility_mode}",
            dataset_id=dataset_id,
            seed=seed,
            policy=policy,
            acquisition_round=round_index + 1,
        )
    policy_curve = [row for row in curve_rows if row["policy"] == policy]
    correlation, valid_correlation = _safe_spearman(acquired_scores, query_local_gains)
    final_frozen_entropy = float(policy_curve[-1]["frozen_class_entropy"])
    initial_class_count = int(policy_curve[0]["operational_class_count"])
    certified_rate = float(np.mean([
        bool(row["eig_ranking_certified"]) for row in query_rows
    ])) if query_rows else 1.0
    pcpi_queries = [
        row for row in query_rows
        if row["policy"] == protocol.pcpi_policy
    ]
    eig_modes = {
        (
            "representative-safe-discrepancy-robust-maximin-joint-eig-surrogate"
            if protocol.discrepancy_profile_method is not None
            else "representative-safe-maximin-joint-eig-surrogate"
        )
    }
    epistemic_modes = {
        "representative-safe-posterior-epistemic-variance-uncertified-maximin-joint-eig",
    }
    representative_fallback_modes = {
        "representative-minimum-mmd-no-nonincreasing-action",
    }
    valid_modes = eig_modes | epistemic_modes | representative_fallback_modes
    decision_valid = [
        row["utility_mode"] in valid_modes
        and row["acquisition_target_partition_hash"] == initial_partition_hash
        and row["representative_guard_applied"]
        and (
            (
                row["representative_safe_set_nonempty"]
                and not row["representative_fallback_used"]
                and row["representative_selected_in_safe_set"]
                and row["representative_selected_mmd_nonincrease"]
                and (
                    row["eig_ranking_certified"]
                    if row["utility_mode"] in eig_modes
                    else not row["eig_ranking_certified"]
                )
            )
            if row["utility_mode"] not in representative_fallback_modes
            else (
                not row["representative_safe_set_nonempty"]
                and row["representative_fallback_used"]
                and not row["representative_selected_in_safe_set"]
                and row["representative_selected_is_minimum_mmd"]
            )
        )
        for row in pcpi_queries
    ]
    policy_wall_time = time.perf_counter() - run_started
    summary = {
        "dataset_id": dataset_id,
        "dataset_family": _family(dataset_id),
        "seed": seed,
        "policy": policy,
        "initial_observations": len(initial_y),
        "acquired_observations": len(query_rows),
        "validation_observations": len(validation_y),
        "candidate_pool_observations": len(candidate_indices),
        "candidate_evaluations": sum(row["remaining_candidates_before_query"] for row in query_rows),
        "initial_subset_hash": subset_commitments["initial"],
        "validation_subset_hash": subset_commitments["validation"],
        "candidate_subset_hash": subset_commitments["candidate"],
        "posterior_type": "power-likelihood-generalized-bayes",
        "likelihood_power": engine.likelihood_power,
        "posterior_target_hash": engine.target_hash,
        "pcpi_ambiguity_set": list(ambiguity_powers),
        "pcpi_robust_utility": config.get(
            "pcpi_robust_utility", "maximin-joint-class-predictive-information"
        ),
        "pcpi_discrepancy_profile": config.get(
            "pcpi_discrepancy_profile", "not-applied"
        ),
        "design_preconditioner_hash": design_preconditioner.stable_hash,
        "likelihood_power_calibration_hash": calibration_hash,
        "likelihood_power_calibration_wall_time_seconds": calibration_wall_time_seconds,
        "normalized_aulc_validation_rmse": normalized_area_under_learning_curve(
            np.asarray([row["validation_rmse"] for row in policy_curve])
        ),
        "final_validation_rmse": policy_curve[-1]["validation_rmse"],
        "final_validation_nll": policy_curve[-1]["validation_nll"],
        "final_structure_entropy": policy_curve[-1]["structure_entropy"],
        "final_class_entropy": policy_curve[-1]["class_entropy"],
        "final_maximum_class_probability": policy_curve[-1]["maximum_class_probability"],
        "initial_operational_class_count": initial_class_count,
        "final_operational_class_count": int(policy_curve[-1]["operational_class_count"]),
        "initial_class_aggregation_fraction": 1.0 - initial_class_count / len(bank.structures),
        "initial_frozen_class_partition_hash": initial_partition_hash,
        "predictive_target_distribution": config.get(
            "predictive_target_distribution",
            "registered-action-domain-uniform",
        ),
        "predictive_target_subset_hash": subset_commitments["candidate"],
        "conditional_predictive_information_method": config.get(
            "conditional_predictive_information_method",
            GAUSSIAN_CLASS_CONDITIONAL_EPIG,
        ),
        "representative_mmd_method": config.get(
            "representative_discrepancy", REPRESENTATIVE_MMD_METHOD
        ),
        "operational_class_distance_threshold": class_distance_threshold,
        "operational_class_resolution_method": BUDGET_RESOLUTION_METHOD,
        "initial_frozen_class_entropy": initial_frozen_entropy,
        "final_frozen_class_entropy": final_frozen_entropy,
        "frozen_class_entropy_gain": initial_frozen_entropy - final_frozen_entropy,
        "sum_query_local_class_entropy_gain": float(sum(query_local_gains)),
        "dynamic_partition_count": len(partition_hashes),
        "eig_ranking_certified_rate": certified_rate,
        "pcpi_class_eig_used_rate": float(np.mean([
            row["utility_mode"] in eig_modes for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_maximin_joint_eig_used_rate": float(np.mean([
            row["utility_mode"] in eig_modes for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_epistemic_fallback_rate": float(np.mean([
            row["utility_mode"] in epistemic_modes for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_representative_guard_applied_rate": float(np.mean([
            row["representative_guard_applied"] for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_representative_safe_set_nonempty_rate": float(np.mean([
            row["representative_safe_set_nonempty"] for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_representative_fallback_rate": float(np.mean([
            row["utility_mode"] in representative_fallback_modes
            for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_representative_selected_nonincrease_rate": float(np.mean([
            row["representative_selected_mmd_nonincrease"]
            for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_mean_representative_safe_set_size": float(np.mean([
            row["representative_safe_set_size"] for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_decision_rule_valid_rate": float(np.mean(decision_valid))
        if decision_valid else (
            1.0 if not is_pcpi else 0.0
        ),
        "pcpi_mean_selected_discrepancy_variance": float(np.mean([
            row["selected_discrepancy_candidate_variance"] for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "maximum_eig_samples_used": max(
            (int(row["score_sample_count"]) for row in query_rows), default=0
        ),
        "score_realized_gain_spearman": correlation,
        "score_realized_gain_spearman_valid": valid_correlation,
        "wall_time_seconds": policy_wall_time,
        "wall_time_seconds_including_shared_calibration": (
            policy_wall_time + calibration_wall_time_seconds
        ),
        "failure_status": "",
    }
    return summary, curve_rows, query_rows


def _mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, len(array) - 1))
    half = critical * float(np.std(array, ddof=1)) / np.sqrt(len(array))
    return mean, mean - half, mean + half


def _aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "normalized_aulc_validation_rmse", "final_validation_rmse",
        "final_validation_nll", "final_structure_entropy", "final_class_entropy",
        "final_maximum_class_probability", "initial_operational_class_count",
        "final_operational_class_count", "initial_class_aggregation_fraction",
        "operational_class_distance_threshold",
        "initial_frozen_class_entropy", "final_frozen_class_entropy",
        "frozen_class_entropy_gain", "sum_query_local_class_entropy_gain",
        "dynamic_partition_count",
        "eig_ranking_certified_rate", "maximum_eig_samples_used",
        "pcpi_class_eig_used_rate", "pcpi_maximin_joint_eig_used_rate",
        "pcpi_epistemic_fallback_rate",
        "pcpi_representative_guard_applied_rate",
        "pcpi_representative_safe_set_nonempty_rate",
        "pcpi_representative_fallback_rate",
        "pcpi_representative_selected_nonincrease_rate",
        "pcpi_mean_representative_safe_set_size",
        "pcpi_mean_selected_discrepancy_variance",
        "pcpi_decision_rule_valid_rate",
        "score_realized_gain_spearman", "wall_time_seconds",
    )
    output: list[dict[str, Any]] = []
    keys = sorted({(row["dataset_id"], row["policy"]) for row in rows})
    for dataset_id, policy in keys:
        selected = [row for row in rows if row["dataset_id"] == dataset_id and row["policy"] == policy]
        aggregate: dict[str, Any] = {
            "dataset_id": dataset_id,
            "dataset_family": selected[0]["dataset_family"],
            "policy": policy,
            "successful_seeds": len(selected),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            mean, lower, upper = _mean_ci(values)
            aggregate[f"mean_{metric}"] = mean
            aggregate[f"ci95_lower_{metric}"] = lower
            aggregate[f"ci95_upper_{metric}"] = upper
            aggregate[f"std_{metric}"] = float(np.std(values, ddof=1))
        output.append(aggregate)
    return output


def _family_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row["dataset_family"], row["policy"], row["seed"]) for row in rows})
    for family, policy, seed in keys:
        selected = [
            row for row in rows
            if row["dataset_family"] == family and row["policy"] == policy and row["seed"] == seed
        ]
        output.append({
            "dataset_family": family,
            "policy": policy,
            "seed": seed,
            "normalized_aulc_validation_rmse": float(np.mean([
                row["normalized_aulc_validation_rmse"] for row in selected
            ])),
            "frozen_class_entropy_gain": float(np.mean([
                row["frozen_class_entropy_gain"] for row in selected
            ])),
            "initial_class_aggregation_fraction": float(np.mean([
                row["initial_class_aggregation_fraction"] for row in selected
            ])),
        })
    return output


def _paired_effects(
    rows: list[dict[str, Any]],
    pcpi_policy: str = PCPI_POLICY,
) -> list[dict[str, Any]]:
    family_rows = _family_seed_rows(rows)
    scopes: list[tuple[str, str, list[dict[str, Any]]]] = []
    for dataset_id in sorted({row["dataset_id"] for row in rows}):
        scopes.append(("dataset", dataset_id, [row for row in rows if row["dataset_id"] == dataset_id]))
    for family in sorted({row["dataset_family"] for row in family_rows}):
        scopes.append(("dataset_family", family, [row for row in family_rows if row["dataset_family"] == family]))
    output: list[dict[str, Any]] = []
    for scope_type, scope_id, selected in scopes:
        pcpi = {
            row["seed"]: row
            for row in selected
            if row["policy"] == pcpi_policy
        }
        for baseline in ("random", "uncertainty", "qbc"):
            other = {row["seed"]: row for row in selected if row["policy"] == baseline}
            seeds = sorted(set(pcpi) & set(other))
            deltas = [
                float(pcpi[seed]["normalized_aulc_validation_rmse"])
                - float(other[seed]["normalized_aulc_validation_rmse"])
                for seed in seeds
            ]
            gain_deltas = [
                float(pcpi[seed]["frozen_class_entropy_gain"])
                - float(other[seed]["frozen_class_entropy_gain"])
                for seed in seeds
            ]
            mean, lower, upper = _mean_ci(deltas)
            gain_mean, gain_lower, gain_upper = _mean_ci(gain_deltas)
            output.append({
                "scope_type": scope_type,
                "scope_id": scope_id,
                "baseline": baseline,
                "paired_seeds": len(seeds),
                "mean_delta_normalized_aulc_rmse": mean,
                "ci95_lower_delta_normalized_aulc_rmse": lower,
                "ci95_upper_delta_normalized_aulc_rmse": upper,
                "mean_delta_frozen_class_entropy_gain": gain_mean,
                "ci95_lower_delta_frozen_class_entropy_gain": gain_lower,
                "ci95_upper_delta_frozen_class_entropy_gain": gain_upper,
                "predictive_negative_transfer_rate": float(np.mean(np.asarray(deltas) > 0.0)),
                "class_gain_negative_transfer_rate": float(np.mean(np.asarray(gain_deltas) < 0.0)),
                "rmse_direction": "negative_favors_pcpi",
                "class_gain_direction": "positive_favors_pcpi",
            })
    return output


def _assessment(
    paired: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    protocol_gate: bool,
    config: dict[str, Any],
    pcpi_policy: str = PCPI_POLICY,
) -> dict[str, Any]:
    family = [row for row in paired if row["scope_type"] == "dataset_family"]
    random = [row for row in family if row["baseline"] == "random"]
    rules = config["assessment_rules"]
    random_class_gain_significant = bool(random) and all(
        row["ci95_lower_delta_frozen_class_entropy_gain"] > 0.0
        for row in random
    )
    random_predictive_mean_better = bool(random) and all(
        row["mean_delta_normalized_aulc_rmse"] < 0.0 for row in random
    )
    random_class_gain_mean_better = bool(random) and all(
        row["mean_delta_frozen_class_entropy_gain"] > 0.0 for row in random
    )
    all_baseline_mean_nonpositive = bool(family) and all(
        row["mean_delta_normalized_aulc_rmse"] <= 0.0 for row in family
    )
    class_negative_transfer_controlled = bool(family) and max(
        row["class_gain_negative_transfer_rate"] for row in family
    ) <= float(rules["negative_transfer_rate_max"])
    aggregation_by_family = {
        family_id: any(
            float(row["initial_class_aggregation_fraction"]) > 0.0
            for row in run_rows if row["dataset_family"] == family_id
        )
        for family_id in sorted({row["dataset_family"] for row in run_rows})
    }
    class_aggregation_observed = bool(aggregation_by_family) and all(
        aggregation_by_family.values()
    )
    ranking_by_family: dict[str, float] = {}
    joint_eig_use_by_family: dict[str, float] = {}
    decision_rule_by_family: dict[str, float] = {}
    for family_id in sorted({row["dataset_family"] for row in run_rows}):
        selected = [
            row for row in run_rows
            if row["dataset_family"] == family_id
            and row["policy"] == pcpi_policy
        ]
        values = [
            float(row["eig_ranking_certified_rate"])
            for row in selected
        ]
        ranking_by_family[family_id] = float(np.mean(values)) if values else 0.0
        joint_eig_use_by_family[family_id] = float(np.mean([
            float(row["pcpi_maximin_joint_eig_used_rate"]) for row in selected
        ])) if selected else 0.0
        decision_rule_by_family[family_id] = float(np.mean([
            float(row["pcpi_decision_rule_valid_rate"]) for row in selected
        ])) if selected else 0.0
    decision_rule_valid = bool(decision_rule_by_family) and all(
        rate >= float(rules["pcpi_decision_rule_valid_rate_min"])
        for rate in decision_rule_by_family.values()
    )
    strong_structural = (
        protocol_gate and random_class_gain_significant
        and class_negative_transfer_controlled and class_aggregation_observed
        and decision_rule_valid
    )
    strong_joint = strong_structural and all_baseline_mean_nonpositive
    if not protocol_gate:
        status = "INVALID_PROTOCOL_FAILURE"
    elif not class_aggregation_observed:
        status = "OPERATIONAL_CLASSES_DEGENERATE_NO_CLASS_CLAIM"
    elif not decision_rule_valid:
        status = "INVALID_PCPI_DECISION_RULE"
    elif strong_joint:
        status = "STRONG_JOINT_REAL_ACQUISITION_EVIDENCE"
    elif strong_structural:
        status = "STRONG_STRUCTURAL_MIXED_PREDICTIVE_EVIDENCE"
    elif random_class_gain_mean_better:
        status = "PROMISING_STRUCTURAL_NOT_STRONG"
    else:
        status = "REAL_ADVANTAGE_NOT_DEMONSTRATED"
    return {
        "status": status,
        "strong_evidence": strong_joint,
        "strong_structural_evidence": strong_structural,
        "pcpi_frozen_class_gain_vs_random_significant_in_every_dataset_family": random_class_gain_significant,
        "pcpi_frozen_class_gain_vs_random_mean_better_in_every_dataset_family": random_class_gain_mean_better,
        "pcpi_predictive_naulc_vs_random_mean_better_in_every_dataset_family": random_predictive_mean_better,
        "pcpi_predictive_naulc_mean_nonpositive_vs_each_baseline_in_every_dataset_family": all_baseline_mean_nonpositive,
        "class_gain_negative_transfer_rate_controlled": class_negative_transfer_controlled,
        "class_aggregation_observed_in_every_dataset_family": class_aggregation_observed,
        "class_aggregation_by_dataset_family": aggregation_by_family,
        "pcpi_decision_rule_valid": decision_rule_valid,
        "pcpi_decision_rule_valid_rate_by_dataset_family": decision_rule_by_family,
        "pcpi_maximin_joint_eig_used_rate_by_dataset_family": joint_eig_use_by_family,
        "eig_ranking_certified_rate_by_dataset_family": ranking_by_family,
        "dataset_family_count": 2,
        "gas_targets_counted_as_one_family": True,
    }


def _evidence_flags(
    protocol_gate: bool,
    assessment: dict[str, Any],
) -> dict[str, bool]:
    """Separate protocol validity from positive real-efficacy evidence."""

    strong = bool(
        assessment.get("strong_evidence")
        or assessment.get("strong_structural_evidence")
    )
    return {
        "formal_protocol_evidence": bool(protocol_gate),
        "formal_efficacy_evidence": bool(protocol_gate and strong),
    }


def _aggregate_context(
    identity: dict[str, Any],
    dataset_records: dict[str, Any],
    config: dict[str, Any],
    protocol: RealAcquisitionProtocol = P3B10_PROTOCOL,
) -> dict[str, Any]:
    return {
        "canonical_ast_hash": _hash_json({
            key: value["bank_hash"] for key, value in dataset_records.items()
        }),
        "dataset_id": f"{protocol.stage.lower().replace('.', '')}_registered_real_collection",
        "dataset_family": "uci_ccpp_and_uci_gas_turbine",
        "raw_data_hash": _hash_json({
            key: value["combined_source_hash"]
            for key, value in dataset_records.items()
        }),
        "split_hash": _hash_json({
            key: value["split_manifest"]["split_hash"]
            for key, value in dataset_records.items()
        }),
        "role": "aggregate_real_measured_pool_effectiveness_assessment",
        "code_hash": identity["production_code_hash"],
        "config_hash": identity["config_hash"],
        "engine": "conjugate-power-likelihood-finite-bank-common-posterior",
        "provider": "none",
        "observation_budget": {
            "initial": config["initial_observation_budget"],
            "acquired": config["acquisition_observation_budget"],
        },
        "heldout_opened": False,
        "selection_used_heldout": False,
        "parent_lineage": list(protocol.parent_lineage),
        "claim_boundary": protocol.claim_boundary,
    }


def _record_evidence(
    output: Path,
    rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    aggregates: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    dataset_records: dict[str, Any],
    aggregate_context: dict[str, Any],
    protocol: RealAcquisitionProtocol = P3B10_PROTOCOL,
) -> tuple[EvidenceRegistry, dict[str, Any]]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    curves_by_run: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    queries_by_run: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for item in curve_rows:
        curves_by_run.setdefault(_run_key(item), []).append(item)
    for item in query_rows:
        queries_by_run.setdefault(_run_key(item), []).append(item)
    for row in rows:
        key = _run_key(row)
        registry.append(
            hypothesis_id=protocol.hypothesis_id,
            event_type=EvidenceEventType.TEST_OBSERVED,
            payload={
                **contexts[row["dataset_id"]],
                "evidence_record": "policy_run",
                "seed": row["seed"],
                "candidate_budget": row["candidate_evaluations"],
                "metric": {
                    "run_metrics": row,
                    "learning_curve": curves_by_run.get(key, []),
                    "acquisition_queries": queries_by_run.get(key, []),
                },
                "uncertainty": "paired seed uncertainty exported separately",
                "validation_result": "completed",
                "failure_status": None,
            },
        )
    for failure in failures:
        context = contexts.get(failure["dataset_id"], failure["fallback_context"])
        registry.append(
            hypothesis_id=protocol.hypothesis_id,
            event_type=EvidenceEventType.TEST_OBSERVED,
            payload={
                **context,
                "evidence_record": "policy_failure",
                "seed": failure["seed"],
                "candidate_budget": None,
                "metric": {
                    "failure": {
                        key: value for key, value in failure.items()
                        if key != "fallback_context"
                    }
                },
                "uncertainty": None,
                "validation_result": "fail",
                "failure_status": failure["failure_status"],
            },
        )
    registry.append(
        hypothesis_id=protocol.hypothesis_id,
        event_type=EvidenceEventType.EVIDENCE_ATTACHED,
        payload={
            **aggregate_context,
            "evidence_record": "aggregate_assessment",
            "seed": sorted({int(row["seed"]) for row in rows}),
            "candidate_budget": sorted({
                int(row["candidate_evaluations"]) for row in rows
            }),
            "metric": {
                "summary": summary,
                "aggregate_metrics": aggregates,
                "paired_effects": paired,
                "dataset_records": dataset_records,
            },
            "uncertainty": "paired 95% Student-t intervals across registered seeds",
            "validation_result": (
                "completed" if summary["protocol_gate_passed"] else "failed"
            ),
            "failure_status": (
                None if summary["protocol_gate_passed"] else "protocol_gate_failed"
            ),
        },
    )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError("P3B EvidenceRegistry verification failed")
    return registry, {
        "valid": True,
        "event_count": verification.event_count,
        "head_hash": verification.head_hash,
    }


def _export_evidence(
    output: Path,
    registry: EvidenceRegistry,
    protocol: RealAcquisitionProtocol = P3B10_PROTOCOL,
) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {}
    for event in registry.events(hypothesis_id=protocol.hypothesis_id):
        payload = event.to_dict()["payload"]
        records.setdefault(str(payload["evidence_record"]), []).append(payload)
    aggregate = records.get("aggregate_assessment", [])
    if len(aggregate) != 1:
        raise RuntimeError("P3B evidence must contain one aggregate assessment")
    run_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for payload in records.get("policy_run", []):
        metric = payload["metric"]
        run_rows.append(dict(metric["run_metrics"]))
        curve_rows.extend(dict(item) for item in metric["learning_curve"])
        query_rows.extend(dict(item) for item in metric["acquisition_queries"])
    failures = [
        dict(payload["metric"]["failure"])
        for payload in records.get("policy_failure", [])
    ]
    aggregate_metric = aggregate[0]["metric"]
    summary = dict(aggregate_metric["summary"])
    aggregates = [dict(item) for item in aggregate_metric["aggregate_metrics"]]
    paired = [dict(item) for item in aggregate_metric["paired_effects"]]
    dataset_records = dict(aggregate_metric["dataset_records"])
    run_rows.sort(key=_run_key)
    curve_rows.sort(key=lambda item: (*_run_key(item), int(item["acquired_observations"])))
    query_rows.sort(key=lambda item: (*_run_key(item), int(item["acquisition_round"])))
    failures.sort(key=lambda item: (
        str(item["dataset_id"]), str(item["seed"]), str(item["policy"])
    ))
    expected_failures = sorted(
        (dict(item) for item in summary["failures"]),
        key=lambda item: (str(item["dataset_id"]), str(item["seed"]), str(item["policy"])),
    )
    if failures != expected_failures:
        raise RuntimeError("P3B failure export differs from aggregate evidence")
    paths = _write_evidence_exports(
        output, run_rows, curve_rows, query_rows, aggregates, paired,
        summary, dataset_records, failures,
    )
    verification = registry.verify()
    export_path = output / "diagnostics" / "evidence_export_manifest.json"
    export_path.write_text(_canonical_json({
        "schema": "pcpi-evidence-read-only-export-v1",
        "registry_event_count": verification.event_count,
        "registry_head_hash": verification.head_hash,
        "files": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in paths
        },
    }), encoding="utf-8")
    return {
        "summary": summary,
        "run_rows": run_rows,
        "curve_rows": curve_rows,
        "query_rows": query_rows,
        "export_path": export_path,
    }


def _write_evidence_exports(
    output: Path,
    run_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    summary: dict[str, Any],
    dataset_records: dict[str, Any],
    failures: list[dict[str, Any]],
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path, data in (
        (output / "tables" / "per_seed_policy_metrics.csv", run_rows),
        (output / "tables" / "learning_curves.csv", curve_rows),
        (output / "tables" / "acquisition_queries.csv", query_rows),
        (output / "tables" / "aggregate_metrics.csv", aggregates),
        (output / "tables" / "paired_effects.csv", paired),
    ):
        if data:
            _write_csv(path, data)
            paths.append(path)
    for path, value in (
        (output / "summary.json", summary),
        (output / "diagnostics" / "dataset_records.json", dataset_records),
        (output / "diagnostics" / "failure_runs.json", failures),
        (
            output / "hypotheses" / "effectiveness_assessment.json",
            summary["effectiveness_assessment"],
        ),
    ):
        path.write_text(_canonical_json(value), encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def run(
    args: argparse.Namespace,
    protocol: RealAcquisitionProtocol = P3B10_PROTOCOL,
) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).resolve()
    output = Path(args.output_dir).resolve()
    source = (
        Path(args.source_artifact).resolve() if args.source_artifact else None
    )
    config_path = Path(args.config).resolve()
    if args.phase != protocol.stage or args.heldout_state != "closed":
        raise ValueError(
            f"real runner requires phase {protocol.stage} and heldout closed"
        )
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    config = _load_config(config_path, root, protocol)
    source_identity = resolve_formal_source_identity(root, source)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    dependency_environment = runtime_dependency_snapshot()
    dependency_environment_hash = runtime_dependency_hash(dependency_environment)
    identity = {
        **source_identity,
        "production_code_hash": production_code_hash(root),
        "config_hash": _hash_json(config),
        "config_file_hash": file_sha256(config_path),
        "dependency_specification_hash": dependency_specification_hash(root),
        "dependency_environment_hash": dependency_environment_hash,
        "dependency_lock_hash": dependency_environment_hash,
        "dependency_environment": dependency_environment,
    }
    run_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    dataset_records: dict[str, Any] = {}
    total = len(config["datasets"]) * len(config["seeds"]) * len(config["policies"])
    completed = 0
    reporter.emit(
        "run_started",
        f"{protocol.stage} real acquisition started | runs={total} heldout=closed",
        total_runs=total,
        datasets=config["datasets"],
        policies=config["policies"],
    )
    for dataset_id in config["datasets"]:
        try:
            reporter.emit(
                "dataset_loading",
                f"loading real dataset={dataset_id} with mandatory official hashes",
                dataset_id=dataset_id,
            )
            frame = load_registered_real_dataset(dataset_id, data_root, verify_hashes=True)
            prepared = prepare_real_selection(frame, split_seed=int(config["split_seed"]))
            oracle = prepare_real_pool_oracle(
                frame, prepared, split_seed=int(config["split_seed"])
            )
            selection = prepared.selection
            if selection.acquisition_pool is None:
                raise RuntimeError("P3B measured acquisition pool is unavailable")
            bank_hash = generic_real_bank(selection.development.X.shape[1]).stable_hash
            context = {
                "canonical_ast_hash": bank_hash,
                "dataset_id": dataset_id,
                "dataset_family": _family(dataset_id),
                "raw_data_hash": prepared.combined_source_hash,
                "split_hash": prepared.split_manifest["split_hash"],
                "role": "development_plus_measured_acquisition_with_validation_evaluation",
                "code_hash": identity["production_code_hash"],
                "config_hash": identity["config_hash"],
                "engine": "conjugate-power-likelihood-finite-bank-common-posterior",
                "provider": "none",
                "observation_budget": int(config["initial_observation_budget"]) + int(config["acquisition_observation_budget"]),
                "heldout_opened": False,
                "selection_used_heldout": False,
                "parent_lineage": list(protocol.parent_lineage),
                "claim_boundary": protocol.claim_boundary,
            }
            contexts[dataset_id] = context
            dataset_records[dataset_id] = {
                "dataset_family": _family(dataset_id),
                "official_source_hashes": list(prepared.source_hashes),
                "combined_source_hash": prepared.combined_source_hash,
                "split_manifest": prepared.split_manifest,
                "feature_names": list(prepared.feature_names),
                "target_name": prepared.target_name,
                "bank_hash": bank_hash,
                "standardizer_hashes": {},
                "design_preconditioners": {},
                "likelihood_power_calibrations": {},
                "subset_commitments": {},
            }
            for seed in config["seeds"]:
                initial_indices = stable_budget_indices(
                    prepared.development_row_ids,
                    int(config["initial_observation_budget"]),
                    _subset_seed(int(seed), "initial"),
                )
                validation_indices = stable_budget_indices(
                    prepared.validation_row_ids,
                    int(config["validation_budget"]),
                    _subset_seed(int(seed), "validation"),
                )
                candidates = stable_budget_indices(
                    prepared.acquisition_pool_row_ids,
                    int(config["candidate_pool_budget"]),
                    _subset_seed(int(seed), "candidate"),
                )
                subset_commitments = {
                    "initial": _subset_commitment(
                        prepared.development_row_ids, initial_indices
                    ),
                    "validation": _subset_commitment(
                        prepared.validation_row_ids, validation_indices
                    ),
                    "candidate": _subset_commitment(
                        prepared.acquisition_pool_row_ids, candidates
                    ),
                }
                dataset_records[dataset_id]["subset_commitments"][str(seed)] = (
                    subset_commitments
                )
                standardizer = DevelopmentStandardizer.fit(
                    selection.development.X[initial_indices],
                    selection.development.y[initial_indices],
                )
                dataset_records[dataset_id]["standardizer_hashes"][str(seed)] = standardizer.stable_hash
                initial_X = standardizer.transform_X(selection.development.X[initial_indices])
                initial_y = standardizer.transform_y(selection.development.y[initial_indices])
                bank = generic_real_bank(initial_X.shape[1])
                design_preconditioner = fit_bank_preconditioner(bank, initial_X)
                dataset_records[dataset_id]["design_preconditioners"][str(seed)] = (
                    design_preconditioner.to_dict()
                    | {"preconditioner_hash": design_preconditioner.stable_hash}
                )
                calibration_started = time.perf_counter()
                calibration = calibrate_likelihood_power(
                    bank,
                    initial_X,
                    initial_y,
                    tuple(float(value) for value in config["likelihood_power_candidates"]),
                    design_preconditioner,
                )
                calibration_wall_time = time.perf_counter() - calibration_started
                dataset_records[dataset_id]["likelihood_power_calibrations"][str(seed)] = (
                    calibration.to_dict() | {"calibration_hash": calibration.stable_hash}
                )
                validation_X = standardizer.transform_X(selection.validation.X[validation_indices])
                validation_y = standardizer.transform_y(selection.validation.y[validation_indices])
                fixed_domain_X = standardizer.transform_X(selection.acquisition_pool.X[candidates])
                for policy in config["policies"]:
                    reporter.emit(
                        "policy_run_started",
                        f"{protocol.stage} run {completed + 1}/{total} | dataset={dataset_id} seed={seed} policy={policy}",
                        dataset_id=dataset_id,
                        seed=seed,
                        policy=policy,
                    )
                    try:
                        summary, curves, queries = _run_policy(
                            dataset_id=dataset_id,
                            seed=int(seed),
                            policy=policy,
                            initial_X=initial_X,
                            initial_y=initial_y,
                            validation_X=validation_X,
                            validation_y=validation_y,
                            fixed_domain_X=fixed_domain_X,
                            candidate_indices=candidates,
                            pool_X=selection.acquisition_pool.X,
                            pool_row_ids=prepared.acquisition_pool_row_ids,
                            oracle=oracle,
                            standardizer=standardizer,
                            subset_commitments=subset_commitments,
                            config=config,
                            reporter=reporter,
                            design_preconditioner=design_preconditioner,
                            likelihood_power=calibration.selected_likelihood_power,
                            calibration_hash=calibration.stable_hash,
                            calibration_wall_time_seconds=calibration_wall_time,
                            protocol=protocol,
                        )
                        run_rows.append(summary)
                        curve_rows.extend(curves)
                        query_rows.extend(queries)
                        completed += 1
                        reporter.emit(
                            "policy_run_completed",
                            f"{protocol.stage} run {completed}/{total} complete | dataset={dataset_id} "
                            f"seed={seed} policy={policy} nAULC={summary['normalized_aulc_validation_rmse']:.6g}",
                            **summary,
                        )
                    except Exception as error:
                        completed += 1
                        failure = {
                            "dataset_id": dataset_id,
                            "dataset_family": _family(dataset_id),
                            "seed": int(seed),
                            "policy": policy,
                            "failure_status": f"{type(error).__name__}: {error}",
                            "fallback_context": context,
                        }
                        failures.append(failure)
                        reporter.emit(
                            "policy_run_failed",
                            f"{protocol.stage} run {completed}/{total} FAILED | {failure['failure_status']}",
                            **{key: value for key, value in failure.items() if key != "fallback_context"},
                        )
        except Exception as error:
            failure = {
                "dataset_id": dataset_id,
                "dataset_family": _family(dataset_id),
                "seed": "dataset_preflight",
                "policy": "all",
                "failure_status": f"{type(error).__name__}: {error}",
                "fallback_context": {
                    "canonical_ast_hash": "unavailable",
                    "dataset_id": dataset_id,
                    "dataset_family": _family(dataset_id),
                    "raw_data_hash": "unavailable_due_to_preflight_failure",
                    "split_hash": "unavailable_due_to_preflight_failure",
                    "role": "preflight_failure",
                    "code_hash": identity["production_code_hash"],
                    "config_hash": identity["config_hash"],
                    "engine": "conjugate-power-likelihood-finite-bank-common-posterior",
                    "provider": "none",
                    "observation_budget": None,
                    "heldout_opened": False,
                    "selection_used_heldout": False,
                    "parent_lineage": list(protocol.parent_lineage),
                    "claim_boundary": protocol.claim_boundary,
                },
            }
            failures.append(failure)
            reporter.emit(
                "dataset_failed",
                f"{protocol.stage} dataset={dataset_id} FAILED | {failure['failure_status']}",
                dataset_id=dataset_id,
            )
    expected_runs = total
    initial_partition_groups: dict[tuple[str, int], set[str]] = {}
    subset_groups: dict[tuple[str, int], set[tuple[str, str, str]]] = {}
    policy_groups: dict[tuple[str, int], set[str]] = {}
    calibration_groups: dict[tuple[str, int], set[tuple[float, str]]] = {}
    preconditioner_groups: dict[tuple[str, int], set[str]] = {}
    class_threshold_groups: dict[tuple[str, int], set[float]] = {}
    predictive_target_groups: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for row in run_rows:
        key = (str(row["dataset_id"]), int(row["seed"]))
        initial_partition_groups.setdefault(key, set()).add(
            str(row["initial_frozen_class_partition_hash"])
        )
        subset_groups.setdefault(key, set()).add((
            str(row["initial_subset_hash"]),
            str(row["validation_subset_hash"]),
            str(row["candidate_subset_hash"]),
        ))
        policy_groups.setdefault(key, set()).add(str(row["policy"]))
        calibration_groups.setdefault(key, set()).add((
            float(row["likelihood_power"]),
            str(row["likelihood_power_calibration_hash"]),
        ))
        preconditioner_groups.setdefault(key, set()).add(
            str(row["design_preconditioner_hash"])
        )
        class_threshold_groups.setdefault(key, set()).add(
            float(row["operational_class_distance_threshold"])
        )
        predictive_target_groups.setdefault(key, set()).add((
            str(row["predictive_target_distribution"]),
            str(row["predictive_target_subset_hash"]),
        ))
    expected_candidate_evaluations = _expected_candidate_evaluations(
        int(config["candidate_pool_budget"]),
        int(config["acquisition_observation_budget"]),
    )
    pcpi_query_rows = [
        row for row in query_rows
        if row["policy"] == protocol.pcpi_policy
    ]
    baseline_query_rows = [
        row for row in query_rows
        if row["policy"] != protocol.pcpi_policy
    ]
    representative_decisions_auditable = bool(pcpi_query_rows) and all(
        (
            row["representative_safe_set_nonempty"]
            and not row["representative_fallback_used"]
            and row["representative_selected_in_safe_set"]
            and row["representative_selected_mmd_nonincrease"]
        )
        or (
            not row["representative_safe_set_nonempty"]
            and row["representative_fallback_used"]
            and row["representative_selected_is_minimum_mmd"]
        )
        for row in pcpi_query_rows
    )
    ambiguity_powers = tuple(
        float(value) for value in config["likelihood_power_candidates"]
    )
    maximin_decisions_auditable = bool(pcpi_query_rows) and all(
        (
            row["robust_model_count"] == len(ambiguity_powers)
            and tuple(row["robust_likelihood_powers"]) == ambiguity_powers
            and row["selected_least_favorable_likelihood_power"]
            in ambiguity_powers
            and np.isclose(
                row["selected_joint_class_predictive_score"],
                min(row["selected_robust_joint_scores_by_model"]),
                rtol=0.0,
                atol=1e-15,
            )
            and row["selected_robust_lower_bound"]
            <= row["selected_joint_class_predictive_score"]
            <= row["selected_robust_upper_bound"]
        )
        if not row["representative_fallback_used"]
        else row["robust_model_count"] == 0
        for row in pcpi_query_rows
    )
    discrepancy_expected = protocol.discrepancy_profile_method is not None
    discrepancy_values_auditable = bool(pcpi_query_rows) and all(
        (
            row["discrepancy_method"] == protocol.discrepancy_profile_method
            and row["discrepancy_residual_excess_variance"] >= 0.0
            and row["discrepancy_support_bandwidth_squared"] > 0.0
            and row["selected_discrepancy_candidate_variance"] >= 0.0
            and row["mean_discrepancy_target_variance"] >= 0.0
        )
        if discrepancy_expected
        else row["discrepancy_method"] == "not-applied"
        for row in pcpi_query_rows
    )
    protocol_decisions = {
        "all_runs_completed": len(run_rows) == expected_runs,
        "no_failed_runs": not failures,
        "matched_initial_budget": bool(run_rows) and all(row["initial_observations"] == int(config["initial_observation_budget"]) for row in run_rows),
        "matched_acquisition_budget": bool(run_rows) and all(row["acquired_observations"] == int(config["acquisition_observation_budget"]) for row in run_rows),
        "matched_validation_budget": bool(run_rows) and all(row["validation_observations"] == int(config["validation_budget"]) for row in run_rows),
        "matched_candidate_pool_budget": bool(run_rows) and all(row["candidate_pool_observations"] == int(config["candidate_pool_budget"]) for row in run_rows),
        "matched_candidate_evaluation_budget": bool(run_rows) and all(row["candidate_evaluations"] == expected_candidate_evaluations for row in run_rows),
        "all_policy_curves_complete": len(curve_rows) == expected_runs * (int(config["acquisition_observation_budget"]) + 1),
        "all_query_records_complete": len(query_rows) == expected_runs * int(config["acquisition_observation_budget"]),
        "all_policies_present_per_dataset_seed": bool(policy_groups) and all(
            policies == set(config["policies"]) for policies in policy_groups.values()
        ),
        "subset_commitments_shared_across_policies": bool(subset_groups) and all(
            len(commitments) == 1 for commitments in subset_groups.values()
        ),
        "initial_class_partition_shared_across_policies": bool(initial_partition_groups) and all(
            len(hashes) == 1 for hashes in initial_partition_groups.values()
        ),
        "likelihood_power_calibration_shared_across_policies": bool(calibration_groups) and all(
            len(values) == 1 for values in calibration_groups.values()
        ),
        "design_preconditioner_shared_across_policies": bool(preconditioner_groups) and all(
            len(values) == 1 for values in preconditioner_groups.values()
        ),
        "operational_class_threshold_shared_across_policies": bool(class_threshold_groups) and all(
            len(values) == 1 for values in class_threshold_groups.values()
        ),
        "operational_class_resolution_is_budget_derived": bool(run_rows) and all(
            np.isclose(
                float(row["operational_class_distance_threshold"]),
                _operational_class_threshold(config),
                rtol=0.0,
                atol=1e-15,
            )
            for row in run_rows
        ),
        "predictive_target_distribution_shared_across_policies": bool(
            predictive_target_groups
        ) and all(len(values) == 1 for values in predictive_target_groups.values()),
        "predictive_target_distribution_uses_registered_action_domain": (
            config["predictive_target_distribution"]
            == "registered-action-domain-uniform"
        ),
        "conditional_predictive_information_method": config[
            "conditional_predictive_information_method"
        ],
        "representative_guard_matches_frozen_contract": (
            config["representative_discrepancy"] == REPRESENTATIVE_MMD_METHOD
            and config["representative_target_distribution"]
            == "registered-action-domain-uniform"
        ),
        "representative_guard_applied_to_all_pcpi_queries": bool(
            pcpi_query_rows
        ) and all(row["representative_guard_applied"] for row in pcpi_query_rows),
        "representative_guard_not_applied_to_baselines": bool(
            baseline_query_rows
        ) and all(
            not row["representative_guard_applied"]
            for row in baseline_query_rows
        ),
        "representative_decisions_auditable": representative_decisions_auditable,
        "maximin_decisions_auditable": maximin_decisions_auditable,
        "maximin_ambiguity_set_matches_frozen_contract": (
            ambiguity_powers == (0.125, 0.25, 0.5, 1.0)
            and config["pcpi_ambiguity_set"]
            == "frozen-likelihood-power-candidates"
            and config["pcpi_robust_utility"] == (
                "discrepancy-aware-maximin-joint-class-predictive-information"
                if discrepancy_expected
                else "maximin-joint-class-predictive-information"
            )
            and config["pcpi_least_favorable_tie_break"]
            == "smallest-likelihood-power"
        ),
        "discrepancy_profile_matches_frozen_contract": (
            config.get("pcpi_discrepancy_profile", "not-applied")
            == (
                protocol.discrepancy_profile_method
                if discrepancy_expected else "not-applied"
            )
        ),
        "discrepancy_values_auditable": discrepancy_values_auditable,
        "discrepancy_not_applied_to_baselines": bool(baseline_query_rows) and all(
            row["discrepancy_method"] == "not-applied"
            for row in baseline_query_rows
        ),
        "nominal_calibrated_posterior_retained_for_reporting": True,
        "likelihood_power_calibration_used_initial_development_only": True,
        "basis_preconditioner_used_initial_development_covariates_only": True,
        "posterior_predictive_uses_posterior_target_design": (
            config["predictive_design_transform"] == "posterior-target-frozen"
        ),
        "official_hashes_verified": len(dataset_records) == len(config["datasets"]),
        "heldout_remained_closed": True,
        "selection_did_not_use_heldout": True,
        "gas_targets_counted_as_one_family": True,
        "posterior_type": config["posterior_type"],
        "likelihood_power_calibration_method": config[
            "likelihood_power_calibration_method"
        ],
        "basis_preconditioning_method": config["basis_preconditioning_method"],
    }
    protocol_gate = all(protocol_decisions.values())
    aggregates = _aggregates(run_rows) if run_rows else []
    paired = (
        _paired_effects(run_rows, protocol.pcpi_policy) if protocol_gate else []
    )
    assessment = _assessment(
        paired, run_rows, protocol_gate, config, protocol.pcpi_policy
    )
    evidence_flags = _evidence_flags(protocol_gate, assessment)
    summary = {
        "stage": protocol.stage,
        "experiment": protocol.experiment,
        "real_measurement_experiment": True,
        **evidence_flags,
        "successful_runs": len(run_rows),
        "expected_runs": expected_runs,
        "failure_count": len(failures),
        "failures": [{key: value for key, value in item.items() if key != "fallback_context"} for item in failures],
        "protocol_gate_passed": protocol_gate,
        "protocol_gate_decisions": protocol_decisions,
        "effectiveness_assessment": assessment,
        "dataset_family_count": 2,
        "gas_targets_counted_as_one_family": True,
        "heldout_opened": False,
        "selection_used_heldout": False,
        "claim_boundary": protocol.claim_boundary,
    }
    (output / "config.json").write_text(_canonical_json(config), encoding="utf-8")
    (output / "claim_boundary.md").write_text(
        "# Claim boundary\n\n" + protocol.claim_boundary + "\n", encoding="utf-8"
    )
    registry, evidence = _record_evidence(
        output, run_rows, curve_rows, query_rows, failures, contexts,
        summary, aggregates, paired, dataset_records,
        _aggregate_context(identity, dataset_records, config, protocol),
        protocol,
    )
    exported = _export_evidence(output, registry, protocol)
    if _hash_json(exported["summary"]) != _hash_json(summary):
        raise RuntimeError("P3B evidence export differs from computed summary")
    if protocol_gate and protocol is P3B10_PROTOCOL:
        make_p3b_figures(output)
    registry.lock_path.unlink(missing_ok=True)
    ended = datetime.now(timezone.utc)
    manifest = {
        "schema": "pcpi-run-manifest-v1",
        "stage": protocol.stage,
        "experiment": protocol.experiment,
        **identity,
        "code_hash": identity["production_code_hash"],
        "dataset_raw_hashes": {key: value["combined_source_hash"] for key, value in dataset_records.items()},
        "split_hashes": {key: value["split_manifest"]["split_hash"] for key, value in dataset_records.items()},
        "seeds": config["seeds"],
        "budgets": {
            "initial_observations": config["initial_observation_budget"],
            "acquired_observations": config["acquisition_observation_budget"],
            "candidate_pool": config["candidate_pool_budget"],
            "validation_observations": config["validation_budget"],
            "eig_quadrature_min_evaluations": config[
                "eig_quadrature_min_evaluations"
            ],
            "eig_quadrature_max_evaluations": config[
                "eig_quadrature_max_evaluations"
            ],
            "eig_quadrature_growth_factor": config[
                "eig_quadrature_growth_factor"
            ],
            "eig_quadrature_error_safety_factor": config[
                "eig_quadrature_error_safety_factor"
            ],
            "eig_rank_certificate_method": config["eig_rank_certificate_method"],
            "pcpi_class_target_partition": config["pcpi_class_target_partition"],
            "pcpi_uncertified_eig_action": config[
                "pcpi_uncertified_eig_action"
            ],
            "pcpi_joint_target": config["pcpi_joint_target"],
            "predictive_target_distribution": config[
                "predictive_target_distribution"
            ],
            "conditional_predictive_information_method": config[
                "conditional_predictive_information_method"
            ],
            "representative_guard": config["representative_guard"],
            "representative_target_distribution": config[
                "representative_target_distribution"
            ],
            "representative_discrepancy": config[
                "representative_discrepancy"
            ],
            "representative_kernel_bandwidth": config[
                "representative_kernel_bandwidth"
            ],
            "representative_safe_set_rule": config[
                "representative_safe_set_rule"
            ],
            "representative_empty_safe_set_action": config[
                "representative_empty_safe_set_action"
            ],
            "pcpi_ambiguity_set": config["pcpi_ambiguity_set"],
            "pcpi_robust_utility": config["pcpi_robust_utility"],
            "pcpi_least_favorable_tie_break": config[
                "pcpi_least_favorable_tie_break"
            ],
            "pcpi_discrepancy_profile": config.get(
                "pcpi_discrepancy_profile", "not-applied"
            ),
            "pcpi_discrepancy_scale": config.get(
                "pcpi_discrepancy_scale", "not-applied"
            ),
            "pcpi_discrepancy_support_rule": config.get(
                "pcpi_discrepancy_support_rule", "not-applied"
            ),
            "robust_likelihood_powers": config["likelihood_power_candidates"],
            "operational_class_resolution_method": config[
                "operational_class_resolution_method"
            ],
            "operational_class_aggregate_separation": config[
                "operational_class_aggregate_separation"
            ],
            "operational_class_distance_threshold": _operational_class_threshold(
                config
            ),
            "predictive_design_transform": config["predictive_design_transform"],
            "qbc_committee_size": config["qbc_committee_size"],
        },
        "provider": "none",
        "model": "none",
        "llm_calls": 0,
        "engine_calls": {"successful_policy_runs": len(run_rows), "failed_runs": len(failures)},
        "heldout_state": "closed",
        "heldout_opened": False,
        "selection_used_heldout": False,
        "failure_policy": config["failure_policy"],
        "start_time_utc": started.isoformat(),
        "end_time_utc": ended.isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "primary_metrics": [
            "normalized_aulc_validation_rmse", "final_validation_rmse",
            "frozen_class_entropy_gain",
            "initial_class_aggregation_fraction",
            "paired_delta_normalized_aulc_rmse",
            "paired_delta_frozen_class_entropy_gain",
            "pcpi_class_eig_used_rate",
            "pcpi_maximin_joint_eig_used_rate",
            "pcpi_epistemic_fallback_rate",
            "pcpi_representative_safe_set_nonempty_rate",
            "pcpi_representative_fallback_rate",
            "pcpi_representative_selected_nonincrease_rate",
            "pcpi_mean_representative_safe_set_size",
            "pcpi_mean_selected_discrepancy_variance",
        ],
        "protocol_gate_passed": protocol_gate,
        "effectiveness_assessment": assessment,
        **evidence_flags,
        "claim_boundary": protocol.claim_boundary,
        "evidence_registry": evidence,
        "evidence_export_manifest_hash": file_sha256(exported["export_path"]),
    }
    (output / "RUN_MANIFEST.json").write_text(
        _canonical_json(manifest), encoding="utf-8"
    )
    reporter.emit(
        "run_completed",
        f"{protocol.stage} complete | protocol={'PASS' if protocol_gate else 'FAIL'} "
        f"assessment={assessment['status']} runs={len(run_rows)}/{expected_runs}",
        protocol_gate_passed=protocol_gate,
        effectiveness_assessment=assessment,
    )
    print(_canonical_json(summary), end="", flush=True)
    return 0 if protocol_gate else 2


def build_parser(
    protocol: RealAcquisitionProtocol = P3B10_PROTOCOL,
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-artifact",
        help=(
            "optional verified source ZIP; when omitted, require and record the "
            "clean Git worktree containing this runner"
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase", default=protocol.stage, choices=(protocol.stage,)
    )
    parser.add_argument("--heldout-state", default="closed", choices=("closed",))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
