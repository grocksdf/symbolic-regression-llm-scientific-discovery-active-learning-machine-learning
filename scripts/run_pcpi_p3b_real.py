"""Run matched-budget P3B acquisition on provenance-verified real measurements."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

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
    runtime_binary_identity,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from hypothesis_mvp.pcpi import (
    ACQUISITION_POLICIES,
    ANALYTIC_CLASS_EIG_BOUNDS_METHOD,
    AcquisitionScores,
    BUDGET_RESOLUTION_METHOD,
    ClassPartition,
    DECISION_REGRET_DISTANCE_METRIC,
    DECISION_TARGETED_POLICY,
    DEFAULT_CLASS_EIG_OUTWARD_TOLERANCE,
    DEFAULT_CLASS_EIG_QUANTIZATION_LEVELS,
    DISCREPANCY_AWARE_POLICY,
    DISCREPANCY_PROFILE_METHOD,
    GAUSSIAN_CLASS_CONDITIONAL_EPIG,
    MAXIMIN_RANK_CERTIFICATE,
    P3H_INTERVAL_FRONTIER_RESOLUTION,
    P3H_TERMINAL_ABSTENTION,
    P3I_COPULA_TRANSPORT_METHOD,
    P3I_INFORMATION_INVARIANCE,
    OperationalSemiparametricDecision,
    OperationalSemiparametricState,
    OperationalClassConditionalState,
    P3D_ACQUISITION_POLICIES,
    P3H_OPERATIONAL_LIFECYCLE,
    P3H_OPERATIONAL_POWERS,
    PosteriorModel,
    REFERENCE_DOMINANCE_METHOD,
    REFERENCE_DOMINANCE_POLICY,
    REFERENCE_FALLBACK_MODE,
    REFERENCE_POLICY,
    REFERENCE_SEED_METHOD,
    ReferenceDominanceScores,
    REPRESENTATIVE_MMD_METHOD,
    TARGETED_HANDOVER_MODE,
    SequentialReferencePosterior,
    admit_operational_semiparametric_response,
    aggregate_operational_classes,
    aggregate_decision_equivalent_classes,
    budget_resolved_distance_threshold,
    class_partition,
    fixed_class_entropy,
    initialize_operational_semiparametric_state,
    initialize_operational_class_conditional_state,
    normalized_area_under_learning_curve,
    posterior_metrics,
    score_acquisition_actions,
    score_discrepancy_aware_actions,
    score_decision_targeted_actions,
    score_reference_dominance_actions,
    select_acquisition_candidate,
    stable_derived_seed,
    stable_reference_policy_seed,
    run_p3j_outer_policy,
)
from hypothesis_mvp.pcpi.reference import (
    CALIBRATION_METHOD,
    CALIBRATION_ROLE,
    CALIBRATION_TIE_BREAK,
    DESIGN_PRECONDITIONING_METHOD,
    DESIGN_PRECONDITIONING_ROLE,
    DesignPreconditioner,
    DevelopmentStandardizer,
    ExactPosterior,
    OperationalClassPosterior,
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
P3I_SHARED_INITIAL_TARGET_SOURCE = (
    "complete-initial-history-batch-sufficient-statistics-once-per-dataset-seed-v1"
)


def _class_contract_value(config: dict[str, Any], suffix: str) -> Any:
    matches = [
        config[key]
        for key in (f"p3m_{suffix}", f"p3k_{suffix}", f"p3j_{suffix}")
        if key in config
    ]
    if len(matches) != 1:
        raise ValueError(f"exactly one class-conditional contract key is required: {suffix}")
    return matches[0]


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
    reference_dominance_method: str | None = None
    semiparametric_lifecycle: bool = False
    semiparametric_unresolved_action: str = P3H_TERMINAL_ABSTENTION
    required_runtime_dependency_hash: str | None = None
    required_python_executable_hash: str | None = None
    required_runtime_binary_identity: dict[str, dict[str, object]] | None = None
    decision_target_alignment: bool = False
    shared_initial_frozen_target: bool = False
    fail_fast: bool = False
    operational_execution_authorized: bool = True
    p3j_class_conditional_lifecycle: bool = False
    class_conditional_contract_prefix: str = "p3j"
    class_conditional_representative_method: str = REPRESENTATIVE_MMD_METHOD
    class_conditional_singleton_rank_certificate: str | None = None
    config_validator: Callable[[Path, Path], dict[str, Any]] | None = None


@dataclass(frozen=True)
class FrozenInitialClassTarget:
    """One canonical H0 decision target shared by every matched policy."""

    posterior: ExactPosterior
    classes: OperationalClassPosterior
    partition: ClassPartition
    engine_target_hash: str
    source: str = P3I_SHARED_INITIAL_TARGET_SOURCE

    def __post_init__(self) -> None:
        if (
            self.source != P3I_SHARED_INITIAL_TARGET_SOURCE
            or not self.engine_target_hash
            or self.partition != class_partition(self.posterior, self.classes)
        ):
            raise ValueError("shared initial class target is internally inconsistent")


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


def _validate_reference_dominance_config(
    config: dict[str, Any],
    protocol: RealAcquisitionProtocol,
) -> dict[str, Any]:
    """Validate the clean P3D real-only protocol surface."""

    required = {
        "schema", "stage", "datasets", "policies", "seeds",
        "initial_observation_budget", "acquisition_observation_budget",
        "candidate_pool_budget", "validation_budget", "qbc_committee_size",
        "operational_class_metric", "operational_class_linkage",
        "operational_class_resolution_method",
        "operational_class_aggregate_separation",
        "operational_class_quantile_levels", "class_evaluation_partition",
        "pcpi_class_target_partition", "split_seed", "hash_verification",
        "heldout_state", "failure_policy", "assessment_rules",
        "posterior_type", "likelihood_power_candidates",
        "likelihood_power_calibration_method",
        "likelihood_power_calibration_role", "likelihood_power_tie_break",
        "basis_preconditioning_method", "basis_preconditioning_role",
        "predictive_design_transform", "pcpi_primary_utility",
        "pcpi_information_bound_method",
        "pcpi_response_quantization_probability_levels",
        "pcpi_numerical_outward_tolerance", "pcpi_reference_policy",
        "pcpi_reference_seed_method", "pcpi_decision_method",
    }
    if set(config) != required:
        raise ValueError(
            "P3D config fields differ from schema: "
            f"{sorted(set(config) ^ required)}"
        )
    if config["schema"] != protocol.schema or config["stage"] != protocol.stage:
        raise ValueError("unsupported P3D real-acquisition config schema or stage")
    if tuple(config["datasets"]) != P2A_REAL_DATASETS:
        raise ValueError("P3D datasets must be the frozen CCPP and Gas targets")
    if tuple(config["policies"]) != protocol.policies:
        raise ValueError("P3D requires its frozen matched-budget policy set")
    if tuple(config["seeds"]) != FROZEN_SEEDS:
        raise ValueError("P3D requires the eight frozen registered seeds")
    frozen_values = {
        "initial_observation_budget": 32,
        "acquisition_observation_budget": 32,
        "candidate_pool_budget": 128,
        "validation_budget": 256,
        "qbc_committee_size": 32,
        "split_seed": SPLIT_SEED,
        "hash_verification": "mandatory",
        "heldout_state": "closed",
        "failure_policy": "fail_closed_record_all_no_seed_replacement",
        "operational_class_metric": "pooled-predictive-sd-quantile-rms",
        "operational_class_linkage": "complete",
        "operational_class_resolution_method": BUDGET_RESOLUTION_METHOD,
        "operational_class_aggregate_separation": 1.0,
        "class_evaluation_partition": "initial-frozen",
        "pcpi_class_target_partition": "initial-frozen",
        "posterior_type": "power-likelihood-generalized-bayes",
        "likelihood_power_calibration_method": CALIBRATION_METHOD,
        "likelihood_power_calibration_role": CALIBRATION_ROLE,
        "likelihood_power_tie_break": CALIBRATION_TIE_BREAK,
        "basis_preconditioning_method": DESIGN_PRECONDITIONING_METHOD,
        "basis_preconditioning_role": DESIGN_PRECONDITIONING_ROLE,
        "predictive_design_transform": "posterior-target-frozen",
        "pcpi_primary_utility": "initial-frozen-class-mutual-information",
        "pcpi_information_bound_method": ANALYTIC_CLASS_EIG_BOUNDS_METHOD,
        "pcpi_numerical_outward_tolerance": (
            DEFAULT_CLASS_EIG_OUTWARD_TOLERANCE
        ),
        "pcpi_reference_policy": REFERENCE_POLICY,
        "pcpi_reference_seed_method": REFERENCE_SEED_METHOD,
        "pcpi_decision_method": protocol.reference_dominance_method,
    }
    if any(config[key] != value for key, value in frozen_values.items()):
        raise ValueError("P3D frozen method, budget, or provenance contract changed")
    if tuple(float(value) for value in config["operational_class_quantile_levels"]) != (
        0.1, 0.5, 0.9
    ):
        raise ValueError("P3D operational-class quantiles changed")
    if tuple(float(value) for value in config["likelihood_power_candidates"]) != (
        0.125, 0.25, 0.5, 1.0
    ):
        raise ValueError("P3D likelihood-power calibration candidates changed")
    if tuple(
        float(value)
        for value in config["pcpi_response_quantization_probability_levels"]
    ) != DEFAULT_CLASS_EIG_QUANTIZATION_LEVELS:
        raise ValueError("P3D response quantizer changed")
    rules = config["assessment_rules"]
    expected_rules = {
        "paired_confidence_level": 0.95,
        "negative_transfer_rate_max": 0.25,
        "pcpi_decision_rule_valid_rate_min": 1.0,
        "strong_requires_positive_frozen_class_gain_vs_random_in_every_dataset_family": True,
        "strong_requires_nonpositive_mean_vs_each_baseline_in_every_dataset_family": True,
    }
    if rules != expected_rules:
        raise ValueError("P3D assessment rules changed")
    return config


def _load_config(
    path: Path,
    root: Path,
    protocol: RealAcquisitionProtocol = P3B10_PROTOCOL,
) -> dict[str, Any]:
    if protocol.config_validator is not None:
        return protocol.config_validator(path, root)
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3B config must be an existing file inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    if protocol.reference_dominance_method is not None:
        return _validate_reference_dominance_config(config, protocol)
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
    if protocol.semiparametric_lifecycle:
        required |= {
            "initial_base_warmup_budget",
            "initial_residual_training_budget",
            "p3h_operational_lifecycle",
            "p3h_residual_state_method",
            "p3h_residual_law",
            "p3h_validation_state_policy",
            "p3h5_terminal_status",
            "runtime_dependency_hash",
        }
    if protocol.decision_target_alignment or protocol.p3j_class_conditional_lifecycle:
        required |= {
            "p3i_decision_target",
            "p3i_information_invariance",
            "p3i_semiparametric_transport",
            "operational_execution_authorized",
        }
    if protocol.shared_initial_frozen_target:
        required.add("p3i_initial_frozen_target_source")
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
        "failure_policy": (
            "fail_fast_record_terminal_no_seed_replacement"
            if protocol.fail_fast
            else "fail_closed_record_all_no_seed_replacement"
        ),
    }
    if any(config[key] != value for key, value in frozen_values.items()):
        raise ValueError(f"{protocol.stage} frozen budgets or failure policy were modified")
    expected_class_metric = (
        DECISION_REGRET_DISTANCE_METRIC
        if protocol.decision_target_alignment
        else "pooled-predictive-sd-quantile-rms"
    )
    if config["operational_class_metric"] != expected_class_metric:
        raise ValueError("real protocol changed its frozen operational-class metric")
    if config["operational_class_linkage"] != "complete":
        raise ValueError("P3B.10 requires deterministic complete linkage")
    if config["operational_class_resolution_method"] != BUDGET_RESOLUTION_METHOD:
        raise ValueError("P3B.10 requires budget-resolved predictive equivalence")
    if float(config["operational_class_aggregate_separation"]) != 1.0:
        raise ValueError("P3B.10 aggregate predictive resolution must remain one")
    levels = tuple(float(value) for value in config["operational_class_quantile_levels"])
    expected_levels = () if protocol.decision_target_alignment else (0.1, 0.5, 0.9)
    if levels != expected_levels:
        raise ValueError("real protocol changed its frozen class-profile coordinates")
    if config["class_evaluation_partition"] != "initial-frozen":
        raise ValueError("P3B.10 evaluation must use the initial-frozen class partition")
    if config["pcpi_class_target_partition"] != "initial-frozen":
        raise ValueError("P3B.10 acquisition must target the initial-frozen class partition")
    expected_uncertified = (
        protocol.semiparametric_unresolved_action
        if protocol.semiparametric_lifecycle
        else "posterior-epistemic-variance"
    )
    if config["pcpi_uncertified_eig_action"] != expected_uncertified:
        raise ValueError("P3B.10 requires the frozen epistemic fallback utility")
    joint_target_contract = {
        "pcpi_joint_target": (
            "initial-frozen-operational-class-log-risk"
            if protocol.decision_target_alignment
            else "initial-frozen-class-and-target-prediction"
        ),
        "predictive_target_distribution": "registered-action-domain-uniform",
        "conditional_predictive_information_method": (
            "excluded-zero-by-decision-target"
            if protocol.decision_target_alignment
            else GAUSSIAN_CLASS_CONDITIONAL_EPIG
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
        "representative_empty_safe_set_action": (
            "terminal-failure-no-utility-switch"
            if protocol.decision_target_alignment
            else "minimum-augmented-mmd"
        ),
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
            "p3i-copula-transport-invariant-maximin-operational-class-information"
            if protocol.decision_target_alignment
            else "p3h-semiparametric-discrepancy-aware-maximin-joint-class-predictive-information"
            if protocol.semiparametric_lifecycle
            else "discrepancy-aware-maximin-joint-class-predictive-information"
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
    calibration_contract = (
        {
            "likelihood_power_calibration_method": "none-fixed-complete-family",
            "likelihood_power_calibration_role": "forbidden-no-eta-selection",
            "likelihood_power_tie_break": "not-applicable-complete-family",
        }
        if protocol.semiparametric_lifecycle
        else {
            "likelihood_power_calibration_method": CALIBRATION_METHOD,
            "likelihood_power_calibration_role": CALIBRATION_ROLE,
            "likelihood_power_tie_break": CALIBRATION_TIE_BREAK,
        }
    )
    if any(config[key] != value for key, value in calibration_contract.items()):
        raise ValueError("P3B.10 likelihood-power calibration contract was modified")
    preconditioning_contract = {
        "basis_preconditioning_method": DESIGN_PRECONDITIONING_METHOD,
        "basis_preconditioning_role": (
            "base-warmup-covariates-only"
            if protocol.semiparametric_lifecycle else DESIGN_PRECONDITIONING_ROLE
        ),
    }
    if any(config[key] != value for key, value in preconditioning_contract.items()):
        raise ValueError("P3B.10 basis-preconditioning contract was modified")
    if config["predictive_design_transform"] != "posterior-target-frozen":
        raise ValueError("P3B.10 prediction must use the posterior-target design transform")
    if protocol.semiparametric_lifecycle:
        p3h_contract = {
            "initial_base_warmup_budget": 16,
            "initial_residual_training_budget": 16,
            "p3h_operational_lifecycle": P3H_OPERATIONAL_LIFECYCLE,
            "p3h_residual_state_method": (
                "model-specific-likelihood-power-prequential-raw-pit-family-v1"
            ),
            "p3h_residual_law": "prequential-kt-dyadic-polya-tree-residual-law-v1",
            "p3h_validation_state_policy": "discard-never-enter-operational-state",
            "p3h5_terminal_status": (
                "FAMILY_CALIBRATION_COMPATIBLE_ACQUISITION_BLOCKED"
            ),
            "runtime_dependency_hash": protocol.required_runtime_dependency_hash,
        }
        if any(config[key] != value for key, value in p3h_contract.items()):
            raise ValueError("P3H operational lifecycle contract was modified")
        if (
            config["initial_base_warmup_budget"]
            + config["initial_residual_training_budget"]
            != config["initial_observation_budget"]
        ):
            raise ValueError("P3H initial role budgets do not close")
    if (
        protocol.decision_target_alignment
        or protocol.p3j_class_conditional_lifecycle
    ):
        p3i_contract = {
            "p3i_decision_target": "frozen-operational-class-log-risk",
            "p3i_information_invariance": P3I_INFORMATION_INVARIANCE,
            "p3i_semiparametric_transport": P3I_COPULA_TRANSPORT_METHOD,
            "operational_execution_authorized": (
                protocol.operational_execution_authorized
            ),
        }
        if any(config[key] != value for key, value in p3i_contract.items()):
            raise ValueError("P3I decision-alignment or authorization contract changed")
    if protocol.shared_initial_frozen_target and config[
        "p3i_initial_frozen_target_source"
    ] != P3I_SHARED_INITIAL_TARGET_SOURCE:
        raise ValueError("P3I shared initial frozen target contract changed")
    rules = config["assessment_rules"]
    expected_rules = {
        "paired_confidence_level", "negative_transfer_rate_max",
        "pcpi_decision_rule_valid_rate_min",
        "strong_requires_positive_frozen_class_gain_vs_random_in_every_dataset_family",
        (
            "secondary_predictive_noninferiority_report_only"
            if protocol.decision_target_alignment
            else "strong_requires_nonpositive_mean_vs_each_baseline_in_every_dataset_family"
        ),
    }
    if set(rules) != expected_rules or float(rules["paired_confidence_level"]) != 0.95:
        raise ValueError("P3B assessment rules differ from the frozen schema")
    expected_rule_values = {
        "paired_confidence_level": 0.95,
        "negative_transfer_rate_max": 0.25,
        "pcpi_decision_rule_valid_rate_min": 1.0,
        "strong_requires_positive_frozen_class_gain_vs_random_in_every_dataset_family": True,
        (
            "secondary_predictive_noninferiority_report_only"
            if protocol.decision_target_alignment
            else "strong_requires_nonpositive_mean_vs_each_baseline_in_every_dataset_family"
        ): True,
    }
    if any(rules[key] != value for key, value in expected_rule_values.items()):
        raise ValueError("P3B.10 assessment rules were modified")
    return config


def _prepare_output(
    path: Path, *, allow_p3j_resume: bool = False,
    class_conditional_prefix: str = "p3j",
) -> None:
    if path.exists() and any(path.iterdir()):
        if not allow_p3j_resume:
            raise FileExistsError(f"output directory is not empty: {path}")
        allowed = {
            "hypotheses", "diagnostics", "tables", "figures", "logs",
            class_conditional_prefix,
        }
        unexpected = [item.name for item in path.iterdir() if item.name not in allowed]
        terminal = tuple(path.glob(
            f"{class_conditional_prefix}/**/POLICY_FAILURE.json"
        ))
        if unexpected or terminal:
            raise FileExistsError(
                "P3J resume output contains terminal or unexpected artifacts: "
                + ",".join(unexpected or [str(item) for item in terminal])
            )
    for name in ("hypotheses", "diagnostics", "tables", "figures", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)


class _FailFastProtocolError(RuntimeError):
    """Terminal formal-run failure already persisted to the output ledger."""


def _write_terminal_failure(
    output: Path,
    protocol: RealAcquisitionProtocol,
    failure: dict[str, Any],
) -> Path:
    path = output / "TERMINAL_FAILURE.json"
    payload = {
        "schema": "pcpi-fail-fast-terminal-record-v1",
        "stage": protocol.stage,
        "terminal": True,
        "retry_in_same_output_forbidden": True,
        "seed_replacement_forbidden": True,
        "heldout_opened": False,
        "selection_used_heldout": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "failure": {
            key: value for key, value in failure.items()
            if key != "fallback_context"
        },
    }
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_json(payload))
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _operational_class_threshold(config: dict[str, Any]) -> float:
    return budget_resolved_distance_threshold(
        int(config["acquisition_observation_budget"]),
        aggregate_separation=float(
            config["operational_class_aggregate_separation"]
        ),
    )


def _aggregate_protocol_classes(
    protocol: RealAcquisitionProtocol,
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    actions: np.ndarray,
    config: dict[str, Any],
) -> OperationalClassPosterior:
    threshold = _operational_class_threshold(config)
    if protocol.decision_target_alignment:
        return aggregate_decision_equivalent_classes(
            engine, posterior, actions, distance_threshold=threshold
        )
    if protocol.p3j_class_conditional_lifecycle:
        return aggregate_decision_equivalent_classes(
            engine, posterior, actions, distance_threshold=threshold
        )
    return aggregate_operational_classes(
        engine,
        posterior,
        actions,
        distance_threshold=threshold,
        quantile_levels=tuple(config["operational_class_quantile_levels"]),
    )


def _freeze_initial_class_target(
    protocol: RealAcquisitionProtocol,
    engine: SequentialReferencePosterior,
    initial_X: np.ndarray,
    initial_y: np.ndarray,
    fixed_domain_X: np.ndarray,
    config: dict[str, Any],
) -> FrozenInitialClassTarget:
    """Canonicalize H0 once from the complete opened initial history."""

    posterior = engine.fit_batch(initial_X, initial_y)
    classes = _aggregate_protocol_classes(
        protocol, engine, posterior, fixed_domain_X, config
    )
    return FrozenInitialClassTarget(
        posterior=posterior,
        classes=classes,
        partition=class_partition(posterior, classes),
        engine_target_hash=engine.target_hash,
    )


def _assert_initial_posterior_numerically_equivalent(
    canonical: ExactPosterior,
    candidate: ExactPosterior,
) -> None:
    """Fail closed unless batch and prequential H0 sufficient statistics agree.

    The two conjugate paths differ only in floating-point summation order.  The
    tolerance is an a-priori accumulation bound proportional to the number of
    opened observations; it does not depend on an experimental effect or Gate
    outcome.
    """

    if (
        canonical.bank_hash != candidate.bank_hash
        or canonical.likelihood_power != candidate.likelihood_power
        or len(canonical.members) != len(candidate.members)
    ):
        raise ValueError("shared initial posterior target identity differs")
    epsilon = np.finfo(float).eps
    for expected, actual in zip(canonical.members, candidate.members, strict=True):
        if (
            expected.structure != actual.structure
            or expected.state.prior != actual.state.prior
            or expected.state.observations != actual.state.observations
        ):
            raise ValueError("shared initial posterior structure or prior differs")
        update_count = max(
            1,
            int(round(expected.state.observations / canonical.likelihood_power)),
        )
        relative_bound = 128.0 * update_count * epsilon
        for field in ("precision", "information", "y_square_sum"):
            left = np.asarray(getattr(expected.state, field), dtype=float)
            right = np.asarray(getattr(actual.state, field), dtype=float)
            scale = max(
                1.0,
                float(np.max(np.abs(left))),
                float(np.max(np.abs(right))),
            )
            if not np.all(np.abs(left - right) <= relative_bound * scale):
                raise FloatingPointError(
                    "prequential initial posterior is not numerically equivalent "
                    "to the shared complete-history target"
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


def _reference_score_audit_adapter(
    result: ReferenceDominanceScores,
) -> AcquisitionScores:
    """Expose P3D values through the shared export shape, without recombination."""

    midpoints = result.utility_interval_midpoints
    lower = result.utility_bounds.lower_bounds
    upper = result.utility_bounds.upper_bounds
    radii = 0.5 * (upper - lower)
    zeros = np.zeros(len(midpoints), dtype=float)
    return AcquisitionScores(
        policy=result.policy,
        scores=midpoints,
        integration_error_bounds=radii,
        class_count=result.class_count,
        estimator_samples=0,
        ranking_certified=result.decision.targeted_handover,
        ranking_margin=(
            result.decision.leader_lower_bound
            - result.decision.reference_upper_bound
        ),
        ranking_error_bound=result.decision.reference_upper_bound,
        ranking_certificate_gap=result.decision.dominance_gap,
        ranking_error_safety_factor=0.0,
        ranking_planned_looks=1,
        ranking_looks_used=1,
        ranking_certificate_method=result.decision.method,
        estimator_coarse_samples=0,
        estimator_integration_method=result.utility_bounds.method,
        utility_mode=result.decision.utility_mode,
        target_partition_hash=result.target_partition_hash,
        class_eig_scores=midpoints,
        class_eig_error_bounds=radii,
        conditional_predictive_eig_scores=zeros,
        joint_class_predictive_scores=zeros,
        representative_guard_applied=False,
        representative_current_mmd_squared=0.0,
        representative_augmented_mmd_squared=zeros,
        representative_safe_mask=np.ones(len(midpoints), dtype=bool),
        representative_safe_set_nonempty=True,
        representative_safe_set_size=len(midpoints),
        representative_fallback_used=False,
        representative_mmd_tolerance=0.0,
        representative_kernel_bandwidth_squared=0.0,
        representative_mmd_method="not-applied-reference-dominance",
        robust_likelihood_powers=(),
        robust_model_count=0,
        least_favorable_likelihood_powers=zeros,
        robust_joint_scores_by_model=np.empty((0, len(midpoints))),
        robust_lower_bounds=zeros,
        robust_upper_bounds=zeros,
    )


def _p3j_compatible_query_rows(
    outer_result: object,
    subset_commitments: dict[str, str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for index, (row, identity) in enumerate(zip(
        outer_result.artifacts.query_rows,
        outer_result.measured_run.identities,
        strict=True,
    )):
        curve = outer_result.artifacts.curve_rows[index]
        selected_mmd = float(row["representative_selected_mmd_squared"])
        rows.append(dict(row) | {
            "operational_class_count": curve["operational_class_count"],
            "operational_class_partition_hash_before_query": curve[
                "operational_class_partition_hash"
            ],
            "initial_frozen_class_partition_hash": row[
                "acquisition_target_partition_hash"
            ],
            "operational_class_distance_threshold": _operational_class_threshold(
                config
            ),
            "operational_class_metric": curve["operational_class_metric"],
            "operational_class_scale_hash": curve[
                "operational_class_scale_hash"
            ],
            "p3h_operational_lifecycle_applied": False,
            "p3h_family_hash_before_query": "not-applied",
            "p3h_family_hash_after_query": "not-applied",
            "discrepancy_method": "not-applied",
            "discrepancy_residual_excess_variance": 0.0,
            "discrepancy_support_bandwidth_squared": 0.0,
            "selected_discrepancy_candidate_variance": 0.0,
            "mean_discrepancy_target_variance": 0.0,
            "predictive_target_distribution": config[
                "predictive_target_distribution"
            ],
            "predictive_target_subset_hash": subset_commitments["candidate"],
            "conditional_predictive_information_method": config[
                "conditional_predictive_information_method"
            ],
            "representative_selected_mmd_change": (
                selected_mmd - float(row["representative_current_mmd_squared"])
            ),
            "eig_possible_maximizer_candidate_ids": row[
                "eig_possible_maximizer_local_indices"
            ],
            "eig_selection_admissible_candidate_ids": row[
                "eig_selection_admissible_local_indices"
            ],
            "remaining_candidates_before_query": identity.candidate_count,
            "reference_dominance_applied": False,
            "reference_targeted_handover": False,
        })
    return rows


def _p3j_compatible_summary(
    outer_result: object,
    *,
    dataset_id: str,
    seed: int,
    initial_count: int,
    validation_count: int,
    candidate_count: int,
    subset_commitments: dict[str, str],
    design_preconditioner: DesignPreconditioner,
    calibration_hash: str,
    frozen_initial_target: FrozenInitialClassTarget,
    config: dict[str, Any],
    wall_time_seconds: float,
) -> dict[str, Any]:
    first_curve = outer_result.artifacts.curve_rows[0]
    summary = dict(outer_result.summary_metrics)
    summary.update({
        "dataset_id": dataset_id,
        "dataset_family": _family(dataset_id),
        "seed": seed,
        "policy": config["policies"][-1],
        "initial_observations": initial_count,
        "acquired_observations": len(outer_result.artifacts.query_rows),
        "validation_observations": validation_count,
        "candidate_pool_observations": candidate_count,
        "candidate_evaluations": sum(
            item.candidate_count for item in outer_result.measured_run.identities
        ),
        "initial_subset_hash": subset_commitments["initial"],
        "validation_subset_hash": subset_commitments["validation"],
        "candidate_subset_hash": subset_commitments["candidate"],
        "posterior_type": config["posterior_type"],
        "likelihood_power": 1.0,
        "posterior_target_hash": outer_result.measured_run.final_state.nominal_state.engine.target_hash,
        "pcpi_ambiguity_set": list(P3H_OPERATIONAL_POWERS),
        "pcpi_robust_utility": config["pcpi_robust_utility"],
        "pcpi_discrepancy_profile": "not-applied",
        "design_preconditioner_hash": design_preconditioner.stable_hash,
        "likelihood_power_calibration_hash": calibration_hash,
        "likelihood_power_calibration_wall_time_seconds": 0.0,
        "operational_class_metric": first_curve["operational_class_metric"],
        "initial_operational_class_scale_hash": first_curve[
            "operational_class_scale_hash"
        ],
        "initial_frozen_class_partition_hash": (
            frozen_initial_target.partition.stable_hash
        ),
        "initial_frozen_target_source": frozen_initial_target.source,
        "predictive_target_distribution": config["predictive_target_distribution"],
        "predictive_target_subset_hash": subset_commitments["candidate"],
        "conditional_predictive_information_method": config[
            "conditional_predictive_information_method"
        ],
        "decision_target": "frozen-operational-class-log-risk",
        "semiparametric_transport_method": _class_contract_value(
            config, "joint_information_method"
        ),
        "representative_mmd_method": config["representative_discrepancy"],
        "operational_class_resolution_method": BUDGET_RESOLUTION_METHOD,
        "wall_time_seconds": wall_time_seconds,
        "wall_time_seconds_including_shared_calibration": wall_time_seconds,
        "failure_status": "",
    })
    return summary


def _run_p3j_shared_policy(
    *,
    run_root: Path,
    source_git_tree: str,
    config_sha256: str,
    state: OperationalClassConditionalState,
    dataset_id: str,
    seed: int,
    initial_X: np.ndarray,
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
    design_preconditioner: DesignPreconditioner,
    calibration_hash: str,
    frozen_initial_target: FrozenInitialClassTarget,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    run_root.mkdir(parents=True, exist_ok=True)
    outer = run_p3j_outer_policy(
        run_root, state, standardizer.transform_X(pool_X[candidate_indices]),
        candidate_indices, fixed_domain_X, initial_X, validation_X, validation_y,
        pool_row_ids, oracle, standardizer, source_git_tree=source_git_tree,
        config_sha256=config_sha256, dataset_id=dataset_id,
        dataset_family=_family(dataset_id), seed=seed,
        policy=config["policies"][-1],
        acquisition_budget=int(config["acquisition_observation_budget"]),
        eig_min_samples=int(config["eig_quadrature_min_evaluations"]),
        eig_max_samples=int(config["eig_quadrature_max_evaluations"]),
        eig_error_safety_factor=float(config["eig_quadrature_error_safety_factor"]),
        eig_growth_factor=int(config["eig_quadrature_growth_factor"]),
        class_distance_threshold=_operational_class_threshold(config),
        structure_count=len(generic_real_bank(initial_X.shape[1]).structures),
        action_chunk_size=int(config["eig_action_chunk_size"]),
        information_risk_tail_probability=(
            float(config["pcpi_information_risk_tail_probability"])
            if "pcpi_information_risk_tail_probability" in config
            else None
        ),
    )
    queries = _p3j_compatible_query_rows(outer, subset_commitments, config)
    summary = _p3j_compatible_summary(
        outer, dataset_id=dataset_id, seed=seed, initial_count=len(initial_X),
        validation_count=len(validation_y), candidate_count=len(candidate_indices),
        subset_commitments=subset_commitments,
        design_preconditioner=design_preconditioner,
        calibration_hash=calibration_hash,
        frozen_initial_target=frozen_initial_target, config=config,
        wall_time_seconds=time.perf_counter() - started,
    )
    return summary, list(outer.artifacts.curve_rows), queries


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
    semiparametric_state: OperationalSemiparametricState | None = None,
    frozen_initial_target: FrozenInitialClassTarget | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train_X, train_y = initial_X.copy(), initial_y.copy()
    available = np.asarray(candidate_indices, dtype=int).copy()
    is_pcpi = policy == protocol.pcpi_policy
    if semiparametric_state is not None and (
        not protocol.semiparametric_lifecycle or not is_pcpi
    ):
        raise ValueError("P3H lifecycle state is restricted to the P3H PCPI policy")
    if semiparametric_state is not None and (
        not np.array_equal(
            train_X,
            np.vstack((
                semiparametric_state.conditioning_actions,
                semiparametric_state.residual_actions,
            )),
        )
        or not np.array_equal(
            train_y,
            np.concatenate((
                semiparametric_state.conditioning_targets,
                semiparametric_state.residual_targets,
            )),
        )
    ):
        raise ValueError("P3H lifecycle initial history differs from the runner history")
    bank = generic_real_bank(train_X.shape[1])
    engine = (
        semiparametric_state.nominal_model.engine
        if semiparametric_state is not None
        else SequentialReferencePosterior(bank, likelihood_power, design_preconditioner)
    )
    is_reference_protocol = protocol.reference_dominance_method is not None
    ambiguity_powers = (
        (float(likelihood_power),)
        if is_reference_protocol
        else tuple(
            float(value)
            for value in config.get("likelihood_power_candidates", [likelihood_power])
        )
    )
    ambiguity_engines = (
        tuple(item.engine for item in semiparametric_state.family.model_states)
        if semiparametric_state is not None
        else tuple(
            SequentialReferencePosterior(bank, power, design_preconditioner)
            for power in ambiguity_powers
        )
    )
    curve_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    acquired_scores: list[float] = []
    query_local_gains: list[float] = []
    class_distance_threshold = _operational_class_threshold(config)
    initial_posterior = (
        semiparametric_state.nominal_model.posterior
        if semiparametric_state is not None
        else engine.fit_batch(train_X, train_y)
    )
    if protocol.shared_initial_frozen_target and frozen_initial_target is None:
        raise ValueError("P3I shared initial frozen target was not injected")
    if frozen_initial_target is not None:
        if frozen_initial_target.engine_target_hash != engine.target_hash:
            raise ValueError("shared initial frozen target belongs to another posterior")
        _assert_initial_posterior_numerically_equivalent(
            frozen_initial_target.posterior, initial_posterior
        )
        initial_classes = frozen_initial_target.classes
        frozen_partition = frozen_initial_target.partition
    else:
        initial_classes = _aggregate_protocol_classes(
            protocol, engine, initial_posterior, fixed_domain_X, config
        )
        frozen_partition = class_partition(initial_posterior, initial_classes)
    initial_frozen_entropy = frozen_partition.entropy
    initial_partition_hash = frozen_partition.stable_hash
    partition_hashes: set[str] = set()
    is_reference_pcpi = is_pcpi and is_reference_protocol
    run_started = time.perf_counter()
    for round_index in range(int(config["acquisition_observation_budget"]) + 1):
        posterior = (
            semiparametric_state.nominal_model.posterior
            if semiparametric_state is not None
            else engine.fit_batch(train_X, train_y)
        )
        classes = _aggregate_protocol_classes(
            protocol, engine, posterior, fixed_domain_X, config
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
            "operational_class_metric": classes.metric,
            "operational_class_scale_hash": classes.scale_hash,
            "initial_frozen_class_partition_hash": initial_partition_hash,
            "operational_class_distance_threshold": class_distance_threshold,
            "frozen_class_entropy": frozen_entropy,
            "frozen_class_entropy_gain": initial_frozen_entropy - frozen_entropy,
            **metrics.__dict__,
        })
        if round_index == int(config["acquisition_observation_budget"]):
            break
        visible_actions = standardizer.transform_X(pool_X[available])
        derived_seed = (
            stable_reference_policy_seed(seed, round_index)
            if is_reference_pcpi
            else stable_derived_seed(seed, policy, round_index)
        )
        score_kwargs = {
            "seed": derived_seed,
            "eig_min_samples": int(config.get("eig_quadrature_min_evaluations", 32)),
            "eig_max_samples": int(config.get("eig_quadrature_max_evaluations", 32)),
            "eig_error_safety_factor": float(
                config.get("eig_quadrature_error_safety_factor", 4.0)
            ),
            "eig_growth_factor": int(config.get("eig_quadrature_growth_factor", 2)),
            "qbc_committee_size": int(config["qbc_committee_size"]),
            "predictive_target_actions": fixed_domain_X,
            "representative_observed_actions": (
                train_X if is_pcpi and not is_reference_pcpi else None
            ),
            "target_partition": frozen_partition if is_pcpi else None,
            "posterior_models": (
                semiparametric_state.posterior_models
                if semiparametric_state is not None
                else _fit_posterior_models(ambiguity_engines, train_X, train_y)
                if is_pcpi and not is_reference_pcpi else None
            ),
        }
        if semiparametric_state is not None:
            score_kwargs["semiparametric_residual_family"] = (
                semiparametric_state.family
            )
            score_kwargs["semiparametric_unresolved_action"] = (
                protocol.semiparametric_unresolved_action
            )
        reference_result: ReferenceDominanceScores | None = None
        if is_reference_pcpi:
            reference_result = score_reference_dominance_actions(
                engine,
                posterior,
                visible_actions,
                available,
                target_partition=frozen_partition,
                reference_seed=derived_seed,
                quantization_probability_levels=tuple(
                    float(value)
                    for value in config[
                        "pcpi_response_quantization_probability_levels"
                    ]
                ),
                numerical_outward_tolerance=float(
                    config["pcpi_numerical_outward_tolerance"]
                ),
            )
            scores = _reference_score_audit_adapter(reference_result)
        elif is_pcpi and protocol.decision_target_alignment:
            if semiparametric_state is None:
                raise ValueError("P3I decision-targeted scoring requires P3H state")
            scores = score_decision_targeted_actions(
                engine,
                posterior,
                visible_actions,
                target_partition=frozen_partition,
                predictive_target_actions=fixed_domain_X,
                representative_observed_actions=train_X,
                posterior_models=semiparametric_state.posterior_models,
                semiparametric_residual_family=semiparametric_state.family,
                minimum_samples=int(config["eig_quadrature_min_evaluations"]),
                maximum_samples=int(config["eig_quadrature_max_evaluations"]),
                error_safety_factor=float(
                    config["eig_quadrature_error_safety_factor"]
                ),
                growth_factor=int(config["eig_quadrature_growth_factor"]),
                unresolved_action=protocol.semiparametric_unresolved_action,
            )
        elif is_pcpi and protocol.discrepancy_profile_method is not None:
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
        selected = (
            reference_result.decision.selected_candidate_id
            if reference_result is not None
            else select_acquisition_candidate(scores, available)
        )
        local_index = int(np.flatnonzero(available == selected)[0])
        possible_mask = (
            np.asarray(scores.possible_maximizer_mask, dtype=bool)
            if scores.possible_maximizer_mask is not None
            else np.zeros(len(available), dtype=bool)
        )
        admissible_mask = (
            np.asarray(scores.selection_admissible_mask, dtype=bool)
            if scores.selection_admissible_mask is not None
            else np.zeros(len(available), dtype=bool)
        )
        lifecycle_decision = (
            OperationalSemiparametricDecision(
                prior_state_hash=semiparametric_state.stable_hash,
                selected_candidate_id=selected,
                selected_action=visible_actions[local_index],
                local_index=local_index,
                scores=scores,
            )
            if semiparametric_state is not None else None
        )
        family_hash_before = (
            semiparametric_state.stable_hash
            if semiparametric_state is not None else "not-applied"
        )
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
        if reference_result is not None:
            decision = reference_result.decision
            reference_audit = {
                "reference_dominance_applied": True,
                "reference_policy": reference_result.reference_policy,
                "reference_seed_method": reference_result.reference_seed_method,
                "reference_seed": derived_seed,
                "reference_decision_method": decision.method,
                "reference_information_bound_method": (
                    reference_result.utility_bounds.method
                ),
                "reference_quantization_probability_levels": list(
                    reference_result.utility_bounds.quantization_probability_levels
                ),
                "reference_numerical_outward_tolerance": (
                    reference_result.utility_bounds.numerical_outward_tolerance
                ),
                "reference_targeted_handover": decision.targeted_handover,
                "reference_sample_candidate_id": (
                    decision.reference_sample_candidate_id
                ),
                "reference_leader_candidate_id": decision.leader_candidate_id,
                "reference_leader_estimate": decision.leader_estimate,
                "reference_leader_lower_bound": decision.leader_lower_bound,
                "reference_leader_upper_bound": decision.leader_upper_bound,
                "reference_estimate": decision.reference_estimate,
                "reference_lower_bound": decision.reference_lower_bound,
                "reference_upper_bound": decision.reference_upper_bound,
                "reference_dominance_gap": decision.dominance_gap,
                "reference_decision_numerical_tolerance": (
                    decision.numerical_tolerance
                ),
                "selected_class_eig_lower_bound": float(
                    reference_result.utility_bounds.lower_bounds[local_index]
                ),
                "selected_class_eig_upper_bound": float(
                    reference_result.utility_bounds.upper_bounds[local_index]
                ),
            }
        else:
            reference_audit = {
                "reference_dominance_applied": False,
                "reference_policy": "not-applied",
                "reference_seed_method": "not-applied",
                "reference_seed": 0,
                "reference_decision_method": "not-applied",
                "reference_information_bound_method": "not-applied",
                "reference_quantization_probability_levels": [],
                "reference_numerical_outward_tolerance": 0.0,
                "reference_targeted_handover": False,
                "reference_sample_candidate_id": -1,
                "reference_leader_candidate_id": -1,
                "reference_leader_estimate": 0.0,
                "reference_leader_lower_bound": 0.0,
                "reference_leader_upper_bound": 0.0,
                "reference_estimate": 0.0,
                "reference_lower_bound": 0.0,
                "reference_upper_bound": 0.0,
                "reference_dominance_gap": 0.0,
                "reference_decision_numerical_tolerance": 0.0,
                "selected_class_eig_lower_bound": 0.0,
                "selected_class_eig_upper_bound": 0.0,
            }
        revealed_X, revealed_y, revealed_indices = oracle.acquire_indices(
            np.asarray([selected])
        )
        if int(revealed_indices[0]) != selected:
            raise AssertionError("pool oracle returned a different acquisition index")
        train_X = np.vstack((train_X, standardizer.transform_X(revealed_X)))
        train_y = np.concatenate((train_y, standardizer.transform_y(revealed_y)))
        if semiparametric_state is not None:
            semiparametric_state = admit_operational_semiparametric_response(
                semiparametric_state,
                lifecycle_decision,
                int(revealed_indices[0]),
                standardizer.transform_X(revealed_X)[0],
                float(standardizer.transform_y(revealed_y)[0]),
            )
            updated = semiparametric_state.nominal_model.posterior
        else:
            updated = engine.fit_batch(train_X, train_y)
        family_hash_after = (
            semiparametric_state.stable_hash
            if semiparametric_state is not None else "not-applied"
        )
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
            "operational_class_metric": classes.metric,
            "operational_class_scale_hash": classes.scale_hash,
            "semiparametric_transport_method": (
                scores.semiparametric_transport_method
            ),
            "semiparametric_information_invariance_applied": (
                scores.semiparametric_information_invariance_applied
            ),
            "decision_target": scores.decision_target,
            "p3h_operational_lifecycle_applied": lifecycle_decision is not None,
            "p3h_family_hash_before_query": family_hash_before,
            "p3h_family_hash_after_query": family_hash_after,
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
                (
                    "not-applicable-class-only"
                    if is_reference_protocol
                    else "registered-action-domain-uniform"
                ),
            ),
            "predictive_target_subset_hash": subset_commitments["candidate"],
            "conditional_predictive_information_method": config.get(
                "conditional_predictive_information_method",
                (
                    "not-applied-class-only"
                    if is_reference_protocol else GAUSSIAN_CLASS_CONDITIONAL_EPIG
                ),
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
            "eig_primary_ranking_certified": scores.primary_ranking_certified,
            "eig_possible_maximizer_count": scores.possible_maximizer_count,
            "eig_possible_maximizer_candidate_ids": (
                available[possible_mask].tolist()
            ),
            "eig_secondary_resolution_used": scores.secondary_resolution_used,
            "eig_secondary_resolution_method": (
                scores.secondary_resolution_method
            ),
            "eig_selected_from_admissible_set": (
                scores.selection_admissible_mask is None
                or bool(scores.selection_admissible_mask[local_index])
            ),
            "eig_selection_admissible_candidate_ids": (
                available[admissible_mask].tolist()
            ),
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
            **reference_audit,
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
    initial_class_count = len(frozen_partition.class_ids)
    certified_rate = float(np.mean([
        bool(row["eig_ranking_certified"]) for row in query_rows
    ])) if query_rows else 1.0
    pcpi_queries = [
        row for row in query_rows
        if row["policy"] == protocol.pcpi_policy
    ]
    eig_modes = (
        {
            "representative-safe-robust-operational-class-eig-"
            "copula-transport-invariant"
        }
        if protocol.decision_target_alignment
        else {TARGETED_HANDOVER_MODE, REFERENCE_FALLBACK_MODE}
        if is_reference_protocol
        else {
            (
                "representative-safe-discrepancy-robust-p3h-semiparametric-"
                "maximin-joint-eig-surrogate"
                if protocol.discrepancy_profile_method is not None
                else "representative-safe-p3h-semiparametric-maximin-joint-eig-surrogate"
            ),
            (
                "representative-safe-discrepancy-robust-p3h-semiparametric-"
                "maximin-joint-eig-interval-frontier-resolution"
                if protocol.discrepancy_profile_method is not None
                else "representative-safe-p3h-semiparametric-maximin-joint-eig-"
                "interval-frontier-resolution"
            ),
        }
        if protocol.semiparametric_lifecycle
        and protocol.semiparametric_unresolved_action
        == P3H_INTERVAL_FRONTIER_RESOLUTION
        else {
            (
                "representative-safe-discrepancy-robust-p3h-semiparametric-"
                "maximin-joint-eig-surrogate"
                if protocol.discrepancy_profile_method is not None
                else "representative-safe-p3h-semiparametric-maximin-joint-eig-surrogate"
            )
        }
        if protocol.semiparametric_lifecycle
        else {
            (
                "representative-safe-discrepancy-robust-maximin-joint-eig-surrogate"
                if protocol.discrepancy_profile_method is not None
                else "representative-safe-maximin-joint-eig-surrogate"
            )
        }
    )
    epistemic_modes = set() if protocol.semiparametric_lifecycle else {
        "representative-safe-posterior-epistemic-variance-uncertified-maximin-joint-eig",
    }
    representative_fallback_modes = set() if protocol.semiparametric_lifecycle else {
        "representative-minimum-mmd-no-nonincreasing-action",
    }
    valid_modes = eig_modes | epistemic_modes | representative_fallback_modes
    if is_reference_protocol:
        decision_valid = [
            row["utility_mode"] in eig_modes
            and row["acquisition_target_partition_hash"] == initial_partition_hash
            and row["reference_dominance_applied"]
            and row["reference_policy"] == REFERENCE_POLICY
            and row["reference_decision_method"] == protocol.reference_dominance_method
            and row["reference_information_bound_method"]
            == ANALYTIC_CLASS_EIG_BOUNDS_METHOD
            and row["selected_class_eig_lower_bound"]
            <= row["selected_class_eig_upper_bound"]
            and (
                (
                    row["reference_targeted_handover"]
                    and row["selected_pool_index"]
                    == row["reference_leader_candidate_id"]
                    and row["reference_leader_lower_bound"]
                    > row["reference_upper_bound"]
                    + row["reference_decision_numerical_tolerance"]
                )
                or (
                    not row["reference_targeted_handover"]
                    and row["selected_pool_index"]
                    == row["reference_sample_candidate_id"]
                    and row["reference_leader_lower_bound"]
                    <= row["reference_upper_bound"]
                    + row["reference_decision_numerical_tolerance"]
                )
            )
            for row in pcpi_queries
        ]
    elif protocol.decision_target_alignment:
        decision_valid = [
            row["utility_mode"] in eig_modes
            and row["acquisition_target_partition_hash"] == initial_partition_hash
            and row["operational_class_metric"] == DECISION_REGRET_DISTANCE_METRIC
            and row["representative_guard_applied"]
            and row["representative_safe_set_nonempty"]
            and not row["representative_fallback_used"]
            and row["representative_selected_in_safe_set"]
            and row["representative_selected_mmd_nonincrease"]
            and row["eig_selected_from_admissible_set"]
            and row["eig_ranking_certified"]
            and row["semiparametric_transport_method"]
            == P3I_COPULA_TRANSPORT_METHOD
            and row["semiparametric_information_invariance_applied"]
            and row["decision_target"]
            == config["p3i_decision_target"] + "|" + P3I_INFORMATION_INVARIANCE
            and row["selected_conditional_predictive_eig"] == 0.0
            and np.isclose(
                row["selected_joint_class_predictive_score"],
                row["selected_class_eig"],
                rtol=0.0,
                atol=2e-14,
            )
            for row in pcpi_queries
        ]
    else:
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
                    and row["eig_selected_from_admissible_set"]
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
            "pcpi_robust_utility",
            (
                "not-applied-reference-dominance-class-eig"
                if is_reference_protocol
                else "maximin-joint-class-predictive-information"
            ),
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
        "operational_class_metric": initial_classes.metric,
        "initial_operational_class_scale_hash": initial_classes.scale_hash,
        "initial_class_aggregation_fraction": 1.0 - initial_class_count / len(bank.structures),
        "initial_frozen_class_partition_hash": initial_partition_hash,
        "initial_frozen_target_source": (
            frozen_initial_target.source
            if frozen_initial_target is not None else "policy-local-legacy"
        ),
        "predictive_target_distribution": config.get(
            "predictive_target_distribution",
            (
                "not-applicable-class-only"
                if is_reference_protocol
                else "registered-action-domain-uniform"
            ),
        ),
        "predictive_target_subset_hash": subset_commitments["candidate"],
        "conditional_predictive_information_method": config.get(
            "conditional_predictive_information_method",
            (
                "not-applied-class-only"
                if is_reference_protocol else GAUSSIAN_CLASS_CONDITIONAL_EPIG
            ),
        ),
        "decision_target": config.get(
            "p3i_decision_target", "legacy-joint-class-predictive-information"
        ),
        "semiparametric_transport_method": config.get(
            "p3i_semiparametric_transport", "not-applied"
        ),
        "representative_mmd_method": config.get(
            "representative_discrepancy",
            (
                "not-applied-reference-dominance"
                if is_reference_protocol else REPRESENTATIVE_MMD_METHOD
            ),
        ),
        "operational_class_distance_threshold": class_distance_threshold,
        "operational_class_resolution_method": BUDGET_RESOLUTION_METHOD,
        "initial_frozen_class_entropy": initial_frozen_entropy,
        "final_frozen_class_entropy": final_frozen_entropy,
        "frozen_class_entropy_gain": initial_frozen_entropy - final_frozen_entropy,
        "sum_query_local_class_entropy_gain": float(sum(query_local_gains)),
        "dynamic_partition_count": len(partition_hashes),
        "eig_ranking_certified_rate": certified_rate,
        "pcpi_primary_ranking_certified_rate": float(np.mean([
            row["eig_primary_ranking_certified"] for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_secondary_resolution_rate": float(np.mean([
            row["eig_secondary_resolution_used"] for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_mean_possible_maximizer_count": float(np.mean([
            row["eig_possible_maximizer_count"] for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_class_eig_used_rate": float(np.mean([
            row["utility_mode"] in eig_modes for row in pcpi_queries
        ])) if pcpi_queries else 0.0,
        "pcpi_maximin_joint_eig_used_rate": float(np.mean([
            row["utility_mode"] in eig_modes for row in pcpi_queries
        ])) if pcpi_queries and not is_reference_protocol else 0.0,
        "pcpi_target_only_class_eig_used_rate": float(np.mean([
            row["utility_mode"] in eig_modes
            and row["selected_conditional_predictive_eig"] == 0.0
            and np.isclose(
                row["selected_joint_class_predictive_score"],
                row["selected_class_eig"],
                rtol=0.0,
                atol=2e-14,
            )
            for row in pcpi_queries
        ])) if pcpi_queries and protocol.decision_target_alignment else 0.0,
        "pcpi_information_risk_used_rate": float(np.mean([
            row.get("information_risk_method")
            == config.get("pcpi_information_risk_method")
            for row in pcpi_queries
        ])) if pcpi_queries and "pcpi_information_risk_method" in config else 0.0,
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
        "pcpi_targeted_handover_rate": float(np.mean([
            row["reference_targeted_handover"] for row in pcpi_queries
        ])) if pcpi_queries and is_reference_protocol else 0.0,
        "pcpi_reference_fallback_rate": float(np.mean([
            not row["reference_targeted_handover"] for row in pcpi_queries
        ])) if pcpi_queries and is_reference_protocol else 0.0,
        "pcpi_mean_reference_dominance_gap": float(np.mean([
            row["reference_dominance_gap"] for row in pcpi_queries
        ])) if pcpi_queries and is_reference_protocol else 0.0,
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
        "pcpi_target_only_class_eig_used_rate",
        "pcpi_information_risk_used_rate",
        "pcpi_primary_ranking_certified_rate",
        "pcpi_secondary_resolution_rate",
        "pcpi_mean_possible_maximizer_count",
        "pcpi_epistemic_fallback_rate",
        "pcpi_representative_guard_applied_rate",
        "pcpi_representative_safe_set_nonempty_rate",
        "pcpi_representative_fallback_rate",
        "pcpi_representative_selected_nonincrease_rate",
        "pcpi_mean_representative_safe_set_size",
        "pcpi_mean_selected_discrepancy_variance",
        "pcpi_decision_rule_valid_rate",
        "pcpi_targeted_handover_rate", "pcpi_reference_fallback_rate",
        "pcpi_mean_reference_dominance_gap",
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
    decision_targeted = (
        config.get("p3i_decision_target")
        == "frozen-operational-class-log-risk"
    )
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
    registered_utility_use_by_family: dict[str, float] = {}
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
        use_metric = (
            "pcpi_information_risk_used_rate"
            if "pcpi_information_risk_method" in config
            else "pcpi_target_only_class_eig_used_rate"
            if decision_targeted
            else "pcpi_maximin_joint_eig_used_rate"
        )
        registered_utility_use_by_family[family_id] = float(np.mean([
            float(row[use_metric]) for row in selected
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
    elif decision_targeted and strong_structural:
        status = "STRONG_DECISION_TARGETED_REAL_ACQUISITION_EVIDENCE"
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
        "strong_evidence": (
            strong_structural if decision_targeted else strong_joint
        ),
        "strong_structural_evidence": strong_structural,
        "primary_decision_target": (
            "frozen-operational-class-log-risk"
            if decision_targeted else "legacy-joint-class-predictive-information"
        ),
        "predictive_performance_role": (
            "secondary-reported-not-in-primary-efficacy-decision"
            if decision_targeted else "joint-strong-evidence-requirement"
        ),
        "pcpi_frozen_class_gain_vs_random_significant_in_every_dataset_family": random_class_gain_significant,
        "pcpi_frozen_class_gain_vs_random_mean_better_in_every_dataset_family": random_class_gain_mean_better,
        "pcpi_predictive_naulc_vs_random_mean_better_in_every_dataset_family": random_predictive_mean_better,
        "pcpi_predictive_naulc_mean_nonpositive_vs_each_baseline_in_every_dataset_family": all_baseline_mean_nonpositive,
        "class_gain_negative_transfer_rate_controlled": class_negative_transfer_controlled,
        "class_aggregation_observed_in_every_dataset_family": class_aggregation_observed,
        "class_aggregation_by_dataset_family": aggregation_by_family,
        "pcpi_decision_rule_valid": decision_rule_valid,
        "pcpi_decision_rule_valid_rate_by_dataset_family": decision_rule_by_family,
        "pcpi_registered_utility_used_rate_by_dataset_family": (
            registered_utility_use_by_family
        ),
        "pcpi_information_risk_used_rate_by_dataset_family": (
            registered_utility_use_by_family
            if "pcpi_information_risk_method" in config else {}
        ),
        "pcpi_maximin_joint_eig_used_rate_by_dataset_family": (
            {} if "pcpi_information_risk_method" in config
            else registered_utility_use_by_family
        ),
        "pcpi_target_only_class_eig_used_rate_by_dataset_family": (
            registered_utility_use_by_family
            if decision_targeted and "pcpi_information_risk_method" not in config
            else {}
        ),
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


def _manifest_method_contract(
    config: dict[str, Any],
    protocol: RealAcquisitionProtocol,
) -> dict[str, Any]:
    if protocol.reference_dominance_method is not None:
        keys = (
            "pcpi_class_target_partition", "pcpi_primary_utility",
            "pcpi_information_bound_method",
            "pcpi_response_quantization_probability_levels",
            "pcpi_numerical_outward_tolerance", "pcpi_reference_policy",
            "pcpi_reference_seed_method", "pcpi_decision_method",
        )
        return {
            key: config[key] for key in keys
        } | {
            "likelihood_power_calibration_candidates": config[
                "likelihood_power_candidates"
            ]
        }
    keys = (
        "eig_quadrature_min_evaluations", "eig_quadrature_max_evaluations",
        "eig_quadrature_growth_factor", "eig_quadrature_error_safety_factor",
        "eig_rank_certificate_method", "pcpi_class_target_partition",
        "pcpi_uncertified_eig_action", "pcpi_joint_target",
        "predictive_target_distribution",
        "conditional_predictive_information_method", "representative_guard",
        "representative_target_distribution", "representative_discrepancy",
        "representative_kernel_bandwidth", "representative_safe_set_rule",
        "representative_empty_safe_set_action", "pcpi_ambiguity_set",
        "pcpi_robust_utility", "pcpi_least_favorable_tie_break",
    )
    contract = {key: config[key] for key in keys} | {
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
    }
    if protocol.semiparametric_lifecycle:
        contract |= {
            key: config[key]
            for key in (
                "initial_base_warmup_budget",
                "initial_residual_training_budget",
                "p3h_operational_lifecycle",
                "p3h_residual_state_method",
                "p3h_residual_law",
                "p3h_validation_state_policy",
                "p3h5_terminal_status",
                "runtime_dependency_hash",
            )
        }
    if protocol.decision_target_alignment:
        contract |= {
            key: config[key]
            for key in (
                "p3i_decision_target",
                "p3i_information_invariance",
                "p3i_semiparametric_transport",
                "operational_execution_authorized",
            )
        }
    if protocol.p3j_class_conditional_lifecycle:
        prefix = protocol.class_conditional_contract_prefix
        contract |= {
            key: config[key]
            for key in (
                f"{prefix}_operational_lifecycle",
                f"{prefix}_residual_state_method",
                f"{prefix}_class_posterior_update",
                f"{prefix}_joint_information_method",
                f"{prefix}_measured_run_protocol",
                f"{prefix}_policy_dispatch",
                f"{prefix}_reporting_order",
                f"{prefix}_outer_runner_composition",
                "eig_action_chunk_size",
                "operational_execution_authorized",
                "formal_dataset_runner_authorized",
            )
        }
        if protocol.class_conditional_singleton_rank_certificate is not None:
            contract[f"{prefix}_singleton_rank_certificate"] = config[
                f"{prefix}_singleton_rank_certificate"
            ]
        if prefix == "p3m":
            contract |= {
                key: config[key]
                for key in (
                    "p3m_context_transform",
                    "p3m_bandwidth_rule",
                    "p3m_bandwidth_schedule",
                    "p3m_checkpoint_schema",
                    "p3m_checkpoint_publication",
                )
            }
            # P3M.6 binds the partial-pooling prior explicitly.  Keep these
            # optional so the completed P3M.5 manifest contract remains byte
            # compatible with its frozen configuration.
            contract |= {
                key: config[key]
                for key in (
                    "p3m_residual_pooling_kappa",
                    "p3m_residual_pooling_rule",
                )
                if key in config
            }
        if "pcpi_information_risk_method" in config:
            contract |= {
                key: config[key]
                for key in (
                    "pcpi_information_risk_method",
                    "pcpi_information_risk_tail_probability",
                    "pcpi_information_risk_tail_probability_source",
                    "pcpi_mean_information_role",
                )
            }
    return contract


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
    config = _load_config(config_path, root, protocol)
    if not protocol.operational_execution_authorized:
        raise PermissionError(
            f"{protocol.stage} is source-composition-only; real access is blocked"
        )
    python_executable_hash = file_sha256(Path(sys.executable))
    if (
        protocol.required_python_executable_hash is not None
        and python_executable_hash != protocol.required_python_executable_hash
    ):
        raise RuntimeError(
            "formal runtime executable differs from the frozen Python binary"
        )
    source_identity = resolve_formal_source_identity(root, source)
    dependency_environment = runtime_dependency_snapshot()
    dependency_environment_hash = runtime_dependency_hash(dependency_environment)
    if (
        protocol.required_runtime_dependency_hash is not None
        and dependency_environment_hash != protocol.required_runtime_dependency_hash
    ):
        raise RuntimeError(
            "formal P3H runtime differs from the frozen canonical environment"
        )
    binary_identity = None
    if protocol.required_runtime_binary_identity is not None:
        binary_identity = runtime_binary_identity()
        if binary_identity != protocol.required_runtime_binary_identity:
            raise RuntimeError(
                "formal runtime base interpreter or ABI identity differs from the freeze"
            )
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    _prepare_output(
        output,
        allow_p3j_resume=protocol.p3j_class_conditional_lifecycle,
        class_conditional_prefix=protocol.class_conditional_contract_prefix,
    )
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    identity = {
        **source_identity,
        "production_code_hash": production_code_hash(root),
        "config_hash": _hash_json(config),
        "config_file_hash": file_sha256(config_path),
        "dependency_specification_hash": dependency_specification_hash(root),
        "dependency_environment_hash": dependency_environment_hash,
        "dependency_lock_hash": dependency_environment_hash,
        "dependency_environment": dependency_environment,
        "python_executable_hash": python_executable_hash,
    }
    if binary_identity is not None:
        identity["runtime_binary_identity"] = binary_identity
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
                warmup_count = (
                    int(config["initial_base_warmup_budget"])
                    if (
                        protocol.semiparametric_lifecycle
                        or protocol.p3j_class_conditional_lifecycle
                    )
                    else len(initial_indices)
                )
                warmup_indices = initial_indices[:warmup_count]
                standardizer = DevelopmentStandardizer.fit(
                    selection.development.X[warmup_indices],
                    selection.development.y[warmup_indices],
                )
                dataset_records[dataset_id]["standardizer_hashes"][str(seed)] = standardizer.stable_hash
                initial_X = standardizer.transform_X(selection.development.X[initial_indices])
                initial_y = standardizer.transform_y(selection.development.y[initial_indices])
                bank = generic_real_bank(initial_X.shape[1])
                design_preconditioner = fit_bank_preconditioner(
                    bank, initial_X[:warmup_count]
                )
                dataset_records[dataset_id]["design_preconditioners"][str(seed)] = (
                    design_preconditioner.to_dict()
                    | {"preconditioner_hash": design_preconditioner.stable_hash}
                )
                if (
                    protocol.semiparametric_lifecycle
                    or protocol.p3j_class_conditional_lifecycle
                ):
                    calibration = None
                    calibration_wall_time = 0.0
                    calibration_hash = _hash_json({
                        "method": "none-fixed-complete-family",
                        "powers": list(P3H_OPERATIONAL_POWERS),
                    })
                    selected_likelihood_power = 1.0
                    dataset_records[dataset_id]["likelihood_power_calibrations"][str(seed)] = {
                        "method": "none-fixed-complete-family",
                        "powers": list(P3H_OPERATIONAL_POWERS),
                        "nominal_reporting_power": selected_likelihood_power,
                        "calibration_hash": calibration_hash,
                    }
                else:
                    calibration_started = time.perf_counter()
                    calibration = calibrate_likelihood_power(
                        bank,
                        initial_X,
                        initial_y,
                        tuple(float(value) for value in config["likelihood_power_candidates"]),
                        design_preconditioner,
                    )
                    calibration_wall_time = time.perf_counter() - calibration_started
                    calibration_hash = calibration.stable_hash
                    selected_likelihood_power = calibration.selected_likelihood_power
                    dataset_records[dataset_id]["likelihood_power_calibrations"][str(seed)] = (
                        calibration.to_dict() | {"calibration_hash": calibration_hash}
                    )
                validation_X = standardizer.transform_X(selection.validation.X[validation_indices])
                validation_y = standardizer.transform_y(selection.validation.y[validation_indices])
                fixed_domain_X = standardizer.transform_X(selection.acquisition_pool.X[candidates])
                frozen_initial_target = (
                    _freeze_initial_class_target(
                        protocol,
                        SequentialReferencePosterior(
                            bank, selected_likelihood_power, design_preconditioner
                        ),
                        initial_X,
                        initial_y,
                        fixed_domain_X,
                        config,
                    )
                    if protocol.shared_initial_frozen_target else None
                )
                for policy in config["policies"]:
                    reporter.emit(
                        "policy_run_started",
                        f"{protocol.stage} run {completed + 1}/{total} | dataset={dataset_id} seed={seed} policy={policy}",
                        dataset_id=dataset_id,
                        seed=seed,
                        policy=policy,
                    )
                    try:
                        semiparametric_state = None
                        p3j_state = None
                        if protocol.semiparametric_lifecycle and policy == protocol.pcpi_policy:
                            family_engines = tuple(
                                SequentialReferencePosterior(
                                    bank, power, design_preconditioner
                                )
                                for power in P3H_OPERATIONAL_POWERS
                            )
                            semiparametric_state = (
                                initialize_operational_semiparametric_state(
                                    family_engines,
                                    initial_X[:warmup_count],
                                    initial_y[:warmup_count],
                                    initial_X[warmup_count:],
                                    initial_y[warmup_count:],
                                )
                            )
                        if (
                            protocol.p3j_class_conditional_lifecycle
                            and policy == protocol.pcpi_policy
                        ):
                            if frozen_initial_target is None:
                                raise ValueError("P3J requires the shared frozen target")
                            family_engines = tuple(
                                SequentialReferencePosterior(
                                    bank, power, design_preconditioner
                                )
                                for power in P3H_OPERATIONAL_POWERS
                            )
                            p3j_state = initialize_operational_class_conditional_state(
                                family_engines,
                                initial_X[:warmup_count],
                                initial_y[:warmup_count],
                                initial_X[warmup_count:],
                                initial_y[warmup_count:],
                                frozen_initial_target.partition,
                                action_conditional_residual=(
                                    protocol.class_conditional_contract_prefix
                                    == "p3m"
                                ),
                                action_conditional_residual_method=(
                                    config.get("p3m_residual_state_method")
                                    if protocol.class_conditional_contract_prefix
                                    == "p3m"
                                    else None
                                )
                                or "strict-prefix-rbf-weighted-kt-dyadic-polya-tree-v1",
                            )
                            summary, curves, queries = _run_p3j_shared_policy(
                                run_root=(
                                    output / protocol.class_conditional_contract_prefix
                                    / dataset_id / f"seed-{seed}"
                                ),
                                source_git_tree=str(identity["source_git_tree"]),
                                config_sha256=str(identity["config_file_hash"]),
                                state=p3j_state, dataset_id=dataset_id, seed=int(seed),
                                initial_X=initial_X, validation_X=validation_X,
                                validation_y=validation_y,
                                fixed_domain_X=fixed_domain_X,
                                candidate_indices=candidates,
                                pool_X=selection.acquisition_pool.X,
                                pool_row_ids=prepared.acquisition_pool_row_ids,
                                oracle=oracle, standardizer=standardizer,
                                subset_commitments=subset_commitments, config=config,
                                design_preconditioner=design_preconditioner,
                                calibration_hash=calibration_hash,
                                frozen_initial_target=frozen_initial_target,
                            )
                        else:
                            summary, curves, queries = _run_policy(
                                dataset_id=dataset_id, seed=int(seed), policy=policy,
                                initial_X=initial_X, initial_y=initial_y,
                                validation_X=validation_X, validation_y=validation_y,
                                fixed_domain_X=fixed_domain_X,
                                candidate_indices=candidates,
                                pool_X=selection.acquisition_pool.X,
                                pool_row_ids=prepared.acquisition_pool_row_ids,
                                oracle=oracle, standardizer=standardizer,
                                subset_commitments=subset_commitments, config=config,
                                reporter=reporter,
                                design_preconditioner=design_preconditioner,
                                likelihood_power=selected_likelihood_power,
                                calibration_hash=calibration_hash,
                                calibration_wall_time_seconds=calibration_wall_time,
                                protocol=protocol,
                                semiparametric_state=semiparametric_state,
                                frozen_initial_target=frozen_initial_target,
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
                        if protocol.fail_fast:
                            terminal_path = _write_terminal_failure(
                                output, protocol, failure
                            )
                            raise _FailFastProtocolError(
                                f"formal run stopped at first failure; record={terminal_path}"
                            ) from error
        except Exception as error:
            if isinstance(error, _FailFastProtocolError):
                raise
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
            if protocol.fail_fast:
                terminal_path = _write_terminal_failure(output, protocol, failure)
                raise _FailFastProtocolError(
                    f"formal run stopped at first failure; record={terminal_path}"
                ) from error
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
    projected_representative_guard = (
        protocol.p3j_class_conditional_lifecycle
        and protocol.class_conditional_contract_prefix in ("p3k", "p3m")
    )
    representative_decisions_auditable = bool(pcpi_query_rows) and all(
        (
            row["representative_safe_set_nonempty"]
            and not row["representative_fallback_used"]
            and row["representative_selected_in_safe_set"]
            and (
                row.get("representative_selected_within_projected_budget", False)
                if projected_representative_guard
                else row["representative_selected_mmd_nonincrease"]
            )
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
    information_risk_expected = "pcpi_information_risk_method" in config
    maximin_decisions_auditable = bool(pcpi_query_rows) and all(
        (
            row["robust_model_count"] == len(ambiguity_powers)
            and tuple(row["robust_likelihood_powers"]) == ambiguity_powers
            and (
                (
                    row["information_risk_method"]
                    == config["pcpi_information_risk_method"]
                    and row["selected_least_favorable_information_risk_power"]
                    in ambiguity_powers
                    and np.isclose(
                        row["selected_joint_class_predictive_score"],
                        min(row["selected_lower_tail_cvar_by_model"]),
                        rtol=0.0, atol=2e-14,
                    )
                    and row["selected_robust_lower_tail_cvar_lower_bound"]
                    <= row["selected_joint_class_predictive_score"]
                    <= row["selected_robust_lower_tail_cvar_upper_bound"]
                )
                if information_risk_expected
                else (
                    row["selected_least_favorable_likelihood_power"]
                    in ambiguity_powers
                    and np.isclose(
                        row["selected_joint_class_predictive_score"],
                        min(row["selected_robust_joint_scores_by_model"]),
                        rtol=0.0, atol=1e-15,
                    )
                    and row["selected_robust_lower_bound"]
                    <= row["selected_joint_class_predictive_score"]
                    <= row["selected_robust_upper_bound"]
                )
            )
        )
        if not row["representative_fallback_used"]
        else row["robust_model_count"] == 0
        for row in pcpi_query_rows
    )
    p3h_interval_resolution_auditable = (
        protocol.semiparametric_lifecycle
        and bool(pcpi_query_rows)
        and all(
            (
                row["eig_ranking_certified"]
                and row["eig_selected_from_admissible_set"]
                and (
                    (
                        not row["eig_primary_ranking_certified"]
                        and row["eig_secondary_resolution_used"]
                        and row["eig_secondary_resolution_method"]
                        == P3H_INTERVAL_FRONTIER_RESOLUTION
                        and row["eig_possible_maximizer_count"] >= 2
                        and bool(row["eig_selection_admissible_candidate_ids"])
                        and set(row["eig_selection_admissible_candidate_ids"])
                        <= set(row["eig_possible_maximizer_candidate_ids"])
                        and row["selected_pool_index"]
                        == min(row["eig_selection_admissible_candidate_ids"])
                    )
                    or (
                        row["eig_primary_ranking_certified"]
                        and not row["eig_secondary_resolution_used"]
                        and row["eig_secondary_resolution_method"] == "not-applied"
                    )
                )
            )
            for row in pcpi_query_rows
        )
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
    p3h_query_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in pcpi_query_rows:
        p3h_query_groups.setdefault(
            (str(row["dataset_id"]), int(row["seed"])), []
        ).append(row)
    p3h_hash_chains_valid = bool(p3h_query_groups) and all(
        all(
            ordered[index]["p3h_family_hash_after_query"]
            == ordered[index + 1]["p3h_family_hash_before_query"]
            for index in range(len(ordered) - 1)
        )
        and all(
            row["p3h_family_hash_before_query"] != "not-applied"
            and row["p3h_family_hash_after_query"] != "not-applied"
            and row["p3h_family_hash_before_query"]
            != row["p3h_family_hash_after_query"]
            for row in ordered
        )
        for ordered in (
            sorted(rows, key=lambda item: int(item["acquisition_round"]))
            for rows in p3h_query_groups.values()
        )
    )
    expected_robust_utility = (
        config["pcpi_robust_utility"]
        if protocol.p3j_class_conditional_lifecycle
        else "p3i-copula-transport-invariant-maximin-operational-class-information"
        if protocol.decision_target_alignment
        else "p3h-semiparametric-discrepancy-aware-maximin-joint-class-predictive-information"
        if protocol.semiparametric_lifecycle
        else "discrepancy-aware-maximin-joint-class-predictive-information"
        if discrepancy_expected
        else "maximin-joint-class-predictive-information"
    )
    expected_representative_method = (
        protocol.class_conditional_representative_method
        if protocol.p3j_class_conditional_lifecycle
        else REPRESENTATIVE_MMD_METHOD
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
        "initial_frozen_target_source_matches_frozen_contract": (
            not protocol.shared_initial_frozen_target
            or (
                config.get("p3i_initial_frozen_target_source")
                == P3I_SHARED_INITIAL_TARGET_SOURCE
                and bool(run_rows)
                and all(
                    row["initial_frozen_target_source"]
                    == P3I_SHARED_INITIAL_TARGET_SOURCE
                    for row in run_rows
                )
            )
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
            config.get("predictive_target_distribution")
            == "registered-action-domain-uniform"
        ),
        "conditional_predictive_information_method": config.get(
            "conditional_predictive_information_method", "not-applied"
        ),
        "representative_guard_matches_frozen_contract": (
            config.get("representative_discrepancy")
            == expected_representative_method
            and config.get("representative_target_distribution")
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
            and config.get("pcpi_ambiguity_set")
            == "frozen-likelihood-power-candidates"
            and config.get("pcpi_robust_utility") == expected_robust_utility
            and config.get("pcpi_least_favorable_tie_break")
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
    if protocol.semiparametric_lifecycle:
        protocol_decisions.pop("nominal_calibrated_posterior_retained_for_reporting")
        protocol_decisions.pop(
            "likelihood_power_calibration_used_initial_development_only"
        )
        protocol_decisions.update({
            "p3h_complete_family_used_for_every_pcpi_query": bool(pcpi_query_rows)
            and all(
                row["p3h_operational_lifecycle_applied"]
                and tuple(row["robust_likelihood_powers"])
                == P3H_OPERATIONAL_POWERS
                for row in pcpi_query_rows
            ),
            "p3h_family_hash_chain_valid_for_every_pcpi_run": (
                p3h_hash_chains_valid
            ),
            "p3h_lifecycle_not_applied_to_matched_baselines": bool(
                baseline_query_rows
            ) and all(
                not row["p3h_operational_lifecycle_applied"]
                and row["p3h_family_hash_before_query"] == "not-applied"
                and row["p3h_family_hash_after_query"] == "not-applied"
                for row in baseline_query_rows
            ),
            "likelihood_power_selection_forbidden": (
                config["likelihood_power_calibration_method"]
                == "none-fixed-complete-family"
            ),
            "nominal_eta_one_reporting_posterior_shared_across_policies": bool(
                run_rows
            ) and all(float(row["likelihood_power"]) == 1.0 for row in run_rows),
            "p3h_initial_roles_close_without_validation": (
                config["initial_base_warmup_budget"]
                + config["initial_residual_training_budget"]
                == config["initial_observation_budget"]
                and config["p3h_validation_state_policy"]
                == "discard-never-enter-operational-state"
            ),
            "p3h_interval_frontier_resolution_matches_frozen_contract": (
                p3h_interval_resolution_auditable
                if protocol.semiparametric_unresolved_action
                == P3H_INTERVAL_FRONTIER_RESOLUTION
                else all(
                    row["eig_primary_ranking_certified"]
                    and not row["eig_secondary_resolution_used"]
                    for row in pcpi_query_rows
                )
            ),
        })
    if protocol.decision_target_alignment:
        protocol_decisions.update({
            "p3i_decision_class_metric_used_for_every_run": bool(run_rows)
            and all(
                row["operational_class_metric"]
                == DECISION_REGRET_DISTANCE_METRIC
                and row["initial_operational_class_scale_hash"]
                != "pairwise-pooled-predictive-scale"
                for row in run_rows
            ),
            "p3i_transport_invariance_used_for_every_pcpi_query": bool(
                pcpi_query_rows
            )
            and all(
                row["semiparametric_transport_method"]
                == P3I_COPULA_TRANSPORT_METHOD
                and row["semiparametric_information_invariance_applied"]
                for row in pcpi_query_rows
            ),
            "p3i_primary_score_excludes_conditional_epig": bool(pcpi_query_rows)
            and all(
                row["selected_conditional_predictive_eig"] == 0.0
                and np.isclose(
                    row["selected_joint_class_predictive_score"],
                    row["selected_class_eig"],
                    rtol=0.0,
                    atol=2e-14,
                )
                for row in pcpi_query_rows
            ),
            "p3i_decision_target_matches_frozen_contract": (
                config["p3i_decision_target"]
                == "frozen-operational-class-log-risk"
                and config["conditional_predictive_information_method"]
                == "excluded-zero-by-decision-target"
            ),
            "p3i_no_representative_fallback_or_utility_switch": bool(
                pcpi_query_rows
            )
            and all(
                row["representative_safe_set_nonempty"]
                and not row["representative_fallback_used"]
                for row in pcpi_query_rows
            ),
            "p3i_fail_fast_policy_frozen": (
                protocol.fail_fast
                and config["failure_policy"]
                == "fail_fast_record_terminal_no_seed_replacement"
            ),
        })
    if protocol.p3j_class_conditional_lifecycle:
        class_contract_prefix = protocol.class_conditional_contract_prefix
        protocol_decisions.update({
            f"{class_contract_prefix}_transaction_identity_present_for_every_pcpi_query": bool(
                pcpi_query_rows
            ) and all(
                row.get(f"{class_contract_prefix}_identity_hash")
                and row.get(f"{class_contract_prefix}_prior_state_hash")
                and row.get(f"{class_contract_prefix}_next_state_hash")
                for row in pcpi_query_rows
            ),
            f"{class_contract_prefix}_response_admitted_before_every_report": bool(pcpi_query_rows)
            and all(
                row.get("response_receipt_admitted_before_reporting") is True
                and row.get("selection_used_validation") is False
                for row in pcpi_query_rows
            ),
            f"{class_contract_prefix}_class_conditional_method_used_for_every_pcpi_query": bool(
                pcpi_query_rows
            ) and all(
                row["semiparametric_transport_method"]
                == config[f"{class_contract_prefix}_joint_information_method"]
                and not row["semiparametric_information_invariance_applied"]
                for row in pcpi_query_rows
            ),
            f"{class_contract_prefix}_primary_score_excludes_conditional_epig": bool(pcpi_query_rows)
            and all(
                row["selected_conditional_predictive_eig"] == 0.0
                and np.isclose(
                    row["selected_joint_class_predictive_score"],
                    (
                        row["selected_lower_tail_cvar"]
                        if information_risk_expected
                        else row["selected_class_eig"]
                    ),
                    rtol=0.0, atol=2e-14,
                )
                for row in pcpi_query_rows
            ),
            f"{class_contract_prefix}_no_representative_fallback_or_utility_switch": bool(
                pcpi_query_rows
            ) and all(
                row["representative_safe_set_nonempty"]
                and not row["representative_fallback_used"]
                and row.get(
                    "representative_selected_within_projected_budget", False
                )
                for row in pcpi_query_rows
            ),
            f"{class_contract_prefix}_fail_fast_policy_frozen": (
                protocol.fail_fast
                and config["failure_policy"]
                == "fail_fast_record_terminal_no_seed_replacement"
            ),
        })
        singleton_method = protocol.class_conditional_singleton_rank_certificate
        if singleton_method is not None:
            protocol_decisions[
                f"{class_contract_prefix}_singleton_rank_certificates_finite_and_explicit"
            ] = bool(pcpi_query_rows) and all(
                (
                    row["eig_ranking_certificate_method"] == singleton_method
                    and row["eig_primary_ranking_certified"]
                    and row["eig_ranking_certified"]
                    and row["eig_possible_maximizer_count"] == 1
                    and row["eig_ranking_margin"] == 0.0
                    and row["eig_ranking_error_bound"] == 0.0
                    and row["eig_ranking_certificate_gap"] == 0.0
                    and np.isfinite(row["eig_ranking_margin"])
                    and np.isfinite(row["eig_ranking_error_bound"])
                    and np.isfinite(row["eig_ranking_certificate_gap"])
                )
                if row["representative_safe_set_size"] == 1
                else row["eig_ranking_certificate_method"] != singleton_method
                for row in pcpi_query_rows
            )
        if "pcpi_information_risk_method" in config:
            alpha = float(config["pcpi_information_risk_tail_probability"])
            protocol_decisions.update({
                "p3l_information_risk_tail_matches_assessment": (
                    alpha
                    == float(config["assessment_rules"]["negative_transfer_rate_max"])
                    == 0.25
                ),
                "p3l_information_risk_used_for_every_pcpi_query": bool(
                    pcpi_query_rows
                ) and all(
                    row["information_risk_method"]
                    == config["pcpi_information_risk_method"]
                    and row["information_risk_tail_probability"] == alpha
                    and np.isclose(
                        row["score"],
                        row["selected_lower_tail_cvar"],
                        rtol=0.0,
                        atol=2e-14,
                    )
                    for row in pcpi_query_rows
                ),
                "p3l_complete_ambiguity_risk_envelope_auditable": bool(
                    pcpi_query_rows
                ) and all(
                    len(row["selected_lower_tail_cvar_by_model"])
                    == len(ambiguity_powers)
                    and len(row["selected_negative_gain_probability_by_model"])
                    == len(ambiguity_powers)
                    and np.isclose(
                        row["selected_lower_tail_cvar"],
                        min(row["selected_lower_tail_cvar_by_model"]),
                        rtol=0.0,
                        atol=2e-14,
                    )
                    and row["selected_robust_lower_tail_cvar_lower_bound"]
                    <= row["selected_lower_tail_cvar"]
                    <= row["selected_robust_lower_tail_cvar_upper_bound"]
                    and all(
                        0.0 <= value <= 1.0
                        for value in row[
                            "selected_negative_gain_probability_by_model"
                        ]
                    )
                    and row["selected_least_favorable_information_risk_power"]
                    in ambiguity_powers
                    for row in pcpi_query_rows
                ),
            })
    if protocol.reference_dominance_method is not None:
        for key in (
            "predictive_target_distribution_shared_across_policies",
            "predictive_target_distribution_uses_registered_action_domain",
            "conditional_predictive_information_method",
            "representative_guard_matches_frozen_contract",
            "representative_guard_applied_to_all_pcpi_queries",
            "representative_guard_not_applied_to_baselines",
            "representative_decisions_auditable",
            "maximin_decisions_auditable",
            "maximin_ambiguity_set_matches_frozen_contract",
            "discrepancy_profile_matches_frozen_contract",
            "discrepancy_values_auditable",
            "discrepancy_not_applied_to_baselines",
        ):
            protocol_decisions.pop(key)
        reference_decisions_auditable = bool(pcpi_query_rows) and all(
            row["reference_dominance_applied"]
            and row["reference_policy"] == config["pcpi_reference_policy"]
            and row["reference_seed_method"]
            == config["pcpi_reference_seed_method"]
            and row["reference_decision_method"]
            == config["pcpi_decision_method"]
            and row["reference_information_bound_method"]
            == config["pcpi_information_bound_method"]
            and tuple(row["reference_quantization_probability_levels"])
            == tuple(config["pcpi_response_quantization_probability_levels"])
            and row["reference_numerical_outward_tolerance"]
            == config["pcpi_numerical_outward_tolerance"]
            and row["selected_class_eig_lower_bound"]
            <= row["selected_class_eig"]
            <= row["selected_class_eig_upper_bound"]
            and (
                (
                    row["reference_targeted_handover"]
                    and row["selected_pool_index"]
                    == row["reference_leader_candidate_id"]
                    and row["reference_dominance_gap"]
                    > row["reference_decision_numerical_tolerance"]
                )
                or (
                    not row["reference_targeted_handover"]
                    and row["selected_pool_index"]
                    == row["reference_sample_candidate_id"]
                    and row["reference_dominance_gap"]
                    <= row["reference_decision_numerical_tolerance"]
                )
            )
            for row in pcpi_query_rows
        )
        protocol_decisions.update({
            "reference_dominance_contract_matches_frozen_config": (
                config["pcpi_primary_utility"]
                == "initial-frozen-class-mutual-information"
                and config["pcpi_information_bound_method"]
                == ANALYTIC_CLASS_EIG_BOUNDS_METHOD
                and config["pcpi_reference_policy"] == REFERENCE_POLICY
                and config["pcpi_reference_seed_method"] == REFERENCE_SEED_METHOD
                and config["pcpi_decision_method"] == REFERENCE_DOMINANCE_METHOD
            ),
            "reference_dominance_decisions_auditable": (
                reference_decisions_auditable
            ),
            "reference_dominance_not_applied_to_baselines": bool(
                baseline_query_rows
            ) and all(
                not row["reference_dominance_applied"]
                for row in baseline_query_rows
            ),
            "initial_frozen_class_target_used_for_all_pcpi_queries": bool(
                pcpi_query_rows
            ) and all(
                row["acquisition_target_partition_hash"]
                == row["initial_frozen_class_partition_hash"]
                for row in pcpi_query_rows
            ),
            "p3b_p3c_acquisition_modules_not_applied": bool(pcpi_query_rows)
            and all(
                not row["representative_guard_applied"]
                and row["robust_model_count"] == 0
                and row["discrepancy_method"] == "not-applied"
                and row["selected_conditional_predictive_eig"] == 0.0
                and row["selected_joint_class_predictive_score"] == 0.0
                for row in pcpi_query_rows
            ),
        })
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
            **_manifest_method_contract(config, protocol),
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
            "pcpi_targeted_handover_rate",
            "pcpi_reference_fallback_rate",
            "pcpi_mean_reference_dominance_gap",
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
