"""Independent predictive contrast for an admitted iterative formula bank.

This compares a Full bank against the *same frozen engine bank* with no LLM
structures. It is an incremental bank contrast, not a second discovery run,
operational-class decision comparison, or measured acquisition experiment.
The reporting responses enter only after DiscoveryAgent.run has returned.
"""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset, SelectionData
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from .pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_discovery_target, model_factory_for_policy,
)
from .realized_drr import _independent_predictive_score


def _covariates(data):
    return {sha256(np.ascontiguousarray(row, dtype=np.float64).tobytes()).hexdigest()
            for row in data.X}


def paired_iterative_bank_reporting(
    result, selection: SelectionData, reporting: RoleDataset,
    cycle_roles: tuple[tuple[RoleDataset, RoleDataset], ...],
    prior: NormalInverseGammaPrior, measurement_budget: int,
):
    """Score each admitted bank and its frozen-engine ablation on one report set."""
    trace = result.system_evaluation.get("iterative_feedback_trace", ())
    gate = result.system_evaluation.get("iterative_feedback_gate") or {}
    # Abstention is an outcome to report, not a reason to drop a task.
    gate_problems = set(gate.get("problems", ()))
    if (not gate or gate_problems - {
                "no_admitted_posterior_update_reached_a_later_gap"}
            or len(trace) < 2 or len(trace) != len(cycle_roles)
            or reporting.role is not DataRole.VALIDATION
            or not isinstance(prior, NormalInverseGammaPrior)
            or type(measurement_budget) is not int or measurement_budget < 1
            or selection.acquisition_pool is None):
        raise ValueError("verified iterative trace and independent reporting required")
    roles = (selection.development, selection.validation,
             selection.acquisition_pool,
             *(role for pair in cycle_roles for role in pair))
    report_rows = _covariates(reporting)
    if any(report_rows & _covariates(role) for role in roles):
        raise ValueError("reporting responses overlap discovery or admission")
    for row, (gap, admission) in zip(trace, cycle_roles, strict=True):
        if (set(row["fit_rows"]) != selection.development.row_fingerprints
                or set(row["selection_rows"]) != selection.validation.row_fingerprints
                or set(row["gap_rows"]) != gap.row_fingerprints
                or set(row["admission_rows"]) != admission.row_fingerprints):
            raise ValueError("iterative trace is not bound to reporting roles")
    policy = EXPANDED_FORMULA_POLICY
    n_features = selection.development.X.shape[1]
    if reporting.X.shape[1] != n_features:
        raise ValueError("reporting feature dimension changed")

    def freeze(rows):
        identity = sha256(json.dumps(rows, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()
        model = model_factory_for_policy(policy)(
            rows, n_features=n_features, prior=prior,
            exploration_identity=identity, coefficient_policy=policy)
        target = freeze_discovery_target(
            model, selection.development, selection.acquisition_pool.X,
            measurement_budget=measurement_budget,
            expected_model_identity=model.stable_hash)
        return model, target

    baseline, baseline_target = freeze(trace[0]["core_rows_before"])
    if (baseline.stable_hash != trace[0]["bank_before"]
            or baseline_target.stable_hash != trace[0]["posterior_before"]):
        raise ValueError("engine-only bank identity changed")
    no_llm_mse, no_llm_log = _independent_predictive_score(
        baseline.engine(baseline.stable_hash),
        baseline_target.initial_posterior, reporting)
    cycles = []
    for index, row in enumerate(trace):
        full, target = freeze(row["bank_rows_after"])
        if (full.stable_hash != row["bank_after"]
                or target.stable_hash != row["posterior_after"]):
            raise ValueError("admitted bank identity changed before reporting")
        mse, log_score = _independent_predictive_score(
            full.engine(full.stable_hash), target.initial_posterior, reporting)
        cycles.append({"cycle": index, "full_mse": mse,
                       "no_llm_mse": no_llm_mse,
                       "no_llm_minus_full_mse": no_llm_mse - mse,
                       "full_log_score": log_score,
                       "no_llm_log_score": no_llm_log,
                       "full_minus_no_llm_log_score": log_score - no_llm_log,
                       "full_bank": full.stable_hash,
                       "no_llm_bank": baseline.stable_hash})
    return {"schema": "iterative-common-report-bank-contrast-v1",
            "reporting_identity": reporting.fingerprint,
            "development_identity": selection.development.fingerprint,
            "action_domain_identity": baseline_target.action_domain_identity,
            "measurement_budget": measurement_budget,
            "cycles": cycles,
            "iterative_feedback_witnessed": bool(gate.get("passed")),
            "reporting_excluded_from_generation_and_admission": True,
            "paired_bank_only": True,
            "measured_action_authorized": False}
