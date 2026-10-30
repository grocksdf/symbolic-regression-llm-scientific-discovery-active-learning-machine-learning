"""Protected direct admission with a response-isolated quarantine appeal.

Quarantine grants no posterior authority.  A candidate that misses the
original direct Gate may receive one additional, disjoint evidence batch.
The final statistic is the fixed 50/50 e-value mixture

    0.5 * E1 + 0.5 * E1 * E2,

which preserves the original direct threshold while allowing independent
appeal evidence to rescue a candidate.  Candidate multiplicity is handled by
the same K/alpha Bonferroni threshold as the direct Gate.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior

from .candidate_region_expansion import FrozenAxisRegions
from .pcpi_adapter import support_parser_for_policy
from .regional_candidate_admission import (
    CLOSED_POLICY,
    _fixed_predictive_logpdf,
    global_protected_evidence_decision,
)


def staged_global_protected_evidence_decision(
    first_log_ratios: np.ndarray,
    first_assignments: np.ndarray,
    appeal_log_ratios: np.ndarray,
    appeal_assignments: np.ndarray,
    eligible_regions: tuple[int, ...],
    threshold: float,
) -> dict:
    """Apply direct evidence, then an e-value-mixture appeal if eligible."""
    first = np.asarray(first_log_ratios, dtype=float).reshape(-1)
    second = np.asarray(appeal_log_ratios, dtype=float).reshape(-1)
    first_labels = np.asarray(first_assignments, dtype=int).reshape(-1)
    second_labels = np.asarray(appeal_assignments, dtype=int).reshape(-1)
    regions = tuple(int(value) for value in eligible_regions)
    if (first.shape != first_labels.shape
            or second.shape != second_labels.shape
            or not len(first) or not len(second)
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))
            or not np.isfinite(threshold)):
        raise ValueError("staged evidence requires finite aligned stages")
    direct = global_protected_evidence_decision(
        first, first_labels, regions, threshold)
    if direct["passed"]:
        return {
            **direct,
            "passed": True,
            "stage": "direct",
            "quarantined": False,
            "appeal_opened": False,
            "first_stage_global_log_evalue": float(np.sum(first)),
            "appeal_global_log_evalue": None,
            "mixture_global_log_evalue": float(np.sum(first)),
        }
    first_global = float(np.sum(first))
    if first_global <= 0.0:
        return {
            **direct,
            "passed": False,
            "stage": "rejected-before-appeal",
            "quarantined": False,
            "appeal_opened": False,
            "first_stage_global_log_evalue": first_global,
            "appeal_global_log_evalue": None,
            "mixture_global_log_evalue": first_global,
            "reason": "nonpositive-first-stage-global-evidence",
        }
    regional = []
    for region in regions:
        first_mask = first_labels == region
        second_mask = second_labels == region
        if not np.any(first_mask) or not np.any(second_mask):
            return {
                **direct,
                "passed": False,
                "stage": "appeal",
                "quarantined": True,
                "appeal_opened": True,
                "first_stage_global_log_evalue": first_global,
                "appeal_global_log_evalue": float(np.sum(second)),
                "mixture_global_log_evalue": None,
                "reason": "empty-eligible-appeal-region",
            }
        regional.append(float(
            np.sum(first[first_mask]) + np.sum(second[second_mask])))
    appeal_global = float(np.sum(second))
    mixture_global = float(np.logaddexp(
        math.log(.5) + first_global,
        math.log(.5) + first_global + appeal_global))
    minimum = float(min(regional))
    passed = bool(mixture_global > threshold and minimum >= 0.0)
    return {
        "passed": passed,
        "stage": "appeal",
        "quarantined": True,
        "appeal_opened": True,
        "first_stage_global_log_evalue": first_global,
        "appeal_global_log_evalue": appeal_global,
        "mixture_global_log_evalue": mixture_global,
        "region_log_predictive_ratios": regional,
        "minimum_region_log_predictive_ratio": minimum,
        "log_evidence_threshold": float(threshold),
        "reason": (
            "independent-staged-global-and-region-safe-evidence"
            if passed else
            "insufficient-staged-global-or-region-safe-evidence"),
    }


def admit_global_protected_with_quarantine(
    engine_rows, candidate_rows, fit: RoleDataset,
    gap_audit: RoleDataset, admission: RoleDataset,
    appeal: RoleDataset, domain: np.ndarray,
    regions: FrozenAxisRegions, eligible_regions: tuple[int, ...],
    prior: NormalInverseGammaPrior, measurement_budget: int,
    *, alpha: float = .05, coefficient_policy: str = CLOSED_POLICY,
) -> tuple[list[dict], dict]:
    """Admit directly or after one disjoint quarantine appeal."""
    roles = (fit, gap_audit, admission, appeal)
    if (fit.role is not DataRole.DEVELOPMENT
            or any(role.role is not DataRole.VALIDATION for role in roles[1:])
            or any(left.row_fingerprints & right.row_fingerprints
                   for index, left in enumerate(roles)
                   for right in roles[index + 1:])
            or not 0 < alpha < 1
            or regions.n_features != fit.X.shape[1]
            or not eligible_regions):
        raise ValueError(
            "staged protected admission needs four disjoint registered roles")
    core = [dict(row) for row in engine_rows]
    proposals = [dict(row) for row in candidate_rows]
    if not core or not proposals:
        return [], {
            "schema": "candidate-staged-global-protected-admission-v1",
            "attempts": len(proposals), "candidates": [],
            "direct_admitted_count": 0, "quarantine_count": 0,
            "appeal_admitted_count": 0,
            "candidate_response_accessed": False,
            "heldout_opened": False}
    parser = support_parser_for_policy(coefficient_policy)
    core_supports = {
        parser(row["expression"], fit.X.shape[1]) for row in core}
    core_first, core_identity = _fixed_predictive_logpdf(
        core, fit, admission, domain, prior, measurement_budget,
        coefficient_policy=coefficient_policy)
    core_appeal, appeal_core_identity = _fixed_predictive_logpdf(
        core, fit, appeal, domain, prior, measurement_budget,
        coefficient_policy=coefficient_policy)
    first_assignments = regions.assign(admission.X)
    appeal_assignments = regions.assign(appeal.X)
    threshold = math.log(len(proposals) / alpha)
    retained, records, admitted_supports = [], [], set()
    for candidate in proposals:
        support = parser(candidate["expression"], fit.X.shape[1])
        common = {
            "candidate_identity": sha256(json.dumps(
                candidate, sort_keys=True, default=str).encode()).hexdigest(),
            "lineage_id": str(candidate.get("lineage_id") or ""),
            "composed_expression": candidate["expression"],
            "support": list(support),
            "attempted_candidate_count": len(proposals),
            "eligible_regions": list(eligible_regions),
            "region_identity": regions.identity,
        }
        if support in core_supports:
            records.append({
                **common, "passed": False, "admitted": False,
                "stage": "rejected-before-appeal",
                "quarantined": False, "appeal_opened": False,
                "reason": "duplicate-engine-support"})
            continue
        candidate_first, first_identity = _fixed_predictive_logpdf(
            [candidate], fit, admission, domain, prior,
            measurement_budget, coefficient_policy=coefficient_policy)
        first_ratio = candidate_first - core_first
        direct = global_protected_evidence_decision(
            first_ratio, first_assignments,
            eligible_regions, threshold)
        if direct["passed"]:
            decision = staged_global_protected_evidence_decision(
                first_ratio, first_assignments,
                np.zeros_like(core_appeal), appeal_assignments,
                eligible_regions, threshold)
            appeal_identity = None
        elif float(np.sum(first_ratio)) > 0.0:
            candidate_appeal, appeal_identity = _fixed_predictive_logpdf(
                [candidate], fit, appeal, domain, prior,
                measurement_budget, coefficient_policy=coefficient_policy)
            decision = staged_global_protected_evidence_decision(
                first_ratio, first_assignments,
                candidate_appeal - core_appeal, appeal_assignments,
                eligible_regions, threshold)
        else:
            decision = staged_global_protected_evidence_decision(
                first_ratio, first_assignments,
                np.zeros_like(core_appeal), appeal_assignments,
                eligible_regions, threshold)
            appeal_identity = None
        duplicate = decision["passed"] and support in admitted_supports
        passed = bool(decision["passed"] and not duplicate)
        if passed:
            retained.append(candidate)
            admitted_supports.add(support)
        records.append({
            **common, **decision,
            "first_stage_expert_identity": first_identity,
            "appeal_expert_identity": appeal_identity,
            "passed": passed, "admitted": passed,
            "reason": (
                "duplicate-admitted-support" if duplicate
                else decision["reason"])})
    return retained, {
        "schema": "candidate-staged-global-protected-admission-v1",
        "core_identity": core_identity,
        "appeal_core_identity": appeal_core_identity,
        "admission_identity": admission.fingerprint,
        "appeal_identity": appeal.fingerprint,
        "gap_identity": gap_audit.fingerprint,
        "region_identity": regions.identity,
        "coefficient_policy": coefficient_policy,
        "attempts": len(proposals),
        "candidates": records,
        "direct_admitted_count": sum(
            row["admitted"] and row["stage"] == "direct"
            for row in records),
        "quarantine_count": sum(row["quarantined"] for row in records),
        "appeal_admitted_count": sum(
            row["admitted"] and row["stage"] == "appeal"
            for row in records),
        "decision_rule": (
            "direct E1 or fixed-half-mixture 0.5*E1+0.5*E1*E2 "
            "above K/alpha, with nonnegative evidence in every region"),
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }
