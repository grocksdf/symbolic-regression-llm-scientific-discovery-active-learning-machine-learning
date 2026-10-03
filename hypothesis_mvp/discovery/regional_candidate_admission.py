"""Independent candidate-specific region evidence for quality-first discovery.

The selector is a modular fixed-predictive-expert model, not the PCPI law
posterior. Screening never consumes acquisition or sealed responses.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior

from .candidate_region_expansion import FrozenAxisRegions, RegionSelectorPosterior
from .pcpi_adapter import (
    freeze_discovery_target, model_factory_for_policy, support_parser_for_policy,
)


CLOSED_POLICY = "discard-fitted-coefficients-refit-closed-basis"


def _fixed_predictive_logpdf(candidates, fit, observations, domain,
                             prior, measurement_budget, *,
                             coefficient_policy=CLOSED_POLICY):
    identity = sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()
    model = model_factory_for_policy(coefficient_policy)(
        candidates, n_features=fit.X.shape[1], prior=prior,
        exploration_identity=identity,
        coefficient_policy=coefficient_policy,
        minimum_supports=1)
    target = freeze_discovery_target(
        model, fit, domain, measurement_budget=measurement_budget,
        expected_model_identity=model.stable_hash)
    components = predictive_components_for_partition(
        model.engine(model.stable_hash), target.initial_posterior,
        target.partition, observations.X)
    component_likelihood = student_t.logpdf(
        observations.y[None, :],
        df=components.degrees_freedom[:, None],
        loc=components.locations, scale=components.scales)
    values = logsumexp(
        np.log(components.structure_probabilities[:, None])
        + component_likelihood, axis=0)
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite fixed-expert predictive scores")
    return values, model.stable_hash


def admit_regional_candidates(
    engine_rows, llm_rows, fit: RoleDataset, gap_audit: RoleDataset,
    admission: RoleDataset, domain: np.ndarray, regions: FrozenAxisRegions,
    eligible_regions: tuple[int, ...], prior: NormalInverseGammaPrior,
    measurement_budget: int, *, alpha: float = .05,
    coefficient_policy: str = CLOSED_POLICY,
) -> tuple[list[dict], dict]:
    """Bonferroni evidence ratios for independent fixed-expert predictive laws.

    Under the declared fixed core predictive joint law, the optional/core
    likelihood ratio has expectation one. The threshold K*R/alpha accounts
    for all attempted candidates and eligible regions, including failures.
    It is not a frequentist guarantee when the core law is misspecified.
    """
    if (fit.role is not DataRole.DEVELOPMENT
            or gap_audit.role is not DataRole.VALIDATION
            or admission.role is not DataRole.VALIDATION
            or admission.row_fingerprints & (
                fit.row_fingerprints | gap_audit.row_fingerprints)
            or gap_audit.row_fingerprints & fit.row_fingerprints
            or not 0 < alpha < 1
            or regions.n_features != fit.X.shape[1]
            or not eligible_regions
            or any(r < 0 or r > len(regions.cuts) for r in eligible_regions)
            or len(set(eligible_regions)) != len(eligible_regions)):
        raise ValueError("regional admission needs disjoint fit, gap and admission roles")
    core = [dict(row) for row in engine_rows]
    proposals = [dict(row) for row in llm_rows]
    if not core or not proposals:
        return [], {"schema": "candidate-regional-admission-v1",
                    "attempts": len(proposals), "candidates": [],
                    "admission_identity": admission.fingerprint,
                    "candidate_response_accessed": False,
                    "heldout_opened": False}
    parser = support_parser_for_policy(coefficient_policy)
    core_supports = {parser(row["expression"], fit.X.shape[1])
                     for row in core}
    core_score, core_identity = _fixed_predictive_logpdf(
        core, fit, admission, domain, prior, measurement_budget,
        coefficient_policy=coefficient_policy)
    assignments = regions.assign(admission.X)
    threshold = math.log(len(proposals) * len(eligible_regions) / alpha)
    retained, results = [], []
    for candidate in proposals:
        support = parser(candidate["expression"], fit.X.shape[1])
        identity = sha256(json.dumps(candidate, sort_keys=True,
                                      default=str).encode()).hexdigest()
        if support in core_supports:
            results.append({"candidate_identity": identity,
                            "lineage_id": str(candidate.get("lineage_id") or ""),
                            "composed_expression": candidate["expression"],
                            "support": list(support),
                            "attempted_candidate_count": len(proposals),
                            "region_identity": regions.identity,
                            "reason": "duplicate-engine-support", "admitted": False})
            continue
        optional_score, law_identity = _fixed_predictive_logpdf(
            [candidate], fit, admission, domain, prior, measurement_budget,
            coefficient_policy=coefficient_policy)
        selector = RegionSelectorPosterior.from_partition(
            regions, ("core", identity),
            np.full((len(regions.cuts) + 1, 2), .5))
        selector = selector.update(
            assignments, np.column_stack([core_score, optional_score]))
        factors = selector.log_bayes_factor_vs_core(identity)
        passed = any(np.isfinite(factors[r]) and factors[r] > threshold
                     for r in eligible_regions)
        if passed:
            retained.append(candidate)
        results.append({"candidate_identity": identity,
                        "lineage_id": str(candidate.get("lineage_id") or ""),
                        "composed_expression": candidate["expression"],
                        "support": list(support),
                        "attempted_candidate_count": len(proposals),
                        "region_identity": regions.identity,
                        "expert_identity": law_identity,
                        "region_log_predictive_ratios": factors.tolist(),
                        "region_selector_mass": selector.probabilities[:, 1].tolist(),
                        "eligible_regions": list(eligible_regions),
                        "log_evidence_threshold": threshold,
                        "admitted": passed,
                        "reason": "independent-region-evidence" if passed
                                  else "insufficient-independent-region-evidence"})
    return retained, {"schema": "candidate-regional-admission-v1",
                      "core_identity": core_identity,
                      "region_identity": regions.identity,
                      "admission_identity": admission.fingerprint,
                      "gap_identity": gap_audit.fingerprint,
                      "coefficient_policy": coefficient_policy,
                      "attempts": len(proposals), "candidates": results,
                      "decision_contribution_assessed": False,
                      "candidate_response_accessed": False,
                      "heldout_opened": False}
