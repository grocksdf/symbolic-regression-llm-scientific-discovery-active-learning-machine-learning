"""Independent, regionwise undercoverage screen of a frozen bank.

The screen is a development-time search signal, never an admission decision
or an estimate of realized action value. Audit outcomes must be disjoint from
all fitting and discovery selection outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, t as student_t

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition

from .candidate_region_expansion import FrozenAxisRegions
from .pcpi_adapter import FrozenDiscoveryModel, FrozenDiscoveryTarget
from .posterior_gap_diagnosis import FrozenBankDiagnosis


@dataclass(frozen=True)
class RegionalAdequacy:
    region: int
    count: int
    tail_misses: int
    tail_miss_lower_bound: float
    insufficient_coverage_evidence: bool


@dataclass(frozen=True)
class IndependentGapEvidence:
    target_identity: str
    audit_identity: str
    region_identity: str
    rows: tuple[RegionalAdequacy, ...]
    alpha: float
    tail_probability: float
    audit_role: str = "dedicated-development-gap-audit"

    @property
    def eligible_regions(self) -> tuple[int, ...]:
        return tuple(row.region for row in self.rows
                     if row.insufficient_coverage_evidence)

    def prompt_brief(self, diagnosis: FrozenBankDiagnosis) -> dict[str, object]:
        if (self.target_identity != diagnosis.target_identity
                or self.region_identity != diagnosis.region_identity):
            raise ValueError("gap evidence and posterior diagnosis changed identity")
        eligible = set(self.eligible_regions)
        selected = [row for row in diagnosis.rows if row.region in eligible]
        payload = diagnosis.proposal_brief(limit=max(1, len(diagnosis.rows)))
        payload["regions"] = [row for row in payload["regions"]
                              if row["region"] in eligible]
        payload["independent_adequacy"] = [
            {"region": row.region, "count": row.count,
             "tail_misses": row.tail_misses,
             "lower_bound": row.tail_miss_lower_bound}
            for row in self.rows if row.region in eligible]
        payload["external_adequacy_checked"] = True
        payload["audit_identity"] = self.audit_identity
        payload["propose_allowed"] = bool(selected)
        payload["interpretation"] = (
            "Independent development responses show excess 90% predictive "
            "interval misses in these regions after a simultaneous one-sided "
            "screen. This indicates model predictive inadequacy here; it "
            "does not identify a missing symbolic interaction or certify an "
            "LLM candidate or action."
        )
        return payload


def screen_independent_adequacy(
    model: FrozenDiscoveryModel,
    target: FrozenDiscoveryTarget,
    diagnosis: FrozenBankDiagnosis,
    regions: FrozenAxisRegions,
    audit: RoleDataset,
    *, discovery_development: RoleDataset,
    discovery_validation: RoleDataset,
    alpha: float = .05,
    tail_probability: float = .10,
) -> IndependentGapEvidence:
    """One-sided exact binomial undercoverage tests with Bonferroni regions.

    Conditional on a frozen calibrated predictive distribution and independent
    audit responses, tail indicators have nominal probability 0.1. Selection
    of the bank on the audit, covariate shift, and distribution misspecification
    invalidate the null; the caller must guarantee split provenance.
    """
    if (audit.role is not DataRole.VALIDATION
            or discovery_development.role is not DataRole.DEVELOPMENT
            or discovery_validation.role is not DataRole.VALIDATION
            or audit.row_fingerprints & discovery_development.row_fingerprints
            or audit.row_fingerprints & discovery_validation.row_fingerprints
            or audit.fingerprint in {discovery_development.fingerprint,
                                     discovery_validation.fingerprint}):
        raise ValueError("gap audit must be a disjoint validation role")
    if (target.stable_hash != diagnosis.target_identity
            or model.stable_hash != diagnosis.model_identity
            or regions.identity != diagnosis.region_identity):
        raise ValueError("gap audit changed frozen model, target, or regions")
    if (not 0 < alpha < 1 or not 0 < tail_probability < 1
            or audit.X.shape[1] != model.n_features):
        raise ValueError("invalid registered adequacy screen")
    components = predictive_components_for_partition(
        model.engine(model.stable_hash), target.initial_posterior,
        target.partition, audit.X)
    cdf = np.sum(components.structure_probabilities[:, None] * student_t.cdf(
        (audit.y[None, :] - components.locations) / components.scales,
        df=components.degrees_freedom[:, None]), axis=0)
    if not np.all(np.isfinite(cdf)):
        raise ValueError("nonfinite frozen-bank predictive CDF")
    misses = (cdf < tail_probability / 2) | (cdf > 1 - tail_probability / 2)
    assigned = regions.assign(audit.X)
    rows = []
    for region in range(len(regions.cuts) + 1):
        mask = assigned == region
        count, errors = int(mask.sum()), int(np.sum(misses[mask]))
        lower = (float(beta.ppf(alpha / (len(regions.cuts) + 1),
                                errors, count - errors + 1))
                 if errors else 0.)
        rows.append(RegionalAdequacy(
            region, count, errors, lower, bool(lower > tail_probability)))
    return IndependentGapEvidence(target.stable_hash, audit.fingerprint,
                                  regions.identity, tuple(rows),
                                  float(alpha), float(tail_probability))
