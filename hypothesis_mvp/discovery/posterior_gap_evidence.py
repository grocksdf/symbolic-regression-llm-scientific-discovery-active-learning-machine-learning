"""Independent, regionwise undercoverage screen of a frozen bank.

The screen is a development-time search signal, never an admission decision
or an estimate of realized action value. Audit outcomes must be disjoint from
all fitting and discovery selection outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
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
    pit_above_median: int = 0
    pit_direction_lower_bound: float = 0.0
    localized_bias_evidence: bool = False
    reference_log_score_advantage: float | None = None
    score_degradation_evidence: bool = False


@dataclass(frozen=True)
class IndependentGapEvidence:
    target_identity: str
    audit_identity: str
    region_identity: str
    rows: tuple[RegionalAdequacy, ...]
    alpha: float
    tail_probability: float
    score_reference_identity: str = ""
    audit_role: str = "dedicated-development-gap-audit"
    audit_row_count: int = 0
    undefined_row_count: int = 0

    @property
    def evaluable_row_count(self) -> int:
        return self.audit_row_count - self.undefined_row_count

    @property
    def eligible_regions(self) -> tuple[int, ...]:
        return tuple(row.region for row in self.rows
                     if row.insufficient_coverage_evidence
                     or row.localized_bias_evidence
                     or row.score_degradation_evidence)

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
             "lower_bound": row.tail_miss_lower_bound,
             "undercoverage": row.insufficient_coverage_evidence,
             "pit_above_median": row.pit_above_median,
             "directional_pit_lower_bound": row.pit_direction_lower_bound,
             "directional_pit_imbalance": row.localized_bias_evidence,
             "reference_log_score_advantage": row.reference_log_score_advantage,
             "score_degradation": row.score_degradation_evidence}
            for row in self.rows if row.region in eligible]
        payload["external_adequacy_checked"] = True
        payload["audit_identity"] = self.audit_identity
        payload["audit_row_count"] = self.audit_row_count
        payload["undefined_row_count"] = self.undefined_row_count
        payload["score_reference_identity"] = self.score_reference_identity
        payload["propose_allowed"] = bool(selected)
        payload["interpretation"] = (
            "Independent development responses show a regionwise coverage, "
            "directional PIT, or frozen-reference log-score gap under a "
            "simultaneous screen. These are model-conditional search cues; "
            "they do not identify a missing symbolic interaction or certify "
            "an LLM candidate or action."
        )
        return payload


def evaluable_predictive_components(
    model: FrozenDiscoveryModel, posterior, partition, values: np.ndarray,
) -> tuple[object, np.ndarray]:
    """Predictive components together with the rows the frozen bank defines.

    The bank is registered on the acquisition domain. An audit row may fall
    outside that domain, where a banked structure is singular, for example a
    division by a feature that attains zero there. Such a row has no defined
    predictive law under the frozen bank: it is excluded from the numeric
    screen and counted, never silently scored as an ordinary observation.
    """
    values = np.asarray(values, dtype=float)
    engine = model.engine(model.stable_hash)
    try:
        components = predictive_components_for_partition(
            engine, posterior, partition, values)
        return components, np.ones(len(values), dtype=bool)
    except ValueError:
        mask = np.zeros(len(values), dtype=bool)
        for index in range(len(values)):
            try:
                predictive_components_for_partition(
                    engine, posterior, partition, values[index:index + 1])
            except ValueError:
                continue
            mask[index] = True
        if not mask.any():
            raise ValueError("frozen bank undefined on every audit row")
        components = predictive_components_for_partition(
            engine, posterior, partition, values[mask])
        return components, mask


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
    score_reference_model: FrozenDiscoveryModel | None = None,
    score_reference_target: FrozenDiscoveryTarget | None = None,
) -> IndependentGapEvidence:
    """Three independently screened, model-conditional regional search cues.

    Coverage and PIT direction have binomial nulls under the frozen calibrated
    core law. The reference/core likelihood ratio has a unit expectation under
    a declared *product* core predictive law. These conditional nulls need not
    describe real outcomes; a score reference must be frozen on development
    rows before opening audit responses.
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
            or target.initial_data_identity != discovery_development.fingerprint
            or regions.identity != diagnosis.region_identity):
        raise ValueError("gap audit changed frozen model, target, or regions")
    if (not 0 < alpha < 1 or not 0 < tail_probability < 1
            or audit.X.shape[1] != model.n_features):
        raise ValueError("invalid registered adequacy screen")
    if (score_reference_model is None) != (score_reference_target is None):
        raise ValueError("score reference model and target must be paired")
    if score_reference_model is not None and (
            score_reference_model.n_features != model.n_features
            or score_reference_target.model_identity != score_reference_model.stable_hash
            or score_reference_target.initial_data_identity != target.initial_data_identity
            or score_reference_target.action_domain_identity != target.action_domain_identity
            or score_reference_target.measurement_budget != target.measurement_budget):
        raise ValueError("score reference changed fit data or target domain")
    components, evaluable = evaluable_predictive_components(
        model, target.initial_posterior, target.partition, audit.X)
    undefined_rows = int(len(audit.X) - int(np.sum(evaluable)))
    screen_X = np.asarray(audit.X, dtype=float)[evaluable]
    screen_y = np.asarray(audit.y, dtype=float)[evaluable]
    cdf = np.sum(components.structure_probabilities[:, None] * student_t.cdf(
        (screen_y[None, :] - components.locations) / components.scales,
        df=components.degrees_freedom[:, None]), axis=0)
    if not np.all(np.isfinite(cdf)):
        raise ValueError("nonfinite frozen-bank predictive CDF")
    core_log_score = logsumexp(
        np.log(components.structure_probabilities[:, None])
        + student_t.logpdf(screen_y[None, :],
            df=components.degrees_freedom[:, None],
            loc=components.locations, scale=components.scales), axis=0)
    if not np.all(np.isfinite(core_log_score)):
        raise ValueError("nonfinite frozen-bank predictive log score")
    reference_log_score = None
    if score_reference_model is not None:
        reference = predictive_components_for_partition(
            score_reference_model.engine(score_reference_model.stable_hash),
            score_reference_target.initial_posterior,
            score_reference_target.partition, screen_X)
        reference_log_score = logsumexp(
            np.log(reference.structure_probabilities[:, None])
            + student_t.logpdf(screen_y[None, :],
                df=reference.degrees_freedom[:, None],
                loc=reference.locations, scale=reference.scales), axis=0)
        if not np.all(np.isfinite(reference_log_score)):
            raise ValueError("nonfinite frozen-reference predictive log score")
    misses = (cdf < tail_probability / 2) | (cdf > 1 - tail_probability / 2)
    assigned = regions.assign(screen_X)
    rows = []
    tests_per_region = 4  # coverage, both PIT directions, reference log score
    local_alpha = alpha / (tests_per_region * (len(regions.cuts) + 1))
    for region in range(len(regions.cuts) + 1):
        mask = assigned == region
        count, errors = int(mask.sum()), int(np.sum(misses[mask]))
        lower = (float(beta.ppf(local_alpha,
                                errors, count - errors + 1))
                 if errors else 0.)
        positive = int(np.sum(cdf[mask] > .5))
        negative = int(np.sum(cdf[mask] < .5))
        dominant = max(positive, negative)
        direction_lower = (float(beta.ppf(local_alpha,
            dominant, count - dominant + 1)) if dominant else 0.)
        advantage = (float(np.sum(reference_log_score[mask] - core_log_score[mask]))
                     if reference_log_score is not None else None)
        rows.append(RegionalAdequacy(
            region, count, errors, lower, bool(lower > tail_probability),
            positive, direction_lower, bool(direction_lower > .5),
            advantage, bool(advantage is not None
                            and advantage > -np.log(local_alpha))))
    return IndependentGapEvidence(target.stable_hash, audit.fingerprint,
                                  regions.identity, tuple(rows),
                                  float(alpha), float(tail_probability),
                                  score_reference_model.stable_hash
                                  if score_reference_model is not None else "",
                                  audit_row_count=int(len(audit.X)),
                                  undefined_row_count=undefined_rows)
