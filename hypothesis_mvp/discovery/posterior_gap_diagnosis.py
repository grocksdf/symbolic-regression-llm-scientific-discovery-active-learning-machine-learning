"""Response-free, model-relative diagnosis of a frozen PCPI structure bank.

Disagreement identifies places where the existing bank predicts differently.
It cannot establish that the bank is missing a structure, nor that a new
structure improves an external decision. This module makes no LLM or pool call.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition

from .candidate_region_expansion import FrozenAxisRegions
from .pcpi_adapter import FrozenDiscoveryModel, FrozenDiscoveryTarget


@dataclass(frozen=True)
class RegionDisagreement:
    region: int
    lower_bound: float | None
    upper_bound: float | None
    target_mass: float
    point_count: int
    between_class_variance: float
    within_class_structure_variance: float
    conditional_predictive_variance: float
    weighted_between_class_variance: float
    class_profile: tuple[tuple[str, float, float], ...]


@dataclass(frozen=True)
class StructureSummary:
    structure_id: str
    basis_terms: tuple[str, ...]
    posterior_probability: float


@dataclass(frozen=True)
class FrozenBankDiagnosis:
    target_identity: str
    model_identity: str
    region_identity: str
    target_weights_identity: str
    region_axis: int
    rows: tuple[RegionDisagreement, ...]
    structures: tuple[StructureSummary, ...]
    ordinary_bayes: bool
    diagnosis_kind: str = "frozen-bank-predictive-disagreement"

    def proposal_brief(self, limit: int = 3) -> dict[str, object]:
        """A covariate-only prompt cue; no claim of predictive inadequacy.

        This is a research payload. The current discovery runner has no
        frozen PCPI posterior during its proposal rounds, so it does not
        automatically send this brief to a provider.
        """
        if type(limit) is not int or limit < 1:
            raise ValueError("brief limit must be positive")
        ranked = sorted(
            (row for row in self.rows if row.point_count),
            key=lambda row: (-row.weighted_between_class_variance, row.region),
        )
        return {
            "schema": "pcpi-frozen-bank-exploration-brief-v1",
            "target_identity": self.target_identity,
            "region_identity": self.region_identity,
            "region_axis": self.region_axis,
            "diagnosis_kind": self.diagnosis_kind,
            "interpretation": (
                "Existing operational classes disagree here under their own "
                "fitted predictive model. This is not evidence that an "
                "interaction, nonlinearity, or missing structure exists."
            ),
            "instruction": (
                "If proposing a hypothesis, describe a testable structural "
                "alternative for these regions using the registered grammar; "
                "do not assert a data-supported deficiency or decision gain."
            ),
            "regions": [
                {"region": row.region, "target_mass": row.target_mass,
                 "lower_bound": row.lower_bound,
                 "upper_bound": row.upper_bound,
                 "weighted_between_class_variance":
                     row.weighted_between_class_variance,
                 "competing_classes": [
                     {"class_id": class_id,
                      "posterior_probability": probability,
                      "mean_prediction": mean}
                     for class_id, probability, mean in row.class_profile]}
                for row in ranked[:limit]
            ],
            "top_existing_structures": [
                {"structure_id": row.structure_id,
                 "basis_terms": list(row.basis_terms),
                 "posterior_probability": row.posterior_probability}
                for row in sorted(self.structures,
                                  key=lambda row: (-row.posterior_probability,
                                                   row.structure_id))[:5]
            ],
            "external_adequacy_checked": False,
            "decision_utility_gain_checked": False,
            "candidate_response_accessed": False,
        }


def diagnose_frozen_bank(
    model: FrozenDiscoveryModel,
    target: FrozenDiscoveryTarget,
    action_domain: np.ndarray,
    target_weights: np.ndarray,
    regions: FrozenAxisRegions,
) -> FrozenBankDiagnosis:
    """Decompose location dispersion on the exact registered action domain.

    The between-class component is sum_c p(c) * (mu_c - mu)^2. The
    within-class component is sum_h p(h) * (mu_h - mu_class(h))^2. Both
    depend on the fitted bank; neither estimates its real-world error.
    """
    if target.model_identity != model.stable_hash:
        raise ValueError("frozen model identity mismatch")
    x = np.ascontiguousarray(action_domain, dtype=np.float64)
    if (x.ndim != 2 or not len(x) or x.shape[1] != model.n_features
            or not np.all(np.isfinite(x))):
        raise ValueError("invalid registered action domain")
    if sha256(str(x.shape).encode() + x.tobytes()).hexdigest() != target.action_domain_identity:
        raise ValueError("action domain identity mismatch")
    if regions.n_features != model.n_features:
        raise ValueError("region map feature identity mismatch")
    weights = np.asarray(target_weights, dtype=np.float64)
    if (weights.shape != (len(x),) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("invalid registered target weights")
    engine = model.engine(target.model_identity)
    posterior = target.initial_posterior
    if posterior.bank_hash != model.bank.stable_hash or posterior.likelihood_power != engine.likelihood_power:
        raise ValueError("posterior bank or likelihood identity mismatch")
    components = predictive_components_for_partition(
        engine, posterior, target.partition, x)
    p = components.structure_probabilities
    mu = components.locations
    membership = np.asarray(target.partition.structure_to_class, dtype=int)
    if (len(membership) != len(p) or np.any(membership < 0)
            or np.any(membership >= len(target.partition.class_ids))):
        raise ValueError("invalid frozen class membership")
    class_mean = np.empty((len(target.partition.class_ids), len(x)))
    class_mass = np.empty(len(class_mean))
    for c in range(len(class_mean)):
        mask = membership == c
        class_mass[c] = np.sum(p[mask])
        if class_mass[c] <= 0:
            raise ValueError("empty or zero-mass operational class")
        class_mean[c] = np.sum(p[mask, None] * mu[mask], axis=0) / class_mass[c]
    overall = np.sum(p[:, None] * mu, axis=0)
    between = np.sum(class_mass[:, None] * (class_mean - overall) ** 2, axis=0)
    within = np.sum(p[:, None] * (mu - class_mean[membership]) ** 2, axis=0)
    degrees = components.degrees_freedom
    if np.any(degrees <= 2):
        raise ValueError("conditional predictive variance has no finite moment")
    conditional = np.sum(
        p[:, None] * components.scales ** 2
        * (degrees / (degrees - 2))[:, None], axis=0)
    if not all(np.all(np.isfinite(v)) for v in (between, within, conditional)):
        raise ValueError("nonfinite predictive decomposition")
    assignment = regions.assign(x)
    rows = []
    for r in range(len(regions.cuts) + 1):
        mask = assignment == r
        w = weights[mask]
        mass = float(w.sum())
        def average(values: np.ndarray) -> float:
            return float(np.dot(w, values[mask]) / mass) if mass else 0.0
        profiles = tuple(
            (class_id, float(class_mass[c]),
             float(np.dot(w, class_mean[c, mask]) / mass) if mass else 0.0)
            for c, class_id in enumerate(target.partition.class_ids)
        ) if mass else ()
        rows.append(RegionDisagreement(
            r, float(regions.cuts[r - 1]) if r else None,
            float(regions.cuts[r]) if r < len(regions.cuts) else None,
            mass, int(mask.sum()), average(between), average(within),
            average(conditional), float(np.dot(w, between[mask])), profiles))
    summaries = tuple(StructureSummary(
        member.structure.structure_id, member.structure.basis_terms,
        float(member.probability)) for member in posterior.members)
    return FrozenBankDiagnosis(
        target.stable_hash, model.stable_hash, regions.identity,
        sha256(json.dumps(weights.tolist(), allow_nan=False).encode()).hexdigest(),
        regions.axis, tuple(rows), summaries, posterior.likelihood_power == 1.,
    )
