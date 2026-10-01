"""Exploratory same-target action ablation under externally calibrated finite laws.

This is a conditional reference calculation. It neither chooses a measured
action nor claims that the finite laws cover the real response distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior

from .candidate_region_expansion import (
    FiniteCommonLaw, FrozenAxisRegions, RegionSelectorPosterior,
    candidate_action_contribution,
)
from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target
from .regional_candidate_admission import _fixed_predictive_logpdf


@dataclass(frozen=True)
class FrozenFiniteActionReference:
    """A frozen common covariate/loss target and independent finite laws."""

    target_identity: str
    calibration_identity: str
    target_covariates: np.ndarray
    target_weights: np.ndarray
    action_covariates: np.ndarray
    response_nodes: np.ndarray  # [action, finite response node]
    laws: tuple[FiniteCommonLaw, ...]

    def __post_init__(self):
        target = np.asarray(self.target_covariates, dtype=float)
        weights = np.asarray(self.target_weights, dtype=float)
        action = np.asarray(self.action_covariates, dtype=float)
        nodes = np.asarray(self.response_nodes, dtype=float)
        if (not self.target_identity or not self.calibration_identity
                or target.ndim != 2 or not len(target)
                or action.ndim != 2 or not len(action)
                or action.shape[1] != target.shape[1]
                or weights.shape != (len(target),)
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)
                or nodes.ndim != 2 or nodes.shape[0] != len(action)
                or nodes.shape[1] < 2 or not self.laws
                or len({law.identity for law in self.laws}) != len(self.laws)
                or any(not law.identity
                       or law.target_identity != self.target_identity
                       or law.response_probabilities.shape != nodes.shape
                       or law.target_mean.shape != (len(target),)
                       or not np.all(np.isfinite(law.response_probabilities))
                       or not np.all(np.isfinite(law.target_mean))
                       or np.any(law.response_probabilities < 0)
                       or not np.allclose(law.response_probabilities.sum(axis=1),
                                          1., rtol=0., atol=1e-12)
                       for law in self.laws)
                or not all(np.all(np.isfinite(a)) for a in
                    (target, weights, action, nodes))):
            raise ValueError("invalid frozen finite common-target action reference")
        for name, value in (("target_covariates", target),
                            ("target_weights", weights),
                            ("action_covariates", action),
                            ("response_nodes", nodes)):
            frozen = np.array(value, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        frozen_laws = []
        for law in self.laws:
            probabilities = np.array(law.response_probabilities, copy=True)
            truth_mean = np.array(law.target_mean, copy=True)
            probabilities.setflags(write=False)
            truth_mean.setflags(write=False)
            frozen_laws.append(FiniteCommonLaw(
                law.identity, law.target_identity, probabilities, truth_mean))
        object.__setattr__(self, "laws", tuple(frozen_laws))

    @property
    def stable_hash(self):
        payload = {"target": self.target_identity,
                   "calibration": self.calibration_identity,
                   "covariates": self.target_covariates.tolist(),
                   "weights": self.target_weights.tolist(),
                   "actions": self.action_covariates.tolist(),
                   "nodes": self.response_nodes.tolist(),
                   "laws": [{"identity": law.identity,
                             "response_probabilities":
                                 law.response_probabilities.tolist(),
                             "target_mean": law.target_mean.tolist()}
                            for law in self.laws]}
        return sha256(json.dumps(payload, sort_keys=True,
                                 allow_nan=False).encode()).hexdigest()


def _expert(rows, fit, domain, prior, budget, target_x, action_x, nodes):
    identity = sha256(json.dumps(rows, sort_keys=True,
                                 default=str).encode()).hexdigest()
    model = freeze_discovery_model(rows, n_features=fit.X.shape[1],
        prior=prior, exploration_identity=identity,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        minimum_supports=1)
    target = freeze_discovery_target(model, fit, domain,
        measurement_budget=budget, expected_model_identity=model.stable_hash)
    engine = model.engine(model.stable_hash)
    means = predictive_components_for_partition(
        engine, target.initial_posterior, target.partition, target_x)
    actions = predictive_components_for_partition(
        engine, target.initial_posterior, target.partition, action_x)
    expert_mean = np.dot(means.structure_probabilities, means.locations)
    logpdf = logsumexp(
        np.log(actions.structure_probabilities[:, None, None])
        + student_t.logpdf(nodes[None, :, :],
            df=actions.degrees_freedom[:, None, None],
            loc=actions.locations[:, :, None],
            scale=actions.scales[:, :, None]), axis=0)
    if not np.all(np.isfinite(expert_mean)) or not np.all(np.isfinite(logpdf)):
        raise ValueError("nonfinite finite-law expert predictions")
    return expert_mean, logpdf


def audit_admitted_candidate_action(
        core_rows, candidate, fit: RoleDataset, gap_audit: RoleDataset,
        admission: RoleDataset, selector_update: RoleDataset,
        calibration: RoleDataset, domain,
        regions: FrozenAxisRegions, prior: NormalInverseGammaPrior,
        measurement_budget: int, reference: FrozenFiniteActionReference,
        *, candidate_identity: str):
    """Compute signed candidate action effect after independent admission.

    Admission selects the candidate; a separate response role updates its
    fixed-expert selector. Calibration and future reporting are separate.
    The caller never uses this diagnostic to select a measured acquisition.
    """
    roles = (fit, gap_audit, admission, selector_update, calibration)
    if (fit.role is not DataRole.DEVELOPMENT
            or gap_audit.role is not DataRole.VALIDATION
            or admission.role is not DataRole.VALIDATION
            or selector_update.role is not DataRole.VALIDATION
            or calibration.role is not DataRole.VALIDATION
            or reference.calibration_identity != calibration.fingerprint
            or not candidate_identity
            or any(a.row_fingerprints & b.row_fingerprints
                   or _covariate_rows(a.X) & _covariate_rows(b.X)
                   for i, a in enumerate(roles) for b in roles[i + 1:])
            or not np.array_equal(reference.action_covariates, domain)
            or not np.array_equal(reference.target_covariates, domain)
            or reference.action_covariates.shape[1] != fit.X.shape[1]
            or regions.n_features != fit.X.shape[1]):
        raise ValueError("action audit requires disjoint roles and one frozen domain")
    core_score, core_identity = _fixed_predictive_logpdf(
        core_rows, fit, selector_update, domain, prior, measurement_budget)
    optional_score, optional_identity = _fixed_predictive_logpdf(
        [candidate], fit, selector_update, domain, prior, measurement_budget)
    selector = RegionSelectorPosterior.from_partition(
        regions, ("core", candidate_identity),
        np.full((len(regions.cuts) + 1, 2), .5))
    selector = selector.update(regions.assign(selector_update.X),
        np.column_stack([core_score, optional_score]))
    means, logpdf = [], []
    for rows in (core_rows, [candidate]):
        mean, density = _expert(rows, fit, domain, prior, measurement_budget,
                                reference.target_covariates,
                                reference.action_covariates,
                                reference.response_nodes)
        means.append(mean)
        logpdf.append(density)
    report = candidate_action_contribution(
        selector, candidate_identity,
        regions.assign(reference.action_covariates),
        np.stack(logpdf, axis=-1),
        regions.assign(reference.target_covariates),
        np.stack(means, axis=-1), reference.target_weights,
        reference.laws, target_identity=reference.target_identity)
    return {**report, "reference_identity": reference.stable_hash,
            "loss_identity": "common-domain-squared-predictive-mean-v1",
            "calibration_identity": calibration.fingerprint,
            "admission_identity": admission.fingerprint,
            "selector_update_identity": selector_update.fingerprint,
            "selector_update_independent_of_admission": True,
            "core_expert_identity": core_identity,
            "candidate_expert_identity": optional_identity,
            "decision_contribution_assessed": False,
            "finite_law_action_contribution_assessed": True,
            "pcpi_operational_class_decision_assessed": False,
            "measured_action_authorized": False,
            "candidate_response_accessed": False,
            "heldout_opened": False}


def _covariate_rows(x: np.ndarray) -> set[bytes]:
    """Disjoint rows even when a reused covariate has a changed response."""
    return {np.ascontiguousarray(np.asarray(row, dtype=np.float64) + 0.).tobytes()
            for row in x}
