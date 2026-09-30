"""Exact finite reference for candidate-by-region predictive expansion.

This is a *modular selector posterior* conditional on fixed predictive experts,
not the PCPI posterior over symbolic laws. Experts and the covariate-only region
map must be frozen before selector observations are revealed. Nothing here
accesses a dataset or authorizes an acquisition-pool response.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class FrozenAxisRegions:
    """Covariate-only partition, frozen before selector responses are opened."""

    n_features: int
    axis: int
    cuts: tuple[float, ...]

    def __post_init__(self) -> None:
        if (type(self.n_features) is not int or self.n_features < 1
                or type(self.axis) is not int
                or not 0 <= self.axis < self.n_features
                or not isinstance(self.cuts, tuple)
                or not self.cuts or any(not np.isfinite(value)
                for value in self.cuts)
                or any(left >= right for left, right in zip(
                    self.cuts, self.cuts[1:]))):
            raise ValueError("invalid frozen covariate partition")

    @property
    def identity(self) -> str:
        material = (self.n_features, self.axis, self.cuts)
        return sha256(json.dumps(material, allow_nan=False).encode()).hexdigest()

    def assign(self, covariates: np.ndarray) -> np.ndarray:
        x = np.asarray(covariates, dtype=float)
        if (x.ndim != 2 or x.shape[1] != self.n_features
                or not np.all(np.isfinite(x))):
            raise ValueError("invalid region assignment covariates")
        return np.searchsorted(self.cuts, x[:, self.axis], side="right")


@dataclass(frozen=True, eq=False)
class RegionSelectorPosterior:
    """Independent categorical expert selectors, one per frozen region.

    Index zero is the protected core predictive expert. Every other index is
    one distinct candidate, even when several share the same source. The
    prior is a *declared* probability model, not a fitted stacking weight.
    """

    region_identity: str
    candidate_ids: tuple[str, ...]
    prior: np.ndarray  # [region, expert]
    log_likelihood: np.ndarray  # cumulative fixed-expert log likelihood
    response_counts: np.ndarray  # [region]

    def __post_init__(self) -> None:
        prior = np.array(self.prior, dtype=float, copy=True)
        logs = np.array(self.log_likelihood, dtype=float, copy=True)
        counts = np.array(self.response_counts, dtype=int, copy=True)
        if (not self.region_identity or len(self.candidate_ids) < 1
                or self.candidate_ids[0] != "core"
                or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or prior.ndim != 2 or prior.shape[1] != len(self.candidate_ids)
                or prior.shape[0] == 0 or logs.shape != prior.shape
                or counts.shape != (prior.shape[0],)
                or np.any(counts < 0) or not np.all(np.isfinite(prior))
                or not np.all(np.isfinite(logs)) or np.any(prior <= 0)
                or not np.allclose(prior.sum(axis=1), 1., rtol=0, atol=1e-12)):
            raise ValueError("invalid frozen region selector")
        for name, value in (("prior", prior), ("log_likelihood", logs),
                            ("response_counts", counts)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def start(cls, region_identity: str, candidate_ids: tuple[str, ...],
              prior: np.ndarray) -> "RegionSelectorPosterior":
        values = np.asarray(prior, dtype=float)
        if values.ndim != 2:
            raise ValueError("region prior must be a matrix")
        return cls(region_identity, candidate_ids, values,
                   np.zeros_like(values), np.zeros(len(values), dtype=int))

    @classmethod
    def from_partition(cls, partition: FrozenAxisRegions,
                       candidate_ids: tuple[str, ...], prior: np.ndarray
                       ) -> "RegionSelectorPosterior":
        values = np.asarray(prior, dtype=float)
        if values.ndim != 2 or values.shape[0] != len(partition.cuts) + 1:
            raise ValueError("region prior does not match frozen partition")
        return cls.start(partition.identity, candidate_ids, values)

    @property
    def probabilities(self) -> np.ndarray:
        logs = np.log(self.prior) + self.log_likelihood
        return np.exp(logs - logsumexp(logs, axis=1, keepdims=True))

    def update(self, regions: np.ndarray, fixed_expert_logpdf: np.ndarray
               ) -> "RegionSelectorPosterior":
        region = np.asarray(regions)
        logpdf = np.asarray(fixed_expert_logpdf, dtype=float)
        if (region.ndim != 1 or region.dtype.kind not in "iu"
                or logpdf.shape != (len(region), len(self.candidate_ids))
                or np.any(region < 0) or np.any(region >= len(self.prior))
                or not np.all(np.isfinite(logpdf))):
            raise ValueError("invalid fixed-expert sequential observations")
        logs, counts = self.log_likelihood.copy(), self.response_counts.copy()
        np.add.at(logs, region, logpdf)
        np.add.at(counts, region, 1)
        return replace(self, log_likelihood=logs, response_counts=counts)

    def without(self, candidate_id: str) -> "RegionSelectorPosterior":
        """Ablate before evidence; transfer candidate prior mass to the core."""
        if candidate_id == "core" or candidate_id not in self.candidate_ids:
            raise ValueError("only an optional candidate can be ablated")
        index = self.candidate_ids.index(candidate_id)
        keep = [i for i in range(len(self.candidate_ids)) if i != index]
        prior = self.prior[:, keep].copy()
        prior[:, 0] += self.prior[:, index]
        return type(self)(self.region_identity,
                          tuple(self.candidate_ids[i] for i in keep), prior,
                          self.log_likelihood[:, keep], self.response_counts)

    def log_bayes_factor_vs_core(self, candidate_id: str) -> np.ndarray:
        if candidate_id == "core" or candidate_id not in self.candidate_ids:
            raise ValueError("unknown optional candidate")
        index = self.candidate_ids.index(candidate_id)
        return self.log_likelihood[:, index] - self.log_likelihood[:, 0]


@dataclass(frozen=True)
class FiniteCommonLaw:
    """Predeclared plausible law, never fitted from reporting responses."""

    identity: str
    target_identity: str
    response_probabilities: np.ndarray  # [action, finite response node]
    target_mean: np.ndarray  # [registered target point]


def _target_inputs(state: RegionSelectorPosterior, target_regions,
                   expert_means, target_weights):
    regions = np.asarray(target_regions)
    means = np.asarray(expert_means, dtype=float)
    weights = np.asarray(target_weights, dtype=float)
    if (regions.ndim != 1 or regions.dtype.kind not in "iu" or not len(regions)
            or np.any(regions < 0) or np.any(regions >= len(state.prior))
            or means.shape != (len(regions), len(state.candidate_ids))
            or weights.shape != (len(regions),)
            or not np.all(np.isfinite(means))
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("invalid common prediction target")
    return regions, means, weights


def common_target_squared_loss(state: RegionSelectorPosterior, target_regions,
                               expert_means, target_weights, truth_mean) -> float:
    """Expected squared decision loss under one specified external mean law."""
    regions, means, weights = _target_inputs(
        state, target_regions, expert_means, target_weights)
    truth = np.asarray(truth_mean, dtype=float)
    if truth.shape != (len(regions),) or not np.all(np.isfinite(truth)):
        raise ValueError("invalid common-target reference law")
    prediction = np.sum(state.probabilities[regions] * means, axis=1)
    return float(np.dot(weights, (prediction - truth) ** 2))


def action_values_under_law(state: RegionSelectorPosterior, action_regions,
                            expert_response_logpdf, law_response_probabilities,
                            target_regions, expert_target_means,
                            target_weights, law_target_mean) -> np.ndarray:
    """One-step loss reduction under an explicit, fixed plausible law.

    The law's action probabilities and target mean refer to the same common
    target in every bank. Negative values are allowed under misspecification.
    The finite response grid is a reference calculation, not a quadrature
    certificate for continuous measured responses.
    """
    action_region = np.asarray(action_regions)
    logpdf = np.asarray(expert_response_logpdf, dtype=float)
    q = np.asarray(law_response_probabilities, dtype=float)
    if (action_region.ndim != 1 or action_region.dtype.kind not in "iu"
            or not len(action_region) or np.any(action_region < 0)
            or np.any(action_region >= len(state.prior))
            or q.ndim != 2 or q.shape[0] != len(action_region)
            or q.shape[1] == 0
            or logpdf.shape != (*q.shape, len(state.candidate_ids))
            or not np.all(np.isfinite(logpdf)) or not np.all(np.isfinite(q))
            or np.any(q < 0)
            or not np.allclose(q.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError("invalid shared-law action response grid")
    before = common_target_squared_loss(
        state, target_regions, expert_target_means, target_weights,
        law_target_mean)
    values = np.empty(len(action_region))
    for action, region in enumerate(action_region):
        after = 0.
        for response, probability in enumerate(q[action]):
            if probability == 0:
                continue
            updated = state.update(np.array([region]),
                                   logpdf[action, response][None, :])
            after += probability * common_target_squared_loss(
                updated, target_regions, expert_target_means, target_weights,
                law_target_mean)
        values[action] = before - after
    return values


def candidate_region_contribution(state: RegionSelectorPosterior, candidate_id,
                                  target_regions, expert_target_means,
                                  target_weights, truth_mean) -> dict:
    """Diagnostic posterior and common-target decision ablation per candidate."""
    index = state.candidate_ids.index(candidate_id)
    ablated = state.without(candidate_id)
    reduced_means = np.delete(np.asarray(expert_target_means), index, axis=1)
    full_loss = common_target_squared_loss(
        state, target_regions, expert_target_means, target_weights, truth_mean)
    reduced_loss = common_target_squared_loss(
        ablated, target_regions, reduced_means, target_weights, truth_mean)
    return {"candidate_id": candidate_id,
            "region_posterior_mass": state.probabilities[:, index].tolist(),
            "region_log_bayes_factor_vs_core":
                state.log_bayes_factor_vs_core(candidate_id).tolist(),
            "common_target_loss_reduction_vs_ablation": reduced_loss - full_loss,
            "reference_law_required": True, "efficacy_demonstrated": False}


def candidate_action_contribution(
    state: RegionSelectorPosterior, candidate_id: str,
    action_regions, expert_response_logpdf, target_regions,
    expert_target_means, target_weights, laws: tuple[FiniteCommonLaw, ...],
    *, target_identity: str,
) -> dict:
    """Compare both action policies under exactly the same finite law(s).

    This is a response-free reference audit, not a numerical certificate for
    continuous observations or a guarantee of coverage by the plausible laws.
    """
    if (not target_identity or not laws or len({law.identity for law in laws})
            != len(laws) or any(not law.identity
            or law.target_identity != target_identity for law in laws)):
        raise ValueError("candidate action audit changed common target or laws")
    index = state.candidate_ids.index(candidate_id)
    reduced = state.without(candidate_id)
    full_means = np.asarray(expert_target_means, dtype=float)
    reduced_means = np.delete(full_means, index, axis=1)
    logpdf = np.asarray(expert_response_logpdf, dtype=float)
    if logpdf.ndim != 3 or logpdf.shape[2] != len(state.candidate_ids):
        raise ValueError("candidate action audit has invalid expert density")
    full_values, reduced_values = [], []
    for law in laws:
        full_values.append(action_values_under_law(
            state, action_regions, logpdf, law.response_probabilities,
            target_regions, full_means, target_weights, law.target_mean))
        reduced_values.append(action_values_under_law(
            reduced, action_regions, np.delete(logpdf, index, axis=2),
            law.response_probabilities, target_regions, reduced_means,
            target_weights, law.target_mean))
    full_values, reduced_values = np.stack(full_values), np.stack(reduced_values)
    full_robust, reduced_robust = full_values.min(axis=0), reduced_values.min(axis=0)
    full_action, reduced_action = int(np.argmax(full_robust)), int(np.argmax(reduced_robust))
    after_loss_reduction = []
    for law, full, ablated in zip(laws, full_values, reduced_values, strict=True):
        full_before = common_target_squared_loss(
            state, target_regions, full_means, target_weights, law.target_mean)
        reduced_before = common_target_squared_loss(
            reduced, target_regions, reduced_means, target_weights, law.target_mean)
        after_loss_reduction.append(float(
            reduced_before - ablated[reduced_action]
            - (full_before - full[full_action])))
    return {"candidate_id": candidate_id, "target_identity": target_identity,
            "law_identities": [law.identity for law in laws],
            "full_action": full_action, "without_candidate_action": reduced_action,
            "region_posterior_mass": state.probabilities[:, index].tolist(),
            "region_log_bayes_factor_vs_core":
                state.log_bayes_factor_vs_core(candidate_id).tolist(),
            "robust_action_value_full": full_robust.tolist(),
            "robust_action_value_without": reduced_robust.tolist(),
            "action_regret_avoided_under_full_target": float(
                full_robust[full_action] - full_robust[reduced_action]),
            "after_action_common_loss_reduction_by_law": after_loss_reduction,
            "numerically_certified": False, "efficacy_demonstrated": False}
