"""Response-free marginal decision influence for frozen discovery banks.

The Gate compares class-EIG intervals on one matched action domain before a
candidate-pool response is revealed.  It is a correctness/screening layer, not
an efficacy estimate: a pass only says that hypotheses unique to the full bank
have a numerically certified effect on the frozen initial acquisition problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.acquisition import (
    EXACT_CLASS_EIG_EPSABS,
    analytic_class_eig_bounds,
    exact_class_eig_shared_actions,
    predictive_components_for_partition,
)
from .pcpi_adapter import (
    FrozenDiscoveryModel, FrozenDiscoveryTarget,
    freeze_discovery_model, freeze_discovery_target,
)


SCHEMA = "scientific-marginal-decision-influence-v1"
PROFILE_SCHEMA = "scientific-initial-class-eig-interval-profile-v1"
QUALITY_SCHEMA = "scientific-source-predictive-quality-profile-v1"


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class InitialEIGIntervalProfile:
    variant: str
    model_identity: str
    target_identity: str
    action_identity: str
    posterior_state: str
    strict_prefix_response_count: int
    support_keys: tuple[tuple[str, ...], ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    exact_scores: tuple[float, ...]
    exact_errors: tuple[float, ...]
    numerical_outward_tolerance: float
    exact_eig_epsabs: float

    def __post_init__(self) -> None:
        count = len(self.lower)
        arrays = [np.asarray(value, dtype=float) for value in (
            self.lower, self.upper, self.exact_scores, self.exact_errors)]
        if (not self.variant or not self.model_identity or not self.target_identity
                or not self.action_identity or count < 1
                or any(array.shape != (count,) for array in arrays)
                or any(not np.all(np.isfinite(array)) for array in arrays)
                or np.any(arrays[0] < 0.0) or np.any(arrays[1] < arrays[0])
                or np.any(arrays[3] < 0.0)
                or not np.isfinite(self.numerical_outward_tolerance)
                or self.numerical_outward_tolerance < 0.0
                or self.exact_eig_epsabs != EXACT_CLASS_EIG_EPSABS):
            raise ValueError("invalid initial class-EIG interval profile")

    @property
    def certified_leader(self) -> int | None:
        lower, upper = np.asarray(self.lower), np.asarray(self.upper)
        leader = int(np.argmax(lower))
        competitors = np.delete(upper, leader)
        return leader if not len(competitors) or lower[leader] > np.max(competitors) else None

    @property
    def identity(self) -> str:
        return _digest({
            "schema": PROFILE_SCHEMA,
            "variant": self.variant,
            "model": self.model_identity,
            "target": self.target_identity,
            "actions": self.action_identity,
            "posterior_state": self.posterior_state,
            "strict_prefix_response_count": self.strict_prefix_response_count,
            "support_keys": self.support_keys,
            "lower": self.lower,
            "upper": self.upper,
            "exact_scores": self.exact_scores,
            "exact_errors": self.exact_errors,
            "numerical_outward_tolerance": self.numerical_outward_tolerance,
            "exact_eig_epsabs": self.exact_eig_epsabs,
        })


@dataclass(frozen=True)
class PredictiveQualityProfile:
    variant: str
    model_identity: str
    target_identity: str
    arbitration_identity: str
    pointwise_log_predictive_density: tuple[float, ...]

    def __post_init__(self):
        values = np.asarray(self.pointwise_log_predictive_density, dtype=float)
        if (not self.variant or not self.model_identity or not self.target_identity
                or not self.arbitration_identity or values.ndim != 1
                or not len(values) or not np.all(np.isfinite(values))):
            raise ValueError("invalid predictive quality profile")

    @property
    def identity(self):
        return _digest({"schema": QUALITY_SCHEMA, "variant": self.variant,
            "model": self.model_identity, "target": self.target_identity,
            "arbitration": self.arbitration_identity,
            "log_predictive_density": self.pointwise_log_predictive_density})


def initial_eig_interval_profile(variant: str, model: FrozenDiscoveryModel,
                                 target: FrozenDiscoveryTarget,
                                 actions: np.ndarray, *,
                                 exact_eig_epsabs: float) -> InitialEIGIntervalProfile:
    """Evaluate the frozen-H0 finite-bank class-EIG intervals for all actions."""
    if exact_eig_epsabs != EXACT_CLASS_EIG_EPSABS:
        raise ValueError("marginal influence Gate must use registered exact EIG tolerance")
    values = np.ascontiguousarray(actions, dtype=float)
    action_identity = sha256(str(values.shape).encode() + values.tobytes()).hexdigest()
    if action_identity != target.action_domain_identity:
        raise ValueError("marginal influence actions crossed frozen target")
    engine = model.engine(target.model_identity)
    components = predictive_components_for_partition(
        engine, target.initial_posterior, target.partition, values)
    analytic = analytic_class_eig_bounds(components)
    exact = exact_class_eig_shared_actions(
        components, epsabs=exact_eig_epsabs)
    outward = float(analytic.numerical_outward_tolerance)
    exact_lower = np.maximum(0.0, np.asarray(exact.scores) - np.asarray(exact.quadrature_errors) - outward)
    exact_upper = np.asarray(exact.scores) + np.asarray(exact.quadrature_errors) + outward
    lower = np.maximum(np.asarray(analytic.lower_bounds), exact_lower)
    upper = np.minimum(np.asarray(analytic.upper_bounds), exact_upper)
    if np.any(lower > upper + outward):
        raise ValueError("inconsistent exact and analytic class-EIG intervals")
    lower = np.minimum(lower, upper)
    support_keys = tuple(sorted(tuple(structure.basis_terms)
                                for structure in model.bank.structures))
    return InitialEIGIntervalProfile(
        variant=variant,
        model_identity=model.stable_hash,
        target_identity=target.stable_hash,
        action_identity=action_identity,
        posterior_state="frozen-initial",
        strict_prefix_response_count=0,
        support_keys=support_keys,
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        exact_scores=tuple(float(value) for value in exact.scores),
        exact_errors=tuple(float(value) for value in exact.quadrature_errors),
        numerical_outward_tolerance=outward,
        exact_eig_epsabs=float(exact_eig_epsabs),
    )


def predictive_quality_profile(variant: str, model: FrozenDiscoveryModel,
                               target: FrozenDiscoveryTarget,
                               arbitration: RoleDataset) -> PredictiveQualityProfile:
    """Proper log score on a registered source-arbitration validation split."""
    if arbitration.role is not DataRole.VALIDATION:
        raise ValueError("source arbitration requires validation-role data")
    values = model.engine(target.model_identity).predictive_logpdf(
        target.initial_posterior, arbitration.X, arbitration.y)
    return PredictiveQualityProfile(
        variant, model.stable_hash, target.stable_hash, arbitration.fingerprint,
        tuple(float(value) for value in values))


def freeze_initial_eig_interval_profile(variant, candidates, initial_data, actions,
                                        *, n_features, prior, exploration_identity,
                                        coefficient_policy, measurement_budget,
                                        exact_eig_epsabs):
    """Freeze one variant and evaluate it without accepting pool responses."""
    model = freeze_discovery_model(
        candidates, n_features=n_features, prior=prior,
        exploration_identity=exploration_identity,
        coefficient_policy=coefficient_policy,
    )
    target = freeze_discovery_target(
        model, initial_data, actions, measurement_budget=measurement_budget,
        expected_model_identity=model.stable_hash,
    )
    return initial_eig_interval_profile(
        variant, model, target, actions, exact_eig_epsabs=exact_eig_epsabs)


def compare_marginal_influence(full: InitialEIGIntervalProfile,
                               baseline: InitialEIGIntervalProfile,
                               *, contribution: str) -> dict:
    """Certify a leave-one-source-out decision under the full-bank utility.

    Cross-bank utility magnitudes target different operational class variables
    and are not treated as comparable scientific values.  The ablated bank may
    nominate an action, but its cost is certified only with the full bank's
    class-EIG intervals: ``L_full(a_full) - U_full(a_ablated)``.  This is a
    lower bound on the full-target regret avoided by retaining the source.
    """
    expected_variant = f"full_without_{contribution.replace(':', '_')}"
    if full.variant != "full" or baseline.variant != expected_variant:
        raise ValueError("invalid leave-one-source-out influence pairing")
    if (full.action_identity != baseline.action_identity
            or len(full.lower) != len(baseline.lower)
            or full.exact_eig_epsabs != baseline.exact_eig_epsabs):
        raise ValueError("marginal influence profiles are not action/tolerance matched")
    if (full.posterior_state != "frozen-initial"
            or baseline.posterior_state != "frozen-initial"
            or full.strict_prefix_response_count != 0
            or baseline.strict_prefix_response_count != 0):
        raise ValueError("marginal influence must use frozen initial posteriors")

    added = tuple(sorted(set(full.support_keys) - set(baseline.support_keys)))
    full_lower, full_upper = np.asarray(full.lower), np.asarray(full.upper)
    full_leader, base_leader = full.certified_leader, baseline.certified_leader
    resolution = float(len(full_lower) * full.exact_eig_epsabs)
    ranking_changed = (full_leader is not None and base_leader is not None
                       and full_leader != base_leader)
    full_regret_reduction = (0.0 if not ranking_changed else max(0.0, float(
        full_lower[full_leader] - full_upper[base_leader])))
    intervals_resolved = full_leader is not None and base_leader is not None
    decisions = {
        "full_has_unique_structural_support": bool(added),
        "both_top_actions_certified": intervals_resolved,
        "certified_top_action_changes": ranking_changed,
        "full_target_regret_reduction_exceeds_familywise_resolution": bool(
            full_regret_reduction > resolution),
    }
    return {
        "schema": SCHEMA,
        "contribution": contribution,
        "ablation": baseline.variant,
        "full_profile_identity": full.identity,
        "baseline_profile_identity": baseline.identity,
        "added_supports": [list(value) for value in added],
        "action_count": len(full_lower),
        "familywise_numerical_resolution_nats": resolution,
        "full_certified_top_action": full_leader,
        "baseline_certified_top_action": base_leader,
        "certified_top_action_changed": ranking_changed,
        "full_target_regret_reduction_lower_bound_nats": full_regret_reduction,
        "utility_comparison_identity": "full-target-certified-regret-v1",
        "decisions": decisions,
        "passed": all(decisions.values()),
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
    }


def audit_marginal_influence(profiles: dict[str, InitialEIGIntervalProfile],
                             *, quality_profiles: dict[str, PredictiveQualityProfile],
                             required_contributions: tuple[str, ...]) -> dict:
    if required_contributions != ("llm", "engine:mcts"):
        raise ValueError("unknown marginal influence contribution registration")
    expected = {"full", *(f"full_without_{name.replace(':', '_')}"
                          for name in required_contributions)}
    if set(profiles) != expected:
        raise ValueError("marginal influence profiles do not match source ablations")
    if set(quality_profiles) != expected:
        raise ValueError("predictive quality profiles do not match source ablations")
    comparisons = {}
    for name in required_contributions:
        ablation = f"full_without_{name.replace(':', '_')}"
        decision = compare_marginal_influence(
            profiles["full"], profiles[f"full_without_{name.replace(':', '_')}"],
            contribution=name)
        full_quality, ablated_quality = quality_profiles["full"], quality_profiles[ablation]
        if full_quality.arbitration_identity != ablated_quality.arbitration_identity:
            raise ValueError("source quality profiles crossed arbitration split")
        full_values = np.asarray(full_quality.pointwise_log_predictive_density)
        ablated_values = np.asarray(ablated_quality.pointwise_log_predictive_density)
        if full_values.shape != ablated_values.shape:
            raise ValueError("source quality profiles are not paired")
        log_ratio = float(np.sum(full_values - ablated_values))
        scale = max(1.0, float(np.sum(np.abs(full_values))),
                    float(np.sum(np.abs(ablated_values))))
        tolerance = float(1024.0 * np.finfo(float).eps * scale)
        quality_passed = log_ratio > tolerance
        comparisons[name] = {**decision,
            "predictive_quality": {
                "schema": "scientific-source-predictive-log-score-contribution-v1",
                "full_profile_identity": full_quality.identity,
                "ablated_profile_identity": ablated_quality.identity,
                "arbitration_identity": full_quality.arbitration_identity,
                "arbitration_observation_count": len(full_values),
                "cumulative_log_predictive_ratio_nats": log_ratio,
                "numerical_tolerance_nats": tolerance,
                "proper_scoring_rule": "posterior-predictive-log-density",
                "passed": quality_passed,
                "acquisition_pool_response_accessed": False,
                "heldout_opened": False},
            "decision_contribution_passed": decision["passed"],
            "hypothesis_quality_contribution_passed": quality_passed,
            "accepted_contribution_role": (
                "decision-and-hypothesis-quality" if decision["passed"] and quality_passed
                else "decision" if decision["passed"] else "hypothesis-quality"
                if quality_passed else None),
            "passed": bool(decision["decisions"]["full_has_unique_structural_support"]
                           and (decision["passed"] or quality_passed))}
    return {
        "schema": "scientific-dual-channel-source-contribution-family-gate-v1",
        "profiles": {name: {"identity": profile.identity,
                             "model": profile.model_identity,
                             "target": profile.target_identity,
                             "certified_top_action": profile.certified_leader}
                     for name, profile in profiles.items()},
        "comparisons": comparisons,
        "passed": all(row["passed"] for row in comparisons.values()),
        "candidate_response_accessed": False,
        "acquisition_pool_response_accessed": False,
        "source_arbitration_validation_responses_accessed": True,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "matched-bank leave-one-source-out source arbitration: acquisition-"
            "pool-response-free decision regret plus opened independent validation "
            "posterior-predictive log score; "
            "not real-data efficacy or superiority evidence"
        ),
    }


def leave_one_source_out_candidates(candidates, contribution: str):
    """Return a source-matched ablation without inspecting any response."""
    if contribution == "llm":
        kept = [row for row in candidates if str(row.get("origin", "")) != "llm"]
    elif contribution == "engine:mcts":
        kept = [row for row in candidates if str(row.get("source", "")) != contribution]
    else:
        raise ValueError("unknown source contribution")
    if len(kept) == len(candidates):
        raise ValueError(f"registered contribution absent from full bank: {contribution}")
    return kept
