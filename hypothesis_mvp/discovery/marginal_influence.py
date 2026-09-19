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
                             *, required_contributions: tuple[str, ...]) -> dict:
    if required_contributions != ("llm", "engine:mcts"):
        raise ValueError("unknown marginal influence contribution registration")
    expected = {"full", *(f"full_without_{name.replace(':', '_')}"
                          for name in required_contributions)}
    if set(profiles) != expected:
        raise ValueError("marginal influence profiles do not match source ablations")
    comparisons = {
        name: compare_marginal_influence(
            profiles["full"], profiles[f"full_without_{name.replace(':', '_')}"],
            contribution=name)
        for name in required_contributions
    }
    return {
        "schema": "scientific-marginal-decision-influence-family-gate-v1",
        "profiles": {name: {"identity": profile.identity,
                             "model": profile.model_identity,
                             "target": profile.target_identity,
                             "certified_top_action": profile.certified_leader}
                     for name, profile in profiles.items()},
        "comparisons": comparisons,
        "passed": all(row["passed"] for row in comparisons.values()),
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "response-free matched-bank leave-one-source-out frozen-H0 "
            "full-target decision-regret screening only; "
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
