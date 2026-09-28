"""Conditional-development system execution over already-frozen proposals.

No data loader or provider call lives here. The caller supplies exploration
output and opened H0; held-out objects are neither accepted nor inspected.
"""
from pathlib import Path

import numpy as np

from hypothesis_mvp.data.oracle import PoolOracle
from hypothesis_mvp.pcpi.discovery_transaction import (
    DiscoveryTransaction, DiscoveryScoringControls, _publish,
)
from hypothesis_mvp.pcpi.acquisition import (
    EXACT_CLASS_EIG_EPSABS, exact_class_decision_risk_reduction_shared_actions,
    predictive_components_for_partition,
)
from hypothesis_mvp.pcpi.action_conditional_residual import (
    bayes_zero_one_decision_risk,
)
from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target
from .inference_router import route_inference
from .system_evidence import validate_system_pairs
from .resource_limits import run_bounded
from hypothesis_mvp.data.roles import DataRole


def _freeze_comparison(candidates, n_features, prior, exploration_identity,
                       coefficient_policy, initial_data, actions, measurement_budget,
                       source_prior_weights=None):
    model = freeze_discovery_model(candidates, n_features=n_features,
        prior=prior, exploration_identity=exploration_identity,
        coefficient_policy=coefficient_policy,
        source_prior_weights=source_prior_weights)
    target = freeze_discovery_target(model, initial_data, actions,
        measurement_budget=measurement_budget,
        expected_model_identity=model.stable_hash)
    return model, target


def audit_frozen_hypothesis_bank(candidates, initial_data, actions, *, n_features,
                                 prior, exploration_identity, coefficient_policy,
                                 measurement_budget, exact_eig_epsabs,
                                 source_prior_weights=None):
    """Response-free capacity Gate before any candidate response is revealed."""
    if exact_eig_epsabs != EXACT_CLASS_EIG_EPSABS:
        raise ValueError("hypothesis-bank Gate must match exact EIG absolute tolerance")
    model, target = _freeze_comparison(
        candidates, n_features, prior, exploration_identity, coefficient_policy,
        initial_data, actions, measurement_budget, source_prior_weights,
    )
    inference = route_inference(model)
    probabilities = np.asarray(target.partition.class_probabilities, dtype=float)
    entropy = float(target.partition.entropy)
    bayes_risk = bayes_zero_one_decision_risk(probabilities)
    familywise_resolution = float(len(actions) * exact_eig_epsabs)
    decisions = {
        "multiple_operational_classes": len(probabilities) >= 2,
        "class_entropy_exceeds_familywise_exact_eig_resolution": (
            entropy > familywise_resolution
        ),
        "class_bayes_risk_exceeds_single_action_exact_eig_resolution": (
            bayes_risk > exact_eig_epsabs
        ),
    }
    return {
        "schema": "scientific-hypothesis-bank-viability-v1",
        "model": model.stable_hash,
        "target": target.stable_hash,
        "inference_plan": inference.to_dict(),
        "inference_plan_identity": inference.stable_hash,
        "candidate_binding_count": len(model.candidate_bindings),
        "distinct_structural_support_count": len(model.bank.structures),
        "operational_class_count": len(probabilities),
        "class_entropy_nats": entropy,
        "effective_class_count": float(np.exp(entropy)),
        "class_bayes_risk": bayes_risk,
        "exact_eig_epsabs": float(exact_eig_epsabs),
        "candidate_action_count": int(len(actions)),
        "familywise_utility_resolution_nats": familywise_resolution,
        "decisions": decisions,
        "passed": all(decisions.values()),
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }


def audit_decision_risk_target(model, target, actions, exact_epsabs):
    """Certify decision-risk utility for one already-frozen target."""
    if exact_epsabs != EXACT_CLASS_EIG_EPSABS:
        raise ValueError("decision-risk Gate must match registered absolute tolerance")
    engine = model.engine(target.model_identity)
    components = predictive_components_for_partition(
        engine, target.initial_posterior, target.partition, actions)
    exact = exact_class_decision_risk_reduction_shared_actions(
        components, epsabs=exact_epsabs)
    lower = np.maximum(0.0, exact.scores - exact.quadrature_errors)
    upper = exact.scores + exact.quadrature_errors
    leader = int(np.argmax(lower))
    resolution = float(len(actions) * exact_epsabs)
    decisions = {
        "all_intervals_finite_and_ordered": bool(
            np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
            and np.all(lower <= upper)),
        "maximum_lower_bound_exceeds_familywise_resolution": bool(
            lower[leader] > resolution),
        "selected_upper_bound_within_prior_bayes_risk": bool(
            upper[leader] <= bayes_zero_one_decision_risk(
                np.asarray(target.partition.class_probabilities))
            + exact.quadrature_errors[leader] + 32.0 * np.finfo(float).eps),
    }
    return {
        "schema": "scientific-response-free-decision-risk-utility-gate-v1",
        "model": model.stable_hash, "target": target.stable_hash,
        "utility": "expected-bayes-zero-one-operational-class-risk-reduction-v1",
        "candidate_action_count": len(actions), "selected_action_index": leader,
        "selected_score": float(exact.scores[leader]),
        "selected_lower_bound": float(lower[leader]),
        "selected_upper_bound": float(upper[leader]),
        "familywise_resolution": resolution, "decisions": decisions,
        "passed": all(decisions.values()), "candidate_response_accessed": False,
        "heldout_opened": False,
    }


def audit_frozen_decision_risk_utility(candidates, initial_data, actions, *, n_features,
                                       prior, exploration_identity, coefficient_policy,
                                       measurement_budget, exact_epsabs,
                                       source_prior_weights=None):
    """Certify a nontrivial initial-prefix decision-risk utility without responses."""
    model, target = _freeze_comparison(
        candidates, n_features, prior, exploration_identity, coefficient_policy,
        initial_data, actions, measurement_budget, source_prior_weights)
    return audit_decision_risk_target(model, target, actions, exact_epsabs)
def _execute_policy(root, model, target, actions, controls, source_identity,
                    policy, random_seed, pool, ids, evaluation):
    transaction = DiscoveryTransaction(root, model, target, actions,
        controls, source_identity=source_identity, query_policy=policy,
        random_seed=random_seed)
    manifest = transaction.run_measured_pool(pool, ids, actions)
    if evaluation is not None:
        posterior = target.initial_posterior
        curve = []
        for index in range(len(transaction.receipts) + 1):
            prediction = sum(member.probability * transaction.engine.predictive_moments(
                member, evaluation.X)[0] for member in posterior.members)
            rmse = float(np.sqrt(np.mean((prediction - evaluation.y) ** 2)))
            if not np.isfinite(rmse):
                raise ValueError("non-finite development evaluation")
            curve.append(rmse)
            if index < len(transaction.receipts):
                receipt = transaction.receipts[index]
                posterior = transaction.engine.update_one(posterior,
                    np.asarray(receipt["action"]), receipt["response"])
        if curve[0] <= 0:
            raise ValueError("zero initial RMSE cannot normalize learning curve")
        reporting = {"schema": "scientific-development-curve-v1", "target": target.stable_hash,
            "evaluation_identity": evaluation.fingerprint, "rmse": curve,
            "normalized_mean_rmse": float(np.mean(curve) / curve[0]),
            "metric_role": "opened-independent-development-evaluation-not-confirmation",
            "heldout_opened": False, "efficacy_demonstrated": False}
        _publish(root / "DEVELOPMENT_CURVE.json", reporting)
        manifest = {**manifest, "development_curve": reporting}
    return manifest


def run_frozen_system_comparison(root, candidates, initial_data, pool,
                                 candidate_ids, *, n_features, prior,
                                 exploration_identity, coefficient_policy,
                                 measurement_budget, controls, source_identity,
                                 random_seed, policy_wall_time_seconds=None,
                                 evaluation_data=None, source_prior_weights=None,
                                 policies=("class_eig", "random"),
                                 inference_mode="auto"):
    """Share one conditional model/H0/class map between EIG and random queries.

    Unsupported proposals abort the entire freeze. No retry, fallback, efficacy
    evaluation or formal authorization is performed by this development API.
    Both policies use the same immutable static oracle; labels are opened only
    through their own durable transaction decisions.
    """
    if type(pool) is not PoolOracle:
        raise TypeError("registered static pool required")
    if evaluation_data is not None and evaluation_data.role is not DataRole.VALIDATION:
        raise ValueError("only opened development evaluation is supported")
    policies = tuple(policies)
    if (len(policies) != 2 or policies[-1] != "random"
            or policies[0] not in {"class_eig", "decision_risk"}):
        raise ValueError("invalid registered system comparison policies")
    root = Path(root)
    if (root / "TERMINAL_FAILURE.json").exists():
        raise ValueError("terminally failed comparison cannot resume")
    ids = np.asarray(candidate_ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or np.any(ids >= len(pool.X_pool))):
        raise ValueError("invalid registered pool IDs")
    actions = pool.X_pool[ids]
    freeze_arguments = (candidates, n_features, prior, exploration_identity,
                        coefficient_policy, initial_data, actions, measurement_budget,
                        source_prior_weights)
    if policy_wall_time_seconds is None:
        model, target = _freeze_comparison(*freeze_arguments)
    else:
        (model, target), _ = run_bounded(_freeze_comparison, args=freeze_arguments,
            seconds=policy_wall_time_seconds, provider_attempts=0)
    inference = route_inference(model, requested_mode=inference_mode)
    contract = {"schema": "conditional-discovery-comparison-v1",
        "model": model.stable_hash, "target": target.stable_hash,
        "source": source_identity, "controls": vars(controls),
        "measurement_budget": measurement_budget, "random_seed": random_seed,
        "policy_wall_time_seconds": policy_wall_time_seconds,
        "evaluation_identity": None if evaluation_data is None else evaluation_data.fingerprint,
        "source_prior_weights": source_prior_weights,
        "inference_plan": inference.to_dict(),
        "inference_plan_identity": inference.stable_hash,
        "policies": list(policies), "heldout_opened": False,
        "hypothesis_audit": {
            "candidate_binding_count": len(model.candidate_bindings),
            "distinct_structural_support_count": len(model.bank.structures),
            "distinct_source_count": len({row[0] for row in model.candidate_bindings}),
            "operational_class_count": len(target.partition.class_ids),
            "initial_class_entropy_nats": float(target.partition.entropy),
            "initial_effective_class_count": float(np.exp(target.partition.entropy)),
            "candidate_response_accessed": False,
        },
        "formal_experiment_authorized": False}
    root.mkdir(parents=True, exist_ok=True)
    _publish(root / "COMPARISON_CONTRACT.json", contract)
    manifests = {}
    for policy in contract["policies"]:
        # Existing oracle contains sealed static labels; no direct label read.
        try:
            arguments = (root / policy, model, target, actions, controls,
                         source_identity, policy, random_seed, pool, ids, evaluation_data)
            if policy_wall_time_seconds is None:
                manifests[policy] = _execute_policy(*arguments)
            else:
                manifests[policy], _ = run_bounded(_execute_policy, args=arguments,
                    seconds=policy_wall_time_seconds, provider_attempts=0)
        except Exception as error:
            _publish(root / "TERMINAL_FAILURE.json", {
                "schema": "conditional-discovery-failure-v1", "policy": policy,
                "error_type": type(error).__name__, "message": str(error),
                "completed_policies": manifests, "heldout_opened": False,
                "protocol_complete": False, "efficacy_demonstrated": False})
            raise
    result = {**contract, "manifests": manifests, "protocol_complete": True,
              "efficacy_demonstrated": False}
    _publish(root / "COMPARISON_MANIFEST.json", result)
    return result


def analyze_system_contract(rows):
    """Require paired ablation integrity before exposing development metrics."""
    gate = validate_system_pairs(rows)
    return {"schema": "scientific-system-analysis-v1", "pair_gate": gate,
            "rows": [dict(row) for row in rows],
            "metric_role": "internal-validation-development-only",
            "superiority_demonstrated": False, "heldout_opened": False}
