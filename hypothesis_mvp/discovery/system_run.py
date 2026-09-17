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
from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target
from .system_evidence import validate_system_pairs
from .resource_limits import run_bounded
from hypothesis_mvp.data.roles import DataRole


def _freeze_comparison(candidates, n_features, prior, exploration_identity,
                       coefficient_policy, initial_data, actions, measurement_budget):
    model = freeze_discovery_model(candidates, n_features=n_features,
        prior=prior, exploration_identity=exploration_identity,
        coefficient_policy=coefficient_policy)
    target = freeze_discovery_target(model, initial_data, actions,
        measurement_budget=measurement_budget,
        expected_model_identity=model.stable_hash)
    return model, target


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
                                 evaluation_data=None):
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
    root = Path(root)
    if (root / "TERMINAL_FAILURE.json").exists():
        raise ValueError("terminally failed comparison cannot resume")
    ids = np.asarray(candidate_ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or np.any(ids >= len(pool.X_pool))):
        raise ValueError("invalid registered pool IDs")
    actions = pool.X_pool[ids]
    freeze_arguments = (candidates, n_features, prior, exploration_identity,
                        coefficient_policy, initial_data, actions, measurement_budget)
    if policy_wall_time_seconds is None:
        model, target = _freeze_comparison(*freeze_arguments)
    else:
        (model, target), _ = run_bounded(_freeze_comparison, args=freeze_arguments,
            seconds=policy_wall_time_seconds, provider_attempts=0)
    contract = {"schema": "conditional-discovery-comparison-v1",
        "model": model.stable_hash, "target": target.stable_hash,
        "source": source_identity, "controls": vars(controls),
        "measurement_budget": measurement_budget, "random_seed": random_seed,
        "policy_wall_time_seconds": policy_wall_time_seconds,
        "evaluation_identity": None if evaluation_data is None else evaluation_data.fingerprint,
        "policies": ["class_eig", "random"], "heldout_opened": False,
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
