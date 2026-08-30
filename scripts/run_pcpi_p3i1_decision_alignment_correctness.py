"""Run the response-free P3I.1 decision-alignment correctness Gate."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi import (
    DECISION_REGRET_DISTANCE_METRIC,
    DECISION_TARGETED_POLICY,
    P3I_COPULA_TRANSPORT_METHOD,
    ClassPartition,
    PosteriorModel,
    PredictiveComponents,
    SequentialReferencePosterior,
    aggregate_decision_equivalent_classes,
    budget_resolved_distance_threshold,
    class_partition,
    copula_transport_class_coupling,
    exact_class_eig,
    reconstruct_conditioned_likelihood_power_residual_family,
    score_decision_targeted_actions,
    semiparametric_class_coupling,
)
from hypothesis_mvp.pcpi.reference import (
    DyadicPolyaTreePredictiveLaw,
    generic_real_bank,
)


CONFIG_SCHEMA = "pcpi-p3i1-decision-alignment-correctness-config-v1"
RESULT_SCHEMA = "pcpi-p3i1-decision-alignment-correctness-result-v1"


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("stage") != "P3I.1":
        raise ValueError("P3I.1 config identity is invalid")
    blocked = (
        "simulated_experiment",
        "real_data_access",
        "validation_response_access",
        "candidate_response_access",
        "heldout_access",
        "operational_execution_authorized",
        "formal_efficacy_evidence",
    )
    if any(config.get(name) is not False for name in blocked):
        raise ValueError("P3I.1 config opened a forbidden evidence surface")
    if config.get("conditional_predictive_information_in_primary_score") is not False:
        raise ValueError("P3I.1 primary target cannot include predictive nuisance")
    return config


def _fixture_components() -> PredictiveComponents:
    return PredictiveComponents(
        np.asarray([0.30, 0.20, 0.50]),
        np.asarray([8.0, 12.0, 20.0]),
        np.asarray([[-2.0, 0.0], [-1.0, 0.2], [2.0, 0.1]]),
        np.asarray([[0.7, 1.0], [0.8, 0.9], [0.6, 1.1]]),
        ClassPartition(("left", "right"), ((0, 1), (2,)), (0.5, 0.5), (0, 0, 1)),
    )


def _fixture_residual_law() -> DyadicPolyaTreePredictiveLaw:
    return DyadicPolyaTreePredictiveLaw(
        2, np.asarray([0.50, 0.10, 0.15, 0.25]), 16
    )


def _fixture_family(powers: tuple[float, ...]):
    x = np.linspace(-1.5, 1.5, 12)[:, None]
    y = 0.75 - 1.2 * x[:, 0] + 0.55 * np.square(x[:, 0])
    engines = tuple(
        SequentialReferencePosterior(generic_real_bank(1), power)
        for power in powers
    )
    family = reconstruct_conditioned_likelihood_power_residual_family(
        engines, x[:4], y[:4], x[4:], y[4:]
    )
    models = tuple(
        PosteriorModel(state.likelihood_power, state.engine, state.posterior)
        for state in family.model_states
    )
    return x, family.model_states[-1], models, family


def _score_fixture(config: dict[str, object]):
    powers = tuple(float(value) for value in config["likelihood_power_candidates"])
    x, nominal, models, family = _fixture_family(powers)
    target = np.linspace(-1.5, 1.5, 24)[:, None]
    threshold = budget_resolved_distance_threshold(int(config["measurement_budget"]))
    classes = aggregate_decision_equivalent_classes(
        nominal.engine, nominal.posterior, target, distance_threshold=threshold
    )
    score = score_decision_targeted_actions(
        nominal.engine,
        nominal.posterior,
        np.asarray([[-1.25], [-0.50], [0.25], [1.00]]),
        target_partition=class_partition(nominal.posterior, classes),
        predictive_target_actions=target,
        representative_observed_actions=x,
        posterior_models=models,
        semiparametric_residual_family=family,
        minimum_samples=int(config["quadrature_minimum_evaluations"]),
        maximum_samples=int(config["quadrature_maximum_evaluations"]),
        error_safety_factor=float(config["quadrature_error_safety_factor"]),
        growth_factor=int(config["quadrature_growth_factor"]),
    )
    return classes, score


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    components, residual = _fixture_components(), _fixture_residual_law()
    nodes = int(config["transport_nodes_per_leaf"])
    old = semiparametric_class_coupling(components, residual, 0, nodes)
    transport = copula_transport_class_coupling(components, residual, 0, nodes)
    exact = exact_class_eig(components, epsabs=1e-11, epsrel=1e-10)
    classes, score = _score_fixture(config)
    source = inspect.getsource(score_decision_targeted_actions)
    decisions = {
        "copula_transport_changes_responses": bool(np.max(np.abs(
            transport.output_response_nodes - transport.base_response_nodes
        )) > 1.0),
        "class_information_is_transport_invariant": bool(
            transport.invariance_error <= 2e-15
            and abs(transport.mutual_information - exact.scores[0]) < 1e-4
        ),
        "legacy_projection_is_not_reused": bool(
            abs(old.mutual_information - transport.mutual_information) > 0.05
        ),
        "decision_class_uses_common_mean_regret_scale": bool(
            classes.metric == DECISION_REGRET_DISTANCE_METRIC
            and classes.quantile_levels == ()
        ),
        "primary_score_is_class_only": bool(
            score.policy == DECISION_TARGETED_POLICY
            and np.all(score.conditional_predictive_eig_scores == 0.0)
            and np.allclose(
                score.joint_class_predictive_scores,
                score.class_eig_scores,
                rtol=0.0,
                atol=2e-14,
            )
        ),
        "production_source_excludes_old_joint_dispatch": not any(
            token in source
            for token in (
                "semiparametric_class_coupling",
                "estimate_semiparametric_class_eig",
                "class_conditional_predictive_eig",
                "validation",
                "heldout",
            )
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3I.1 decision-alignment correctness Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": "P3I.1",
        "status": "passed-correctness-no-real-experiment-authorized",
        "decisions": decisions,
        "transport_method": P3I_COPULA_TRANSPORT_METHOD,
        "transport_class_marginal_quadrature_error": (
            transport.maximum_marginal_error
        ),
        "transport_information_invariance_error": transport.invariance_error,
        "legacy_projection_information": old.mutual_information,
        "transport_information": transport.mutual_information,
        "simulated_experiment": False,
        "real_data_access": False,
        "candidate_response_access": False,
        "heldout_access": False,
        "operational_execution_authorized": False,
        "formal_efficacy_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3i_1_decision_alignment_correctness.json"),
    )
    args = parser.parse_args()
    print(json.dumps(_evaluate(_load_config(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
