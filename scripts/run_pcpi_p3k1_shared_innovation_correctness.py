"""Run the no-data P3K.1 shared-innovation correctness Gate."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD,
    P3K_SHARED_INNOVATION_JOINT_METHOD,
    P3K_SHARED_INNOVATION_RESIDUAL_METHOD,
    ClassConditionalResidualState,
    ClassPartition,
    PredictiveComponents,
    SequentialReferencePosterior,
    advance_calibrated_class_posterior,
    advance_class_conditional_residual_state,
    class_conditional_semiparametric_coupling,
    estimate_class_conditional_semiparametric_eig,
    exact_class_eig,
    initialize_calibrated_class_posterior,
    initialize_class_conditional_residual_state,
    reconstruct_class_conditional_residual_state,
)
from hypothesis_mvp.pcpi import class_conditional_semiparametric as p3k
from hypothesis_mvp.pcpi.reference import (
    DyadicPolyaTreeState,
    NormalInverseGammaPrior,
    ReferenceBank,
    ReferenceStructure,
)


CONFIG = PROJECT_ROOT / "configs" / "p3k_1_shared_innovation_correctness.json"
RESULT_SCHEMA = "pcpi-p3k1-shared-innovation-correctness-result-v1"
POWERS = (0.125, 0.25, 0.5, 1.0)


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "failed_predecessor", "scientific_target",
        "joint_law", "residual_state_method", "posterior_update",
        "residual_coordinate", "within_power_sharing", "across_power_sharing",
        "initial_state_policy", "residual_depth_schedule",
        "residual_split_prior", "operational_execution_authorized",
        "real_data_access", "candidate_response_access",
        "validation_response_access", "heldout_access", "simulated_experiment",
    }
    if set(config) != required:
        raise ValueError("P3K.1 correctness config fields changed")
    if (
        config["schema"]
        != "pcpi-p3k1-shared-innovation-correctness-config-v1"
        or config["stage"] != "P3K.1"
        or config["failed_predecessor"]
        != "P3J.15/REAL_ADVANTAGE_NOT_DEMONSTRATED"
        or config["joint_law"] != P3K_SHARED_INNOVATION_JOINT_METHOD
        or config["residual_state_method"]
        != P3K_SHARED_INNOVATION_RESIDUAL_METHOD
        or config["posterior_update"]
        != P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD
        or config["residual_coordinate"]
        != "current-posterior-base-mixture-raw-pit"
        or config["within_power_sharing"]
        != "one-observable-innovation-law-shared-by-all-frozen-classes"
        or config["across_power_sharing"]
        != "sharing-forbidden-separate-strict-prefix-state-per-likelihood-power"
        or any(config[key] for key in (
            "operational_execution_authorized", "real_data_access",
            "candidate_response_access", "validation_response_access",
            "heldout_access", "simulated_experiment",
        ))
    ):
        raise ValueError("P3K.1 correctness contract changed")
    return config


def _bank() -> ReferenceBank:
    definitions = (
        ("constant", "b0", ("1",)),
        ("linear", "b0 + b1*x", ("1", "x")),
        ("linear_alias", "b0 + b1*x_alias", ("1", "x_alias")),
        ("quadratic", "b0 + b1*x + b2*x^2", ("1", "x", "x2")),
        ("cubic", "b0 + b1*x + b2*x^2 + b3*x^3", ("1", "x", "x2", "x3")),
        ("sinusoid", "b0 + b1*sin(x)", ("1", "sin_x")),
        ("reciprocal", "b0 + b1/(1+x^2)", ("1", "reciprocal_1_plus_x2")),
    )
    return ReferenceBank(
        tuple(ReferenceStructure(*item, 1.0 / len(definitions)) for item in definitions),
        NormalInverseGammaPrior(0.0, 0.5, 3.0, 0.05),
    )


def _posterior_case(power: float):
    actions = np.linspace(-1.5, 1.5, 12)[:, None]
    targets = (
        0.65 - 0.9 * actions[:, 0] + 0.4 * np.square(actions[:, 0])
        + np.asarray([
            0.03, -0.02, 0.01, -0.04, 0.02, 0.00,
            -0.01, 0.04, -0.03, 0.02, -0.02, 0.01,
        ])
    )
    engine = SequentialReferencePosterior(_bank(), power)
    fitted = engine.fit_batch(actions, targets)
    groups = ((0, 1, 2), tuple(range(3, len(fitted.members))))
    partition = ClassPartition(
        class_ids=("low-order", "higher-order"),
        member_indices=groups,
        class_probabilities=tuple(
            sum(fitted.members[index].probability for index in group)
            for group in groups
        ),
        structure_to_class=(0, 0, 0, 1, 1, 1, 1),
    )
    residual, sequential = reconstruct_class_conditional_residual_state(
        engine, actions[:4], targets[:4], actions[4:], targets[4:], partition
    )
    return engine, partition, residual, sequential


def _ranking_case() -> tuple[PredictiveComponents, ClassConditionalResidualState]:
    partition = ClassPartition(
        class_ids=("a", "b"), member_indices=((0,), (1,)),
        class_probabilities=(0.5, 0.5), structure_to_class=(0, 1),
    )
    components = PredictiveComponents(
        structure_probabilities=np.asarray([0.5, 0.5]),
        degrees_freedom=np.asarray([10.8773, 8.8448]),
        locations=np.asarray([[-0.08615, 0.11469], [-0.54845, 0.47277]]),
        scales=np.asarray([[0.42346, 0.97704], [1.24644, 1.62866]]),
        partition=partition,
    )
    state = ClassConditionalResidualState(
        class_ids=partition.class_ids,
        residual_state=DyadicPolyaTreeState(tuple([0.9] * 16)),
        observation_count=16,
        target_partition_hash=partition.stable_hash,
    )
    return components, state


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    components, shared = _ranking_case()
    uniform = initialize_class_conditional_residual_state(components.partition)
    base_scores = exact_class_eig(components, epsabs=1e-11, epsrel=1e-10).scores
    uniform_scores = estimate_class_conditional_semiparametric_eig(
        components, uniform, nodes_per_leaf=128
    ).scores
    changed = estimate_class_conditional_semiparametric_eig(
        components, shared, nodes_per_leaf=128
    )
    coupling = class_conditional_semiparametric_coupling(
        components, shared, action_index=1, nodes_per_leaf=128
    )

    prior = initialize_class_conditional_residual_state(components.partition)
    advanced, raw_pits, shared_pit, factors = (
        advance_class_conditional_residual_state(
            components, prior, action_index=0, response=0.25
        )
    )
    mixture_pit_error = abs(
        shared_pit
        - float(np.asarray(components.partition.class_probabilities) @ raw_pits)
    )

    identity_errors: dict[str, float] = {}
    powered_weight_gaps: dict[str, float] = {}
    power_state_hashes: list[str] = []
    for power in POWERS:
        engine, partition, residual, posterior = _posterior_case(power)
        state = initialize_calibrated_class_posterior(
            engine, posterior, partition, residual
        )
        powered_update = engine.update_one(
            posterior, np.asarray([0.35]), 0.52
        )
        next_state, audit = advance_calibrated_class_posterior(
            state, np.asarray([0.35]), response=0.52
        )
        identity_errors[str(power)] = float(np.max(np.abs(
            audit.class_probabilities_after
            - audit.joint_law_class_probabilities_after
        )))
        powered_weight_gaps[str(power)] = float(np.max(np.abs(
            np.asarray([
                item.probability for item in powered_update.members
            ])
            - np.asarray([
                item.probability for item in next_state.base_posterior.members
            ])
        )))
        power_state_hashes.append(next_state.residual_state.stable_hash)

    module_source = inspect.getsource(p3k)
    update_source = inspect.getsource(advance_calibrated_class_posterior)
    decisions = {
        "one_shared_state_cannot_encode_independent_class_distortions": (
            not hasattr(shared, "residual_states")
            and not hasattr(shared, "residual_laws")
            and shared.residual_law is not None
        ),
        "shared_state_is_updated_once_per_reveal": (
            advanced.observation_count == 1
            and len(advanced.residual_state.raw_pits) == 1
            and advanced.residual_state.raw_pits == (shared_pit,)
        ),
        "shared_raw_pit_is_the_observable_base_mixture_pit": (
            mixture_pit_error <= 2e-15
        ),
        "uniform_shared_law_recovers_base_eig": bool(np.allclose(
            uniform_scores, base_scores, rtol=0.0, atol=2e-5
        )),
        "nonidentity_shared_law_changes_a_certified_ranking": bool(
            int(np.argmax(base_scores)) == 0
            and int(np.argmax(changed.scores)) == 1
            and changed.scores[1] - changed.error_bounds[1]
            > changed.scores[0] + changed.error_bounds[0]
        ),
        "every_conditional_density_preserves_registered_class_mass": bool(
            coupling.maximum_conditional_normalization_error < 2e-15
            and np.array_equal(
                coupling.class_probabilities,
                np.asarray(components.partition.class_probabilities),
            )
        ),
        "every_likelihood_power_matches_its_direct_class_bayes_update": (
            max(identity_errors.values()) <= 8.0 * np.finfo(float).eps
        ),
        "tempered_powered_weights_are_not_misreported_as_predictive_bayes": (
            min(powered_weight_gaps[str(power)] for power in POWERS[:-1])
            > 1e-3
            and powered_weight_gaps["1.0"]
            <= 32.0 * np.finfo(float).eps
        ),
        "likelihood_powers_retain_separate_prequential_states": (
            len(set(power_state_hashes)) == len(POWERS)
        ),
        "normalized_predictive_update_precedes_direct_identity_check": (
            update_source.index("structure_log_predictive")
            < update_source.index("_prequential_base_update")
            < update_source.index("direct_log_joint")
            and "nominal_identity" not in update_source
        ),
        "external_response_and_experiment_surfaces_are_absent": (
            not any(token in module_source for token in (
                "candidate_targets", "validation_targets", "heldout_targets",
                "PoolOracle", "default_rng", "random.",
            ))
            and not config["operational_execution_authorized"]
        ),
    }
    decisions = {key: bool(value) for key, value in decisions.items()}
    if not all(decisions.values()):
        raise AssertionError("P3K.1 shared-innovation correctness Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": "P3K.1",
        "status": "passed-correctness-no-real-experiment-authorized",
        "decisions": decisions,
        "base_fixture_scores": [float(value) for value in base_scores],
        "shared_innovation_fixture_scores": [
            float(value) for value in changed.scores
        ],
        "shared_innovation_fixture_error_bounds": [
            float(value) for value in changed.error_bounds
        ],
        "maximum_power_class_update_identity_error": max(identity_errors.values()),
        "power_class_update_identity_errors": identity_errors,
        "powered_vs_normalized_structure_weight_gaps": powered_weight_gaps,
        "mixture_pit_identity_error": mixture_pit_error,
        "initial_calibration_factors": [float(value) for value in factors],
        "fixture_role": "hand-authored-algebraic-correctness-only",
        "simulated_experiment": False,
        "real_data_access": False,
        "candidate_response_access": False,
        "validation_response_access": False,
        "heldout_access": False,
        "operational_execution_authorized": False,
        "formal_efficacy_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(_evaluate(_load_config(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
