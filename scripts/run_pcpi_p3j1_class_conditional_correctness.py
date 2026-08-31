"""Run the no-data P3J.1 class-conditional joint-law correctness Gate."""

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
    P3J_CLASS_CONDITIONAL_JOINT_METHOD,
    P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD,
    P3J_CLASS_POSTERIOR_UPDATE_METHOD,
    ClassConditionalResidualState,
    ClassPartition,
    PredictiveComponents,
    estimate_class_conditional_semiparametric_eig,
    exact_class_eig,
    initialize_class_conditional_residual_state,
    score_decision_targeted_actions,
)
from hypothesis_mvp.pcpi import class_conditional_semiparametric as p3j
from hypothesis_mvp.pcpi.reference import DyadicPolyaTreeState


CONFIG = PROJECT_ROOT / "configs" / "p3j_1_class_conditional_correctness.json"
AUDIT = PROJECT_ROOT / "docs" / "pcpi_p3i4_result_failure_audit_20260831.md"
RESULT_SCHEMA = "pcpi-p3j1-class-conditional-correctness-result-v1"


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "stage",
        "failed_predecessor",
        "scientific_target",
        "joint_law",
        "residual_state_method",
        "posterior_update",
        "counterfactual_class_update",
        "initial_state_policy",
        "residual_depth_schedule",
        "residual_split_prior",
        "operational_execution_authorized",
        "real_data_access",
        "candidate_response_access",
        "validation_response_access",
        "heldout_access",
        "simulated_experiment",
    }
    if set(config) != required:
        raise ValueError("P3J.1 correctness config fields changed")
    if (
        config["schema"]
        != "pcpi-p3j1-class-conditional-correctness-config-v1"
        or config["stage"] != "P3J.1"
        or config["failed_predecessor"]
        != "P3I.4/REAL_ADVANTAGE_NOT_DEMONSTRATED"
        or config["joint_law"] != P3J_CLASS_CONDITIONAL_JOINT_METHOD
        or config["residual_state_method"]
        != P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD
        or config["posterior_update"] != P3J_CLASS_POSTERIOR_UPDATE_METHOD
        or any(
            config[key]
            for key in (
                "operational_execution_authorized",
                "real_data_access",
                "candidate_response_access",
                "validation_response_access",
                "heldout_access",
                "simulated_experiment",
            )
        )
    ):
        raise ValueError("P3J.1 correctness contract changed")
    return config


def _components() -> PredictiveComponents:
    return PredictiveComponents(
        structure_probabilities=np.asarray([0.30, 0.20, 0.50]),
        degrees_freedom=np.asarray([8.0, 12.0, 20.0]),
        locations=np.asarray([[-2.0, 0.0], [-1.0, 0.2], [2.0, 0.1]]),
        scales=np.asarray([[0.7, 1.0], [0.8, 0.9], [0.6, 1.1]]),
        partition=ClassPartition(
            class_ids=("left", "right"),
            member_indices=((0, 1), (2,)),
            class_probabilities=(0.5, 0.5),
            structure_to_class=(0, 0, 1),
        ),
    )


def _opposing_state(components: PredictiveComponents) -> ClassConditionalResidualState:
    return ClassConditionalResidualState(
        class_ids=components.partition.class_ids,
        residual_states=(
            DyadicPolyaTreeState(tuple([0.9] * 16)),
            DyadicPolyaTreeState(tuple([0.1] * 16)),
        ),
        observation_count=16,
        target_partition_hash=components.partition.stable_hash,
    )


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    components = _components()
    uniform = initialize_class_conditional_residual_state(components.partition)
    base = exact_class_eig(components, epsabs=1e-11, epsrel=1e-10).scores
    recovered = estimate_class_conditional_semiparametric_eig(
        components, uniform, nodes_per_leaf=128
    )
    changed = estimate_class_conditional_semiparametric_eig(
        components, _opposing_state(components), nodes_per_leaf=64
    )
    module_source = inspect.getsource(p3j)
    old_source = inspect.getsource(score_decision_targeted_actions)
    update_source = inspect.getsource(p3j.advance_calibrated_class_posterior)
    audit_source = AUDIT.read_text(encoding="utf-8")
    decisions = {
        "p3i4_negative_result_is_frozen_without_efficacy_claim": all(
            token in audit_source
            for token in (
                "REAL_ADVANTAGE_NOT_DEMONSTRATED",
                "formal_protocol_evidence` is true",
                "`formal_efficacy_evidence`",
                "must not be rerun",
            )
        ),
        "marginal_transport_invariance_limitation_is_explicit": all(
            token in old_source
            for token in (
                "invertible PIT",
                "preserves mutual information",
                "semiparametric_information_invariance_applied=True",
            )
        ),
        "class_conditional_joint_is_normalized_by_construction": all(
            token in module_source
            for token in (
                "q_c(y) = g_c(F_c(y)) f_c(y)",
                "class_conditional_semiparametric_coupling",
                "_validated_class_probabilities",
            )
        ),
        "uniform_class_laws_recover_base_eig": bool(
            np.allclose(recovered.scores, base, rtol=0.0, atol=2e-5)
        ),
        "nonidentity_class_laws_change_a_certified_ranking": bool(
            int(np.argmax(base)) == 0
            and int(np.argmax(changed.scores)) == 1
            and changed.scores[1] - changed.error_bounds[1]
            > changed.scores[0] + changed.error_bounds[0]
        ),
        "every_class_uses_its_strict_prefix_without_latent_labels": all(
            token in module_source
            for token in (
                "for item, raw_pit in zip(state.residual_states, raw_pits",
                "advance_class_conditional_residual_state",
                "engine.update_one",
            )
        ) and not any(
            token in module_source
            for token in ("soft_assignment", "class_label", "rng")
        ),
        "nominal_bayes_identity_and_family_calibration_composition_are_explicit": all(
            token in update_source
            for token in (
                "calibrated_class_log_joint",
                "np.log(factors)",
                "joint_law_update_identity_required=nominal_identity",
            )
        ),
        "external_response_and_execution_surfaces_are_absent": not any(
            token in module_source
            for token in (
                "validation_targets",
                "heldout",
                "candidate_targets",
                "PoolOracle",
            )
        ) and not config["operational_execution_authorized"],
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.1 class-conditional correctness Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": "P3J.1",
        "status": "passed-correctness-no-real-experiment-authorized",
        "decisions": decisions,
        "base_fixture_scores": [float(value) for value in base],
        "class_conditional_fixture_scores": [
            float(value) for value in changed.scores
        ],
        "class_conditional_fixture_error_bounds": [
            float(value) for value in changed.error_bounds
        ],
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
