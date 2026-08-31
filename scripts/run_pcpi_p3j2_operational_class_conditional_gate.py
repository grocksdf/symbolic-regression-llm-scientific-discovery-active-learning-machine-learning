"""Run the response-free P3J.2 operational source-composition Gate."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3H_OPERATIONAL_POWERS,
    P3J_OPERATIONAL_LIFECYCLE,
)
from hypothesis_mvp.pcpi import operational_class_conditional as operational


CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "p3j_2_operational_class_conditional_correctness.json"
)
RESULT_SCHEMA = "pcpi-p3j2-operational-class-conditional-gate-v1"


def _load(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "stage",
        "predecessor",
        "lifecycle",
        "likelihood_power_family",
        "initial_response_roles",
        "initial_calibration_evidence_policy",
        "revealed_response_policy",
        "unresolved_ranking_action",
        "operational_execution_authorized",
        "real_data_access",
        "candidate_response_access",
        "validation_response_access",
        "heldout_access",
        "simulated_experiment",
    }
    if set(config) != expected:
        raise ValueError("P3J.2 config fields changed")
    if (
        config["schema"]
        != "pcpi-p3j2-operational-class-conditional-correctness-config-v1"
        or config["stage"] != "P3J.2"
        or config["lifecycle"] != P3J_OPERATIONAL_LIFECYCLE
        or tuple(config["likelihood_power_family"]) != P3H_OPERATIONAL_POWERS
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
        raise ValueError("P3J.2 frozen source contract changed")
    return config


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    source = inspect.getsource(operational)
    score_source = inspect.getsource(
        operational.score_operational_class_conditional_candidates
    )
    admit_source = inspect.getsource(
        operational.admit_operational_class_conditional_response
    )
    initialize_source = inspect.getsource(
        operational.initialize_operational_class_conditional_state
    )
    decisions = {
        "complete_frozen_power_family_is_required": all(
            token in source
            for token in ("P3H_OPERATIONAL_POWERS", "complete frozen family")
        ),
        "initial_roles_are_the_only_initial_response_inputs": all(
            token in initialize_source
            for token in (
                "conditioning_targets",
                "residual_targets",
                "reconstruct_class_conditional_residual_state",
            )
        ),
        "raw_initial_history_is_erased_after_reconstruction": all(
            token not in inspect.getsource(operational.OperationalClassConditionalState)
            for token in ("conditioning_targets", "residual_targets")
        ),
        "same_calibrated_states_drive_acquisition": all(
            token in score_source
            for token in (
                "score_class_conditional_decision_actions",
                "calibrated_posterior_states=state.model_states",
            )
        ),
        "response_free_decision_precedes_selection": (
            score_source.index("score_class_conditional_decision_actions")
            < score_source.index("selected = select_acquisition_candidate")
        ),
        "matching_decision_guard_precedes_all_model_updates": (
            admit_source.index("decision.prior_state_hash")
            < admit_source.index("advance_calibrated_class_posterior")
        ),
        "every_model_advances_once_without_retry": (
            "for item in state.model_states" in admit_source
            and "while " not in admit_source
            and "retry" not in admit_source
        ),
        "external_response_surfaces_are_absent": not any(
            token in source
            for token in (
                "candidate_targets",
                "validation",
                "heldout",
                "PoolOracle",
                "rng",
            )
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.2 operational source-composition Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": config["stage"],
        "status": "passed-source-composition-no-real-experiment-authorized",
        "decisions": decisions,
        "likelihood_power_family": list(P3H_OPERATIONAL_POWERS),
        "formal_efficacy_evidence": False,
        "operational_execution_authorized": False,
        "real_data_access": False,
        "candidate_response_access": False,
        "validation_response_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(_evaluate(_load(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
