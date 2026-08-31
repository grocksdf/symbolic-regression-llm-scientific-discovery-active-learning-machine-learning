"""Run the no-data P3J.6 checkpointed-estimator source Gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import P3J_CHECKPOINTED_ESTIMATOR
from hypothesis_mvp.pcpi import p3j_checkpointed


CONFIG = PROJECT_ROOT / "configs" / "p3j_6_checkpointed_estimator_gate.json"
RESULT_SCHEMA = "pcpi-p3j6-checkpointed-estimator-gate-result-v1"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        config.get("schema")
        != "pcpi-p3j6-checkpointed-estimator-gate-config-v1"
        or config.get("stage") != "P3J.6"
        or config.get("estimator") != P3J_CHECKPOINTED_ESTIMATOR
        or config.get("action_chunk_size") != 16
        or any(config.get(key) for key in (
            "real_data_access",
            "candidate_response_access",
            "validation_access",
            "heldout_access",
            "simulated_experiment",
            "operational_execution_authorized",
            "runner_execution_authorized",
        ))
    ):
        raise ValueError("P3J.6 checkpointed-estimator contract changed")
    complete = inspect.getsource(p3j_checkpointed.complete_p3j_quadrature_grid)
    estimate = inspect.getsource(
        p3j_checkpointed.checkpointed_class_conditional_semiparametric_eig
    )
    module = inspect.getsource(p3j_checkpointed)
    decisions = {
        "resume_begins_at_complete_prefix_boundary": (
            "start_action=snapshot.completed_action_count" in complete
        ),
        "each_grid_is_complete_before_score_release": (
            complete.index("append_p3j_checkpoint_chunk")
            < complete.index("require_complete_p3j_scores(snapshot)")
        ),
        "fine_and_coarse_scores_require_complete_checkpoints": (
            estimate.count("require_complete_p3j_scores") == 2
        ),
        "refinement_binds_state_partition_order_and_error_rule": all(
            token in estimate for token in (
                "order != 2 * preceding.nodes_per_leaf",
                "preceding.residual_state_hash != state.stable_hash",
                "preceding.target_partition_hash != components.partition.stable_hash",
                "preceding.error_safety_factor",
            )
        ),
        "no_response_validation_heldout_or_rng_surface": not any(
            token in module for token in (
                "candidate_targets", "validation", "heldout", "PoolOracle", "rng"
            )
        ),
        "formal_runner_remains_blocked": (
            not config["runner_execution_authorized"]
            and not config["operational_execution_authorized"]
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.6 checkpointed-estimator Gate failed")
    result = {
        "schema": RESULT_SCHEMA,
        "stage": "P3J.6",
        "status": "passed-checkpointed-estimator-formal-runner-blocked",
        "decisions": decisions,
        "estimator": P3J_CHECKPOINTED_ESTIMATOR,
        "real_data_access": False,
        "candidate_response_access": False,
        "validation_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "operational_execution_authorized": False,
        "runner_execution_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
