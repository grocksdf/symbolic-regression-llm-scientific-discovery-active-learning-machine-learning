"""Run the response-free P3J.5 durable-checkpoint source Gate."""

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
    P3J_CHECKPOINT_PUBLICATION,
    P3J_CHECKPOINT_SCHEMA,
)
from hypothesis_mvp.pcpi import p3j_checkpoint


CONFIG = PROJECT_ROOT / "configs" / "p3j_5_durable_checkpoint_gate.json"
RESULT_SCHEMA = "pcpi-p3j5-durable-checkpoint-gate-result-v1"


def _load(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "checkpoint_schema", "publication",
        "prefix_policy", "resume_identity", "partial_output_policy",
        "stale_staging_policy", "runner_composition_authorized",
        "operational_execution_authorized", "real_data_access",
        "candidate_response_access", "validation_response_access",
        "heldout_access", "simulated_experiment",
    }
    if set(config) != required:
        raise ValueError("P3J.5 checkpoint config fields changed")
    if (
        config["schema"] != "pcpi-p3j5-durable-checkpoint-gate-config-v1"
        or config["stage"] != "P3J.5"
        or config["checkpoint_schema"] != P3J_CHECKPOINT_SCHEMA
        or config["publication"] != P3J_CHECKPOINT_PUBLICATION
        or any(
            config[key]
            for key in (
                "runner_composition_authorized",
                "operational_execution_authorized",
                "real_data_access",
                "candidate_response_access",
                "validation_response_access",
                "heldout_access",
                "simulated_experiment",
            )
        )
    ):
        raise ValueError("P3J.5 checkpoint authorization contract changed")
    return config


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    publish = inspect.getsource(p3j_checkpoint._publish)
    plan = inspect.getsource(p3j_checkpoint.build_p3j_chunk_plan)
    load = inspect.getsource(p3j_checkpoint.load_p3j_checkpoint)
    append = inspect.getsource(p3j_checkpoint.append_p3j_checkpoint_chunk)
    release = inspect.getsource(p3j_checkpoint.require_complete_p3j_scores)
    decisions = {
        "publication_fsyncs_staging_before_atomic_replace": (
            publish.index("os.fsync") < publish.index("os.replace")
            and 'staging.open("x"' in publish
        ),
        "plan_binds_all_predictive_and_discretization_identities": all(
            token in plan
            for token in (
                "_components_hash(components)",
                "state.stable_hash",
                "components.partition.stable_hash",
                "nodes_per_leaf",
                "action_chunk_size",
            )
        ),
        "resume_verifies_hash_chained_contiguous_prefix": all(
            token in load
            for token in (
                "item[\"start\"] != completed",
                "item[\"previous_hash\"] != previous",
                "chunk hash mismatch",
            )
        ),
        "append_accepts_only_next_identity_bound_chunk": all(
            token in append
            for token in (
                "chunk.start != snapshot.completed_action_count",
                "chunk.residual_state_hash != plan.residual_state_hash",
                "chunk.target_partition_hash != plan.target_partition_hash",
            )
        ),
        "partial_checkpoint_cannot_release_scores": (
            "not checkpoint.complete" in release
            and "argmax" not in release
            and "checkpoint.scores[" not in release
        ),
        "runner_and_real_execution_remain_blocked": (
            not config["runner_composition_authorized"]
            and not config["operational_execution_authorized"]
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.5 durable-checkpoint Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": "P3J.5",
        "status": "passed-durable-checkpoint-runner-composition-blocked",
        "decisions": decisions,
        "checkpoint_schema": P3J_CHECKPOINT_SCHEMA,
        "publication": P3J_CHECKPOINT_PUBLICATION,
        "partial_output_released": False,
        "runner_composition_authorized": False,
        "operational_execution_authorized": False,
        "real_data_access": False,
        "candidate_response_access": False,
        "validation_response_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "formal_efficacy_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(_evaluate(_load(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
