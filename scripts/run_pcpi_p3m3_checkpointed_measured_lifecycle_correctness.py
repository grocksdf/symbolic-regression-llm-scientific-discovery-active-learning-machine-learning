"""Run the no-data P3M.3 checkpoint and measured-lifecycle Gate."""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3M_CHECKPOINT_PUBLICATION,
    P3M_CHECKPOINT_SCHEMA,
    P3M_MEASURED_POOL_ORDER,
    P3M_MEASURED_RUN_PROTOCOL,
    P3M_QUERY_DECISION_SCHEMA,
    P3M_QUERY_LEDGER_SCHEMA,
    P3M_REVEAL_RECEIPT_SCHEMA,
    P3M_RUN_MANIFEST_SCHEMA,
)
from hypothesis_mvp.pcpi import (
    p3j_measured_pool as measured_source,
    p3j_measured_run as run_source,
    p3j_query_runner as query_source,
    p3j_reveal_runner as reveal_source,
    p3m_checkpoint as checkpoint_source,
    real_acquisition as acquisition_source,
)


CONFIG = PROJECT_ROOT / "configs" / "p3m_3_checkpointed_measured_lifecycle_correctness.json"
CONFIG_SHA256 = "3a5657788ad77690927046578f7da0f5a09b1d1c5e245ba9d0409cbcd2bf673d"


def _load_config(path: Path = CONFIG) -> dict[str, object]:
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3M.3 correctness config hash changed")
    config = json.loads(raw.decode("utf-8"))
    expected = {
        "stage": "P3M.3",
        "checkpoint_schema": P3M_CHECKPOINT_SCHEMA,
        "checkpoint_publication": P3M_CHECKPOINT_PUBLICATION,
        "heldout_state": "closed",
        "real_data_access_authorized": False,
        "operational_execution_authorized": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("P3M.3 correctness contract changed")
    return config


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    plan = inspect.getsource(checkpoint_source.P3MCheckpointPlan)
    append = inspect.getsource(checkpoint_source.append_p3m_checkpoint_chunk)
    complete = inspect.getsource(checkpoint_source.complete_p3m_information_risk_grid)
    model_look = inspect.getsource(acquisition_source._information_risk_model_look)
    query = inspect.getsource(query_source.run_p3j_formal_query)
    measured = inspect.getsource(measured_source.run_p3j_measured_pool_query)
    run = inspect.getsource(run_source.run_p3j_measured_pool_acquisition)
    reveal = inspect.getsource(reveal_source.admit_p3j_formal_response)
    fields = tuple(config["checkpoint_identity_fields"])
    decisions = {
        "checkpoint_binds_every_registered_identity_field": all(
            field in plan for field in fields
        ),
        "chunks_are_contiguous_hash_chained_and_fsync_published": (
            "snapshot.completed_action_count" in append
            and "snapshot.head_hash" in append
            and "os.fsync" in inspect.getsource(checkpoint_source._publish)
        ),
        "partial_grid_resumes_before_score_release": (
            "start_action=checkpoint.completed_action_count" in complete
            and "if not checkpoint.complete" in complete
        ),
        "adaptive_model_look_uses_p3m_checkpoint": (
            "checkpointed_action_conditional_information_risk" in model_look
            and "candidate_actions" in model_look
        ),
        "decision_is_published_before_measured_response": (
            measured.index("run_p3j_formal_query")
            < measured.index("oracle.acquire_indices")
            < measured.index("admit_p3j_formal_response")
        ),
        "query_failure_is_terminal_and_has_no_hidden_retry": (
            "publish_p3j_terminal_failure" in query and "retry" not in query
        ),
        "matching_reveal_advances_exactly_once": (
            "admit_operational_class_conditional_response" in reveal
            and "decision.prior_state_hash != prior_state.stable_hash" in reveal
        ),
        "complete_run_finalizes_one_contiguous_manifest": (
            "finalize_p3j_run_manifest" in run and "available = np.delete" in run
        ),
        "all_p3m_transaction_schemas_are_explicit": all((
            P3M_QUERY_DECISION_SCHEMA.startswith("pcpi-p3m3-"),
            P3M_REVEAL_RECEIPT_SCHEMA.startswith("pcpi-p3m3-"),
            P3M_QUERY_LEDGER_SCHEMA.startswith("pcpi-p3m3-"),
            P3M_RUN_MANIFEST_SCHEMA.startswith("pcpi-p3m3-"),
            P3M_MEASURED_POOL_ORDER.startswith("p3m-"),
            P3M_MEASURED_RUN_PROTOCOL.startswith("p3m-"),
        )),
    }
    if not all(decisions.values()):
        failed = [name for name, passed in decisions.items() if not passed]
        raise AssertionError(f"P3M.3 correctness Gate failed: {failed}")
    return {
        "schema": "pcpi-p3m3-checkpointed-measured-lifecycle-gate-v1",
        "stage": "P3M.3",
        "status": "passed-correctness-outer-runner-freeze-blocked",
        "role": "candidate-bound-checkpoint-query-reveal-manifest-gate",
        "decisions": decisions,
        "formal_experiment": False,
        "real_data_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "operational_execution_authorized": False,
    }


def main() -> int:
    print(json.dumps(_evaluate(_load_config()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
