"""Run the no-data P3J.12 durable measured-run source gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import P3J_MEASURED_RUN_PROTOCOL
from hypothesis_mvp.pcpi import p3j_measured_pool, p3j_measured_run, p3j_reveal_runner


CONFIG = PROJECT_ROOT / "configs" / "p3j_12_measured_run_gate.json"
RESULT_SCHEMA = "pcpi-p3j12-measured-run-gate-result-v1"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocked = (
        "real_data_access", "real_oracle_invoked", "validation_access",
        "heldout_access", "simulated_experiment", "operational_execution_authorized",
        "formal_dataset_runner_authorized",
    )
    coordinator = inspect.getsource(
        p3j_measured_run.run_p3j_measured_pool_acquisition
    )
    recovery = inspect.getsource(
        p3j_measured_pool.resume_or_run_p3j_measured_pool_query
    )
    receipt_recovery = inspect.getsource(
        p3j_reveal_runner.resume_p3j_formal_response
    )
    decisions = {
        "protocol_frozen": (
            config.get("protocol") == P3J_MEASURED_RUN_PROTOCOL
            and config.get("durable_reveal_recovery") is True
            and config.get("contiguous_manifest_required") is True
        ),
        "durable_receipt_precedes_any_new_oracle_access": (
            recovery.index("receipt_path.is_file")
            < recovery.index("return run_p3j_measured_pool_query(")
        ),
        "receipt_recovery_rechecks_identity_and_reconstructs_advance": (
            "identity_hash" in receipt_recovery
            and "admit_p3j_formal_response" in receipt_recovery
        ),
        "coordinator_uses_only_guarded_query_adapter": (
            "resume_or_run_p3j_measured_pool_query" in coordinator
            and "oracle.acquire_indices" not in coordinator
        ),
        "candidate_domain_shrinks_after_matching_reveal": (
            coordinator.index("resume_or_run_p3j_measured_pool_query")
            < coordinator.index("np.delete")
        ),
        "complete_contiguous_manifest_is_terminal": (
            coordinator.index("np.delete")
            < coordinator.index("finalize_p3j_run_manifest")
        ),
        "formal_dataset_runner_remains_blocked": not any(
            config.get(key) for key in blocked
        ),
    }
    if (
        config.get("schema") != "pcpi-p3j12-measured-run-gate-config-v1"
        or config.get("stage") != "P3J.12"
        or config.get("completed_query_oracle_reopen_authorized") is not False
        or not all(decisions.values())
    ):
        raise AssertionError("P3J.12 measured-run Gate failed")
    print(json.dumps({
        "schema": RESULT_SCHEMA,
        "stage": "P3J.12",
        "status": "passed-durable-measured-run-dataset-runner-blocked",
        "decisions": decisions,
        "protocol": P3J_MEASURED_RUN_PROTOCOL,
        **{key: False for key in blocked},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
