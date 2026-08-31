"""Run the no-data P3J.10 reveal/advance/manifest source Gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3J_QUERY_LEDGER_SCHEMA,
    P3J_REVEAL_RECEIPT_SCHEMA,
    P3J_RUN_MANIFEST_SCHEMA,
)
from hypothesis_mvp.pcpi import p3j_reveal_runner


CONFIG = PROJECT_ROOT / "configs" / "p3j_10_reveal_manifest_gate.json"
RESULT_SCHEMA = "pcpi-p3j10-reveal-manifest-gate-result-v1"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocked = (
        "real_data_access", "real_candidate_response_access", "validation_access",
        "heldout_access", "simulated_experiment", "operational_execution_authorized",
        "formal_dataset_runner_authorized",
    )
    expected_order = [
        "load-published-decision", "validate-matching-reveal",
        "publish-no-overwrite-receipt", "advance-complete-family-once",
        "publish-state-linked-query-ledger",
    ]
    if (
        config.get("schema") != "pcpi-p3j10-reveal-manifest-gate-config-v1"
        or config.get("stage") != "P3J.10"
        or config.get("reveal_receipt_schema") != P3J_REVEAL_RECEIPT_SCHEMA
        or config.get("query_ledger_schema") != P3J_QUERY_LEDGER_SCHEMA
        or config.get("run_manifest_schema") != P3J_RUN_MANIFEST_SCHEMA
        or config.get("admission_order") != expected_order
        or config.get("response_admission_composed") is not True
        or any(config.get(key) for key in blocked)
    ):
        raise ValueError("P3J.10 reveal/manifest contract changed")
    admit = inspect.getsource(p3j_reveal_runner.admit_p3j_formal_response)
    receipt = inspect.getsource(p3j_reveal_runner._load_or_publish_receipt)
    ledger = inspect.getsource(p3j_reveal_runner._query_ledger_payload)
    manifest = inspect.getsource(p3j_reveal_runner.finalize_p3j_run_manifest)
    decisions = {
        "decision_precedes_receipt_and_advance": (
            admit.index("_load_decision")
            < admit.index("_load_or_publish_receipt")
            < admit.index("admit_operational_class_conditional_response")
        ),
        "reveal_receipt_is_no_overwrite_and_exact_match": (
            "observed != expected" in receipt
            and "_publish_no_overwrite" in receipt
        ),
        "all_four_models_advance_exactly_once": (
            "admit_operational_class_conditional_response" in admit
            and "next_residual_observation_count" in ledger
            and "next_calibrated_update_count" in ledger
        ),
        "query_ledger_is_state_linked_and_heldout_closed": all(
            token in ledger for token in (
                '"prior_state_hash"', '"next_state_hash"',
                '"heldout_opened": False', '"selection_used_heldout": False',
            )
        ),
        "run_manifest_requires_contiguous_linked_queries": all(
            token in manifest for token in (
                "identity.query_index != expected_query",
                'ledger.get("prior_state_hash") != lineage[key]',
                'ledger.get("complete") is not True',
                'failure_count": 0',
            )
        ),
        "first_advance_error_is_terminal": (
            "except Exception as error" in admit
            and "publish_p3j_terminal_failure" in admit
        ),
        "formal_dataset_runner_remains_blocked": not any(
            config[key] for key in blocked
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.10 reveal/manifest Gate failed")
    print(json.dumps({
        "schema": RESULT_SCHEMA,
        "stage": "P3J.10",
        "status": "passed-reveal-manifest-dataset-runner-blocked",
        "decisions": decisions,
        "reveal_receipt_schema": P3J_REVEAL_RECEIPT_SCHEMA,
        "query_ledger_schema": P3J_QUERY_LEDGER_SCHEMA,
        "run_manifest_schema": P3J_RUN_MANIFEST_SCHEMA,
        "response_admission_composed": True,
        **{key: False for key in blocked},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
