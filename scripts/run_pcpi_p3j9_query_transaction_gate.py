"""Run the no-data P3J.9 indivisible query-transaction source Gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import P3J_QUERY_DECISION_SCHEMA
from hypothesis_mvp.pcpi import p3j_query_runner


CONFIG = PROJECT_ROOT / "configs" / "p3j_9_query_transaction_gate.json"
RESULT_SCHEMA = "pcpi-p3j9-query-transaction-gate-result-v1"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocked = (
        "real_data_access", "candidate_response_access", "validation_access",
        "heldout_access", "simulated_experiment", "response_admission_authorized",
        "operational_execution_authorized", "formal_dataset_runner_authorized",
    )
    expected_order = [
        "reject-prior-terminal-failure",
        "resume-complete-decision-without-rescoring",
        "score-identity-bound-complete-family",
        "publish-one-response-free-decision",
        "on-first-error-publish-terminal-no-retry",
    ]
    if (
        config.get("schema") != "pcpi-p3j9-query-transaction-gate-config-v1"
        or config.get("stage") != "P3J.9"
        or config.get("decision_schema") != P3J_QUERY_DECISION_SCHEMA
        or config.get("transaction_order") != expected_order
        or any(config.get(key) for key in blocked)
    ):
        raise ValueError("P3J.9 query-transaction contract changed")
    run = inspect.getsource(p3j_query_runner.run_p3j_formal_query)
    load = inspect.getsource(p3j_query_runner._load_decision)
    payload = inspect.getsource(p3j_query_runner._decision_payload)
    progress = inspect.getsource(p3j_query_runner.p3j_query_progress_snapshot)
    module = inspect.getsource(p3j_query_runner)
    decisions = {
        "prior_terminal_precedes_resume_or_scoring": (
            run.index("terminal_failure_path.exists")
            < run.index("decision_path.exists")
            < run.index("score_identity_bound_p3j_query")
        ),
        "complete_decision_resumes_without_rescoring": (
            "return _load_decision" in run
            and run.index("return _load_decision")
            < run.index("score_identity_bound_p3j_query")
        ),
        "decision_is_response_free_and_identity_bound": (
            '"response_opened": False' in payload
            and '"identity_hash": workspace.identity.stable_hash' in payload
            and "payload[\"identity_hash\"] != workspace.identity.stable_hash" in load
        ),
        "first_error_is_terminal_and_propagated": (
            "except Exception as error" in run
            and "publish_p3j_terminal_failure" in run
            and run.rstrip().endswith(
                "return _load_decision(workspace, candidate_actions, candidate_ids)"
            )
        ),
        "progress_is_response_free_checkpoint_inspection": (
            'glob("model-*")' in progress
            and 'glob("*nodes-*.json")' in progress
            and 'payload.get("complete")' in progress
        ),
        "no_oracle_response_validation_heldout_or_rng_surface": not any(
            token in module for token in (
                "PoolOracle", "candidate_targets", "validation", "heldout", "rng"
            )
        ),
        "formal_dataset_runner_remains_blocked": not any(
            config[key] for key in blocked
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.9 query-transaction Gate failed")
    print(json.dumps({
        "schema": RESULT_SCHEMA,
        "stage": "P3J.9",
        "status": "passed-query-transaction-dataset-runner-blocked",
        "decisions": decisions,
        "decision_schema": P3J_QUERY_DECISION_SCHEMA,
        **{key: False for key in blocked},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
