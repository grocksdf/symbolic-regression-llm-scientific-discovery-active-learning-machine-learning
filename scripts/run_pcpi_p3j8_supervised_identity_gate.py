"""Run the no-data P3J.8 supervised identity source Gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import P3J_RUN_IDENTITY_SCHEMA, P3J_RUN_PUBLICATION
from hypothesis_mvp.pcpi import p3j_run_identity


CONFIG = PROJECT_ROOT / "configs" / "p3j_8_supervised_identity_gate.json"
SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3j8_supervised.ps1"
RESULT_SCHEMA = "pcpi-p3j8-supervised-identity-gate-result-v1"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocked = (
        "real_data_access", "candidate_response_access", "validation_access",
        "heldout_access", "simulated_experiment",
        "operational_execution_authorized", "formal_runner_authorized",
    )
    expected_fields = {
        "source_git_tree", "config_sha256", "dataset_id", "seed", "query_index",
        "candidate_ids_hash", "candidate_actions_hash",
        "predictive_target_actions_hash", "representative_observed_actions_hash",
        "operational_state_hash",
    }
    if (
        config.get("schema") != "pcpi-p3j8-supervised-identity-gate-config-v1"
        or config.get("stage") != "P3J.8"
        or config.get("query_identity_schema") != P3J_RUN_IDENTITY_SCHEMA
        or config.get("publication") != P3J_RUN_PUBLICATION
        or set(config.get("identity_fields", ())) != expected_fields
        or config.get("supervisor_mode") != "preflight-only"
        or any(config.get(key) for key in blocked)
    ):
        raise ValueError("P3J.8 supervised identity contract changed")
    build = inspect.getsource(p3j_run_identity.build_p3j_formal_query_identity)
    scoring = inspect.getsource(p3j_run_identity.score_identity_bound_p3j_query)
    terminal = inspect.getsource(p3j_run_identity.publish_p3j_terminal_failure)
    publication = inspect.getsource(p3j_run_identity._publish_no_overwrite)
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    decisions = {
        "query_identity_binds_all_selection_inputs": all(
            field in build for field in expected_fields
        ),
        "identity_check_precedes_checkpointed_scoring": (
            scoring.index("observed != frozen")
            < scoring.index("score_checkpointed_operational_class_conditional_candidates")
        ),
        "terminal_failure_is_fsync_no_overwrite": (
            publication.index("os.fsync") < publication.index("os.link")
            and '"retry_authorized": False' in terminal
        ),
        "supervisor_checks_source_config_python_and_clean_tree": all(
            token in supervisor for token in (
                "status --porcelain=v1", "rev-parse HEAD", "HEAD^{tree}",
                "Get-FileHash", "ExpectedPythonHash",
            )
        ),
        "supervisor_is_preflight_only": (
            "if (-not $PreflightOnly)" in supervisor
            and "formal execution remains blocked" in supervisor
            and "Start-Process" not in supervisor
        ),
        "formal_runner_remains_blocked": not any(config[key] for key in blocked),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.8 supervised identity Gate failed")
    print(json.dumps({
        "schema": RESULT_SCHEMA,
        "stage": "P3J.8",
        "status": "passed-supervised-identity-formal-runner-blocked",
        "decisions": decisions,
        "query_identity_schema": P3J_RUN_IDENTITY_SCHEMA,
        "publication": P3J_RUN_PUBLICATION,
        **{key: False for key in blocked},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
