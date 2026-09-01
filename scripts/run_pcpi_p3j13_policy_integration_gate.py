"""Run the no-data P3J.13 policy integration source gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import P3J_POLICY_DISPATCH, P3J_REPORTING_ORDER
from hypothesis_mvp.pcpi import p3j_policy_integration, p3j_reporting


CONFIG = PROJECT_ROOT / "configs" / "p3j_13_policy_integration_gate.json"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocked = (
        "real_data_access", "real_oracle_invoked", "validation_access",
        "heldout_access", "simulated_experiment", "operational_execution_authorized",
        "formal_dataset_runner_authorized",
    )
    dispatch = inspect.getsource(
        p3j_policy_integration.dispatch_p3j_matched_policy
    )
    reporting = inspect.getsource(p3j_reporting.build_p3j_policy_artifacts)
    failure = inspect.getsource(
        p3j_policy_integration.publish_p3j_policy_failure_snapshot
    )
    decisions = {
        "protocols_frozen": (
            config.get("policy_dispatch") == P3J_POLICY_DISPATCH
            and config.get("reporting_order") == P3J_REPORTING_ORDER
        ),
        "pcpi_only_uses_transactional_path": (
            dispatch.index("policy == pcpi_policy") < dispatch.index("p3j_runner()")
        ),
        "matched_baselines_remain_legacy": (
            config.get("matched_baselines_use_legacy_runner") is True
            and dispatch.index("p3j_state is not None")
            < dispatch.index("legacy_runner()")
        ),
        "reporting_consumes_complete_query_results": (
            "measured_run.query_results" in reporting
            and "measured_run.final_state" in reporting
        ),
        "failure_snapshot_contains_no_response_values": (
            config.get("failure_snapshot_no_response_values") is True
            and '"response_values_recorded": False' in failure
        ),
        "no_direct_oracle_surface": "oracle.acquire_indices" not in (
            dispatch + reporting + failure
        ),
        "formal_dataset_runner_remains_blocked": not any(
            config.get(key) for key in blocked
        ),
    }
    if (
        config.get("schema") != "pcpi-p3j13-policy-integration-gate-config-v1"
        or config.get("stage") != "P3J.13"
        or not all(decisions.values())
    ):
        raise AssertionError("P3J.13 policy integration Gate failed")
    print(json.dumps({
        "schema": "pcpi-p3j13-policy-integration-gate-result-v1",
        "stage": "P3J.13",
        "status": "passed-policy-integration-formal-runner-blocked",
        "decisions": decisions,
        **{key: False for key in blocked},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
