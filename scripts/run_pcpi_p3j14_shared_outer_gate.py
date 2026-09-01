"""Run the no-data P3J.14 shared outer-runner gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import p3j_outer_runner
from scripts import run_pcpi_p3j14_shared_outer as runner


CONFIG = PROJECT_ROOT / "configs" / "p3j_14_shared_outer_runner.json"


def main() -> int:
    config = runner._load_config(CONFIG, PROJECT_ROOT)
    compose = inspect.getsource(runner._compose_policy_call)
    outer = inspect.getsource(p3j_outer_runner.run_p3j_outer_policy)
    blocked = inspect.getsource(runner.run)
    decisions = {
        "frozen_config_hash_verified": runner.CONFIG_SHA256 == (
            __import__("hashlib").sha256(CONFIG.read_bytes()).hexdigest()
        ),
        "pcpi_routes_to_transactional_outer_policy": (
            "dispatch_p3j_matched_policy" in compose
            and "p3j_call(**p3j_kwargs)" in compose
        ),
        "baselines_route_to_unchanged_shared_policy_runner": (
            "legacy_call(**legacy_kwargs)" in compose
            and runner._run_policy.__module__ == "scripts.run_pcpi_p3b_real"
        ),
        "transaction_precedes_reporting_and_summary": (
            outer.index("run_p3j_measured_pool_acquisition")
            < outer.index("build_p3j_policy_artifacts")
            < outer.index("summarize_p3j_policy_artifacts")
        ),
        "failure_snapshot_wraps_complete_outer_composition": (
            outer.index("try:") < outer.index("publish_p3j_policy_failure_snapshot")
        ),
        "real_access_fails_before_any_dataset_loader": (
            "PermissionError" in blocked
            and "load_registered_real_dataset" not in blocked
        ),
        "authorization_remains_closed": (
            config["operational_execution_authorized"] is False
            and config["formal_dataset_runner_authorized"] is False
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.14 shared outer-runner Gate failed")
    print(json.dumps({
        "schema": "pcpi-p3j14-shared-outer-runner-gate-result-v1",
        "stage": "P3J.14",
        "status": "passed-shared-outer-runner-real-execution-blocked",
        "decisions": decisions,
        "real_data_access": False,
        "real_oracle_invoked": False,
        "validation_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "operational_execution_authorized": False,
        "formal_dataset_runner_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
