"""Run the P3J.15 no-data formal execution freeze gate."""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.hypotheses import runtime_dependency_hash, runtime_dependency_snapshot
from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3j15_formal_real_acquisition as runner


CONFIG = PROJECT_ROOT / "configs" / "p3j_15_formal_real_acquisition.json"
SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3j15_supervised.ps1"


def main() -> int:
    config = runner.validate_p3j15_config(CONFIG, PROJECT_ROOT)
    run_source = inspect.getsource(shared.run)
    adapter = inspect.getsource(shared._run_p3j_shared_policy)
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    decisions = {
        "config_hash_frozen": (
            sha256(CONFIG.read_bytes()).hexdigest() == runner.CONFIG_SHA256
        ),
        "runtime_identity_frozen": (
            runtime_dependency_hash(runtime_dependency_snapshot())
            == runner.RUNTIME_HASH
        ),
        "p3j_state_precedes_transactional_policy": (
            run_source.index("initialize_operational_class_conditional_state")
            < run_source.index("_run_p3j_shared_policy")
        ),
        "pcpi_has_no_direct_oracle_surface": (
            "run_p3j_outer_policy" in adapter
            and "oracle.acquire_indices" not in adapter
        ),
        "baselines_retain_shared_policy_path": "else:" in run_source
        and "summary, curves, queries = _run_policy(" in run_source,
        "resume_is_checkpoint_only_and_terminal_fail_closed": (
            "allow_p3j_resume" in inspect.getsource(shared._prepare_output)
            and "POLICY_FAILURE.json" in inspect.getsource(shared._prepare_output)
        ),
        "supervisor_freezes_source_config_python": all(
            token in supervisor for token in (
                "ExpectedCommit", "ExpectedTree", "ExpectedConfigHash",
                "ExpectedPythonHash",
            )
        ),
        "supervisor_reports_query_checkpoint_progress": all(
            token in supervisor for token in (
                "PROGRESS.json", "completed_models", "nodes_per_leaf",
            )
        ),
        "heldout_closed_and_execution_explicit": (
            config["heldout_state"] == "closed"
            and config["operational_execution_authorized"] is True
            and config["formal_dataset_runner_authorized"] is True
        ),
        "gate_accesses_no_registered_data": not hasattr(
            runner, "load_registered_real_dataset"
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.15 formal freeze Gate failed")
    print(json.dumps({
        "schema": "pcpi-p3j15-formal-execution-freeze-gate-result-v1",
        "stage": "P3J.15",
        "status": "passed-user-real-execution-authorized",
        "decisions": decisions,
        "config_sha256": runner.CONFIG_SHA256,
        "runtime_dependency_hash": runner.RUNTIME_HASH,
        "real_data_access": False,
        "real_oracle_invoked": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "formal_experiment": False,
        "user_execution_authorized": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
