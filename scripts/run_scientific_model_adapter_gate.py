"""Execute algebraic adapter checks only; never open real measurement files."""
from pathlib import Path
import json
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "tests/test_discovery_pcpi_adapter.py", "tests/test_system_candidate_registry.py",
               "tests/test_system_evidence.py", "tests/test_system_pair_gate.py",
               "tests/test_discovery_transaction.py", "tests/test_system_run.py",
               "tests/test_system_ablation.py", "tests/test_pcpi_leakage_boundaries.py", "tests/test_integrity.py"]
    command.extend(["tests/test_system_resource_limits.py", "tests/test_system_data_protocol.py",
                    "tests/test_system_executor.py"])
    result = subprocess.run(command, cwd=root, check=False)
    print(json.dumps({"schema": "scientific-model-adapter-correctness-gate-v1",
        "passed": result.returncode == 0, "real_data_access": False,
        "heldout_access": False, "user_execution_authorized": False,
        "scope": "structural refit, H0, measured transactions, matched exploration orchestration and freeze rejection",
        "matched_exploration_correctness_verified": result.returncode == 0,
        "hard_resource_limits_verified": result.returncode == 0,
        "registered_data_boundary_verified": result.returncode == 0,
        "end_to_end_wiring_verified": result.returncode == 0,
        "clean_source_formal_freeze_passed": False,
        "plan_admit_recovery_verified": result.returncode == 0,
        "static_measured_pool_order_verified": result.returncode == 0,
        "formal_experiment_authorized": False}, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
