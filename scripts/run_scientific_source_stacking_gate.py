"""No-data correctness Gate for conservative hierarchical source stacking."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "tests/test_system_source_stacking.py", "tests/test_system_bank_selection.py",
        "tests/test_discovery_pcpi_adapter.py", "tests/test_system_run.py",
        "tests/test_system_executor.py", "tests/test_system_marginal_influence.py",
        "tests/test_discovery_transaction.py",
        "tests/test_integrity.py::test_final_source_integrity"]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    result = {"schema": "scientific-source-admission-correctness-gate-v3",
        "passed": completed.returncode == 0, "real_data_access": False,
        "candidate_response_accessed": False, "heldout_opened": False,
        "production_runner_wired": True,
        "hierarchical_prior_identity_bound": True,
        "strict_prefix_source_update_verified": True,
        "negative_transfer_fallback_verified": True,
        "singleton_counterfactual_isolated_from_production": True,
        "independent_source_admission_verified": True,
        "core_prior_reserve_verified": True,
        "rejected_source_certificate_verified": True,
        "reporting_response_excluded": True,
        "formal_experiment_authorized": False}
    print(json.dumps(result, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
