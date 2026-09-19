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
                    "tests/test_closed_basis_refit.py", "tests/test_seed_budget_contract.py",
                    "tests/test_exploration_prediction_baseline.py",
                    "tests/test_symbolic_closed_contract.py",
                    "tests/test_system_contribution_audit.py",
                    "tests/test_system_marginal_influence.py",
                    "tests/test_system_executor.py",
                    "tests/test_pcpi_p3d2_reference_acquisition.py"])
    result = subprocess.run(command, cwd=root, check=False)
    print(json.dumps({"schema": "scientific-model-adapter-correctness-gate-v1",
        "passed": result.returncode == 0, "real_data_access": False,
        "heldout_access": False, "user_execution_authorized": False,
        "scope": "registered scientific context, provenance-bound multi-engine/LLM composition, structural refit, premeasurement hypothesis-bank viability, response-free identifiability, shared-action exact EIG, measured transactions, matched exploration orchestration and freeze rejection",
        "matched_exploration_correctness_verified": result.returncode == 0,
        "hard_resource_limits_verified": result.returncode == 0,
        "registered_data_boundary_verified": result.returncode == 0,
        "end_to_end_wiring_verified": result.returncode == 0,
        "response_free_identifiability_verified": result.returncode == 0,
        "shared_action_exact_eig_verified": result.returncode == 0,
        "hypothesis_provenance_verified": result.returncode == 0,
        "registered_scientific_context_verified": result.returncode == 0,
        "premeasurement_viability_gate_verified": result.returncode == 0,
        "marginal_decision_influence_gate_verified": result.returncode == 0,
        "offline_source_influence_screen_registered": (
            root / "scripts/run_scientific_source_influence_screen.py").is_file(),
        "clean_source_formal_freeze_passed": False,
        "plan_admit_recovery_verified": result.returncode == 0,
        "static_measured_pool_order_verified": result.returncode == 0,
        "formal_experiment_authorized": False}, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
