"""Run only the response-free CERT.14 certified common-target Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert13 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert13_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert13["EXPECTED_CHECKS"])
EXPECTED_CHECKS[
    "tests/test_pcpi_p3f4_certified_function_space_common_target.py"
] = (
    "test_cert14_authorizes_only_standalone_common_target_composition",
    "test_common_plan_binds_provider_bridge_local_rj_cdf_and_sparse_hashes",
    "test_cert13_predictive_and_cert14_collapsed_targets_share_one_prior_builder",
    "test_beta_zero_is_exact_prior_target_with_zero_collapsed_log_mass",
    "test_full_inactive_collapsed_ball_contains_independent_parameter_space_identity",
    "test_bridge_potential_is_outward_difference_of_same_state_targets",
    "test_earlier_bridge_is_independent_of_later_frozen_response_value",
    "test_exact_local_rj_forward_reverse_ratios_share_certified_target",
    "test_local_rj_rejects_crossed_bridge_or_endpoint_identity",
    "test_finite_exact_mh_matrix_is_reversible_and_target_invariant",
    "test_sparse_fixed_candidate_adapter_retains_common_target_identity",
    "test_resident_engine_blocks_before_data_and_retires_float_target_branch",
    "test_arb_path_has_no_inverse_retry_regularization_or_float_factor_basis",
    "test_operational_guard_precedes_state_result_particle_and_candidate_access",
)


def _syntax_check(root: Path) -> int:
    files = tuple(
        path
        for path in root.rglob("*.py")
        if not any(part in {".git", ".venv", "evidence"} for part in path.parts)
    )
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return len(files)


def main() -> int:
    root = _repository_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    observed_flint = metadata.version("python-flint")
    if observed_flint != "0.8.0":
        raise RuntimeError("CERT.14 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 123:
        raise RuntimeError("CERT.14 response-free check identity is not exactly 123")
    syntax_count = _syntax_check(root)
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert14-certified-function-space-common-target-"
                    "response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "role": (
                    "response-free exact-H0 weighted function-space marginal, "
                    "bridge/local-RJ common-target and sparse candidate composition"
                ),
                "python_flint_version": observed_flint,
                "arb_working_precision_bits": 512,
                "validated_arb_solve_algorithm": "precond",
                "standalone_common_target_composition_authorized": True,
                "cert13_predictive_prior_reused": True,
                "factorisation_free_function_space_covariance_verified": True,
                "weighted_collapsed_log_marginal_verified": True,
                "weighted_system_identity": "I+sqrtW-P-sqrtW",
                "bridge_incremental_potential_verified": True,
                "exact_local_rj_proposal_ratio_verified": True,
                "complete_forward_reverse_support_verified": True,
                "unit_jacobian_verified": True,
                "finite_mh_detailed_balance_verified": True,
                "target_invariance_verified": True,
                "sparse_fixed_candidate_target_adapter_verified": True,
                "future_response_access": False,
                "floating_factor_basis_resident_target_authorized": False,
                "rounded_snapshot_arrays_treated_as_exact": False,
                "matrix_inverse_used": False,
                "approximate_arb_algorithm_used": False,
                "diagonal_jitter_or_regularizer_used": False,
                "result_dependent_precision_retry_used": False,
                "tolerance_rank_decision_used": False,
                "posthoc_normalization_applied": False,
                "full_class_vector_materialized": False,
                "operational_target_result_access_authorized": False,
                "operational_sparse_result_access_authorized": False,
                "selection_island_execution_authorized": False,
                "confirmation_island_execution_authorized": False,
                "resident_smc_integration_authorized": False,
                "resident_smc_invoked": False,
                "simulated_experiment": False,
                "formal_experiment": False,
                "real_data_access": False,
                "heldout_access": False,
                "confirmatory_materialization": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
