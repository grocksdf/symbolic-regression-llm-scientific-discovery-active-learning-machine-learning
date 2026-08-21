"""Run only the response-free CERT.13 full-H0/sparse-projector Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert12 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert12_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert12["EXPECTED_CHECKS"])
EXPECTED_CHECKS[
    "tests/test_pcpi_p3f4_h0_parameter_balls_sparse_projector.py"
] = (
    "test_cert13_retains_all_execution_guards_and_authorizes_only_pure_constructor",
    "test_frozen_h0_and_standardizer_hash_bind_exact_binary_inputs",
    "test_exact_polynomial_evaluation_has_no_float_or_raw_ast_dependence",
    "test_projected_rbf_uses_schur_complement_and_certifies_orthogonality",
    "test_rbf_kernel_enclosures_are_symmetric_and_contain_high_precision_reference",
    "test_zero_polynomial_has_vacuous_projection_and_complete_state_support",
    "test_inactive_component_matches_exact_conjugate_nig_identity",
    "test_validated_arb_solve_is_pinned_without_inverse_retry_or_regularizer",
    "test_full_h0_provider_outputs_every_action_threshold_parameter_ball",
    "test_cert13_parameters_feed_cert12_arb_kernel_with_monotone_outward_cdfs",
    "test_provider_rejects_cross_target_kernel_and_component_identities",
    "test_sparse_candidate_projection_propagates_boundary_uncertain_mass",
    "test_sparse_candidate_bounds_match_complete_sparse_projection_query",
    "test_sparse_lower_bound_composes_with_fixed_candidate_map_certificate",
    "test_operational_guard_precedes_result_state_provider_and_no_smuggling_source",
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
        raise RuntimeError("CERT.13 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 109:
        raise RuntimeError("CERT.13 response-free check identity is not exactly 109")
    syntax_count = _syntax_check(root)
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert13-full-h0-parameter-balls-sparse-"
                    "candidate-projector-response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "role": (
                    "response-free exact-H0 reconstruction, Arb projected-RBF/"
                    "validated-linear-algebra fixtures and sparse fixed-candidate "
                    "combinatorics"
                ),
                "python_flint_version": observed_flint,
                "arb_parameter_provider_working_precision_bits": 512,
                "arb_cdf_working_precision_bits": 256,
                "standalone_h0_parameter_ball_construction_authorized": True,
                "full_state_parameter_ball_provider_verified": True,
                "exact_h0_binary_identity_verified": True,
                "exact_polynomial_state_evaluation_verified": True,
                "exact_active_standardizer_selection_verified": True,
                "factorisation_free_projected_rbf_verified": True,
                "projected_rbf_schur_complement_psd_verified": True,
                "projected_rbf_structure_orthogonality_verified": True,
                "zero_polynomial_vacuous_constraint_verified": True,
                "eigen_or_svd_basis_used": False,
                "tolerance_rank_decision_used": False,
                "validated_arb_linear_solve_verified": True,
                "validated_arb_solve_algorithm": "precond",
                "approximate_arb_algorithm_used": False,
                "matrix_inverse_used": False,
                "diagonal_jitter_or_regularizer_used": False,
                "result_dependent_precision_retry_used": False,
                "rounded_snapshot_arrays_treated_as_exact": False,
                "cdf_beta_unit_domain_intersection_verified": True,
                "cdf_zero_crossing_complement_identity_verified": True,
                "nonnegative_map_regret_upper_verified": True,
                "sparse_candidate_projector_verified": True,
                "boundary_uncertainty_preserved": True,
                "full_class_vector_materialized": False,
                "full_class_space_enumerated": False,
                "posthoc_normalization_applied": False,
                "operational_h0_access_authorized": False,
                "operational_cdf_result_access_authorized": False,
                "sparse_projector_result_access_authorized": False,
                "cert11_projector_result_access_authorized": False,
                "split_product_source_materialization_authorized": False,
                "split_island_execution_authorized": False,
                "resident_smc_integration_authorized": False,
                "resident_smc_invoked": False,
                "future_response_access": False,
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
