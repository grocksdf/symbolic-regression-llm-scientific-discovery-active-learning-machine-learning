"""Run only the response-free CERT.12 rigorous-CDF/split-MAP Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert11 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert11_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert11["EXPECTED_CHECKS"])
EXPECTED_CHECKS[
    "tests/test_pcpi_p3f4_rigorous_cdf_split_confirmation.py"
] = (
    "test_cert12_retains_every_execution_and_result_access_guard",
    "test_python_flint_backend_is_exactly_pinned_in_both_dependency_manifests",
    "test_dyadic_interval_encoding_is_exact_and_rejects_decimal_surrogates",
    "test_arb_kernel_contract_forbids_point_padding_and_unproved_parameter_balls",
    "test_arb_student_t_kernel_contains_cauchy_analytic_identities",
    "test_arb_parameter_balls_propagate_outward_without_nearest_bin_assignment",
    "test_arb_kernel_source_uses_rigorous_beta_and_exact_outward_endpoints_only",
    "test_operational_oracle_binds_full_support_provider_but_stays_guarded",
    "test_rounded_snapshot_parameter_provider_is_explicitly_rejected",
    "test_split_confirmation_budget_is_dimension_free_in_implicit_class_count",
    "test_selection_and_confirmation_coordinates_are_disjoint_product_factors",
    "test_conditional_fixed_candidate_failure_needs_no_class_union_bound",
    "test_majority_mass_certificate_implies_map_regret_on_complete_small_simplexes",
    "test_split_map_certificate_uses_frozen_threshold_or_abstains",
    "test_cert12_source_has_no_class_enumeration_retry_or_execution_smuggling",
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
        raise RuntimeError("CERT.12 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 94:
        raise RuntimeError("CERT.12 response-free check identity is not exactly 94")
    syntax_count = _syntax_check(root)
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert12-rigorous-cdf-split-map-"
                    "response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "role": (
                    "response-free rigorous special-function analytic fixtures "
                    "and exact finite conditional-product combinatorics"
                ),
                "python_flint_version": observed_flint,
                "arb_student_t_kernel_verified": True,
                "arb_regularized_incomplete_beta_verified": True,
                "arb_working_precision_bits": 256,
                "exact_dyadic_parameter_endpoint_encoding_verified": True,
                "exact_outward_binary_endpoint_export_verified": True,
                "cauchy_analytic_identities_enclosed": True,
                "ordinary_floating_cdf_used": False,
                "nextafter_or_point_padding_used": False,
                "approximate_arb_algorithm_used": False,
                "result_dependent_precision_retry_used": False,
                "full_state_parameter_ball_provider_verified": False,
                "rounded_snapshot_arrays_treated_as_exact": False,
                "operational_cdf_oracle_run_authorized": False,
                "cert11_cdf_oracle_implementation_authorized": False,
                "cert11_projector_result_access_authorized": False,
                "dimension_free_split_map_budget_verified": True,
                "conditional_fixed_candidate_theorem_verified": True,
                "selection_confirmation_product_partition_verified": True,
                "selection_island_count": 1,
                "registered_confirmation_island_count": 9,
                "registered_confirmation_failure_upper": "6413/131072",
                "registered_failure_budget": "1/20",
                "implicit_class_counts_checked": ("6^7", "6^700"),
                "class_count_union_bound_used": False,
                "full_class_space_enumerated": False,
                "selection_confirmation_island_reuse": False,
                "adaptive_retry_authorized": False,
                "result_derived_threshold_used": False,
                "confirmation_median_threshold": "1/2",
                "map_regret_implication_verified": True,
                "abstention_path_verified": True,
                "posthoc_normalization_applied": False,
                "posterior_probability_vector_claimed": False,
                "external_independence_premise_proved_by_source": False,
                "split_product_source_materialization_authorized": False,
                "split_island_execution_authorized": False,
                "map_result_access_authorized": False,
                "cert10_island_executor_run_authorized": False,
                "cert10_product_source_authorized": False,
                "cert10_projector_authorized": False,
                "future_response_access": False,
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
