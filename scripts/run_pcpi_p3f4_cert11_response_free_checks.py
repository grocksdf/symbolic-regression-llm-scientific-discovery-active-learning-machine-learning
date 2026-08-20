"""Run only the response-free CERT.11 product-source/projector Gate."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert10 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert10_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert10["EXPECTED_CHECKS"])
EXPECTED_CHECKS[
    "tests/test_pcpi_p3f4_resident_product_source_projector.py"
] = (
    "test_cert11_retains_all_cert10_and_new_execution_guards",
    "test_product_source_contract_binds_every_ordered_coordinate_directly",
    "test_key_manifest_is_one_key_per_coordinate_auditable_and_fail_closed",
    "test_actual_product_source_is_guarded_before_entropy_or_coordinate_access",
    "test_operational_estimand_freezes_h0_grid_budget_and_claim_domain",
    "test_implicit_operational_class_rank_is_a_complete_bijection",
    "test_class_identity_is_support_extension_and_population_order_invariant",
    "test_exact_interval_binning_obeys_boundaries_without_nearest_rounding",
    "test_sparse_exact_pushforward_preserves_mass_order_and_support_splits",
    "test_boundary_uncertainty_propagates_sparse_exact_class_mass_bounds",
    "test_fixed_vector_adapter_requires_exact_and_every_occupied_class_registered",
    "test_actual_projector_is_plan_bound_and_guarded_before_result_or_oracle_access",
    "test_no_uncertified_cdf_or_cartesian_class_enumeration_is_smuggled_in",
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
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 79:
        raise RuntimeError("CERT.11 response-free check identity is not exactly 79")
    syntax_count = _syntax_check(root)
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert11-product-source-full-support-"
                    "projector-response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "role": (
                    "response-free auditable direct-key product-source and "
                    "full-support operational-projector source audit"
                ),
                "auditable_philox_direct_key_source_verified": True,
                "ordered_product_coordinate_binding_verified": True,
                "one_key_per_coordinate_no_retry_verified": True,
                "external_independence_premise_proved_by_source": False,
                "distinct_integer_seeds_treated_as_independent": False,
                "seedsequence_spawn_used": False,
                "root_key_derivation_used": False,
                "jumped_streams_used": False,
                "favourable_key_selection_authorized": False,
                "full_support_operational_class_map_verified": True,
                "implicit_class_space_rank_bijection_verified": True,
                "support_extension_invariance_verified": True,
                "grid_restricted_claim_only": True,
                "exact_polynomial_classes_used": False,
                "exact_boundary_uncertainty_propagation_verified": True,
                "nearest_bin_rounding_used": False,
                "sparse_class_mass_bounds_verified": True,
                "full_class_space_enumerated_during_projection": False,
                "fixed_vector_adapter_fail_closed_verified": True,
                "posthoc_normalization_applied": False,
                "system_entropy_capture_authorized": False,
                "product_stream_materialization_authorized": False,
                "certified_cdf_interval_oracle_implementation_authorized": False,
                "projector_result_access_authorized": False,
                "cert10_island_executor_run_authorized": False,
                "cert10_product_source_authorized": False,
                "cert10_projector_authorized": False,
                "future_response_access": False,
                "independent_island_execution_authorized": False,
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
