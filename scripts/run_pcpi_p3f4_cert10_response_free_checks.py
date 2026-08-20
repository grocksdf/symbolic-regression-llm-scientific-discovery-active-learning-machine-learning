"""Run only the response-free CERT.10 independent-island source Gate."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert9 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert9_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert9["EXPECTED_CHECKS"])
EXPECTED_CHECKS[
    "tests/test_pcpi_p3f4_resident_independent_island_executor.py"
] = (
    "test_cert10_plan_binds_cert9_config_estimand_and_claim_boundary",
    "test_cert10_plan_rejects_cross_target_config_or_class_identity",
    "test_product_coordinates_and_finite_product_law_are_exact",
    "test_random_stream_aliases_or_cross_coordinates_are_rejected",
    "test_each_island_outcome_is_a_bound_normalized_pushforward",
    "test_componentwise_median_scores_do_not_claim_simplex_normalization",
    "test_simultaneous_median_union_budget_matches_the_product_law",
    "test_missing_duplicate_or_cross_plan_outcomes_fail_closed",
    "test_all_computational_failures_propagate_without_partial_aggregate",
    "test_actual_executor_source_is_isolated_and_guarded_before_every_access",
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
    if len(passed) != 66:
        raise RuntimeError("CERT.10 response-free check identity is not exactly 66")
    syntax_count = _syntax_check(root)
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert10-independent-island-executor-"
                    "aggregation-response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "role": (
                    "response-free independent-product stream isolation, "
                    "failure propagation, plan binding, and median decision-error audit"
                ),
                "independent_product_randomness_contract_verified": True,
                "distinct_integer_seeds_treated_as_independent": False,
                "random_stream_alias_rejected": True,
                "exact_plan_identity_binding_verified": True,
                "operational_estimand_binding_verified": True,
                "all_island_failures_propagated": True,
                "retry_or_island_replacement_authorized": False,
                "partial_island_aggregation_authorized": False,
                "componentwise_median_score_aggregation_verified": True,
                "componentwise_median_normalization_assumed": False,
                "posterior_probability_vector_claimed": False,
                "simultaneous_union_error_budget_verified": True,
                "map_decision_regret_certificate_verified": True,
                "independent_product_random_source_implementation_authorized": False,
                "operational_class_projector_implementation_authorized": False,
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
