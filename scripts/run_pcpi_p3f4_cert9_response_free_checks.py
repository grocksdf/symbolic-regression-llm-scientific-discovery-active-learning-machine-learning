"""Run only the response-free CERT.9 finite-N/island error-budget Gate."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert8 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert8_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert8["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_resident_finite_n_island_budget.py"] = (
    "test_cert9_registers_the_exact_fixed_path_theorem_assumptions",
    "test_particle_lower_bound_and_class_decision_budget_are_derived",
    "test_exact_independent_island_median_budget_is_simultaneous",
    "test_multinomial_product_law_is_not_systematic_shared_offset",
    "test_prior_local_kernel_mixture_is_invariant_and_globally_minorized",
    "test_every_certified_bridge_receives_a_preparticle_mixing_budget",
    "test_bridge_mixing_and_cross_target_fail_closed",
    "test_runtime_identity_rejects_unregistered_counts_or_controls",
    "test_actual_source_uses_theorem_resampling_preflight_and_kernel_mixture",
    "test_cert9_run_guard_precedes_data_preflight_and_particle_sampling",
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
    if len(passed) != 56:
        raise RuntimeError("CERT.9 response-free check identity is not exactly 56")
    syntax_count = _syntax_check(root)
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert9-finite-n-independent-island-"
                    "error-budget-response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "role": (
                    "response-free finite-N L2 path, countably-open kernel "
                    "minorization, and independent-island decision-error audit"
                ),
                "finite_n_l2_particle_bound_verified": True,
                "decision_regret_budget_verified": True,
                "multinomial_product_resampling_verified": True,
                "systematic_finite_n_theorem_transfer_rejected": True,
                "countably_open_kernel_minorization_verified": True,
                "prior_local_kernel_mixture_invariance_verified": True,
                "finite_n_bridge_preflight_verified": True,
                "independent_island_product_law_verified": True,
                "independent_island_median_budget_verified": True,
                "within_island_particle_independence_assumed": False,
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
