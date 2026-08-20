"""Run only the response-free CERT.8 resident Feynman--Kac composition Gate."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert7 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert7_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert7["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_resident_feynman_kac_composition.py"] = (
    "test_cert8_registers_only_full_open_certified_common_target_controls",
    "test_analytic_bridge_selector_uses_largest_certified_grid_step",
    "test_uncertified_path_fails_closed_without_forced_terminal_step",
    "test_incremental_potential_telescopes_and_binds_one_target",
    "test_systematic_resampling_law_is_exactly_unbiased",
    "test_finite_feynman_kac_resample_move_composition_is_invariant_and_mixing",
    "test_actual_resident_local_rj_finite_kernel_has_positive_spectral_gap",
    "test_actual_resident_source_threads_one_bridge_through_all_operations",
    "test_rejuvenation_and_run_fail_before_unbound_target_or_data_access",
    "test_cert8_plan_rejects_cross_target_or_incomplete_binding",
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
    if len(passed) != 46:
        raise RuntimeError("CERT.8 response-free check identity is not exactly 46")
    syntax_count = _syntax_check(root)
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert8-resident-feynman-kac-composition-"
                    "response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "role": (
                    "response-free analytic Feynman-Kac path, common-target "
                    "weight/resample/move, and finite-transition audit"
                ),
                "analytic_population_relative_ess_path_verified": True,
                "largest_certified_beta_grid_step_verified": True,
                "forced_terminal_completion_absent": True,
                "current_response_prefix_only_verified": True,
                "future_response_access": False,
                "bridge_target_identity_verified": True,
                "incremental_potential_telescope_verified": True,
                "systematic_resampling_unbiasedness_verified": True,
                "actual_common_target_source_threading_verified": True,
                "finite_resample_move_invariance_verified": True,
                "finite_transition_spectral_gap_verified": True,
                "target_invariance_verified": True,
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
