"""Run only the response-free CERT.15 certified comparison/sampling Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert14 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert14_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert14["EXPECTED_CHECKS"])
EXPECTED_CHECKS[
    "tests/test_pcpi_p3f4_certified_comparison_sampling.py"
] = (
    "test_cert15_authorizes_only_standalone_comparison_sampling",
    "test_sampling_plan_binds_cert14_common_target_and_frozen_budgets",
    "test_outward_log_normalization_contains_equal_mass_law_and_is_shift_invariant",
    "test_outward_log_normalization_preserves_interval_and_cumulative_contract",
    "test_exact_bit_threshold_is_complete_half_open_cell_bijection",
    "test_finite_dyadic_inverse_cdf_enumeration_matches_exact_law",
    "test_multinomial_inverse_cdf_resolves_complete_batch",
    "test_multinomial_unresolved_comparison_aborts_without_partial_output",
    "test_multinomial_unresolved_probability_bound_is_explicit",
    "test_mh_uniform_comparison_certifies_accept_and_reject",
    "test_mh_uniform_comparison_fails_closed_when_unresolved",
    "test_cross_plan_purpose_and_coordinate_identity_are_rejected",
    "test_source_has_no_midpoint_float_rng_retry_or_partial_sampling",
    "test_operational_sampler_guards_precede_all_input_access",
)


def _tracked_python_files(root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ("git", "-C", str(root), "ls-files", "-z", "--", "*.py"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    relatives = tuple(
        Path(item.decode("utf-8"))
        for item in completed.stdout.split(b"\0")
        if item
    )
    if not relatives:
        raise RuntimeError("CERT.15 syntax scope contains no tracked Python source")
    if any(path.is_absolute() or ".." in path.parts for path in relatives):
        raise RuntimeError("CERT.15 received an invalid tracked Python path")
    files = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("CERT.15 tracked Python source is absent from the worktree")
    return files


def _syntax_check(root: Path) -> int:
    files = _tracked_python_files(root)
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return len(files)


def main() -> int:
    root = _repository_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    observed_flint = metadata.version("python-flint")
    if observed_flint != "0.8.0":
        raise RuntimeError("CERT.15 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 138:
        raise RuntimeError("CERT.15 response-free check identity is not exactly 138")
    syntax_count = _syntax_check(root)
    if syntax_count != 232:
        raise RuntimeError("CERT.15 tracked Python source identity is not exactly 232")
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert15-certified-comparison-partial-sampling-"
                    "response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "python_syntax_scope": "git-tracked-python-files",
                "role": (
                    "response-free outward normalization, exact-bit dyadic-cell "
                    "multinomial and MH comparison composition"
                ),
                "python_flint_version": observed_flint,
                "arb_working_precision_bits": 512,
                "uniform_threshold_bit_count": 256,
                "uniform_threshold_semantics": (
                    "complete-half-open-dyadic-prefix-cell"
                ),
                "outward_log_normalization_verified": True,
                "exact_dyadic_inverse_cdf_law_verified": True,
                "multinomial_inverse_cdf_comparison_verified": True,
                "mh_uniform_comparison_verified": True,
                "unresolved_probability_bounds_verified": True,
                "unresolved_policy": "abort-complete-operation-no-retry",
                "standalone_comparison_sampling_authorized": True,
                "adaptive_bit_extension_used": False,
                "result_dependent_precision_retry_used": False,
                "probability_midpoint_used": False,
                "floating_categorical_sampling_used": False,
                "partial_sampling_output_authorized": False,
                "product_bits_materialization_authorized": False,
                "resident_resampling_authorized": False,
                "resident_mh_decision_authorized": False,
                "island_execution_authorized": False,
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
