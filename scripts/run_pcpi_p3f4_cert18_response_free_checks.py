"""Run only the response-free CERT.18 actual Arb refinement Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert17 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert17_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert17["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_actual_arb_refinement.py"] = (
    "test_cert18_authorizes_only_standalone_actual_evaluator_composition",
    "test_actual_plan_binds_provider_common_sampling_integration_and_refinement",
    "test_actual_collapsed_target_refines_same_exact_state_across_rounds",
    "test_actual_mh_path_composes_into_cert17_before_threshold_access",
    "test_linear_normalization_contains_equal_mass_law_at_both_rounds",
    "test_linear_normalization_overlaps_high_precision_point_reference",
    "test_linear_normalization_overlaps_every_small_interval_corner_reference",
    "test_linear_normalization_is_shift_invariant_under_refinement",
    "test_full_registered_normalization_complexity_is_linear_not_quadratic",
    "test_normalization_source_has_no_pairwise_particle_loop",
    "test_actual_convergence_contract_is_pointwise_and_external_backend_explicit",
    "test_crossed_provider_sampling_and_refinement_identities_fail_closed",
    "test_operational_guard_precedes_response_particle_and_threshold_access",
    "test_actual_refinement_source_has_no_rng_retry_or_threshold_materialization",
    "test_cert18_retains_cert17_bit_and_execution_guards",
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
        raise RuntimeError("CERT.18 syntax scope contains no tracked Python source")
    if any(path.is_absolute() or ".." in path.parts for path in relatives):
        raise RuntimeError("CERT.18 received an invalid tracked Python path")
    files = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("CERT.18 tracked Python source is absent from the worktree")
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
        raise RuntimeError("CERT.18 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 178:
        raise RuntimeError("CERT.18 response-free check identity is not exactly 178")
    syntax_count = _syntax_check(root)
    if syntax_count != 241:
        raise RuntimeError("CERT.18 tracked Python source identity is not exactly 241")
    print(
        json.dumps(
            {
                "schema": "pcpi-p3f4-cert18-actual-arb-refinement-response-free-checks-v1",
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "python_syntax_scope": "git-tracked-python-files",
                "role": (
                    "response-free actual CERT.13/14 Arb refinement binding "
                    "and rigorous linear-time normalization"
                ),
                "python_flint_version": observed_flint,
                "precision_schedule": "p_r=512*2^r-before-threshold-access",
                "actual_collapsed_target_refinement_bound": True,
                "actual_mh_prebit_composition_verified": True,
                "normalization_time_complexity": "O(N)",
                "normalization_auxiliary_memory_complexity": "O(N)",
                "quadratic_pair_count_materialized": 0,
                "full_registered_particle_count_audited": 212408,
                "flint_inclusion_and_convergence_premise_required": True,
                "unconditional_third_party_software_correctness_claimed": False,
                "pointwise_not_uniform_runtime_claim": True,
                "threshold_bits_observed_during_refinement": False,
                "adaptive_threshold_bit_extension_used": False,
                "scientific_result_dependent_tuning_used": False,
                "external_ideal_bit_product_law_required": True,
                "external_ideal_bit_product_law_implementation_authorized": False,
                "operational_refinement_authorized": False,
                "product_bits_materialization_authorized": False,
                "island_batch_execution_authorized": False,
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
