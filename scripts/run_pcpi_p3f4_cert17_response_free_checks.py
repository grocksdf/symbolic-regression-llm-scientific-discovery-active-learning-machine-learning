"""Run only the response-free CERT.17 pre-bit refinement Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert16 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert16_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert16["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_prebit_refinement.py"] = (
    "test_cert17_authorizes_only_standalone_prebit_refinement_theorem",
    "test_refinement_plan_binds_cert16_budget_and_preregistered_schedule",
    "test_full_multinomial_and_mh_grid_floors_fit_exact_allocation",
    "test_nested_intersection_tightens_without_threshold_access",
    "test_first_budget_eligible_round_is_selected_before_bit_access",
    "test_insufficient_prefix_requests_more_precision_without_partial_output",
    "test_crossed_schedule_disjoint_and_incomplete_boundaries_fail_closed",
    "test_convergence_lemma_is_pointwise_not_one_fixed_precision_claim",
    "test_operational_guard_precedes_evaluator_response_particle_and_bits",
    "test_source_has_no_rng_response_threshold_or_empirical_tuning_surface",
    "test_cert17_retains_cert16_bit_and_execution_guards",
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
        raise RuntimeError("CERT.17 syntax scope contains no tracked Python source")
    if any(path.is_absolute() or ".." in path.parts for path in relatives):
        raise RuntimeError("CERT.17 received an invalid tracked Python path")
    files = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("CERT.17 tracked Python source is absent from the worktree")
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
        raise RuntimeError("CERT.17 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 163:
        raise RuntimeError("CERT.17 response-free check identity is not exactly 163")
    syntax_count = _syntax_check(root)
    if syntax_count != 238:
        raise RuntimeError("CERT.17 tracked Python source identity is not exactly 238")
    print(
        json.dumps(
            {
                "schema": "pcpi-p3f4-cert17-prebit-refinement-response-free-checks-v1",
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "python_syntax_scope": "git-tracked-python-files",
                "role": (
                    "response-free threshold-blind refinement theorem for the "
                    "frozen CERT.16 per-comparison allocation"
                ),
                "python_flint_version": observed_flint,
                "initial_arb_precision_bits": 512,
                "precision_schedule": "p_r=512*2^r-before-threshold-access",
                "uniform_threshold_bit_count": 256,
                "full_multinomial_grid_floor_below_allocation": True,
                "mh_grid_floor_below_allocation": True,
                "pointwise_finite_round_for_convergent_enclosures_verified": True,
                "one_uniform_precision_round_claimed": False,
                "reachable_state_evaluator_convergence_verified": False,
                "threshold_bits_observed_during_refinement": False,
                "adaptive_threshold_bit_extension_used": False,
                "scientific_result_dependent_tuning_used": False,
                "philox_pseudorandomness_promoted_to_mathematical_independence": False,
                "external_ideal_bit_product_law_required": True,
                "external_ideal_bit_product_law_implementation_authorized": False,
                "operational_evaluator_refinement_authorized": False,
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
