"""Run the identity-bound CERT.21 guarded-runner source Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert20 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert20_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert20["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_cert21_exact_rejection_runner.py"] = (
    "test_cert21_authorizes_only_pure_state_machine_and_atomic_writer",
    "test_runner_plan_binds_cert20_source_actual_evaluator_and_premise",
    "test_coordinate_sources_are_logically_disjoint_and_account_every_byte",
    "test_fixed_batch_returns_complete_acceptances_and_stable_transcript",
    "test_cap_abstention_erases_every_partial_acceptance",
    "test_draw_failure_becomes_terminal_abstention_without_partial_marks",
    "test_independent_selection_and_first_confirmation_stage_publish_one_result",
    "test_no_confirmation_boundary_returns_only_indivisible_abstention",
    "test_selection_or_confirmation_draw_failure_never_leaks_candidate",
    "test_atomic_terminal_ledger_round_trip_and_retry_prohibition",
    "test_terminal_ledger_hash_detects_tampering",
    "test_operational_guard_precedes_h0_entropy_and_output_access",
    "test_crossed_product_coordinate_domains_fail_before_any_draw",
    "test_cert21_runner_freeze_matches_source_and_failure_boundaries",
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
    if not relatives or any(path.is_absolute() or ".." in path.parts for path in relatives):
        raise RuntimeError("CERT.21 tracked Python source identity is invalid")
    files = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("CERT.21 tracked Python source is absent from the worktree")
    return files


def _syntax_check(root: Path) -> int:
    files = _tracked_python_files(root)
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return len(files)


def _freeze_ledger(root: Path) -> dict[str, object]:
    path = root / "configs/p3f_4_cert21_guarded_runner_freeze.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    return {
        "runner_execution_order": freeze["execution_order"],
        "selection_coordinate_domain": freeze["coordinate_policy"]["selection"],
        "confirmation_coordinate_domain": freeze["coordinate_policy"]["confirmation"],
        "draw_failure_policy": freeze["failure_policy"][
            "arb_or_entropy_draw_failure"
        ],
        "programming_assertion_policy": freeze["failure_policy"][
            "programming_assertion"
        ],
        "ledger_publication": freeze["ledger"]["publication"],
        "existing_ledger_policy": freeze["ledger"]["existing_target"],
    }


def main() -> int:
    root = _repository_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    observed_flint = metadata.version("python-flint")
    if observed_flint != "0.8.0":
        raise RuntimeError("CERT.21 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 214:
        raise RuntimeError("CERT.21 response-free check identity is not exactly 214")
    syntax_count = _syntax_check(root)
    if syntax_count != 252:
        raise RuntimeError("CERT.21 tracked Python source identity is not exactly 252")
    result = {
        "schema": "pcpi-p3f4-cert21-guarded-exact-rejection-runner-gate-v1",
        "status": "passed",
        "checks_passed": passed,
        "check_count": len(passed),
        "python_files_syntax_checked": syntax_count,
        "python_syntax_scope": "git-tracked-python-files",
        "python_flint_version": observed_flint,
        "role": "guarded-selection-confirmation-runner-and-indivisible-ledger-gate",
        "selection_confirmation_coordinate_separation_verified": True,
        "cap_and_draw_failure_partial_erasure_verified": True,
        "programming_assertions_masked_as_abstention": False,
        "first_crossed_confirmation_stage_stops": True,
        "abstention_candidate_or_partial_state_leak": False,
        "atomic_no_overwrite_terminal_ledger_verified": True,
        "terminal_ledger_tamper_detection_verified": True,
        "operational_execution_authorized": False,
        "operational_h0_access_authorized": False,
        "system_entropy_access_authorized": False,
        "future_response_access": False,
        "simulated_experiment": False,
        "formal_experiment": False,
        "real_data_access": False,
        "heldout_access": False,
        "acquisition_access": False,
        "confirmatory_materialization": False,
    }
    result.update(_freeze_ledger(root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
