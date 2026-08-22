"""Run the identity-bound CERT.22 operational preflight NO-GO Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert21 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert21_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert21["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_cert22_operational_preflight.py"] = (
    "test_cert22_authorizes_preflight_only",
    "test_registered_family_is_three_targets_eight_seeds_and_twenty_four_h0",
    "test_registered_feature_dimensions_match_registry_without_loading_data",
    "test_raw_ast_recurrence_matches_registered_one_dimensional_evidence",
    "test_monomial_lower_bound_is_analytic_and_dimension_aware",
    "test_target_ball_lower_bound_covers_all_registered_h0",
    "test_cert20_executable_fixture_is_one_dimensional_cutoff_one_not_j17",
    "test_dimension_cutoff_and_missing_h0_are_explicit_blockers",
    "test_missing_time_and_storage_are_not_replaced_by_guesses",
    "test_output_identity_is_unique_but_cannot_be_materialized",
    "test_h0_hash_validation_fails_closed",
    "test_freeze_matches_exact_no_go_decision",
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
        raise RuntimeError("CERT.22 tracked Python source identity is invalid")
    files = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("CERT.22 tracked Python source is absent from the worktree")
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
        raise RuntimeError("CERT.22 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    namespace: dict[str, object] | None = None
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 226:
        raise RuntimeError("CERT.22 response-free check identity is not exactly 226")
    syntax_count = _syntax_check(root)
    if syntax_count != 255:
        raise RuntimeError("CERT.22 tracked Python source identity is not exactly 255")
    if namespace is None:
        raise AssertionError("CERT.22 check namespace is absent")
    preflight = namespace["_preflight"]()
    family = namespace["_family"]()
    result = {
        "schema": "pcpi-p3f4-cert22-operational-preflight-gate-v1",
        "status": "passed-no-go",
        "checks_passed": passed,
        "check_count": len(passed),
        "python_files_syntax_checked": syntax_count,
        "python_syntax_scope": "git-tracked-python-files",
        "python_flint_version": observed_flint,
        "role": "response-free-operational-identity-scale-and-feasibility-no-go-gate",
        "registered_dataset_targets": [item.dataset_id for item in family.tasks],
        "registered_feature_counts": [item.feature_count for item in family.tasks],
        "registered_seed_count": len(family.seeds),
        "required_h0_artifact_count": family.required_h0_artifact_count,
        "bound_h0_artifact_count": len(family.bound_h0_artifact_hashes),
        "frozen_target_feature_count": preflight.frozen_target_feature_count,
        "frozen_target_maximum_nodes": preflight.frozen_target_maximum_nodes,
        "executable_source_feature_count": preflight.executable_source_feature_count,
        "executable_source_maximum_nodes": preflight.executable_source_maximum_nodes,
        "target_ball_lower_bound_across_registered_h0": (
            preflight.target_ball_lower_bound_family
        ),
        "blockers": list(preflight.blockers),
        "output_identity": preflight.output_identity,
        "output_publication": preflight.output_publication,
        "operational_execution_authorized": False,
        "operational_h0_binding_authorized": False,
        "system_entropy_access_authorized": False,
        "output_materialized": False,
        "future_response_access": False,
        "simulated_experiment": False,
        "formal_experiment": False,
        "real_data_access": False,
        "heldout_access": False,
        "acquisition_access": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
