"""Run the identity-bound CERT.23 lazy complete-prior rejection Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert22 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert22_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert22["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_cert23_lazy_prior_rejection.py"] = (
    "test_cert23_authorizes_only_standalone_lazy_source",
    "test_lazy_kernel_binds_actual_target_anchor_envelope_and_premise",
    "test_complete_prior_proposal_has_no_cutoff_core_table_or_atom_grid",
    "test_lazy_source_code_does_not_import_semantic_shell_enumeration",
    "test_exact_complete_prior_draw_returns_matching_raw_and_component_mass",
    "test_complete_prior_draw_is_dimension_generic_for_registered_real_widths",
    "test_finite_accepted_law_is_exact_prior_times_likelihood",
    "test_acceptance_interval_cancels_proposal_prior_exactly",
    "test_anchor_derives_positive_response_frozen_cap_without_class_union",
    "test_actual_cert18_path_evaluates_only_the_proposed_state",
    "test_global_envelope_violation_fails_closed",
    "test_crossed_target_identity_fails_before_evaluation",
    "test_cert23_freeze_matches_lazy_complexity_and_authorization",
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
        raise RuntimeError("CERT.23 tracked Python source identity is invalid")
    files = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("CERT.23 tracked Python source is absent from the worktree")
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
        raise RuntimeError("CERT.23 requires exactly python-flint 0.8.0")
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
    if len(passed) != 239:
        raise RuntimeError("CERT.23 response-free check identity is not exactly 239")
    syntax_count = _syntax_check(root)
    if syntax_count != 258:
        raise RuntimeError("CERT.23 tracked Python source identity is not exactly 258")
    if namespace is None:
        raise AssertionError("CERT.23 check namespace is absent")
    source = namespace["_source_fixture"]()[4]
    freeze = json.loads(
        (root / "configs/p3f_4_cert23_lazy_prior_rejection_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    result = {
        "schema": "pcpi-p3f4-cert23-lazy-prior-rejection-gate-v1",
        "status": "passed",
        "checks_passed": passed,
        "check_count": len(passed),
        "python_files_syntax_checked": syntax_count,
        "python_syntax_scope": "git-tracked-python-files",
        "python_flint_version": observed_flint,
        "role": "dimension-generic-lazy-complete-prior-exact-rejection-source-gate",
        "proposal_law": source.kernel.proposal_law,
        "acceptance_rule": source.kernel.acceptance_rule,
        "semantic_core_enumerated": source.kernel.semantic_core_enumerated,
        "maximum_nodes_used": source.kernel.maximum_nodes_used,
        "dyadic_core_atom_tickets_used": source.kernel.dyadic_atom_tickets_used,
        "anchor_family": freeze["rejection"]["anchor_family"],
        "fixture_anchor_count": len(source.kernel.anchors),
        "algebraic_fixture_selection_cap": source.selection_proposal_cap,
        "algebraic_fixture_confirmation_cap": source.confirmation_proposal_cap,
        "algebraic_fixture_cap_supports_operational_feasibility": False,
        "operational_anchor_complexity": freeze["rejection"][
            "anchor_target_ball_complexity"
        ],
        "per_proposal_target_ball_count": 1,
        "selection_confirmation_coordinate_separation_verified": True,
        "incomplete_batch_policy": source.incomplete_batch_policy,
        "operational_h0_access_authorized": False,
        "operational_execution_authorized": False,
        "system_entropy_access_authorized": False,
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
