"""Run the identity-bound CERT.20 exact-rejection source Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert19 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert19_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert19["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_cert20_exact_rejection_source.py"] = (
    "test_cert20_accepts_only_the_explicit_external_bit_premise",
    "test_actual_cert14_balls_bind_the_complete_exact_core_ticket_grid",
    "test_outward_exponential_and_acceptance_keep_exact_rational_bounds",
    "test_exact_ticket_source_uses_core_lift_and_analytic_tail_lift",
    "test_actual_cert18_rounds_refine_the_rejection_boundary_before_bits",
    "test_lazy_uniform_comparison_evaluates_each_arb_boundary_before_bits",
    "test_lazy_uniform_extreme_prefixes_match_exact_bernoulli_decisions",
    "test_cap_state_erases_partial_samples_and_forbids_retry",
    "test_fixed_candidate_confirmation_releases_only_a_complete_result",
    "test_independent_selection_engine_freezes_mode_and_lexicographic_tie_break",
    "test_production_source_freeze_matches_the_implemented_contract",
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
        raise RuntimeError("CERT.20 tracked Python source identity is invalid")
    files = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("CERT.20 tracked Python source is absent from the worktree")
    return files


def _syntax_check(root: Path) -> int:
    files = _tracked_python_files(root)
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return len(files)


def _freeze_ledger(root: Path) -> dict[str, object]:
    path = root / "configs/p3f_4_cert20_exact_rejection_source_freeze.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    return {
        "proposal_ticket_bits": freeze["proposal"]["ticket_bits"],
        "selection_accepted_samples": freeze["candidate_selection"][
            "accepted_sample_count"
        ],
        "confirmation_accepted_sample_stages": freeze[
            "fixed_candidate_confirmation"
        ]["accepted_sample_stages"],
        "proposal_cap_failure_probability": freeze["proposal"][
            "proposal_cap_failure_probability"
        ],
        "selection_coordinate_domain": freeze["candidate_selection"][
            "coordinate_domain"
        ],
        "confirmation_coordinate_domain": freeze[
            "fixed_candidate_confirmation"
        ]["coordinate_domain"],
        "ideal_byte_materialization": freeze["randomness_premise"][
            "materialization"
        ],
    }


def main() -> int:
    root = _repository_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    observed_flint = metadata.version("python-flint")
    if observed_flint != "0.8.0":
        raise RuntimeError("CERT.20 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 200:
        raise RuntimeError("CERT.20 response-free check identity is not exactly 200")
    syntax_count = _syntax_check(root)
    if syntax_count != 249:
        raise RuntimeError("CERT.20 tracked Python source identity is not exactly 249")
    result = {
        "schema": "pcpi-p3f4-cert20-exact-rejection-source-gate-v1",
        "status": "passed",
        "checks_passed": passed,
        "check_count": len(passed),
        "python_files_syntax_checked": syntax_count,
        "python_syntax_scope": "git-tracked-python-files",
        "python_flint_version": observed_flint,
        "role": "actual-arb exact-rejection source-composition gate",
        "actual_cert14_core_atom_balls_bound": True,
        "cert17_cert18_rejection_boundary_composed": True,
        "exact_core_and_tail_ticket_source_implemented": True,
        "almost_sure_exact_lazy_uniform_comparison_implemented": True,
        "external_ideal_independent_byte_premise_accepted": True,
        "physical_independence_proved_by_source": False,
        "deterministic_prng_promoted_to_ideal_law": False,
        "incomplete_batch_policy": "erase-and-abstain-no-retry-no-partial-output",
        "operational_h0_target_access_authorized": False,
        "resident_smc_invoked": False,
        "future_response_access": False,
        "simulated_experiment": False,
        "formal_experiment": False,
        "real_data_access": False,
        "heldout_access": False,
        "confirmatory_materialization": False,
    }
    result.update(_freeze_ledger(root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
