"""Run the response-free CERT.19 decision-budget replacement Gate."""

from __future__ import annotations

from fractions import Fraction
from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert18 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert18_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert18["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_cert19_direct_confidence.py"] = (
    "test_direct_confidence_allocation_closes_the_failure_budget",
    "test_direct_confidence_particle_count_replaces_median_amplification",
    "test_particle_formula_is_dimension_free_and_monotone_in_alpha",
    "test_envelope_anchor_strictly_improves_the_prior_minorization",
    "test_direct_confidence_guards_fail_closed",
)
EXPECTED_CHECKS["tests/test_pcpi_p3f4_cert19_rejection_confirmation.py"] = (
    "test_exact_ticket_proposal_has_complete_support_and_domination",
    "test_rejection_correction_returns_the_exact_target_law",
    "test_envelope_plan_improves_the_frozen_ac_prior_baseline",
    "test_exact_stage_boundaries_close_the_familywise_error_budget",
    "test_response_frozen_proposal_cap_charges_low_acceptance_to_abstention",
    "test_confirmation_and_operational_boundaries_fail_closed",
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
        raise RuntimeError("CERT.19 syntax scope contains no tracked Python source")
    if any(path.is_absolute() or ".." in path.parts for path in relatives):
        raise RuntimeError("CERT.19 received an invalid tracked Python path")
    files = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("CERT.19 tracked Python source is absent from the worktree")
    return files


def _syntax_check(root: Path) -> int:
    files = _tracked_python_files(root)
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return len(files)


def _budget_ledger() -> dict[str, object]:
    from hypothesis_mvp.pcpi.open_target import (
        P3F4_CERT19_DIRECT_CONFIDENCE_SCHEMA,
        P3F4_CERT19_DIRECT_CONFIDENCE_THEOREM,
        ResidentDirectConfidencePlan,
        build_dyadic_envelope_rejection_plan,
        rejection_proposal_cap,
    )

    direct = ResidentDirectConfidencePlan(
        schema=P3F4_CERT19_DIRECT_CONFIDENCE_SCHEMA,
        theorem=P3F4_CERT19_DIRECT_CONFIDENCE_THEOREM,
        contract_hash="response-free-contract-identity",
        feynman_kac_plan_hash="response-free-feynman-kac-identity",
        operational_estimand_hash="response-free-estimand-identity",
        class_projector_hash="response-free-projector-identity",
        path_step_bound=320,
        relative_ess_floor=Fraction(4, 5),
        map_regret_budget=Fraction(1, 50),
        failure_probability=Fraction(1, 20),
        maximum_rejuvenation_steps_per_bridge=200,
    )
    core = Fraction("0.1467437810166268")
    envelope = Fraction("5639.272478769489")
    rejection = build_dyadic_envelope_rejection_plan(
        "frozen-ac-ledger-cost-audit-only",
        (("semantic-core", core, core),),
        Fraction(8, 125) * envelope,
        proposal_ticket_bits=32,
    )
    cap = rejection_proposal_cap(
        512,
        rejection.acceptance_probability_lower,
        Fraction(1, 100),
    )
    return {
        "legacy_cert18_comparison_coordinates": 368_876_229_120,
        "direct_confidence_particle_count": direct.particle_count,
        "direct_confidence_confirmation_island_count": 1,
        "direct_smc_worst_case_target_evaluations": (
            direct.maximum_confirmation_target_evaluations
        ),
        "direct_smc_still_operationally_infeasible": True,
        "ac_ledger_rejection_acceptance_lower": float(
            rejection.acceptance_probability_lower
        ).hex(),
        "ac_ledger_proposal_cap_for_512_accepts_at_one_percent": cap,
    }


def main() -> int:
    root = _repository_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    observed_flint = metadata.version("python-flint")
    if observed_flint != "0.8.0":
        raise RuntimeError("CERT.19 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 189:
        raise RuntimeError("CERT.19 response-free check identity is not exactly 189")
    syntax_count = _syntax_check(root)
    if syntax_count != 246:
        raise RuntimeError("CERT.19 tracked Python source identity is not exactly 246")
    result = {
        "schema": "pcpi-p3f4-cert19-response-free-decision-budget-gate-v1",
        "status": "passed",
        "checks_passed": passed,
        "check_count": len(passed),
        "python_files_syntax_checked": syntax_count,
        "python_syntax_scope": "git-tracked-python-files",
        "python_flint_version": observed_flint,
        "role": "exact rejection fixed-candidate confirmation theorem gate",
        "single_island_arbitrary_alpha_corollary_verified": True,
        "exact_ticket_full_support_proposal_verified": True,
        "finite_rejection_target_law_verified": True,
        "exact_binomial_stage_boundaries_verified": True,
        "proposal_cap_failure_policy": "abstain-no-retry-no-replacement",
        "local_mixing_theorem_adopted": False,
        "operational_target_ball_access_authorized": False,
        "ideal_uniform_premise_accepted": False,
        "rejection_execution_authorized": False,
        "resident_smc_invoked": False,
        "future_response_access": False,
        "simulated_experiment": False,
        "formal_experiment": False,
        "real_data_access": False,
        "heldout_access": False,
        "confirmatory_materialization": False,
    }
    result.update(_budget_ledger())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
