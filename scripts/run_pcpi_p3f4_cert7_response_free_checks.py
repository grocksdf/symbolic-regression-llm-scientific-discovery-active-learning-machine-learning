"""Run only the response-free CERT.7 resident local/RJ composition Gate."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert6 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert6_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert6["EXPECTED_CHECKS"])
EXPECTED_CHECKS["tests/test_pcpi_p3f4_resident_local_rj_composition.py"] = (
    "test_cert7_registers_only_full_open_terminal_source_composition",
    "test_resident_endpoint_helper_preserves_exact_ratio_and_augmented_balance",
    "test_resident_endpoint_helper_fails_closed_on_semantic_or_mass_mismatch",
    "test_actual_rejuvenate_branch_delegates_proposal_and_acceptance_to_proofs",
    "test_resident_run_guard_precedes_data_or_particle_access",
    "test_resident_composed_finite_transition_is_reversible_and_invariant",
    "test_resident_composition_plan_rejects_cross_target_binding",
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
    if len(passed) != 36:
        raise RuntimeError("CERT.7 response-free check identity is not exactly 36")
    syntax_count = _syntax_check(root)
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert7-resident-local-rj-composition-"
                    "response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "role": (
                    "response-free resident endpoint, exact local/RJ source-"
                    "composition, and finite-transition audit"
                ),
                "resident_rejuvenation_import_verified": True,
                "actual_rejuvenate_delegation_verified": True,
                "resident_endpoint_mass_identity_verified": True,
                "resident_full_open_terminal_only_verified": True,
                "complete_forward_reverse_support_verified": True,
                "exact_proposal_ratio_verified": True,
                "finite_composed_transition_invariance_verified": True,
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
