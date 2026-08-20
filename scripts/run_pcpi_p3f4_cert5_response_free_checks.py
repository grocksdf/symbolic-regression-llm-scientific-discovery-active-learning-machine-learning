"""Run the response-free CERT.5 raw-state local/RJ proof checks only."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


EXPECTED_CHECKS = {
    "tests/test_pcpi_p3f4_semantic_lift.py": (
        "test_semantic_unranking_is_a_bijection_on_complete_small_shells",
        "test_core_lift_plan_exactly_reconstructs_original_raw_prior",
        "test_class_constant_anchor_factor_lifts_to_raw_target_exactly",
        "test_ticket_endpoints_map_to_valid_raw_asts_without_uint64_limits",
        "test_semantic_lift_fails_closed_for_invalid_class_or_rank",
    ),
    "tests/test_pcpi_p3f4_raw_state_anchor.py": (
        "test_component_prior_is_exact_and_includes_spike_and_kernel_mass",
        "test_raw_unranking_is_a_bijection_and_has_no_uint64_ceiling",
        "test_tail_draw_uses_exact_rational_geometric_and_large_shell_unranking",
        "test_complete_anchor_normalizes_and_recovers_every_core_raw_target_mass",
        "test_raw_state_mh_mass_satisfies_pairwise_detailed_balance",
        "test_anchor_sampler_carries_the_same_mass_used_by_mh",
        "test_anchor_fails_closed_for_incomplete_mass_identity_or_envelope",
    ),
    "tests/test_pcpi_p3f4_resident_kernel_static_audit.py": (
        "test_resident_independence_formulas_are_invariant_on_finite_support",
        "test_resident_raw_evaluation_counterexample_blocks_semantic_lumpability",
        "test_resident_open_sampler_uint64_ceiling_blocks_full_reverse_support",
    ),
    "tests/test_pcpi_p3f4_raw_state_local_rj.py": (
        "test_raw_ast_paths_and_subtree_replacement_are_exact_involutions",
        "test_root_refresh_gives_complete_bidirectional_raw_state_support",
        "test_exact_local_rj_draw_has_no_uint64_shell_ceiling",
        "test_target_evaluator_is_exactly_semantic_class_constant_by_interface",
        "test_pathwise_proposal_ratio_and_unit_jacobian_give_detailed_balance",
        "test_finite_censored_reference_is_stochastic_reversible_and_invariant",
        "test_local_rj_contract_fails_closed_without_authorizing_resident_smc",
    ),
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    syntax_count = _syntax_check(root)
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert5-raw-state-local-rj-response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "role": (
                    "response-free exact combinatorial, involutive-RJ, and "
                    "finite-transition audit"
                ),
                "standalone_raw_state_local_rj_authorized": True,
                "complete_forward_reverse_support_verified": True,
                "exact_proposal_ratio_verified": True,
                "target_invariance_verified": True,
                "common_target_semantic_api_verified": True,
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
