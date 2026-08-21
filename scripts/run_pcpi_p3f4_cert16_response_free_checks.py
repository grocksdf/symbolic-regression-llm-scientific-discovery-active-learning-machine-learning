"""Run only the response-free CERT.16 joint-budget/product-bit Gate."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


_cert15 = runpy.run_path(
    str(_repository_root() / "scripts/run_pcpi_p3f4_cert15_response_free_checks.py")
)
EXPECTED_CHECKS = dict(_cert15["EXPECTED_CHECKS"])
EXPECTED_CHECKS[
    "tests/test_pcpi_p3f4_certified_comparison_integration.py"
] = (
    "test_cert16_authorizes_only_standalone_integration_theorem",
    "test_conditional_joint_failure_identity_uses_only_exact_cert9_slack",
    "test_frozen_resampling_mh_and_total_comparison_counts_are_exact",
    "test_finite_coordinate_rank_unrank_is_a_complete_bijection",
    "test_full_coordinate_space_endpoints_and_role_boundaries_are_exact",
    "test_every_island_coordinate_binds_cert11_manifest_commitment",
    "test_philox_comparison_addresses_are_domain_separated_and_injective",
    "test_coordinate_purpose_indices_and_identity_fail_closed",
    "test_crossed_common_finite_source_and_manifest_plans_are_rejected",
    "test_bound_at_allocation_is_certified_without_materializing_bits",
    "test_over_budget_bound_aborts_before_bits_or_partial_output",
    "test_unresolved_comparison_aborts_complete_island_batch",
    "test_guard_and_source_order_precede_bound_key_and_bit_access",
    "test_cert16_retains_every_earlier_operational_guard",
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
        raise RuntimeError("CERT.16 syntax scope contains no tracked Python source")
    if any(path.is_absolute() or ".." in path.parts for path in relatives):
        raise RuntimeError("CERT.16 received an invalid tracked Python path")
    files = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("CERT.16 tracked Python source is absent from the worktree")
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
        raise RuntimeError("CERT.16 requires exactly python-flint 0.8.0")
    passed: list[str] = []
    for relative, names in EXPECTED_CHECKS.items():
        namespace = runpy.run_path(str(root / relative))
        for name in names:
            check = namespace.get(name)
            if not callable(check):
                raise RuntimeError(f"registered response-free check is absent: {name}")
            check()
            passed.append(name)
    if len(passed) != 152:
        raise RuntimeError("CERT.16 response-free check identity is not exactly 152")
    syntax_count = _syntax_check(root)
    if syntax_count != 235:
        raise RuntimeError("CERT.16 tracked Python source identity is not exactly 235")
    print(
        json.dumps(
            {
                "schema": (
                    "pcpi-p3f4-cert16-joint-failure-product-bit-"
                    "response-free-checks-v1"
                ),
                "status": "passed",
                "checks_passed": passed,
                "check_count": len(passed),
                "python_files_syntax_checked": syntax_count,
                "python_syntax_scope": "git-tracked-python-files",
                "role": (
                    "response-free joint finite-N/comparison failure theorem "
                    "and implicit manifest-bound product-bit address space"
                ),
                "python_flint_version": observed_flint,
                "arb_working_precision_bits": 512,
                "uniform_threshold_bit_count": 256,
                "conditional_joint_failure_identity_verified": True,
                "finite_n_slack_only_used": True,
                "uniform_reachable_state_comparison_bound_verified": False,
                "complete_coordinate_bijection_verified": True,
                "cert11_manifest_binding_verified": True,
                "philox_counter_domain_separation_verified": True,
                "pre_bit_bound_check_verified": True,
                "whole_batch_abort_verified": True,
                "selective_reporting_authorized": False,
                "adaptive_bit_extension_used": False,
                "result_dependent_precision_retry_used": False,
                "philox_pseudorandomness_promoted_to_mathematical_independence": False,
                "external_ideal_bit_product_law_required": True,
                "external_ideal_bit_product_law_implementation_authorized": False,
                "product_bits_materialization_authorized": False,
                "resident_comparison_integration_authorized": False,
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
