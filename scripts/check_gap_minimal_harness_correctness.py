"""No-provider correctness Gate for the shared-E0 micro-experiment."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAINLINE = ROOT.parent / "hypothesis_mvp"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("correctness output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    process = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q",
         "-p", "no:cacheprovider",
         "tests/test_gap_minimal_harness_contract.py",
         "tests/test_gap_candidate_marginal_runner.py"],
        cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    mainline_process = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q",
         "-p", "no:cacheprovider",
         "tests/test_symbolic_closed_contract.py",
         "tests/test_feedback_generation_repair.py",
         "tests/test_global_protected_admission.py",
         "tests/test_localized_gap_refinement.py"],
        cwd=MAINLINE, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    checks = {
        "protocol_schema":
            protocol.get("schema")
            == "gap-candidate-marginal-value-protocol-v2",
        "user_authorized":
            protocol.get("authorization", {}).get(
                "user_authorized_experiment") is True,
        "registered_request_count":
            protocol.get("selection", {}).get("slot_requests")
            == (protocol.get("selection", {}).get("blocks", 0)
                * len(protocol.get("selection", {}).get(
                    "conditions", ()))
                * int(protocol.get(
                    "search_rich_generation", {}).get("rounds", 1))),
        "registered_conditions":
            bool(protocol.get("selection", {}).get("conditions"))
            and set(protocol["selection"]["conditions"]) <= {
                "B0", "G_hard0", "G_soft0", "G_placebo0"},
        "quota_zero":
            protocol.get("shared_block_contract", {}).get(
                "same_new_skeleton_quota") == 0,
        "shared_E0_required":
            protocol.get("shared_block_contract", {}).get(
                "same_E0_all_conditions") is True,
        "shared_parent_required":
            protocol.get("shared_block_contract", {}).get(
                "same_parent_all_conditions") is True,
        "generation_freeze_required":
            protocol.get("outcome", {}).get(
                "generation_freeze_before_reporting") is True,
        "registered_admission_method":
            protocol.get("admission_method", "regional-any-region-v1")
            in {"regional-any-region-v1",
                "global-protected-fixed-expert-v1"},
        "localized_requires_global_protection":
            (not protocol.get("localized_gap_refinement")
             or protocol.get("admission_method")
             == "global-protected-fixed-expert-v1"),
        "localized_no_target_fails_closed":
            (not protocol.get("localized_gap_refinement")
             or protocol.get(
                "localized_no_target_policy", {}).get("action")
             == "code-owned-semantic-abstention"),
        "posterior_transaction_roles_registered":
            (not protocol.get("localized_gap_refinement")
             or protocol.get(
                "posterior_update_transaction", {}).get(
                    "admission_responses_reused_for_update") is False
             and protocol.get(
                "posterior_update_transaction", {}).get(
                    "arm_generation_frozen_before_admission_response")
             is True),
        "search_rich_budget_is_matched":
            (not protocol.get("search_rich_generation")
             or protocol["search_rich_generation"].get(
                 "same_budget_all_conditions") is True
             and protocol["search_rich_generation"].get(
                 "maximum_candidate_opportunities_per_arm")
             == protocol["shared_block_contract"].get(
                 "same_candidate_slots")),
        "contract_tests_passed": process.returncode == 0,
        "mainline_scheduler_and_feedback_tests_passed":
            mainline_process.returncode == 0,
        "production_runner_present": (
            ROOT / "scripts/run_gap_candidate_marginal_value.py").is_file(),
        "freeze_builder_present": (
            ROOT / "scripts/build_gap_candidate_marginal_freeze.py").is_file(),
    }
    result = {
        "schema": "gap-minimal-harness-correctness-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "pytest_return_code": process.returncode,
        "pytest_output": process.stdout,
        "mainline_pytest_return_code": mainline_process.returncode,
        "mainline_pytest_output": mainline_process.stdout,
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": digest(args.protocol),
        "contract_source_sha256": digest(
            ROOT / "scripts/gap_minimal_harness_contract.py"),
        "test_source_sha256": digest(
            ROOT / "tests/test_gap_minimal_harness_contract.py"),
        "runner_test_source_sha256": digest(
            ROOT / "tests/test_gap_candidate_marginal_runner.py"),
        "production_runner_sha256": digest(
            ROOT / "scripts/run_gap_candidate_marginal_value.py"),
        "freeze_builder_sha256": digest(
            ROOT / "scripts/build_gap_candidate_marginal_freeze.py"),
        "provider_calls": 0,
        "scientific_data_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": False
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": checks}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
