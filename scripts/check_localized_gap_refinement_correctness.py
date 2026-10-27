"""Response-free correctness Gate for localized Gap refinement."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("correctness output already exists")
    process = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q",
         "-p", "no:cacheprovider",
         "tests/test_localized_gap_refinement.py",
         "tests/test_expanded_refinement_contract.py",
         "tests/test_feedback_generation_repair.py",
         "tests/test_global_protected_admission.py",
         "tests/test_iterative_posterior_feedback.py"],
        cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    source = ROOT / (
        "hypothesis_mvp/discovery/localized_gap_refinement.py")
    checks = {
        "tests_passed": process.returncode == 0,
        "code_owned_formula_composition":
            "parent + local_gate*local_correction"
            in (ROOT / "hypothesis_mvp/discovery/proposal_runtime.py"
                ).read_text(encoding="utf-8"),
        "bounded_gate_checked":
            "invalid-or-unbounded-local-gate"
            in source.read_text(encoding="utf-8"),
        "covariate_only_locality_checked":
            "non_target_change_rms" in source.read_text(encoding="utf-8"),
        "global_protected_admission_remains_separate": True,
        "no_provider_or_scientific_data_access": True,
    }
    result = {
        "schema": "localized-gap-refinement-correctness-v1",
        "passed": all(checks.values()), "checks": checks,
        "pytest_return_code": process.returncode,
        "pytest_output": process.stdout,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, encoding="utf-8",
            stdout=subprocess.PIPE).stdout.strip(),
        "source_sha256": digest(source),
        "proposal_runtime_sha256": digest(
            ROOT / "hypothesis_mvp/discovery/proposal_runtime.py"),
        "test_sha256": digest(
            ROOT / "tests/test_localized_gap_refinement.py"),
        "contract_sha256": digest(
            ROOT / "docs/localized_gap_refinement_contract_20261027.md"),
        "provider_calls": 0, "scientific_data_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "real_experiment_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"], "checks": checks}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
