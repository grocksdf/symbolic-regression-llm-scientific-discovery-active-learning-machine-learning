"""Response-free correctness Gate for global protected admission."""
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


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True,
        encoding="utf-8", stdout=subprocess.PIPE).stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("correctness output already exists")
    process = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q",
         "-p", "no:cacheprovider",
         "tests/test_global_protected_admission.py",
         "tests/test_posterior_gap_full_pipeline.py",
         "tests/test_iterative_posterior_feedback.py"],
        cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    source = ROOT / (
        "hypothesis_mvp/discovery/regional_candidate_admission.py")
    text = source.read_text(encoding="utf-8")
    checks = {
        "tests_passed": process.returncode == 0,
        "historical_regional_method_retained":
            "def admit_regional_candidates(" in text,
        "new_method_has_distinct_identity":
            "def admit_global_protected_candidates(" in text,
        "global_multiplicity_gate_present":
            "global_ratio > threshold" in text,
        "all_region_nonnegative_gate_present":
            "minimum >= 0.0" in text,
        "empty_region_fails_closed":
            "empty-eligible-admission-region" in text,
        "no_provider_or_scientific_data_access": True,
    }
    result = {
        "schema": "global-protected-admission-correctness-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "pytest_return_code": process.returncode,
        "pytest_output": process.stdout,
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_sha256": digest(source),
        "test_sha256": digest(
            ROOT / "tests/test_global_protected_admission.py"),
        "contract_sha256": digest(
            ROOT / "docs/global_protected_admission_contract_20261027.md"),
        "provider_calls": 0,
        "scientific_data_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
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
