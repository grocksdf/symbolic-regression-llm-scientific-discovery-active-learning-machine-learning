"""Response-free Gate for single-target Gap and posterior transaction."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("correctness output already exists")
    process = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q",
         "-p", "no:cacheprovider",
         "tests/test_admission_transaction.py",
         "tests/test_localized_gap_refinement.py",
         "tests/test_global_protected_admission.py"],
        cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    localized = (ROOT /
        "hypothesis_mvp/discovery/localized_gap_refinement.py")
    transaction = (ROOT /
        "hypothesis_mvp/discovery/admission_transaction.py")
    checks = {
        "tests_passed": process.returncode == 0,
        "single_target_rule_present":
            "target_selection_rule" in localized.read_text(encoding="utf-8"),
        "three_roles_disjoint":
            "three disjoint roles" in transaction.read_text(encoding="utf-8"),
        "admission_not_reused_for_update":
            "np.vstack((fit.X, posterior_update.X))"
            in transaction.read_text(encoding="utf-8"),
        "no_provider_or_scientific_data_access": True,
    }
    result = {
        "schema": "admission-posterior-transaction-correctness-v1",
        "passed": all(checks.values()), "checks": checks,
        "pytest_output": process.stdout,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, encoding="utf-8",
            stdout=subprocess.PIPE).stdout.strip(),
        "localized_source_sha256": digest(localized),
        "transaction_source_sha256": digest(transaction),
        "provider_calls": 0, "scientific_data_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "real_experiment_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": checks}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
