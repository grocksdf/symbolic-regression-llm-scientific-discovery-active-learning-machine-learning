"""Artifact-only forensic audit for a completed AISTATS DRR run."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _failure_signature(text):
    rules = (
        ("missing-lsr-bench-data", "Missing required lsr_bench_data.hdf5"),
        ("provider-content-missing", "provider-content-missing"),
        ("incomplete-or-inconsistent-plan", "incomplete-or-inconsistent-plan"),
        ("non-string-scientist-text", "non-string-scientist-text"),
        ("invalid-scientist-text-container", "invalid-scientist-text-container"),
    )
    return next((name for name, marker in rules if marker in text), "other")


def audit(source):
    source = Path(source).resolve()
    result_path = source / "AISTATS_DRR_RESULT.json"
    csv_path = source / "TASK_SEED_RESULTS.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    status = Counter((row["condition"], row["status"]) for row in rows)
    readiness = Counter(
        (row["condition"], row["ready"], row["indicator"]) for row in rows)
    child_failures, certificate_failures, decisions = Counter(), Counter(), Counter()
    seen_children = set()
    for row in rows:
        run_dir = Path(row["run_dir"])
        ledger = run_dir / "DRR_RUN_RESULT.json"
        if row["status"] == "failure" and run_dir not in seen_children:
            seen_children.add(run_dir)
            log = run_dir / "stdout.log"
            text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            child_failures[_failure_signature(text)] += 1
        if not ledger.is_file():
            continue
        entries = json.loads(ledger.read_text(encoding="utf-8"))
        entry = next((
            value for value in entries
            if value.get("condition") == row["condition"]), None)
        if not entry or entry.get("status") != "success":
            continue
        certificate = entry.get("certificate") or {}
        message = certificate.get("error_message")
        if message:
            certificate_failures[str(message)] += 1
        for name, passed in certificate.get("decisions", {}).items():
            if passed is not True:
                decisions[name] += 1
    return {
        "schema": "scientific-aistats-drr-forensic-audit-v1",
        "source_output": str(source),
        "source_result_sha256": _sha(result_path),
        "source_task_seed_csv_sha256": _sha(csv_path),
        "protocol_complete": result.get("protocol_complete") is True,
        "primary_passed": result.get(
            "primary_analysis", {}).get("passed") is True,
        "condition_result_count": len(rows),
        "registered_failure_count": int(result.get("failure_count", -1)),
        "status_counts": {
            f"{condition}/{state}": count
            for (condition, state), count in sorted(status.items())
        },
        "readiness_counts": {
            f"{condition}/ready={ready}/indicator={indicator}": count
            for (condition, ready, indicator), count in sorted(readiness.items())
        },
        "unique_child_failure_causes": dict(sorted(child_failures.items())),
        "successful_child_certificate_failures":
            dict(sorted(certificate_failures.items())),
        "failed_readiness_decisions": dict(sorted(decisions.items())),
        "diagnoses": {
            "lsr_data_preflight_missing":
                child_failures["missing-lsr-bench-data"] > 0,
            "provider_or_plan_protocol_instability":
                sum(child_failures[name] for name in (
                    "provider-content-missing",
                    "incomplete-or-inconsistent-plan",
                    "non-string-scientist-text",
                    "invalid-scientist-text-container")) > 0,
            "protected_core_candidate_contract_mismatch":
                certificate_failures[
                    "no capacity bank preserves registered provenance"] > 0,
            "operational_partition_or_utility_degenerate":
                decisions["bank_viability_passed"] > 0
                or decisions["decision_risk_utility_passed"] > 0,
        },
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "task_arrays_reopened": False,
        "engines_or_llm_reran": False,
        "efficacy_demonstrated": False,
        "repair_authorized_by_audit": False,
        "claim_boundary": (
            "Read-only forensic classification of immutable run artifacts; "
            "not a corrected analysis, rerun authorization, efficacy result, "
            "or superiority claim."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("forensic audit output must be new")
    result = audit(args.source_output)
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_DRR_FORENSIC_AUDIT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
