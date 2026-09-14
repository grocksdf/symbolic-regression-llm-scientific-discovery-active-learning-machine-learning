"""Build a read-only, hash-linked correction certificate for legacy P3M.7 logs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd

from hypothesis_mvp.pcpi import P3M7_CHECKPOINT_SCHEMA, P3M7_DECISION_RISK_UTILITY


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path) -> dict[str, object]:
    root = root.resolve()
    summary_path = root / "summary.json"
    manifest_path = root / "RUN_MANIFEST.json"
    query_path = root / "tables" / "acquisition_queries.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = pd.read_csv(query_path)
    pcpi = queries[queries["policy"].astype(str).str.contains("pcpi_")]
    schemas: Counter[str] = Counter()
    utilities: Counter[str] = Counter()
    checkpoint_count = 0
    bad: list[str] = []
    for path in root.joinpath("p3m").rglob("risk-nodes-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan = payload.get("plan", {})
        schema, utility = str(plan.get("schema")), str(plan.get("utility_method"))
        schemas[schema] += 1
        utilities[utility] += 1
        checkpoint_count += 1
        if schema != P3M7_CHECKPOINT_SCHEMA or utility != P3M7_DECISION_RISK_UTILITY:
            bad.append(str(path))
    old_methods = Counter(str(v) for v in pcpi["information_risk_method"])
    checks = {
        "all_runs_completed": int(summary["successful_runs"]) == int(summary["expected_runs"]) == 96,
        "no_failures": int(summary["failure_count"]) == 0,
        "heldout_closed": not bool(summary["heldout_opened"]),
        "pcpi_query_count": len(pcpi) == 768,
        "all_checkpoints_decision_risk": checkpoint_count == 11664 and not bad,
        "legacy_ledger_identity_mismatch_is_explicit": old_methods == Counter({"action-conditional-lower-tail-cvar-of-frozen-class-entropy-reduction-v1": 768}),
    }
    return {
        "schema": "pcpi-p3m7-reporting-correction-certificate-v1",
        "source_output": str(root),
        "source_summary_sha256": digest(summary_path),
        "source_manifest_sha256": digest(manifest_path),
        "source_query_ledger_sha256": digest(query_path),
        "source_manifest_git_commit": manifest.get("source_git_commit"),
        "source_manifest_git_tree": manifest.get("source_git_tree"),
        "checks": checks,
        "checkpoint_count": checkpoint_count,
        "checkpoint_schemas": dict(schemas),
        "checkpoint_utilities": dict(utilities),
        "legacy_query_methods": dict(old_methods),
        "corrected_utility_method": P3M7_DECISION_RISK_UTILITY,
        "correction_scope": "metadata-only-ledger-identity-reconciliation-no-rescoring-no-selection-no-response-access",
        "original_artifacts_immutable": True,
        "read_only": True,
        "status": "eligible-for-corrected-reporting-only" if all(checks.values()) else "not-eligible",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--certificate", type=Path)
    args = parser.parse_args()
    certificate = build(args.output)
    target = args.certificate or (args.output / "p3m7_correction_certificate.json")
    target.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if certificate["status"] == "eligible-for-corrected-reporting-only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
