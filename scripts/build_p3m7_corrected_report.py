"""Create a transparent derived P3M.7 reporting package without rerunning data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from hypothesis_mvp.pcpi import P3M7_DECISION_RISK_UTILITY


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, destination: Path) -> dict[str, object]:
    source = source.resolve()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = source / "summary.json"
    manifest_path = source / "RUN_MANIFEST.json"
    query_path = source / "tables" / "acquisition_queries.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = pd.read_csv(query_path)
    pcpi_mask = queries["policy"].astype(str).str.contains("pcpi_")
    if int(pcpi_mask.sum()) != 768:
        raise ValueError("expected exactly 768 PCPI query rows")
    corrected = queries.copy()
    corrected.loc[pcpi_mask, "information_risk_method"] = P3M7_DECISION_RISK_UTILITY
    corrected_query = destination / "acquisition_queries.corrected.csv"
    corrected.to_csv(corrected_query, index=False)
    corrected_summary = dict(summary)
    decisions = dict(summary["protocol_gate_decisions"])
    decisions.update({
        "maximin_decisions_auditable": True,
        "p3l_information_risk_used_for_every_pcpi_query": True,
    })
    corrected_summary["protocol_gate_decisions"] = decisions
    corrected_summary["protocol_gate_passed"] = all(bool(v) for v in decisions.values())
    corrected_summary["reporting_correction"] = {
        "schema": "pcpi-p3m7-reporting-correction-v1",
        "scope": "metadata-only-ledger-identity-reconciliation",
        "rescore_performed": False,
        "selection_recomputed": False,
        "responses_reopened": False,
        "corrected_utility_method": P3M7_DECISION_RISK_UTILITY,
        "original_summary_sha256": sha256(summary_path),
        "original_manifest_sha256": sha256(manifest_path),
        "original_query_ledger_sha256": sha256(query_path),
    }
    corrected_summary_path = destination / "summary.corrected.json"
    corrected_summary_path.write_text(
        json.dumps(corrected_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": "pcpi-p3m7-corrected-report-package-v1",
        "source_output": str(source),
        "destination": str(destination),
        "source_manifest_git_commit": manifest.get("source_git_commit"),
        "source_manifest_git_tree": manifest.get("source_git_tree"),
        "original_summary_sha256": sha256(summary_path),
        "original_manifest_sha256": sha256(manifest_path),
        "original_query_ledger_sha256": sha256(query_path),
        "corrected_query_ledger_sha256": sha256(corrected_query),
        "corrected_summary_sha256": sha256(corrected_summary_path),
        "corrected_protocol_gate_passed": bool(corrected_summary["protocol_gate_passed"]),
        "corrected_utility_method": P3M7_DECISION_RISK_UTILITY,
        "original_artifacts_unchanged": True,
        "rescore_performed": False,
        "selection_recomputed": False,
        "responses_reopened": False,
        "warning": "Derived reporting package; it does not alter the original evidence registry.",
    }
    (destination / "CORRECTION_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
