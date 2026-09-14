"""Read-only forensic audit for a completed P3M.7 output directory."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

from hypothesis_mvp.pcpi import P3M7_CHECKPOINT_SCHEMA, P3M7_DECISION_RISK_UTILITY


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    queries = pd.read_csv(root / "tables" / "acquisition_queries.csv")
    pcpi = queries[queries["policy"].astype(str).str.contains("pcpi_")]
    schemas: collections.Counter[str] = collections.Counter()
    utilities: collections.Counter[str] = collections.Counter()
    bad: list[str] = []
    for path in root.joinpath("p3m").rglob("risk-nodes-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan = payload.get("plan", {})
        schemas[str(plan.get("schema"))] += 1
        utilities[str(plan.get("utility_method"))] += 1
        if (plan.get("schema") != P3M7_CHECKPOINT_SCHEMA
                or plan.get("utility_method") != P3M7_DECISION_RISK_UTILITY):
            bad.append(str(path))
    return {
        "output": str(root),
        "successful_runs": int(summary["successful_runs"]),
        "expected_runs": int(summary["expected_runs"]),
        "failure_count": int(summary["failure_count"]),
        "original_protocol_gate_passed": bool(summary["protocol_gate_passed"]),
        "manifest_complete": bool(manifest.get("complete")),
        "heldout_opened": bool(summary["heldout_opened"]),
        "pcpi_query_count": int(len(pcpi)),
        "pcpi_information_risk_methods": pcpi["information_risk_method"].value_counts(dropna=False).to_dict(),
        "checkpoint_file_count": int(sum(schemas.values())),
        "checkpoint_schemas": dict(schemas),
        "checkpoint_utilities": dict(utilities),
        "bad_checkpoint_count": len(bad),
        "bad_checkpoint_examples": bad[:3],
        "read_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
