"""Build an immutable, read-only reporting certificate for the P3M.8 pilot."""
from __future__ import annotations

import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import pandas as pd
from hypothesis_mvp.pcpi import P3M8_CHECKPOINT_SCHEMA, P3M8_DECISION_RISK_PENALIZED_UTILITY

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(root: Path) -> dict[str, object]:
    root = root.resolve()
    summary_path, manifest_path = root / "summary.json", root / "RUN_MANIFEST.json"
    query_path = root / "tables" / "acquisition_queries.csv"
    summary, manifest = json.loads(summary_path.read_text()), json.loads(manifest_path.read_text())
    queries = pd.read_csv(query_path)
    pcpi = queries[queries["policy"].astype(str).str.contains("pcpi_")]
    schemas, utilities = Counter(), Counter(); bad = []
    checkpoints = list(root.joinpath("p3m").rglob("risk-nodes-*.json"))
    for path in checkpoints:
        plan = json.loads(path.read_text(encoding="utf-8")).get("plan", {})
        schema, utility = str(plan.get("schema")), str(plan.get("utility_method"))
        schemas[schema] += 1; utilities[utility] += 1
        if schema != P3M8_CHECKPOINT_SCHEMA or utility != P3M8_DECISION_RISK_PENALIZED_UTILITY:
            bad.append(str(path))
    old = Counter(str(v) for v in pcpi["information_risk_method"])
    checks = {
        "all_runs_completed": int(summary.get("successful_runs", -1)) == int(summary.get("expected_runs", -2)) == 24,
        "no_failures": int(summary.get("failure_count", -1)) == 0,
        "heldout_closed": not bool(summary.get("heldout_opened", True)),
        "pcpi_query_count": len(pcpi) == 192,
        "all_checkpoints_p3m8": len(checkpoints) >= 192 and not bad,
        "legacy_ledger_identity_mismatch_is_explicit": old == Counter({"action-conditional-lower-tail-cvar-of-frozen-class-entropy-reduction-v1": 192}),
        "original_protocol_gate_failed_for_reporting_only": (
            not bool(summary.get("protocol_gate_passed", True))
            and not bool(summary.get("formal_protocol_evidence", True))
            and int(summary.get("failure_count", -1)) == 0
        ),
    }
    return {
        "schema": "pcpi-p3m8-reporting-correction-certificate-v1",
        "source_output": str(root), "source_summary_sha256": digest(summary_path),
        "source_manifest_sha256": digest(manifest_path), "source_query_ledger_sha256": digest(query_path),
        "source_manifest_git_commit": manifest.get("source_git_commit"),
        "source_manifest_git_tree": manifest.get("source_git_tree"),
        "checks": checks, "checkpoint_count": len(checkpoints),
        "checkpoint_schemas": dict(schemas), "checkpoint_utilities": dict(utilities),
        "legacy_query_methods": dict(old),
        "corrected_utility_method": P3M8_DECISION_RISK_PENALIZED_UTILITY,
        "correction_scope": "metadata-only-ledger-identity-reconciliation-no-rescoring-no-selection-no-response-access",
        "original_artifacts_immutable": True, "read_only": True,
        "status": "eligible-for-corrected-reporting-only" if all(checks.values()) else "not-eligible",
        "efficacy_source": "original_summary_unchanged; correction does not upgrade efficacy",
    }

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("output", type=Path); p.add_argument("--certificate", type=Path)
    a = p.parse_args(); root = a.output.resolve(); target = (a.certificate or root / "p3m8_correction_certificate.json").resolve()
    if target.exists(): raise SystemExit(f"refusing to overwrite existing certificate: {target}")
    cert = build(root); target.write_text(json.dumps(cert, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(cert, indent=2, sort_keys=True)); return 0 if cert["status"] == "eligible-for-corrected-reporting-only" else 2
if __name__ == "__main__": raise SystemExit(main())
