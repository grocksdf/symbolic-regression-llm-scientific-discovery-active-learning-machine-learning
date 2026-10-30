"""Audit-corrected re-aggregation for the v3 feedback-repair execution.

ROOT CAUSE OF THE AUDIT FINDINGS
--------------------------------
`reaggregate_v3_results_utf8.py` (the UTF-8 recovery pass) rebuilt every child row
from on-disk artifacts but recomputed ONLY `bank_rows` / `engine_usage` /
`artifact_sha256`. It dropped three runner-computed fields that `cost_summary()`
depends on:

  * `provider_cost`            -> recorded_transport_attempts collapsed to 0
                                  and known_total_tokens to null
  * `elapsed_seconds`          -> sum_child_wall_seconds collapsed to 0.0
  * `candidate_diagnostics`    -> independent_admissions collapsed to 0

So the "missing provider cost", "zero transport attempts / wall seconds" and
"independent_admissions == 0" findings are ONE regression, not three. The
underlying evidence was never lost: every non-inherited child still has its
`PROVIDER_COST_CAPTURE.json` ledger on disk.

WHAT THIS SCRIPT DOES
---------------------
Rebuilds all 72 rows with the full field set and re-runs the reporting/freeze step.
It makes NO provider call and re-executes NO child (zero marginal cost).

Field provenance (explicitly recorded, never guessed):
  * provider_cost / capture sha / provenance  <- PROVIDER_COST_CAPTURE.json, whose
    `source` field gives the provenance; capture sha256 is sha256(file bytes),
    exactly as `persist_provider_cost` returns.
  * elapsed_seconds                           <- child-reported `wall_time_seconds`.
    The runner's parent-side `elapsed_seconds` was never persisted, so this is a
    documented substitute, flagged in `elapsed_seconds_provenance`.
    Inherited children report 0.0 (they make no v3 execution).
  * candidate_diagnostics                     <- recomputed from the artifact with
    the identical expression used by `_run_child`.

EVIDENCE SAFETY
---------------
Existing REPAIR_RESULT.json / GENERATION_FREEZE.json are copied into
outputs/.../audit_20261026/ with a TIMESTAMPED name before being replaced, so this
script never clobbers prior evidence (the old script overwrote a fixed backup name).

Run from the llm-srbench root with the v3 venv python, PYTHONUTF8=1.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_feedback_repair as runner  # noqa: E402
from scripts.feedback_repair_contract import (  # noqa: E402
    engine_usage, final_admitted_rows)

REG_PATH = ROOT / "outputs/feedback_repair_r_v3_acceptance_20261024/REGISTRATION.json"
EXEC_DIR = ROOT / "outputs/feedback_repair_r_v3_EXECUTION_20261024"
AUDIT = EXEC_DIR / "audit_20261026"
AUDIT.mkdir(parents=True, exist_ok=True)


def _candidate_diagnostics(artifact):
    """Identical to the expression `_run_child` stores for a successful child."""
    runtime = artifact["scientific_discovery_runtime"]
    trace = ((runtime.get("scientist_agent_system_evaluation") or {})
             .get("iterative_feedback_trace", ()))
    return {
        "generation_slots": [slot for cycle in trace
                             for slot in cycle.get("generation_slots", ())],
        "evaluated_bank_count": len(runtime.get("evaluated_hypothesis_bank", ())),
        "cycles": [{key: cycle.get(key) for key in (
            "attempted_candidate_count", "proposal_batch_count", "admitted_lineages",
            "admitted_supports", "admission_report", "generation_diagnostics")}
            for cycle in trace],
    }


def _protect_existing(name):
    """Copy an existing artifact into audit/ with a timestamped, non-clobbering name."""
    src = EXEC_DIR / name
    if not src.is_file():
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = AUDIT / f"SNAPSHOT_{stamp}_{name}"
    shutil.copy2(src, dest)
    print(f"protected evidence: {name} -> {dest.name}")
    return dest


def main():
    registration = json.loads(REG_PATH.read_text(encoding="utf-8"))
    # 1. Protect current evidence before rewriting.
    for name in ("REPAIR_RESULT.json", "GENERATION_FREEZE.json"):
        _protect_existing(name)

    rows = []
    notes = {"no_wall_time": [], "no_cost_capture_noninherited": [],
             "cost_capture_on_inherited": []}
    for child in registration["children"]:
        label = f"{child['task']}__{child['seed']}__{child['arm']}"
        folder = EXEC_DIR / "children" / label
        res_path = folder / "THREE_ARM_CHILD_RESULT.json"
        cap_path = folder / "PROVIDER_COST_CAPTURE.json"
        row = {**child}
        if not res_path.is_file():
            rows.append({**row, "status": "failed",
                         "error": "no child result file (retained failure)"})
            continue
        src = json.loads(res_path.read_text(encoding="utf-8"))
        inherited = bool(src.get("inherited_from_v2"))
        row["child_result_sha256"] = hashlib.sha256(res_path.read_bytes()).hexdigest()
        row["output_dir"] = str(folder)
        if inherited:
            row["inherited_from_v2"] = True
            row["inherited_source_result_sha256"] = src.get("inherited_source_result_sha256")
            # Inherited children make no v3 provider call and no v3 execution.
            row["elapsed_seconds"] = 0.0
            row["elapsed_seconds_provenance"] = "inherited_no_v3_execution"
        else:
            wall = src.get("wall_time_seconds")
            row["elapsed_seconds"] = float(wall) if isinstance(wall, (int, float)) else 0.0
            row["elapsed_seconds_provenance"] = (
                "child_reported_wall_time_seconds" if isinstance(wall, (int, float))
                else "unavailable_child_wall_time_missing")
            if not isinstance(wall, (int, float)):
                notes["no_wall_time"].append(label)
            # Provider cost: rebuild from the persisted ledger.
            if cap_path.is_file():
                cap = json.loads(cap_path.read_text(encoding="utf-8"))
                row["provider_cost"] = cap.get("provider_cost", [])
                row["provider_cost_capture_path"] = str(cap_path)
                row["provider_cost_capture_sha256"] = hashlib.sha256(
                    cap_path.read_bytes()).hexdigest()
                row["provider_cost_provenance"] = str(cap.get("source")) + "_verified"
                row["provider_cost_capture_source_file"] = cap.get("source_file")
                row["provider_cost_capture_source_sha256"] = cap.get("source_sha256")
            else:
                notes["no_cost_capture_noninherited"].append(label)
        if inherited and cap_path.is_file():
            notes["cost_capture_on_inherited"].append(label)

        # Artifact identity + derived scoring fields (same functions as _run_child).
        artifact_path = None
        if isinstance(src.get("artifact"), str) and Path(src["artifact"]).is_file():
            artifact_path = Path(src["artifact"])
        else:
            cands = sorted((folder / "pcpi_artifacts").glob("*.json"))
            artifact_path = cands[0] if cands else None
        if artifact_path is None or not artifact_path.is_file():
            rows.append({**row, "status": "failed", "error": "artifact missing"})
            continue
        row["artifact"] = str(artifact_path)
        row["artifact_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        row["bank_rows"] = final_admitted_rows(artifact)
        row["engine_usage"] = engine_usage(artifact["scientific_discovery_runtime"])
        row["candidate_diagnostics"] = _candidate_diagnostics(artifact)
        row["status"] = "generated"
        rows.append(row)

    # 2. write_new() uses exclusive-create; clear the canonical outputs (already
    #    protected above) so the corrected freeze/report can be written.
    for stale in (EXEC_DIR / "GENERATION_FREEZE.json", EXEC_DIR / "REPAIR_RESULT.json"):
        if stale.is_file():
            stale.unlink()

    result = runner.report_after_freeze(registration, rows, EXEC_DIR)

    # 3. Audit summary of the rebuilt fields.
    costs = result.get("costs", {})
    admissions = sum(len(c.get("admitted_supports") or ())
                     for r in rows for c in r.get("candidate_diagnostics", {}).get("cycles", ()))
    llm_bank = sum(1 for r in rows for b in (r.get("bank_rows") or ())
                   if str(b.get("source", "")).startswith("llm_"))
    audit = {
        "protocol_complete": result.get("protocol_complete"),
        "failure_count": result.get("failure_count"),
        "child_count": result.get("child_count"),
        "costs": costs,
        "rebuilt_independent_admissions_total": admissions,
        "final_bank_rows_with_llm_source": llm_bank,
        "inherited_children": sum(r.get("inherited_from_v2") is True for r in rows),
        "children_with_provider_cost": sum("provider_cost" in r for r in rows),
        "notes": notes,
    }
    (AUDIT / "CORRECTED_AGGREGATION_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("\nwrote", AUDIT / "CORRECTED_AGGREGATION_AUDIT.json")
    print("wrote", EXEC_DIR / "REPAIR_RESULT.json")


if __name__ == "__main__":
    main()
