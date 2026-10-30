"""Zero-cost recovery re-aggregation for the v3 execution.

The first run of run_feedback_repair.py aggregated child artifacts with the
Windows default (gbk) text encoding, which raised UnicodeDecodeError on
UTF-8 provider content and marked 3 genuinely-generated children as
`reporting_failed`. Their on-disk artifacts are valid. This script re-reads
all 72 child outcomes with UTF-8 and re-runs only the reporting/freeze step,
so the 3 children score as `success` without any provider call.

Run with PYTHONUTF8=1 so every read_text() in the runner is UTF-8.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_feedback_repair as runner  # noqa: E402


def main():
    reg_path = ROOT / "outputs/feedback_repair_r_v3_acceptance_20261024/REGISTRATION.json"
    exec_dir = ROOT / "outputs/feedback_repair_r_v3_EXECUTION_20261024"
    registration = json.loads(reg_path.read_text(encoding="utf-8"))

    # Back up the buggy report + freeze for provenance before overwriting.
    buggy = exec_dir / "REPAIR_RESULT.json"
    if buggy.is_file():
        shutil.copy2(buggy, exec_dir / "REPAIR_RESULT_pre_utf8_recovery.json")
        print("backed up buggy REPAIR_RESULT.json -> REPAIR_RESULT_pre_utf8_recovery.json")
    freeze = exec_dir / "GENERATION_FREEZE.json"
    if freeze.is_file():
        shutil.copy2(freeze, exec_dir / "GENERATION_FREEZE_pre_utf8_recovery.json")
        print("backed up buggy GENERATION_FREEZE.json -> GENERATION_FREEZE_pre_utf8_recovery.json")
    # write_new() opens with exclusive-create ("x"); clear the stale outputs
    # produced by the first (gbk) run so the corrected freeze/report can be written.
    for stale in (freeze, buggy):
        if stale.is_file():
            stale.unlink()
            print("removed stale", stale.name)

    # Reconstruct rows from on-disk child results. We recompute the derived
    # fields (bank_rows / engine_usage / artifact_sha256) directly from each
    # child's artifact, exactly as _run_child does, because the child process's
    # own result file omits those runner-computed fields.
    import hashlib
    rows = []
    missing = []
    child_dir = exec_dir / "children"
    for child in registration["children"]:
        label = f"{child['task']}__{child['seed']}__{child['arm']}"
        result_path = child_dir / label / "THREE_ARM_CHILD_RESULT.json"
        row = {**child}
        if result_path.is_file():
            try:
                src = json.loads(result_path.read_text(encoding="utf-8"))
                artifact_path = None
                if isinstance(src.get("artifact"), str):
                    cand = Path(src["artifact"])
                    if cand.is_file():
                        artifact_path = cand
                if artifact_path is None:
                    cands = sorted((child_dir / label / "pcpi_artifacts").glob("*.json"))
                    artifact_path = cands[0] if cands else None
                if artifact_path is None or not artifact_path.is_file():
                    raise ValueError("artifact missing for " + label)
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                bank = runner.final_admitted_rows(artifact)
                usage = runner.engine_usage(artifact["scientific_discovery_runtime"])
                row.update(status="generated", artifact=str(artifact_path),
                           artifact_sha256=hashlib.sha256(
                               artifact_path.read_bytes()).hexdigest(),
                           bank_rows=bank, engine_usage=usage,
                           inherited_from_v2=bool(src.get("inherited_from_v2")))
                rows.append(row)
            except Exception as error:
                rows.append({**child, "status": "failed",
                             "error": f"reconstruct failed: {type(error).__name__}: {error}"})
                missing.append(label)
        else:
            rows.append({**child, "status": "failed",
                         "error": "no child result file (retained failure)"})
            missing.append(label)

    statuses = {}
    for r in rows:
        statuses[r.get("status")] = statuses.get(r.get("status"), 0) + 1
    print("reconstructed rows:", len(rows), "statuses:", statuses)
    if missing:
        print("retained failures (no result file):", missing)

    # Re-run only the reporting/freeze step (no provider, no child execution).
    result = runner.report_after_freeze(registration, rows, exec_dir)
    print("protocol_complete:", result.get("protocol_complete"),
          "failure_count:", result.get("failure_count"),
          "child_count:", result.get("child_count"))
    # Tally final statuses
    final = {}
    for r in result["children"]:
        final[r.get("status")] = final.get(r.get("status"), 0) + 1
    print("FINAL child statuses:", final)
    print("wrote", exec_dir / "REPAIR_RESULT.json")


if __name__ == "__main__":
    main()
