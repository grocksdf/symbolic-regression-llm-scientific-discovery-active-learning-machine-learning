"""Audit inventory for the v3 feedback-repair execution (cheap pass, no big artifacts).

Reconstructs, for every one of the 72 registered children, what on-disk evidence
actually exists: child result file, provider-cost capture ledger, inherited flag,
per-child wall time, and the artifact's SHA-256 (bytes only -- no JSON parse of the
multi-MB artifacts).

This is read-only with respect to existing evidence: it writes ONLY new files under
outputs/feedback_repair_r_v3_EXECUTION_20261024/audit_20261026/.

Run from the llm-srbench root with the v3 venv python.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "outputs/feedback_repair_r_v3_acceptance_20261024/REGISTRATION.json"
EXEC = ROOT / "outputs/feedback_repair_r_v3_EXECUTION_20261024"
OUT = EXEC / "audit_20261026"
OUT.mkdir(parents=True, exist_ok=True)

registration = json.loads(REG.read_text(encoding="utf-8"))
rows = []
for child in registration["children"]:
    label = f"{child['task']}__{child['seed']}__{child['arm']}"
    folder = EXEC / "children" / label
    res_p = folder / "THREE_ARM_CHILD_RESULT.json"
    cap_p = folder / "PROVIDER_COST_CAPTURE.json"
    entry: dict = {"label": label, "arm": child["arm"], "task": child["task"],
                   "seed": child["seed"]}
    entry["has_result_file"] = res_p.is_file()
    if res_p.is_file():
        r = json.loads(res_p.read_text(encoding="utf-8"))
        entry["inherited_from_v2"] = bool(r.get("inherited_from_v2"))
        entry["child_status"] = r.get("status")
        pc = r.get("provider_cost")
        entry["result_provider_cost_requests"] = len(pc) if isinstance(pc, list) else None
        entry["child_wall_time_seconds"] = r.get("wall_time_seconds")
        entry["inherited_source_result_sha256"] = r.get("inherited_source_result_sha256")
        entry["result_artifact"] = r.get("artifact")
    entry["has_cost_capture"] = cap_p.is_file()
    if cap_p.is_file():
        c = json.loads(cap_p.read_text(encoding="utf-8"))
        entry["capture_source"] = c.get("source")
        entry["capture_sha256"] = hashlib.sha256(cap_p.read_bytes()).hexdigest()
        entry["capture_source_file"] = c.get("source_file")
        entry["capture_source_sha256"] = c.get("source_sha256")
        pcc = c.get("provider_cost")
        entry["capture_requests"] = len(pcc) if isinstance(pcc, list) else None
        toks = [x.get("total_tokens") for x in pcc if isinstance(x, dict)]
        known = [t for t in toks if t is not None]
        entry["capture_known_total_tokens"] = sum(known) if known else None
        entry["capture_attempts_without_tokens"] = len(toks) - len(known)
    arts = sorted((folder / "pcpi_artifacts").glob("*.json")) if (
        folder / "pcpi_artifacts").is_dir() else []
    entry["n_artifacts"] = len(arts)
    if arts:
        entry["artifact"] = str(arts[0])
        entry["artifact_sha256"] = hashlib.sha256(arts[0].read_bytes()).hexdigest()
    rows.append(entry)

# ---- summary -------------------------------------------------------------
def by_arm(pred):
    return {arm: sum(pred(e) for e in rows if e["arm"] == arm) for arm in ("C", "D", "R")}

summary = {
    "total_children": len(rows),
    "with_result_file": sum(e["has_result_file"] for e in rows),
    "inherited": sum(e.get("inherited_from_v2") is True for e in rows),
    "inherited_by_arm": by_arm(lambda e: e.get("inherited_from_v2") is True),
    "with_cost_capture": sum(e["has_cost_capture"] for e in rows),
    "with_cost_capture_by_arm": by_arm(lambda e: e["has_cost_capture"]),
    "total_capture_requests": sum(e.get("capture_requests") or 0 for e in rows),
    "capture_requests_by_arm": {arm: sum(e.get("capture_requests") or 0
                                         for e in rows if e["arm"] == arm)
                                for arm in ("C", "D", "R")},
    "known_total_tokens": sum(e.get("capture_known_total_tokens") or 0 for e in rows),
    "known_total_tokens_by_arm": {arm: sum(e.get("capture_known_total_tokens") or 0
                                           for e in rows if e["arm"] == arm)
                                  for arm in ("C", "D", "R")},
    "sum_child_wall_seconds": sum(e.get("child_wall_time_seconds") or 0. for e in rows),
}
for e in rows:
    if e.get("capture_requests") is None and e.get("result_provider_cost_requests"):
        summary.setdefault("note", []).append(
            f"{e['label']}: no capture file but result carries "
            f"{e['result_provider_cost_requests']} requests")

(OUT / "CHILD_INVENTORY.json").write_text(
    json.dumps({"summary": summary, "children": rows}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8")

print(json.dumps(summary, indent=2))
print("\nwrote", OUT / "CHILD_INVENTORY.json")
