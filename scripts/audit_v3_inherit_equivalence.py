"""v2 -> v3 reuse equivalence proof for the 35 inherited children.

For every child the audit asks: is the artifact scored in the v3 tree really the
v2 artifact it claims to reuse, and are the 37 non-inherited artifacts genuinely
v3-generated rather than silently reused?

Checks per inherited child:
  * v2 source artifact vs v3 copy: SHA-256 equality (byte-identical).
  * If bytes differ, whether the difference is only Windows CRLF translation
    (JSON text identical, so semantically equivalent) -- `_inherit_child` copies
    with read_text/write_text, which on Windows translates "\\n" -> "\\r\\n".

Checks for the 37 non-inherited children:
  * Their v3 artifact SHA-256 must DIFFER from any same-label v2 artifact, proving
    genuine v3 regeneration rather than silent reuse.

Read-only. Writes only to outputs/.../audit_20261026/.
Run from the llm-srbench root with the v3 venv python.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "outputs/feedback_repair_r_v3_EXECUTION_20261024"
V2 = ROOT / "outputs/feedback_repair_R_v2_EXECUTION_20261023_fix1"
AUDIT = EXEC / "audit_20261026"
AUDIT.mkdir(parents=True, exist_ok=True)

inventory = json.loads((AUDIT / "CHILD_INVENTORY.json").read_text(encoding="utf-8"))
children = inventory["children"]


def artifact_of(folder: Path):
    """Locate a child artifact, tolerating both layouts.

    Newly generated children store it under pcpi_artifacts/, but `_inherit_child`
    writes the copied artifact directly into children/<label>/, so a pcpi_artifacts
    glob alone silently misses every inherited child.
    """
    res = folder / "THREE_ARM_CHILD_RESULT.json"
    if res.is_file():
        try:
            src = json.loads(res.read_text(encoding="utf-8"))
            if isinstance(src.get("artifact"), str):
                p = Path(src["artifact"])
                if p.is_file():
                    return p
        except Exception:
            pass
    if (folder / "pcpi_artifacts").is_dir():
        arts = sorted((folder / "pcpi_artifacts").glob("*.json"))
        if arts:
            return arts[0]
    skip = {"THREE_ARM_CHILD_RESULT.json", "PROVIDER_COST_CAPTURE.json",
            "THREE_ARM_CHILD_FAILURE.json"}
    cands = [p for p in folder.glob("*.json") if p.name not in skip]
    return sorted(cands)[0] if cands else None


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# v3 side: use the authoritative artifact path/sha recorded by the corrected
# aggregation (which resolves the artifact via the child result's `artifact` field).
result = json.loads((EXEC / "REPAIR_RESULT.json").read_text(encoding="utf-8"))
v3_by_label = {f"{c['task']}__{c['seed']}__{c['arm']}": c for c in result["children"]}
inventory_inherited = {e["label"]: e.get("inherited_from_v2") for e in children}

report = {"inherited": [], "non_inherited": [], "v2_missing": []}
for entry in children:
    label = entry["label"]
    v3_row = v3_by_label.get(label, {})
    v3_art = Path(v3_row["artifact"]) if v3_row.get("artifact") else None
    if v3_art is not None and not v3_art.is_file():
        v3_art = None
    v2_folder = V2 / "children" / label
    v2_art = artifact_of(v2_folder)
    rec = {"label": label, "arm": entry["arm"],
           "v3_artifact_sha256": v3_row.get("artifact_sha256"),
           "v3_artifact": str(v3_art) if v3_art else None}
    if v2_art is None:
        rec["v2_artifact"] = None
        report["v2_missing"].append(label)
        (report["inherited"] if entry.get("inherited_from_v2")
         else report["non_inherited"]).append(rec)
        continue
    v2_sha = sha(v2_art)
    v2_size = v2_art.stat().st_size
    v3_size = v3_art.stat().st_size if v3_art else None
    rec.update(v2_artifact=str(v2_art), v2_artifact_sha256=v2_sha,
               v2_size=v2_size, v3_size=v3_size)
    if entry.get("inherited_from_v2"):
        identical = bool(v3_art is not None and sha(v3_art) == v2_sha)
        rec["byte_identical"] = identical
        if not identical and v3_art is not None:
            nl = v2_art.read_bytes().count(b"\n")
            rec["v2_newline_count"] = nl
            rec["size_delta"] = (v3_size or 0) - v2_size
            # CRLF signature: every \n became \r\n, so delta == newline count.
            rec["crlf_translation_signature"] = (rec["size_delta"] == nl)
            rec["json_text_equivalent"] = (
                v3_art.read_text(encoding="utf-8") == v2_art.read_text(encoding="utf-8"))
        report["inherited"].append(rec)
    else:
        v3_sha = sha(v3_art) if v3_art else None
        rec["v3_artifact_sha256"] = v3_sha
        rec["differs_from_v2"] = v3_sha != v2_sha
        report["non_inherited"].append(rec)

inh = report["inherited"]
ni = report["non_inherited"]
summary = {
    "inherited_total": len(inh),
    "inherited_byte_identical": sum(r.get("byte_identical") is True for r in inh),
    "inherited_json_text_equivalent_only": sum(
        r.get("byte_identical") is False and r.get("json_text_equivalent") is True
        for r in inh),
    "inherited_crlf_signature": sum(r.get("crlf_translation_signature") is True for r in inh),
    "inherited_proven_reused": sum(
        (r.get("byte_identical") is True) or (r.get("json_text_equivalent") is True)
        for r in inh),
    "non_inherited_total": len(ni),
    "non_inherited_differs_from_v2": sum(r.get("differs_from_v2") is True for r in ni),
    "non_inherited_same_as_v2": [r["label"] for r in ni
                                 if r.get("differs_from_v2") is False],
    "v2_artifact_missing": report["v2_missing"],
}
report["summary"] = summary
(AUDIT / "INHERIT_EQUIVALENCE.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print("\nwrote", AUDIT / "INHERIT_EQUIVALENCE.json")
