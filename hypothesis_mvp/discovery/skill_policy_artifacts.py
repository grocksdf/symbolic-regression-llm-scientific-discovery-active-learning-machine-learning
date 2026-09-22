"""Read immutable source-admission artifacts into task-level skill evidence."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from hypothesis_mvp.symbolic.registry import registered_engine_names
from .skill_policy import SkillTaskEvidence


def _digest(value) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _representative(rows):
    scored = [row for row in rows
              if len(row.get("fold_log_score_gains_vs_core", [])) >= 2]
    if not scored:
        return (0.0, 0.0), (0.0, 0.0)
    admitted = [row for row in scored if row.get("admitted") is True]
    pool = admitted or scored
    selected = max(pool, key=lambda row: (
        min(float(gain) - float(tolerance) for gain, tolerance in zip(
            row["fold_log_score_gains_vs_core"],
            row["fold_numerical_tolerances"])),
        str(row.get("candidate_identity", ""))))
    return (tuple(float(value) for value in
                  selected["fold_log_score_gains_vs_core"]),
            tuple(float(value) for value in
                  selected["fold_numerical_tolerances"]))


def extract_skill_task_evidence(source_output: str | Path):
    root = Path(source_output).resolve()
    if (root / "SYSTEM_CONTRACT.json").is_file():
        contract_name = "SYSTEM_CONTRACT.json"
        contract = json.loads((root / contract_name).read_text(
            encoding="utf-8"))
        config = contract.get("registration", {})
    elif (root / "CONTINUATION_CONTRACT.json").is_file():
        contract_name = "CONTINUATION_CONTRACT.json"
        contract = json.loads((root / contract_name).read_text(
            encoding="utf-8"))
        config = contract.get("registration", {})
    else:
        raise ValueError("skill evidence source has no immutable contract")
    if len(config.get("data", [])) != 1 or len(config.get("seeds", [])) != 1:
        raise ValueError("skill evidence source must contain one coordinate")
    dataset, seed = config["data"][0]["dataset"], config["seeds"][0]
    workspace = root / dataset / str(seed)
    manifest = json.loads((workspace / "DATA_MANIFEST.json").read_text(
        encoding="utf-8"))
    admission = json.loads((workspace / "CANDIDATE_ADMISSION.json").read_text(
        encoding="utf-8"))
    if (admission.get("candidate_response_accessed") is not False
            or admission.get("heldout_opened") is not False):
        raise ValueError("skill evidence source crossed response boundary")
    certificates = admission["variants"]["full"]["candidate_certificates"]
    grouped = {}
    for row in certificates:
        family = str(row.get("family", ""))
        if family.startswith("engine:"):
            grouped.setdefault(family.removeprefix("engine:"), []).append(row)
    task_identity = _digest({
        "source_output": str(root), "dataset": dataset, "seed": seed,
        "system_contract": contract})
    evidence = []
    for skill, rows in sorted(grouped.items()):
        gains, tolerances = _representative(rows)
        evidence.append(SkillTaskEvidence(
            task_identity, str(manifest["family"]), skill,
            gains, tolerances))
    artifacts = {
        contract_name: sha256(
            (root / contract_name).read_bytes()).hexdigest(),
        f"{dataset}/{seed}/DATA_MANIFEST.json": sha256(
            (workspace / "DATA_MANIFEST.json").read_bytes()).hexdigest(),
        f"{dataset}/{seed}/CANDIDATE_ADMISSION.json": sha256(
            (workspace / "CANDIDATE_ADMISSION.json").read_bytes()).hexdigest()}
    return tuple(evidence), {
        "source_output": str(root), "dataset": dataset, "seed": seed,
        "dataset_family": manifest["family"],
        "task_identity": task_identity, "artifacts": artifacts,
        "registered_engines": list(config.get("agent", {}).get(
            "engines", registered_engine_names()))}


__all__ = ["extract_skill_task_evidence"]
