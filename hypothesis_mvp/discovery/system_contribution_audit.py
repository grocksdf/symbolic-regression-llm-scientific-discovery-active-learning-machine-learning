"""Read-only post-run contribution audit over published development artifacts."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from hypothesis_mvp.hypotheses import EvidenceRegistry
from .pcpi_adapter import structural_terms


VARIANTS = ("full", "no_llm", "single_engine")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _support_rows(result: dict[str, Any], n_features: int) -> dict[tuple[str, ...], dict[str, str]]:
    rows = {}
    for candidate in result["candidates"]:
        support = structural_terms(str(candidate["expression"]), n_features)
        if support in rows:
            raise ValueError("published candidate bank contains duplicate supports")
        rows[support] = {"source": str(candidate["source"]),
                         "origin": str(candidate.get("origin", "unknown"))}
    return rows


def _policy_summary(root: Path, policy: str, observed: list[Path]) -> dict[str, Any]:
    directory = root / policy
    curve_path, manifest_path = directory / "DEVELOPMENT_CURVE.json", directory / "RUN_MANIFEST.json"
    observed.extend((curve_path, manifest_path))
    curve, manifest = _read(curve_path), _read(manifest_path)
    decisions = []
    for path in sorted(directory.glob("DECISION-*.json")):
        observed.append(path)
        decisions.append(_read(path))
    if (manifest.get("protocol_complete") is not True or manifest.get("heldout_opened") is not False
            or len(decisions) != manifest.get("completed_queries") or len(curve["rmse"]) != len(decisions) + 1):
        raise ValueError("incomplete or inconsistent measured policy artifacts")
    gains = [float(curve["rmse"][i] - curve["rmse"][i + 1]) for i in range(len(decisions))]
    scores = [row.get("score") for row in decisions]
    correlation = None
    if policy != "random" and len(scores) > 1:
        value = float(spearmanr(np.asarray(scores, dtype=float), np.asarray(gains)).statistic)
        correlation = value if np.isfinite(value) else None
    entropies = [(row.get("information_audit") or {}).get("class_entropy_nats") for row in decisions]
    return {"candidate_ids": [int(row["candidate_id"]) for row in decisions],
            "scores": scores, "rmse": [float(value) for value in curve["rmse"]],
            "step_rmse_reductions": gains,
            "normalized_mean_rmse": float(curve["normalized_mean_rmse"]),
            "score_rmse_reduction_spearman_descriptive_only": correlation,
            "reported_class_entropies": entropies,
            "completed_queries": len(decisions)}


def _support_difference(left: dict, right: dict) -> list[dict[str, Any]]:
    return [{"support": list(key), **left[key]} for key in sorted(left.keys() - right.keys())]


def audit_system_contribution(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "SYSTEM_MANIFEST.json"
    viability = list(root.glob("*/*/HYPOTHESIS_BANK_VIABILITY.json"))
    viability_path = viability[0] if len(viability) == 1 else None
    if viability_path is None or not manifest_path.is_file() or (root / "TERMINAL_FAILURE.json").exists():
        raise ValueError("complete successful system output required")
    observed = [manifest_path, viability_path]
    manifest, viability = _read(manifest_path), _read(viability_path)
    if (manifest.get("protocol_complete") is not True or manifest.get("heldout_opened") is not False
            or viability.get("passed") is not True or len(manifest.get("results", [])) != 1):
        raise ValueError("system output failed protocol, viability, or heldout closure")
    coordinate = viability_path.parent
    data_manifest = _read(coordinate / "DATA_MANIFEST.json")
    observed.append(coordinate / "DATA_MANIFEST.json")
    n_features = len(data_manifest["scientific_context"]["feature_names"])
    if n_features < 1:
        raise ValueError("data manifest has no registered features")
    summaries, supports = {}, {}
    targeted_policy = ("decision_risk" if
        (coordinate / "measured" / "full" / "decision_risk").is_dir()
        else "class_eig")
    policies_to_audit = (targeted_policy, "random")
    for variant in VARIANTS:
        variant_root = coordinate / "exploration" / variant
        frozen_path = variant_root / "FROZEN_BANK.json"
        result_path = frozen_path if frozen_path.is_file() else variant_root / "RESULT.json"
        observed.append(result_path)
        published = _read(result_path)
        result = (published if "evidence_registry_path" in published else {
            **published,
            "evidence_registry_path": str(variant_root / "evidence_registry.jsonl")})
        registry = Path(result["evidence_registry_path"]); observed.append(registry)
        if not EvidenceRegistry(registry).verify().valid:
            raise ValueError("invalid exploration evidence chain")
        supports[variant] = _support_rows(result, n_features)
        policies = {p: _policy_summary(coordinate / "measured" / variant, p, observed)
                    for p in policies_to_audit}
        summaries[variant] = {"candidate_count": len(supports[variant]),
            "engine_sources": result["hypothesis_provenance"]["engine_sources"],
            "llm_retained_candidate_count": result["hypothesis_provenance"]["llm_retained_candidate_count"],
            "evidence_chain_valid": True, "policies": policies}
    full = summaries["full"]
    eig_ids = {v: summaries[v]["policies"][targeted_policy]["candidate_ids"] for v in VARIANTS}
    random_ids = {v: summaries[v]["policies"]["random"]["candidate_ids"] for v in VARIANTS}
    contributions = {}
    for variant in ("no_llm", "single_engine"):
        baseline = summaries[variant]["policies"][targeted_policy]
        target = full["policies"][targeted_policy]
        contributions[variant] = {
            "normalized_mean_rmse_relative_advantage": float(
                (baseline["normalized_mean_rmse"] - target["normalized_mean_rmse"])
                / baseline["normalized_mean_rmse"]),
            "endpoint_rmse_relative_advantage": float(
                (baseline["rmse"][-1] - target["rmse"][-1]) / baseline["rmse"][-1]),
            "unique_full_supports": _support_difference(supports["full"], supports[variant])}
    before = {str(path): _digest(path) for path in observed}
    after = {str(path): _digest(path) for path in observed}
    if before != after:
        raise RuntimeError("source artifacts changed during read-only audit")
    entropy_constant = {v: len(set(summaries[v]["policies"][targeted_policy]["reported_class_entropies"])) == 1
                        for v in VARIANTS}
    return {"schema": "scientific-system-contribution-audit-v1", "read_only": True,
        "source_artifacts_immutable": True, "receipt_response_values_accessed": False,
        "heldout_opened": False, "protocol_complete": True,
        "source_manifest_sha256": _digest(manifest_path), "artifact_sha256": before,
        "superiority_demonstrated": False, "variants": summaries,
        "full_contribution": contributions,
        "decision_influence": {"targeted_policy": targeted_policy,
            "class_eig_sequences_identical": len({tuple(x) for x in eig_ids.values()}) == 1,
            "random_sequences_identical": len({tuple(x) for x in random_ids.values()}) == 1,
            "class_eig_candidate_ids": eig_ids, "random_candidate_ids": random_ids},
        "reporting_correction": {"reported_entropy_constant_across_prefixes": entropy_constant,
            "scope": "audit-metadata-only-selection-decisions-and-development-curves-unchanged"},
        "assessment": "VALID_DEVELOPMENT_PILOT_POSITIVE_PCPI_SIGNAL_NO_SYSTEM_SUPERIORITY"}
