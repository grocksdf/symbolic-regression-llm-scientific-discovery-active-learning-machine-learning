"""Read-only audit of registered utility versus realized development outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rho(left, right):
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = float(spearmanr(np.asarray(left), np.asarray(right)).statistic)
    return value if np.isfinite(value) else None


def _policy(root: Path, policy: str):
    directory = root / policy
    curve = _read(directory / "DEVELOPMENT_CURVE.json")
    decisions = [_read(path) for path in sorted(directory.glob("DECISION-*.json"))]
    if curve.get("heldout_opened") is True or any(
            row.get("heldout_opened") is True for row in decisions):
        raise ValueError("held-out artifact encountered")
    rmse = [float(value) for value in curve["rmse"]]
    reductions = [rmse[i] - rmse[i + 1] for i in range(len(rmse) - 1)]
    scores = [row.get("score") for row in decisions]
    scores = [None if value is None else float(value) for value in scores]
    entropy = [(row.get("information_audit") or {}).get("class_entropy_nats")
               for row in decisions]
    entropy = [None if value is None else float(value) for value in entropy]
    entropy_reductions = ([entropy[i - 1] - entropy[i]
                           for i in range(1, len(entropy))]
                          if entropy and entropy[0] is not None else [])
    score_pairs = [(s, g) for s, g in zip(scores, reductions) if s is not None]
    entropy_pairs = [(s, g) for s, g in zip(entropy_reductions, reductions)
                     if s is not None]
    return {
        "policy": policy,
        "completed_queries": len(decisions),
        "normalized_mean_rmse": float(curve["normalized_mean_rmse"]),
        "rmse": rmse,
        "realized_rmse_reductions": reductions,
        "registered_scores": scores,
        "class_entropy_nats": entropy,
        "class_entropy_reductions": entropy_reductions,
        "score_rmse_reduction_spearman": _rho(*zip(*score_pairs)) if score_pairs else None,
        "entropy_rmse_reduction_spearman": (_rho(*zip(*entropy_pairs))
            if entropy_pairs else None),
    }


def audit(output: str | Path):
    root = Path(output).resolve()
    manifest = _read(root / "SYSTEM_MANIFEST.json")
    if manifest.get("protocol_complete") is not True or manifest.get("heldout_opened") is not False:
        raise ValueError("system manifest is not a completed held-out-closed run")
    coordinate = next(path for path in root.glob("*/*") if path.is_dir())
    variants = {}
    observed = [root / "SYSTEM_MANIFEST.json"]
    for variant in ("full", "no_llm", "single_engine"):
        base = coordinate / "measured" / variant
        variants[variant] = {policy: _policy(base, policy)
                             for policy in ("class_eig", "random")}
        observed.extend((base / policy / "DEVELOPMENT_CURVE.json"
                         for policy in ("class_eig", "random")))
    full = variants["full"]["class_eig"]
    random = variants["full"]["random"]
    return {
        "schema": "scientific-utility-outcome-alignment-audit-v1",
        "read_only": True, "heldout_opened": False,
        "receipt_response_values_accessed": False,
        "claim_boundary": "opened-development utility alignment diagnostic; not efficacy or superiority evidence",
        "output": str(root),
        "source_artifact_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in observed},
        "variants": variants,
        "full_class_eig_vs_random": {
            "normalized_mean_rmse_delta": full["normalized_mean_rmse"] - random["normalized_mean_rmse"],
            "endpoint_rmse_delta": full["rmse"][-1] - random["rmse"][-1],
            "score_alignment": full["score_rmse_reduction_spearman"],
            "entropy_alignment": full["entropy_rmse_reduction_spearman"],
        },
        "superiority_demonstrated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--output", dest="report", type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
