"""Freeze the provider-free JEGPE frozen-pool development diagnostic."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAINLINE = ROOT.parent / "hypothesis_mvp"


def digest(path):
    value = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("JEGPE Phase-1 freeze exists")
    source_result = json.loads(
        (args.source_output / "RESULT.json").read_text())
    coordinates = sorted({
        (str(row["family"]), str(row["task"]), int(row["seed"]))
        for row in source_result["rows"]})
    if len(coordinates) != 106:
        raise ValueError("JEGPE Phase-1 requires 106 frozen coordinates")
    artifacts, registered = [], []
    for family, task, seed in coordinates:
        root = (args.source_output / "generation" / family / task /
                f"seed{seed}")
        failure = root / "COORDINATE_FAILURE.json"
        if failure.exists():
            artifacts.append({
                "path": str(failure.resolve()), "sha256": digest(failure)})
            registered.append({
                "family": family, "task": task, "seed": seed,
                "coordinate_failure": True})
            continue
        paths = [root / "SHARED_BLOCK.json"]
        for condition in ("B0", "G_hard0", "G_placebo0"):
            paths.append(root / condition / "GENERATION_RESULT.json")
        for path in paths:
            if not path.exists():
                raise ValueError(f"missing frozen JEGPE artifact: {path}")
            artifacts.append({
                "path": str(path.resolve()), "sha256": digest(path)})
        registered.append({
            "family": family, "task": task, "seed": seed,
            "coordinate_failure": False,
            "coordinate_root": str(root.resolve())})
    runner = ROOT / "scripts" / "run_jegpe_frozen_pool_diagnostic.py"
    module = (
        MAINLINE / "hypothesis_mvp" / "discovery" /
        "joint_evidence_gated_expansion.py")
    freeze = {
        "schema": "jegpe-frozen-pool-development-freeze-v1",
        "execution_authorized": True,
        "claim_boundary": (
            "Provider-free post-reporting development analysis over the "
            "historical 53-task frozen candidate pools. It may diagnose "
            "joint complementarity and compare fixed methods, but is not "
            "independent confirmation or an anytime-valid guarantee."),
        "source_output": str(args.source_output.resolve()),
        "source_result_sha256":
            digest(args.source_output / "RESULT.json"),
        "source_generation_freeze_sha256":
            digest(args.source_output / "GENERATION_FROZEN.json"),
        "hdf5": str(args.hdf5.resolve()),
        "hdf5_sha256": digest(args.hdf5),
        "coordinates": registered,
        "coordinate_count": len(registered),
        "candidate_conditions": ["B0", "G_hard0", "G_placebo0"],
        "arms": [
            "original_protected", "staged_protected",
            "jegpe_joint", "predictive_stacking",
            "accept_all", "same_data"],
        "bundle_selection": {
            "role": "decision_calibration",
            "maximum_size": 2,
            "evaluation_budget": 6,
            "objective": "full-bank mean predictive log-score gain",
            "tie_break": (
                "smaller bundle, lower support complexity, lower support "
                "Jaccard redundancy, stable identity"),
            "reporting_response_accessed": False},
        "joint_admission": {
            "role": "gap_admission",
            "alpha": 0.025,
            "tested_bundle_count": 1,
            "global_threshold": "log(M/alpha)",
            "region_rule": "nonnegative cumulative evidence in every region",
            "fixed_batch_only": True,
            "anytime_valid_claimed": False},
        "posterior_update_role": "decision_selector_update",
        "reporting_role": "reporting",
        "analysis": {
            "unit": "task", "seed_aggregation": "mean",
            "task_weighting": "equal",
            "bootstrap_seed": 20261030,
            "resamples": 10000},
        "artifact_manifest": artifacts,
        "file_hashes": {
            str(runner.resolve()): digest(runner),
            str(module.resolve()): digest(module),
            str((ROOT / "scripts/run_gap_candidate_marginal_value.py"
                 ).resolve()): digest(
                     ROOT / "scripts/run_gap_candidate_marginal_value.py"),
            str((ROOT / "scripts/run_protected_admission_ablation.py"
                 ).resolve()): digest(
                     ROOT / "scripts/run_protected_admission_ablation.py")},
        "provider_calls_authorized": False,
        "result_dependent_changes_authorized": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "coordinate_count": len(registered),
        "artifact_count": len(artifacts),
        "arms": freeze["arms"],
        "provider_calls_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
