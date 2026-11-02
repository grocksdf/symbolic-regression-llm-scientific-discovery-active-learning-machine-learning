"""Freeze the provider-free predictive-stacking gain-attribution audit."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
        raise ValueError("stacking attribution freeze exists")
    source_result = json.loads(
        (args.source_output / "RESULT.json").read_text())
    coordinates = sorted({
        (str(row["family"]), str(row["task"]), int(row["seed"]))
        for row in source_result["rows"]})
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
        paths.extend(
            root / condition / "GENERATION_RESULT.json"
            for condition in ("B0", "G_hard0", "G_placebo0"))
        for path in paths:
            artifacts.append({
                "path": str(path.resolve()), "sha256": digest(path)})
        registered.append({
            "family": family, "task": task, "seed": seed,
            "coordinate_failure": False,
            "coordinate_root": str(root.resolve())})
    runner = ROOT / "scripts" / "run_stacking_gain_attribution.py"
    utility = ROOT / "scripts" / "stacking_diagnostics.py"
    freeze = {
        "schema": "stacking-gain-attribution-freeze-v1",
        "execution_authorized": True,
        "claim_boundary": (
            "Provider-free post-reporting development attribution over the "
            "frozen 53-task candidate pools. It diagnoses candidate coverage, "
            "weight optimization and posterior-state matching. It is not "
            "independent confirmation or a new-method efficacy result."),
        "source_output": str(args.source_output.resolve()),
        "source_result_sha256":
            digest(args.source_output / "RESULT.json"),
        "hdf5": str(args.hdf5.resolve()),
        "hdf5_sha256": digest(args.hdf5),
        "coordinates": registered,
        "coordinate_count": len(registered),
        "candidate_conditions": ["B0", "G_hard0", "G_placebo0"],
        "primary_stacking_state": "matched_updated_state",
        "diagnostic_states": [
            "cross_state", "matched_frozen_state"],
        "arms": [
            "e0", "protected_bma", "full_bma",
            "full_stacking_matched", "full_stacking_cross",
            "full_stacking_frozen", "admitted_only_stacking",
            "rejected_only_stacking", "engine_only_stacking",
            "uniform_full_mixture"],
        "practical_zero_tolerance": 1e-8,
        "optimizer": {
            "method": "SLSQP with deterministic EM fixed-point fallback",
            "random_seed": 20261031,
            "random_starts": 4,
            "simplex_constraint": True,
            "maximum_iterations": 2000,
            "ftol": 1e-12},
        "analysis": {
            "unit": "task", "seed_aggregation": "mean",
            "task_weighting": "equal",
            "bootstrap_seed": 20261031,
            "resamples": 10000,
            "large_task_sensitivity_count": 2},
        "artifact_manifest": artifacts,
        "file_hashes": {
            str(runner.resolve()): digest(runner),
            str(utility.resolve()): digest(utility),
            str((ROOT / "scripts/run_gap_candidate_marginal_value.py"
                 ).resolve()): digest(
                     ROOT / "scripts/run_gap_candidate_marginal_value.py")},
        "provider_calls_authorized": False,
        "result_dependent_changes_authorized": False,
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
        "practical_zero_tolerance": 1e-8,
        "provider_calls_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
