"""Freeze a provider-free admission ablation over historical frozen B0 banks."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAINLINE = ROOT.parent / "hypothesis_mvp"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path):
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _source_identity(root: Path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True).strip()
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(git("status", "--short", "--untracked-files=no")),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("admission ablation freeze already exists")
    result_path = args.source_output / "RESULT.json"
    generation_freeze = args.source_output / "GENERATION_FROZEN.json"
    source_result = _read(result_path)
    if (source_result.get("protocol_complete") is not True
            or int(source_result.get("row_count", 0)) != 318):
        raise ValueError("source retrospective result is incomplete")
    if not generation_freeze.exists():
        raise ValueError("source generation freeze is absent")
    coordinates = sorted({
        (str(row["family"]), str(row["task"]), int(row["seed"]))
        for row in source_result["rows"]
        if str(row["condition"]) == "B0"})
    if len(coordinates) != 106:
        raise ValueError("expected 106 frozen B0 coordinates")
    artifacts = []
    coordinate_rows = []
    for family, task, seed in coordinates:
        root = (args.source_output / "generation" / family / task /
                f"seed{seed}")
        failure = root / "COORDINATE_FAILURE.json"
        if failure.exists():
            artifacts.append({
                "path": str(failure.resolve()), "sha256": _digest(failure)})
            coordinate_rows.append({
                "family": family, "task": task, "seed": seed,
                "coordinate_failure": True})
            continue
        shared = root / "SHARED_BLOCK.json"
        generation = root / "B0" / "GENERATION_RESULT.json"
        for path in (shared, generation):
            if not path.exists():
                raise ValueError(f"frozen source artifact missing: {path}")
            artifacts.append({
                "path": str(path.resolve()), "sha256": _digest(path)})
        coordinate_rows.append({
            "family": family, "task": task, "seed": seed,
            "coordinate_failure": False,
            "shared_block": str(shared.resolve()),
            "generation_result": str(generation.resolve())})
    benchmark = _source_identity(ROOT)
    mainline = _source_identity(MAINLINE)
    if benchmark["tracked_dirty"] or mainline["tracked_dirty"]:
        raise ValueError("tracked source worktree is dirty")
    runner = ROOT / "scripts" / "run_protected_admission_ablation.py"
    source_runner = ROOT / "scripts" / "run_gap_candidate_marginal_value.py"
    admission_core = (
        MAINLINE / "hypothesis_mvp" / "discovery" /
        "regional_candidate_admission.py")
    freeze = {
        "schema": "protected-admission-ablation-freeze-v1",
        "execution_authorized": True,
        "claim_boundary": (
            "Provider-free retrospective admission-mechanism ablation over "
            "the exact frozen B0 candidate banks from the 53-task Run-0007 "
            "benchmark. This is not untouched confirmation and does not "
            "attribute candidate generation quality to LLM reasoning."),
        "source_output": str(args.source_output.resolve()),
        "source_result": str(result_path.resolve()),
        "source_result_sha256": _digest(result_path),
        "generation_freeze": str(generation_freeze.resolve()),
        "generation_freeze_sha256": _digest(generation_freeze),
        "hdf5": str(args.hdf5.resolve()),
        "hdf5_sha256": _digest(args.hdf5),
        "coordinates": coordinate_rows,
        "coordinate_count": len(coordinate_rows),
        "task_count": len({row["task"] for row in coordinate_rows}),
        "candidate_condition": "B0",
        "arms": [
            "independent_protected", "same_data", "accept_all"],
        "arm_contract": {
            "same_frozen_candidate_bank": True,
            "same_E0": True,
            "same_posterior_update": True,
            "same_reporting_scorer": True,
            "independent_protected": (
                "gap_admission then selector_update screens"),
            "same_data": "gap_audit reused for both admission screens",
            "accept_all": "all all-domain-defined optional candidates",
            "coordinate_failures_included_as_zero_gain": True},
        "analysis": {
            "unit": "task", "seed_aggregation": "mean",
            "task_weighting": "equal", "bootstrap_seed": 20261030,
            "resamples": 10000, "interval": "95% percentile"},
        "source": {"benchmark": benchmark, "mainline": mainline},
        "interpreter": sys.executable,
        "file_hashes": {
            str(runner.resolve()): _digest(runner),
            str(source_runner.resolve()): _digest(source_runner),
            str(admission_core.resolve()): _digest(admission_core)},
        "artifact_manifest": artifacts,
        "provider_calls_authorized": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "task_count": freeze["task_count"],
        "coordinate_count": freeze["coordinate_count"],
        "artifact_count": len(artifacts),
        "provider_calls_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
