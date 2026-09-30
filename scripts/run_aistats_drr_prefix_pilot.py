"""Run the frozen fresh-task short-prefix DRR screening pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_aistats_drr_benchmark import (
    CONFIGS, DATASET_MAP, RunSpec, _artifact, _command, _provider_key,
    _run_one, _sha,
)


def _verify(freeze):
    if freeze.get("schema") not in {
            "scientific-aistats-drr-prefix-pilot-freeze-v1",
            "scientific-aistats-drr-prefix-pilot-freeze-v2"}:
        raise ValueError("invalid prefix pilot freeze")
    for path, expected in freeze["files"].items():
        if _sha(path) != expected:
            raise ValueError(f"prefix pilot frozen file changed: {path}")
    from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("prefix pilot benchmark source changed")
    if verify_clean_git_source(
            Path(freeze["mainline_root"])) != freeze["mainline_source"]:
        raise ValueError("prefix pilot mainline source changed")
    hdf5 = Path(freeze["hdf5"]["path"])
    if (_sha(hdf5) != freeze["hdf5"]["sha256"]
            or hdf5.stat().st_size != freeze["hdf5"]["size_bytes"]):
        raise ValueError("prefix pilot HDF5 identity changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("prefix pilot output must be new")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    _verify(freeze)
    args.output_dir.mkdir(parents=True)
    protocol = {
        "budget": {
            "evaluated_hypotheses_per_condition_task_seed": 256,
            "wall_time_seconds_per_condition_task_seed": 900,
            "maximum_llm_calls_per_condition_task_seed": 4,
            "maximum_provider_attempts_per_condition_task_seed": 12,
        }}
    registration = {
        "benchmark_data_root": freeze["benchmark_data_root"],
        "benchmark_cache_root": freeze["benchmark_cache_root"],
        "mainline_root": freeze["mainline_root"],
        "provider_env": freeze["provider_env"],
    }
    rows = []
    total = freeze["child_run_count"]
    index = 0
    for family, task in freeze["selected_tasks"].items():
        for seed in freeze["seeds"]:
            for condition in freeze["conditions"]:
                index += 1
                spec = RunSpec(
                    family, DATASET_MAP[family], task, int(seed), condition,
                    args.output_dir / "runs" / family / task /
                    f"seed{seed}" / condition)
                print(
                    f"[{index}/{total}] {condition} "
                    f"{family}/{task} seed={seed}", flush=True)
                child = _run_one(spec, registration, protocol)
                main = next(
                    row for row in child if row["condition"] == condition)
                aulc = 0.0
                error = main["failure"]
                if main["status"] == "success":
                    try:
                        report, _ = _artifact(spec)
                        curve = report["drr_prefix_curve"]
                        if (curve["prefixes"] != [8, 16, 32]
                                or curve["candidate_response_accessed"] is not False
                                or curve["action_response_accessed"] is not False):
                            raise ValueError("invalid prefix curve")
                        aulc = float(curve["normalized_aulc"])
                    except Exception as exc:
                        error = f"prefix-artifact-invalid:{type(exc).__name__}"
                rows.append({
                    "family": family, "task": task, "seed": int(seed),
                    "condition": condition,
                    "normalized_aulc": aulc,
                    "status": (
                        "success" if not error else "failure"),
                    "failure": error,
                })
                (args.output_dir / "PREFIX_RESULTS.json").write_text(
                    json.dumps(rows, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    by = {
        (row["family"], row["seed"], row["condition"]): row
        for row in rows}
    effects = {}
    for family in freeze["selected_tasks"]:
        full = np.mean([
            by[(family, seed, "full_scientist_v6")]["normalized_aulc"]
            for seed in freeze["seeds"]])
        no_llm = np.mean([
            by[(family, seed, "no_llm_v6")]["normalized_aulc"]
            for seed in freeze["seeds"]])
        effects[family] = float(full - no_llm)
    mean = float(np.mean(list(effects.values())))
    positive = sum(value > 0.0 for value in effects.values())
    decisions = {
        "all_24_child_runs_accounted_for": len(rows) == 24,
        "all_four_task_families_included": len(effects) == 4,
        "paired_mean_strictly_positive": mean > 0.0,
        "at_least_three_positive_task_families": positive >= 3,
        "candidate_responses_stayed_closed": True,
        "test_and_ood_stayed_closed": True,
    }
    result = {
        "schema": "scientific-aistats-drr-prefix-pilot-result-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "paired_task_effects": effects,
        "paired_mean_full_minus_no_llm": mean,
        "positive_task_family_count": positive,
        "failure_count": sum(row["status"] != "success" for row in rows),
        "rows": rows,
        "candidate_response_accessed": False,
        "action_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Fresh-task short-prefix development screening only; not realized "
            "acquisition efficacy, superiority, or confirmation."),
    }
    path = args.output_dir / "AISTATS_DRR_PREFIX_PILOT_RESULT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
