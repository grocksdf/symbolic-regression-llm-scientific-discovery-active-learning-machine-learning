"""Run the frozen fresh-task synthesis-only screening pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_aistats_drr_benchmark import (
    DATASET_MAP, RunSpec, _artifact, _run_one, _sha,
)
from scripts.run_aistats_synthesis_only_smoke_gate import (
    _engine_identity, _synthesis_path_is_audited,
)


def _verify(freeze):
    if freeze.get("schema") != (
            "scientific-aistats-synthesis-only-pilot-freeze-v1"):
        raise ValueError("invalid synthesis-only pilot freeze")
    for path, expected in freeze["files"].items():
        if _sha(path) != expected:
            raise ValueError(f"synthesis-only frozen file changed: {path}")
    from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("synthesis-only benchmark source changed")
    if verify_clean_git_source(
            Path(freeze["mainline_root"])) != freeze["mainline_source"]:
        raise ValueError("synthesis-only mainline source changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("synthesis-only pilot output must be new")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    _verify(freeze)
    args.output_dir.mkdir(parents=True)
    protocol = {"budget": {
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
    rows, reports = [], {}
    total, index = freeze["child_run_count"], 0
    for family, task in freeze["selected_tasks"].items():
        for seed in freeze["seeds"]:
            for condition in freeze["conditions"]:
                index += 1
                spec = RunSpec(
                    family, DATASET_MAP[family], task, seed, condition,
                    args.output_dir / "runs" / family / task /
                    f"seed{seed}" / condition)
                print(
                    f"[{index}/{total}] {condition} "
                    f"{family}/{task} seed={seed}", flush=True)
                child = _run_one(spec, registration, protocol)
                main = next(
                    row for row in child if row["condition"] == condition)
                aulc, report, failure = 0.0, {}, main["failure"]
                if main["status"] == "success":
                    try:
                        report, _ = _artifact(spec)
                        aulc = float(
                            report["drr_prefix_curve"]["normalized_aulc"])
                    except Exception as error:
                        failure = f"artifact-invalid:{type(error).__name__}"
                rows.append({
                    "family": family, "task": task, "seed": seed,
                    "condition": condition, "normalized_aulc": aulc,
                    "status": "success" if not failure else "failure",
                    "failure": failure,
                })
                reports[(family, seed, condition)] = report
                (args.output_dir / "SYNTHESIS_RESULTS.json").write_text(
                    json.dumps(rows, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    tolerance = freeze["screening_decision"]["numerical_tolerance"]
    effects, engine_identity, novel = {}, {}, set()
    for family in freeze["selected_tasks"]:
        deltas = []
        for seed in freeze["seeds"]:
            full = reports[(family, seed, "full_scientist_v6")]
            no_llm = reports[(family, seed, "no_llm_v6")]
            same = bool(full and no_llm
                        and _engine_identity(full) == _engine_identity(no_llm))
            engine_identity[f"{family}/{seed}"] = same
            audits = full.get("scientist_agent_synthesis_audits") or []
            if any(audit.get("compiled_candidate_count", 0) > 0
                   for audit in audits):
                novel.add(family)
            full_row = next(row for row in rows if
                row["family"] == family and row["seed"] == seed
                and row["condition"] == "full_scientist_v6")
            no_row = next(row for row in rows if
                row["family"] == family and row["seed"] == seed
                and row["condition"] == "no_llm_v6")
            deltas.append(
                full_row["normalized_aulc"]
                - no_row["normalized_aulc"])
        effects[family] = float(np.mean(deltas))
    mean = float(np.mean(list(effects.values())))
    positive = sum(value > tolerance for value in effects.values())
    decisions = {
        "all_16_child_runs_accounted_for": len(rows) == 16,
        "zero_child_failures": all(
            row["status"] == "success" for row in rows),
        "identical_engine_execution_within_every_pair":
            all(engine_identity.values()),
        "paired_mean_strictly_positive": mean > tolerance,
        "at_least_two_positive_families": positive >= 2,
        "at_least_two_novel_synthesis_families": len(novel) >= 2,
        "no_negative_transfer_family":
            all(value >= -tolerance for value in effects.values()),
    }
    result = {
        "schema": "scientific-aistats-synthesis-only-pilot-result-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "paired_family_effects": effects,
        "paired_mean_full_minus_no_llm": mean,
        "positive_family_count": positive,
        "novel_synthesis_families": sorted(novel),
        "engine_identity": engine_identity,
        "failure_count": sum(row["status"] != "success" for row in rows),
        "rows": rows,
        "candidate_response_accessed": False,
        "action_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Fresh-task synthesis-only development screening; not realized "
            "acquisition efficacy, superiority, or confirmation."),
    }
    path = args.output_dir / "AISTATS_SYNTHESIS_ONLY_PILOT_RESULT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
