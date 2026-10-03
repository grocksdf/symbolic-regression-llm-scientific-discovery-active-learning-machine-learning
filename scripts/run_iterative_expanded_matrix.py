"""Run the registered iterative Full / frozen-engine bank predictive matrix.

An entirely new output root is required. Every generation artifact is frozen
before *any* train-reporting response is decoded. No retry or resume is allowed.
This never executes measured acquisition, ground truth, test or OOD.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.iterative_matrix_contract import (  # noqa: E402
    CONDITION, SCHEMA, digest, read, source_identity, validate_config,
)


def verify_freeze(freeze: dict) -> None:
    if (freeze.get("schema") != SCHEMA
            or freeze.get("condition") != CONDITION
            or freeze.get("execution_authorized") is not True
            or freeze.get("candidate_response_accessed") is not False
            or freeze.get("heldout_opened") is not False
            or freeze.get("test_or_ood_accessed") is not False):
        raise ValueError("not a registered iterative predictive freeze")
    contract = freeze.get("bank_contract") or {}
    if (contract.get("engine_jobs_shared_by_construction") is not True
            or contract.get("matched_non_llm_skeleton_attempts") is not False
            or contract.get("formula_recovery_attribution_authorized") is not False):
        raise ValueError("iterative matrix search attribution contract changed")
    main_root = Path(freeze["mainline_root"]).resolve(strict=True)
    bench_source, main_source = source_identity(ROOT, main_root)
    if (bench_source != freeze["benchmark_source"]
            or main_source != freeze["mainline_source"]):
        raise ValueError("frozen mainline or benchmark source changed")
    config_path = ROOT / "configs/aistats_three_arm_formula_expanded_candidate.yaml"
    config = validate_config(config_path)
    if digest(config_path) != freeze["config_sha256"]:
        raise ValueError("registered config bytes changed")
    agent = config["agent_config"]
    for key, value in freeze["engine_schedule"].items():
        if agent[key] != value:
            raise ValueError("engine schedule changed")
    for key, value in freeze["candidate_search_budget"].items():
        if agent[key] != value:
            raise ValueError("candidate budget changed")
    for path, expected in {**freeze["files"],
                           **freeze["exclusion_freezes"]}.items():
        if digest(path) != expected:
            raise ValueError(f"frozen file changed: {path}")
    if (digest(freeze["hdf5"]["path"]) != freeze["hdf5"]["sha256"]
            or Path(freeze["hdf5"]["path"]).stat().st_size
            != freeze["hdf5"]["size_bytes"]):
        raise ValueError("registered HDF5 bytes changed")
    for row in freeze["development_metadata_files"].values():
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("frozen input metadata changed")
    if not Path(freeze["provider_env"]).is_file():
        raise ValueError("registered provider env is unavailable")
    if digest(freeze["provider_env"]) != freeze["provider_env_sha256"]:
        raise ValueError("registered provider credential identity changed")
    if (str(Path(sys.executable).resolve()) != freeze["interpreter"]
            or sys.version != freeze["python_version"]):
        raise ValueError("registered Python interpreter changed")
    for name, expected in freeze["dependency_versions"].items():
        actual = str(getattr(importlib.import_module(name),
                             "__version__", "present"))
        if actual != expected:
            raise ValueError(f"registered dependency changed: {name}")
    gate = read(freeze["gates"]["provider"])
    if (gate.get("classification") != "transport_2xx"
            or gate.get("http_status") != 200
            or gate.get("openai_completion_contract_valid") is not True
            or gate.get("base_url") != config["llm_api_url"].rstrip("/")
            or gate.get("model") != config["llm_model"]
            or gate.get("provider_env_path") != freeze["provider_env"]):
        raise ValueError("registered provider transport is unhealthy")
    tasks = freeze["selected_tasks"]
    seeds = freeze["seeds"]
    if (not isinstance(tasks, dict) or not tasks
            or not isinstance(seeds, list) or len(set(seeds)) != len(seeds)
            or sum(map(len, tasks.values())) * len(seeds)
            != freeze["pair_count"]):
        raise ValueError("registered matrix dimensions changed")
    for family, task_names in tasks.items():
        if not task_names or len(task_names) != len(set(task_names)):
            raise ValueError(f"duplicate or empty task identities: {family}")
        for task in task_names:
            if (task in freeze["excluded_tasks"][family]
                    or not freeze["input_symbols_by_task"].get(
                        f"{family}/{task}")):
                raise ValueError("task overlaps exclusion or has no input mapping")


def _plan(freeze, root):
    return [{"family": family, "task": task, "seed": seed,
             "run_dir": str(root / "children" / family / task / str(seed))}
            for family, tasks in sorted(freeze["selected_tasks"].items())
            for task in tasks for seed in freeze["seeds"]]


def _child_command(freeze, item):
    family, task = item["family"], item["task"]
    return [sys.executable, str(ROOT / "scripts/run_aistats_three_arm_formula_child.py"),
            "--family", family, "--task", task, "--seed", str(item["seed"]),
            "--condition", CONDITION,
            "--config", str(ROOT / "configs/aistats_three_arm_formula_expanded_candidate.yaml"),
            "--hdf5", freeze["hdf5"]["path"],
            "--provider-env", freeze["provider_env"],
            "--metadata-parquet", freeze["development_metadata_files"][family]["path"],
            "--input-symbols-json", json.dumps(
                freeze["input_symbols_by_task"][f"{family}/{task}"]),
            "--output-dir", item["run_dir"]]


def _run(command, env, timeout):
    return subprocess.run(command, cwd=ROOT, env=env, timeout=timeout,
                          capture_output=True, text=True, check=False)


def _check_child(item):
    path = Path(item["run_dir"])
    child = read(path / "THREE_ARM_CHILD_RESULT.json")
    artifact = path / "pcpi_artifacts" / f"{item['task']}.json"
    if (child.get("status") != "success"
            or child.get("condition") != CONDITION
            or any(child.get(k) != item[k] for k in ("family", "task", "seed"))
            or Path(child.get("artifact", "")).resolve() != artifact.resolve()
            or child.get("result_count") != 1
            or child.get("ground_truth_expression_opened") is not False
            or child.get("test_or_ood_accessed") is not False
            or child.get("heldout_opened") is not False
            or "gap_admission" not in child.get("decoded_response_roles", ())
            or not isinstance(child.get("provider_cost"), list)
            or any(not 200 <= call.get("http_status", 0) < 300
                   for call in child["provider_cost"])):
        raise ValueError("child violated iterative transport or role contract")
    frozen = read(artifact)
    report = frozen.get("scientific_discovery_runtime") or {}
    evaluation = report.get("scientist_agent_system_evaluation") or {}
    if (frozen.get("task_name") != item["task"]
            or report.get("drr_condition") != CONDITION
            or not report.get("paired_bank_predictive_report_available")
            or len(evaluation.get("iterative_feedback_trace", ())) != 2
            or len(report.get("iterative_cycle_role_row_indices", ())) != 2
            or report.get("heldout_opened") is not False
            or report.get("test_or_ood_accessed") is not False
            or report.get("measured_pair_authorized") is not False):
        raise ValueError("generation artifact has no frozen iterative trace")
    return {"child_sha256": digest(path / "THREE_ARM_CHILD_RESULT.json"),
            "artifact_sha256": digest(artifact),
            "provider_requests": len(child["provider_cost"]),
            "admitted_feedback_gate": bool((evaluation.get(
                "iterative_feedback_gate") or {}).get("passed"))}


def _check_report(item, output, child_identity):
    result = read(output)
    cycles = result.get("cycles") or ()
    if (result.get("schema") != "iterative-common-report-bank-contrast-v1"
            or len(cycles) != 2
            or result.get("paired_bank_only") is not True
            or result.get("measured_action_authorized") is not False
            or result.get("reporting_excluded_from_generation_and_admission") is not True
            or result.get("iterative_feedback_witnessed")
            != child_identity["admitted_feedback_gate"]
            or cycles[0]["no_llm_bank"] != cycles[1]["no_llm_bank"]
            or cycles[0]["no_llm_log_score"] != cycles[1]["no_llm_log_score"]):
        raise ValueError("common-report score or bank identity changed")
    if any(not isinstance(cycle.get(key), (int, float))
           or not math.isfinite(cycle[key])
           for cycle in cycles for key in (
               "full_mse", "no_llm_mse", "full_log_score",
               "no_llm_log_score", "no_llm_minus_full_mse",
               "full_minus_no_llm_log_score")):
        raise ValueError("nonfinite common-report score")
    row = cycles[-1]
    return {"family": item["family"], "task": item["task"],
            "seed": item["seed"],
            "full_bank": row["full_bank"], "no_llm_bank": row["no_llm_bank"],
            "full_minus_no_llm_log_score": row["full_minus_no_llm_log_score"],
            "no_llm_minus_full_mse": row["no_llm_minus_full_mse"],
            "iterative_feedback_witnessed": result["iterative_feedback_witnessed"],
            "reporting_identity": result["reporting_identity"],
            "report_sha256": digest(output), **child_identity}


def summarize(freeze, rows):
    if len(rows) != freeze["pair_count"]:
        raise ValueError("missing registered rows")
    task_means = {}
    for family, tasks in freeze["selected_tasks"].items():
        for task in tasks:
            values = [row["full_minus_no_llm_log_score"] for row in rows
                      if row["family"] == family and row["task"] == task]
            if len(values) != len(freeze["seeds"]):
                raise ValueError("task has missing seeds")
            task_means[f"{family}/{task}"] = sum(values) / len(values)
    overall = sum(task_means.values()) / len(task_means)
    witnessed = sum(row["iterative_feedback_witnessed"] for row in rows)
    return {"schema": "iterative-expanded-bank-matrix-result-v1",
            "pair_count": len(rows), "task_means": task_means,
            "task_mean_log_score_gain": overall,
            "verified_feedback_pairs": witnessed,
            "passed": overall > 0 and witnessed > 0,
            "rows": rows, "paired_bank_only": True,
            "matched_non_llm_skeleton_attempts": False,
            "formula_recovery_attribution_authorized": False,
            "measured_action_authorized": False,
            "confirmation_responses_opened": False,
            "test_or_ood_accessed": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output_root.exists():
        raise ValueError("matrix output root already exists (resume forbidden)")
    freeze_path = a.freeze.resolve(strict=True)
    freeze = read(freeze_path)
    verify_freeze(freeze)
    a.output_root.mkdir(parents=True, exist_ok=False)
    plan = _plan(freeze, a.output_root.resolve())
    (a.output_root / "PLAN.json").write_text(json.dumps({
        "freeze_sha256": digest(freeze_path), "items": plan,
        "reporting_responses_opened": False}, indent=2, sort_keys=True) + "\n")
    env = dict(os.environ)
    env["HYPOTHESIS_MVP_ROOT"] = freeze["mainline_root"]
    env["PYTHONPATH"] = os.pathsep.join((
        freeze["mainline_root"], str(ROOT), env.get("PYTHONPATH", "")))
    env["FORMULA_PROVIDER_LOCK"] = str(a.output_root.resolve() / "provider.lock")
    env["THREE_ARM_PROMPT_BYTES"] = "65536"  # upper bound only, no padding
    identities = []
    try:
        for item in plan:
            path = Path(item["run_dir"])
            if path.exists():
                raise ValueError("child output path already exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            process = _run(_child_command(freeze, item), env, 2400)
            if process.returncode:
                raise RuntimeError(f"child failed: {item['family']}/{item['task']} "
                                   f"seed={item['seed']} exit={process.returncode}")
            identities.append(_check_child(item))
        (a.output_root / "GENERATION_FROZEN.json").write_text(json.dumps({
            "freeze_sha256": digest(freeze_path), "items": identities,
            "all_generation_artifacts_frozen_before_reporting": True},
            indent=2, sort_keys=True) + "\n")
        rows = []
        for item, identity in zip(plan, identities, strict=True):
            artifact = Path(item["run_dir"]) / "pcpi_artifacts" / f"{item['task']}.json"
            output = a.output_root / "reports" / item["family"] / item["task"] / (
                str(item["seed"]) + ".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / "scripts/audit_iterative_bank_reporting.py"),
                   "--artifact", str(artifact), "--hdf5", freeze["hdf5"]["path"],
                   "--family", item["family"], "--task", item["task"],
                   "--seed", str(item["seed"]), "--output", str(output)]
            process = _run(cmd, env, 600)
            if process.returncode:
                raise RuntimeError(f"report failed: {item['family']}/{item['task']} "
                                   f"seed={item['seed']} exit={process.returncode}")
            rows.append(_check_report(item, output, identity))
        result = summarize(freeze, rows)
        result["freeze_sha256"] = digest(freeze_path)
        (a.output_root / "MATRIX_RESULT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except Exception as exc:
        (a.output_root / "MATRIX_FAILURE.json").write_text(json.dumps({
            "schema": "iterative-expanded-bank-matrix-failure-v1",
            "error_type": type(exc).__name__, "reason": str(exc),
            "generation_frozen": (a.output_root / "GENERATION_FROZEN.json").is_file(),
            "resume_authorized": False}, indent=2, sort_keys=True) + "\n")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
