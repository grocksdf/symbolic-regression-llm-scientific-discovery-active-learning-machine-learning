"""Frozen AISTATS DRR benchmark runner with failure-inclusive paired inference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_MAP = {
    "lsr_transform": "lsrtransform",
    "bio_pop_growth": "bio_pop_growth",
    "chem_react": "chem_react",
    "matsci": "matsci",
}
CONFIGS = {
    "full_scientist_v6": "configs/aistats_drr_full_v6.yaml",
    "no_llm_v6": "configs/aistats_drr_no_llm_v6.yaml",
    "full_llmchannel_v7": "configs/aistats_drr_full_llmchannel_v7.yaml",
    "three_arm_e_v1": "configs/aistats_three_arm_e_v1.yaml",
    "three_arm_l_blind_v1": "configs/aistats_three_arm_l_blind_v1.yaml",
    "three_arm_l_gap_v1": "configs/aistats_three_arm_l_gap_v1.yaml",
    "single_engine_v6": "configs/aistats_drr_single_engine_v6.yaml",
}


@dataclass(frozen=True)
class RunSpec:
    family: str
    dataset: str
    task: str
    seed: int
    condition: str
    run_dir: Path


def _sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provider_key(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith("OPENAI_API_KEY="):
            value = text.split("=", 1)[1].strip()
            if value:
                return value
    raise ValueError("registered provider env has no OPENAI_API_KEY")


def _lsr_hdf5_identity(root):
    root = Path(root).resolve()
    candidates = (
        root / "lsr_bench_data.hdf5",
        root / "data" / "lsr_bench_data.hdf5",
    )
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            "exactly one registered lsr_bench_data.hdf5 is required")
    path = present[0]
    return {
        "path": str(path),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _load_registration(path):
    registration = json.loads(Path(path).read_text(encoding="utf-8"))
    if (registration.get("schema") !=
            "scientific-aistats-drr-execution-registration-v1"):
        raise ValueError("invalid AISTATS DRR execution registration")
    for name, expected in registration["files"].items():
        if _sha(name) != expected:
            raise ValueError("AISTATS DRR registered file changed")
    mainline_root = Path(registration["mainline_root"]).resolve()
    parent = str(mainline_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from hypothesis_mvp.hypotheses.source_identity import (
        verify_clean_git_source,
    )
    if verify_clean_git_source(ROOT) != registration["benchmark_source"]:
        raise ValueError("AISTATS DRR benchmark source identity changed")
    if verify_clean_git_source(mainline_root) != registration["mainline_source"]:
        raise ValueError("AISTATS DRR mainline source identity changed")
    gate = json.loads(Path(
        registration["adapter_gate"]).read_text(encoding="utf-8"))
    if (gate.get("passed") is not True
            or gate.get("benchmark_source") != registration["benchmark_source"]
            or gate.get("mainline_source") != registration["mainline_source"]):
        raise ValueError("AISTATS DRR adapter correctness Gate is invalid")
    resource = json.loads(Path(
        registration["resource_gate"]).read_text(encoding="utf-8"))
    if (resource.get("passed") is not True
            or resource.get("benchmark_source") !=
                registration["benchmark_source"]
            or resource.get("mainline_source") !=
                registration["mainline_source"]
            or resource.get("provider", {}).get("public_identity") !=
                registration["provider_public_identity"]):
        raise ValueError("AISTATS DRR provider/resource Gate is invalid")
    expected_data = resource.get(
        "benchmark_data_preflight", {}).get("lsr_transform_hdf5")
    actual_data = _lsr_hdf5_identity(
        registration["benchmark_data_root"])
    if expected_data != actual_data:
        raise ValueError("AISTATS DRR benchmark data identity changed")
    live = json.loads(Path(
        registration["provider_live_smoke"]).read_text(encoding="utf-8"))
    if (live.get("passed") is not True
            or live.get("provider_public_identity") !=
                registration["provider_public_identity"]
            or live.get("model_requested") != "glm-5.3"
            or live.get("scientific_prompt_sent") is not False
            or live.get("benchmark_task_arrays_accessed") is not False):
        raise ValueError("AISTATS DRR provider live smoke is invalid")
    runtime = json.loads(Path(
        registration["runtime_gate"]).read_text(encoding="utf-8"))
    if (runtime.get("passed") is not True
            or runtime.get("runtime_public_identity") !=
                registration["runtime_public_identity"]
            or Path(runtime.get("executable", "")).resolve() !=
                Path(sys.executable).resolve()):
        raise ValueError("AISTATS DRR dedicated runtime identity changed")
    smoke = json.loads(Path(
        registration["end_to_end_smoke_gate"]).read_text(encoding="utf-8"))
    if (smoke.get("passed") is not True
            or smoke.get("benchmark_source") !=
                registration["benchmark_source"]
            or smoke.get("mainline_source") !=
                registration["mainline_source"]
            or smoke.get("fixture_task") != "BPG2"):
        raise ValueError("AISTATS DRR end-to-end smoke Gate is invalid")
    if not Path(registration["provider_env"]).is_file():
        raise ValueError("AISTATS DRR provider env is unavailable")
    return registration


def _build_specs(registration, output):
    freeze = json.loads(Path(
        registration["benchmark_freeze"]).read_text(encoding="utf-8"))
    protocol = json.loads(Path(
        registration["protocol"]).read_text(encoding="utf-8"))
    specs = []
    for family, tasks in freeze["selected_tasks"].items():
        for task in tasks:
            for seed in protocol["seeds"]:
                for condition in (
                        "full_scientist_v6", "no_llm_v6",
                        "single_engine_v6"):
                    specs.append(RunSpec(
                        family, DATASET_MAP[family], task, int(seed), condition,
                        output / "runs" / family / task / f"seed{seed}" /
                        condition))
    if len(specs) != 270:
        raise ValueError("AISTATS DRR plan must contain exactly 270 child runs")
    return specs, protocol, freeze


def _command(spec, registration, protocol):
    return [
        sys.executable, str(ROOT / "eval.py"),
        "--dataset", spec.dataset,
        "--searcher_config", str(ROOT / CONFIGS[spec.condition]),
        "--problem_name", spec.task,
        "--ds_root_folder", registration["benchmark_data_root"],
        "--resume_from", str(spec.run_dir),
        "--seed", str(spec.seed),
        "--budget", str(protocol["budget"][
            "evaluated_hypotheses_per_condition_task_seed"]),
    ]


def _artifact(spec):
    path = spec.run_dir / "pcpi_artifacts" / f"{spec.task}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = payload["scientific_discovery_runtime"]
    drr = report["drr_readiness"]
    return report, drr


def _condition_row(spec, condition, report, drr, returncode, failure):
    indicator = int(drr.get("indicator", 0)) if not failure else 0
    ready = bool(drr.get("ready", False)) if not failure else False
    return {
        "family": spec.family, "dataset": spec.dataset, "task": spec.task,
        "seed": spec.seed, "condition": condition,
        "indicator": indicator, "ready": ready,
        "status": "success" if returncode == 0 and not failure else "failure",
        "failure": failure, "returncode": returncode,
        "run_dir": str(spec.run_dir),
        "certificate": dict(drr.get("certificate", {})),
        "logical_llm_call_count": int(
            report.get("scientist_agent_logical_llm_call_count") or 0),
        "provider_attempt_count": int(
            report.get("scientist_agent_provider_attempt_count") or 0),
    }


def _run_one(spec, registration, protocol):
    spec.run_dir.mkdir(parents=True)
    cache_root = Path(registration["benchmark_cache_root"]).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": str(spec.seed),
        "HYPOTHESIS_MVP_ROOT": registration["mainline_root"],
        "PCPI_DRR_CONDITION": spec.condition,
        "PCPI_DISABLE_HELDOUT_PROMPT_DATA": "1",
        "PCPI_STRICT_TRAIN_VAL_ONLY": "1",
        "OPENAI_API_KEY": _provider_key(registration["provider_env"]),
        "HF_HOME": str(cache_root / "huggingface"),
        "HF_DATASETS_CACHE": str(cache_root / "datasets"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
    })
    command = _command(spec, registration, protocol)
    try:
        process = subprocess.run(
            command, cwd=ROOT, env=env, check=False,
            timeout=float(protocol["budget"][
                "wall_time_seconds_per_condition_task_seed"]),
            stdout=(spec.run_dir / "stdout.log").open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT)
        returncode, failure = process.returncode, ""
    except subprocess.TimeoutExpired:
        returncode, failure = 124, "wall-time-timeout"
    report, drr = {}, {}
    if returncode == 0:
        try:
            report, drr = _artifact(spec)
            if int(report.get(
                    "scientist_agent_logical_llm_call_count") or 0) > int(
                    protocol["budget"][
                        "maximum_llm_calls_per_condition_task_seed"]):
                failure = "llm-call-budget-exceeded"
            attempt_limit = protocol["budget"].get(
                "maximum_provider_attempts_per_condition_task_seed")
            if (attempt_limit is not None
                    and int(report.get(
                        "scientist_agent_provider_attempt_count") or 0)
                    > int(attempt_limit)):
                failure = "provider-attempt-budget-exceeded"
        except Exception as exc:
            failure = f"artifact-invalid:{type(exc).__name__}"
    elif not failure:
        failure = f"child-returncode-{returncode}"
    rows = [_condition_row(
        spec, spec.condition, report, drr, returncode, failure)]
    if spec.condition == "full_scientist_v6":
        alternate = report.get("drr_alternate_readiness") or {}
        alternate_failure = failure
        if returncode == 0 and not alternate:
            alternate_failure = "alternate-v5-certificate-missing"
        rows.append(_condition_row(
            spec, "entropy_portfolio_v5", report, alternate,
            returncode, alternate_failure))
    (spec.run_dir / "DRR_RUN_RESULT.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return rows


def _resume_rows(spec):
    path = spec.run_dir / "DRR_RUN_RESULT.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        {"full_scientist_v6", "entropy_portfolio_v5"}
        if spec.condition == "full_scientist_v6"
        else {spec.condition})
    if ({row.get("condition") for row in rows} != expected
            or any(row.get("task") != spec.task
                   or row.get("seed") != spec.seed for row in rows)):
        raise ValueError("DRR resume ledger identity changed")
    return rows


def _write_csv(path, rows):
    fields = [
        "family", "dataset", "task", "seed", "condition", "indicator",
        "ready", "status", "failure", "returncode", "run_dir",
        "logical_llm_call_count", "provider_attempt_count"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _primary_analysis(rows, protocol):
    by_key = {(row["task"], row["seed"], row["condition"]): row
              for row in rows}
    task_effects = []
    tasks = sorted({row["task"] for row in rows})
    for task in tasks:
        full = np.mean([
            by_key[(task, seed, "full_scientist_v6")]["indicator"]
            for seed in protocol["seeds"]])
        no_llm = np.mean([
            by_key[(task, seed, "no_llm_v6")]["indicator"]
            for seed in protocol["seeds"]])
        task_effects.append(float(full - no_llm))
    effects = np.asarray(task_effects, dtype=float)
    rng = np.random.default_rng(
        protocol["primary_metric"]["uncertainty"]["seed"])
    draws = np.asarray([
        np.mean(rng.choice(effects, size=len(effects), replace=True))
        for _ in range(protocol["primary_metric"]["uncertainty"]["resamples"])])
    lower, upper = np.quantile(draws, [.025, .975])
    mean = float(np.mean(effects))
    return {
        "schema": "scientific-aistats-drr-primary-analysis-v1",
        "task_count": len(effects), "paired_mean_drr_difference": mean,
        "paired_bootstrap_interval_95": [float(lower), float(upper)],
        "minimum_effect": 0.10,
        "passed": bool(mean >= .10 and lower > 0.0),
        "all_tasks_included": len(effects) == 30,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    registration = _load_registration(args.registration)
    if args.output_dir.exists() and not args.resume:
        raise ValueError("AISTATS DRR output must be new")
    specs, protocol, freeze = _build_specs(registration, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    plan = [{
        "family": row.family, "dataset": row.dataset, "task": row.task,
        "seed": row.seed, "condition": row.condition,
        "run_dir": str(row.run_dir),
        "command": _command(row, registration, protocol)}
        for row in specs]
    plan_path = args.output_dir / "PLAN.json"
    plan_text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if plan_path.exists():
        if plan_path.read_text(encoding="utf-8") != plan_text:
            raise ValueError("AISTATS DRR resume plan identity changed")
    else:
        plan_path.write_text(plan_text, encoding="utf-8")
    if args.preflight_only:
        preflight = {
            "schema": "scientific-aistats-drr-preflight-v1",
            "passed": True, "planned_child_runs": len(plan),
            "planned_condition_results": 360,
            "selected_task_identity": freeze["selected_task_identity"],
            "task_arrays_accessed": False, "llm_called": False,
            "execution_started": False,
            "execution_authorized": registration[
                "execution_authorized"],
            "benchmark_source": registration["benchmark_source"],
            "mainline_source": registration["mainline_source"],
            "benchmark_data_preflight": {
                "lsr_transform_hdf5": _lsr_hdf5_identity(
                    registration["benchmark_data_root"]),
                "task_arrays_accessed": False,
            },
        }
        (args.output_dir / "AISTATS_DRR_PREFLIGHT.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if registration.get("execution_authorized") is not True:
        raise ValueError("AISTATS DRR execution is not authorized")
    rows = []
    for index, spec in enumerate(specs, 1):
        resumed = _resume_rows(spec) if args.resume else None
        if resumed is not None:
            print(f"[{index}/{len(specs)}] resume-preserved "
                  f"{spec.condition} {spec.dataset}/{spec.task} "
                  f"seed={spec.seed}", flush=True)
            rows.extend(resumed)
            _write_csv(args.output_dir / "TASK_SEED_RESULTS.csv", rows)
            continue
        print(f"[{index}/{len(specs)}] {spec.condition} "
              f"{spec.dataset}/{spec.task} seed={spec.seed}", flush=True)
        rows.extend(_run_one(spec, registration, protocol))
        _write_csv(args.output_dir / "TASK_SEED_RESULTS.csv", rows)
    analysis = _primary_analysis(rows, protocol)
    result = {
        "schema": "scientific-aistats-drr-result-v1",
        "protocol_complete": len(rows) == 360,
        "primary_analysis": analysis,
        "failure_count": sum(row["status"] != "success" for row in rows),
        "heldout_used_for_selection": False,
        "test_or_ood_used_for_prompt_gate_tuning": False,
        "claim_boundary": (
            "Frozen primary DRR analysis; ID/OOD metrics are secondary and "
            "cannot alter task inclusion or the primary decision."),
    }
    (args.output_dir / "AISTATS_DRR_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
