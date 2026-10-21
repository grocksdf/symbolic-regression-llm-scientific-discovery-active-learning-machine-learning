"""Run the frozen A/B/C/D feedback mechanism development factorial."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.feedback_mechanism_factorial_contract import (  # noqa: E402
    ARMS, summarize,
)
from scripts.score_mechanism_arm import BANK_FIELDS, score_arm  # noqa: E402


CONFIGS = {
    "A": "configs/feedback_mechanism_A_no_gap_no_update.yaml",
    "B": "configs/feedback_mechanism_B_gap_generation_no_update.yaml",
    "C": "configs/feedback_mechanism_C_masked_generation_update.yaml",
    "D": "configs/feedback_mechanism_D_full.yaml",
}
CHILD = ROOT / "scripts/run_aistats_three_arm_formula_child.py"
RESULT = "THREE_ARM_CHILD_RESULT.json"


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8").stdout.strip()


def _verify_authorization(freeze_path, freeze, authorization_path):
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8"))
    if (authorization.get("schema")
            != "feedback-mechanism-factorial-execution-authorization-v1"
            or authorization.get("execution_authorized") is not True
            or authorization.get("optimization_authorized") is not False
            or authorization.get("freeze_sha256") != _digest(freeze_path)
            or authorization.get("benchmark_commit")
            != _git("rev-parse", "HEAD")
            or authorization.get("benchmark_tree")
            != _git("rev-parse", "HEAD^{tree}")
            or _git("status", "--porcelain")
            or authorization.get("runner_sha256") != _digest(Path(__file__))
            or authorization.get("provider_env_sha256")
            != _digest(Path(freeze["provider_env"]["path"]))):
        raise ValueError("factorial execution authorization identity failed")
    provider_gate = Path(authorization["provider_gate"]["path"])
    gate = json.loads(provider_gate.read_text(encoding="utf-8"))
    if (_digest(provider_gate)
            != authorization["provider_gate"]["sha256"]
            or gate.get("schema")
            != "response-free-provider-transport-preflight-v2"
            or gate.get("classification") != "transport_2xx"
            or int(gate.get("http_status", 0)) != 200
            or gate.get("openai_completion_contract_valid") is not True
            or gate.get("strict_json_content_valid") is not True
            or gate.get("scientific_data_accessed") is not False
            or gate.get("base_url")
            != authorization["provider_gate"]["base_url"]
            or gate.get("model") != authorization["provider_gate"]["model"]):
        raise ValueError("provider transport authorization gate failed")
    return authorization


def _common_bank_field(paths):
    runtimes = [
        json.loads(path.read_text(encoding="utf-8"))[
            "scientific_discovery_runtime"]
        for path in paths
    ]
    for field in BANK_FIELDS:
        if all(runtime.get(field) for runtime in runtimes):
            return field
    raise ValueError("factorial arms lack one common bank field")


def _command(freeze, item, arm, output):
    config = ROOT / CONFIGS[arm]
    return [
        sys.executable, str(CHILD),
        "--family", item["family"], "--task", item["task"],
        "--seed", str(item["seed"]),
        "--condition", freeze["arms"][arm]["condition"],
        "--config", str(config),
        "--hdf5", freeze["hdf5"]["path"],
        "--provider-env", freeze["provider_env"]["path"],
        "--metadata-parquet", freeze["metadata"]["path"],
        "--input-symbols-json", json.dumps(
            freeze["input_symbols_by_task"][
                f'{item["family"]}/{item["task"]}']),
        "--output-dir", str(output),
    ]


def _usage(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    calls = payload.get("provider_cost") or []
    return {
        "provider_attempts": len(calls),
        "successful_provider_responses": sum(
            int(row.get("http_status", 0)) == 200 for row in calls),
        "prompt_tokens": sum(
            int(row.get("prompt_tokens") or 0) for row in calls),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in calls),
        "wall_time_seconds":
            float(payload.get("wall_time_seconds") or 0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--child-timeout", type=float, default=10800.)
    args = parser.parse_args(argv)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    authorization = _verify_authorization(
        args.freeze.resolve(strict=True), freeze,
        args.authorization.resolve(strict=True))
    if args.output_root.exists():
        raise ValueError("factorial output root already exists; resume forbidden")
    if args.workers != int(freeze["execution"]["workers"]):
        raise ValueError("worker count differs from freeze")
    if args.child_timeout != float(freeze["execution"]["child_timeout_s"]):
        raise ValueError("child timeout differs from freeze")
    root = args.output_root
    (root / "children").mkdir(parents=True)
    plan = [
        {"family": family, "task": task, "seed": int(seed), "arm": arm}
        for family, tasks in sorted(freeze["selected_tasks"].items())
        for task in sorted(tasks) for seed in sorted(freeze["seeds"])
        for arm in ARMS
    ]
    (root / "PLAN.json").write_text(json.dumps({
        "schema": "feedback-mechanism-factorial-plan-v1",
        "freeze_sha256": _digest(args.freeze),
        "authorization_sha256": _digest(args.authorization),
        "children": plan,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env["FORMULA_PROVIDER_LOCK"] = str(root / "provider.lock")
    env["FORMULA_PROVIDER_LOCK_TIMEOUT"] = "7200"

    def execute(item):
        output = (root / "children" / item["family"] / item["task"]
                  / str(item["seed"]) / item["arm"])
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.parent / f'{item["arm"]}.child.log'
        command = _command(freeze, item, item["arm"], output)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    timeout=args.child_timeout, check=False)
                code, timed_out = process.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        return {**item, "return_code": code, "timed_out": timed_out,
                "elapsed_seconds": time.monotonic() - started,
                "log": str(log.relative_to(root))}

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, item) for item in plan]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            (root / "PROGRESS.json").write_text(json.dumps({
                "completed": len(records), "of": len(plan),
            }, indent=2) + "\n", encoding="utf-8")
            if record["timed_out"] or record["return_code"] != 0:
                (root / "MATRIX_FAILURE.json").write_text(json.dumps({
                    "schema": "feedback-mechanism-factorial-failure-v1",
                    "failed_record": record, "records": records,
                    "retry_authorized": False,
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                raise RuntimeError("factorial child failed; preserve run")

    generation = []
    for item in plan:
        output = (root / "children" / item["family"] / item["task"]
                  / str(item["seed"]) / item["arm"])
        result = output / RESULT
        artifact = output / "pcpi_artifacts" / f'{item["task"]}.json'
        payload = json.loads(result.read_text(encoding="utf-8"))
        if (payload.get("status") != "success"
                or payload.get("test_or_ood_accessed") is not False
                or payload.get("heldout_opened") is not False):
            raise ValueError("child result violates frozen safety contract")
        generation.append({
            **item, "result_sha256": _digest(result),
            "artifact_sha256": _digest(artifact),
        })
    (root / "GENERATION_FROZEN.json").write_text(json.dumps({
        "schema": "feedback-mechanism-factorial-generation-frozen-v1",
        "children": generation, "child_count": len(generation),
        "reporting_opened": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows, usage = [], []
    for family, tasks in sorted(freeze["selected_tasks"].items()):
        for task in sorted(tasks):
            for seed in sorted(freeze["seeds"]):
                base = root / "children" / family / task / str(seed)
                artifacts = {
                    arm: base / arm / "pcpi_artifacts" / f"{task}.json"
                    for arm in ARMS
                }
                field = _common_bank_field(list(artifacts.values()))
                scores = {
                    arm: score_arm(
                        arm_artifact=artifacts[arm],
                        hdf5=Path(freeze["hdf5"]["path"]),
                        family=family, task=task, seed=int(seed),
                        bank_field=field)
                    for arm in ARMS
                }
                rows.append({
                    "family": family, "task": task, "seed": int(seed),
                    "bank_field": field, "scores": scores,
                })
                usage.append({
                    "family": family, "task": task, "seed": int(seed),
                    "usage": {
                        arm: _usage(base / arm / RESULT) for arm in ARMS},
                })
    summary = summarize(freeze, rows)
    summary["rows"] = rows
    summary["generation_frozen_before_reporting"] = True
    (root / "FACTORIAL_RESULT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    totals = {
        arm: {
            key: sum(row["usage"][arm][key] for row in usage)
            for key in (
                "provider_attempts", "successful_provider_responses",
                "prompt_tokens", "completion_tokens", "wall_time_seconds")
        } for arm in ARMS
    }
    (root / "COMPUTE_DIAGNOSTIC.json").write_text(json.dumps({
        "schema": "feedback-mechanism-factorial-compute-v1",
        "adaptive_compute_is_reported_not_a_gate": True,
        "totals": totals, "per_coordinate": usage,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "contrasts": summary["contrasts"], "compute": totals,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
