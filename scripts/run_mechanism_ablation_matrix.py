"""Run the Experiment 2 coupled-block ablation matrix.

Arms: L_full (A), L_without_feedback_block (B), L_full_copy (A duplicate, negative control).
All three run under one benchmark commit, one mainline commit, one provider, one seed set.

Harness note (amendment 002)
----------------------------
The per-child guard used to be a hard-coded 2400 s.  The measured baseline for a
single Arm A child at ``lsr_transform/I.12.2_3_0/2201`` is 1898.7 s, so that guard
carried ~26% headroom and the first v2 child was killed at 40 min.  The guard is
now an explicit CLI value (default 5400 s), children may be scheduled
concurrently with ``--workers``, and a half-finished arm directory is renamed
aside instead of blocking the retry (the child contract requires its own
``--output-dir`` to not exist).  None of this touches a child's inputs: same
command line, same config, same seed, same artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mechanism_ablation_contract import (  # noqa: E402
    summarize_ablation, placebo_halves, task_effects,
)
from scripts.score_mechanism_arm import BANK_FIELDS, score_arm  # noqa: E402

ARM_CONFIG = {
    "L_full": "configs/aistats_three_arm_formula_expanded_candidate.yaml",
    "L_without_feedback_block": "configs/aistats_three_arm_formula_no_gap_block.yaml",
    "L_full_copy": "configs/aistats_three_arm_formula_expanded_candidate.yaml",
}
ARM_ORDER = ("L_full", "L_without_feedback_block", "L_full_copy")
CHILD = ROOT / "scripts/run_aistats_three_arm_formula_child.py"
RESULT_NAME = "THREE_ARM_CHILD_RESULT.json"
# Measured Arm A baseline is 1898.7 s at one coordinate; 2400 s had ~26% headroom
# and killed a healthy child.  5400 s is guard headroom, never a budget change.
DEFAULT_CHILD_TIMEOUT_S = 5400.0
# Children hold the provider lock only around provider transport, so concurrent
# children still serialise their requests when --workers > 1.
PROVIDER_LOCK_TIMEOUT_S = "3600"


def _child_command(freeze, item, arm, out_dir):
    return [
        sys.executable, str(CHILD),
        "--family", item["family"], "--task", item["task"], "--seed", str(item["seed"]),
        "--condition", freeze["arm_configs"][arm]["condition"],
        "--config", str(ROOT / ARM_CONFIG[arm]),
        "--hdf5", freeze["hdf5"]["path"], "--provider-env", freeze["provider_env"],
        "--metadata-parquet",
        freeze["development_metadata_files"][item["family"]]["path"],
        "--input-symbols-json",
        json.dumps(freeze["input_symbols_by_task"][f"{item['family']}/{item['task']}"]),
        "--output-dir", str(out_dir),
    ]


def _child_env(root, workers):
    if workers <= 1:
        return None
    env = dict(os.environ)
    env["FORMULA_PROVIDER_LOCK"] = str(root / "provider.lock")
    env.setdefault("FORMULA_PROVIDER_LOCK_TIMEOUT", PROVIDER_LOCK_TIMEOUT_S)
    return env


def _run_child(command, log_path, timeout, env=None):
    """Return (return_code, elapsed_seconds).  ``return_code`` is None on timeout."""
    with log_path.open("w", encoding="utf-8") as stream:
        started = time.monotonic()
        try:
            process = subprocess.run(command, cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", timeout=timeout, check=False,
                                     env=env)
        except subprocess.TimeoutExpired:
            return None, time.monotonic() - started
    return process.returncode, time.monotonic() - started


def _prepare_arm_dir(out_dir):
    """True when the arm still has to run.

    The child refuses an existing ``--output-dir``, and an interrupted run leaves
    one behind.  Renaming it aside keeps the residue as evidence and makes the
    coordinate retryable without deleting anything.
    """
    if not out_dir.exists():
        return True
    if (out_dir / RESULT_NAME).exists():
        return False
    stale = out_dir.parent / (
        f"{out_dir.name}.stale-{time.strftime('%Y%m%dT%H%M%S')}")
    os.rename(out_dir, stale)
    return True


def _usage(result_path):
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    cost = payload.get("provider_cost") or []
    return {
        "provider_requests": len(cost),
        "provider_attempts": sum(1 for row in cost),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in cost
                             if row.get("prompt_tokens") is not None),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in cost
                                 if row.get("completion_tokens") is not None),
        "wall_time_seconds": payload.get("wall_time_seconds"),
        "decoded_response_count": payload.get("decoded_response_count"),
        "decoded_response_roles": payload.get("decoded_response_roles"),
    }


def _common_bank_field(artifacts):
    runtime_keys = []
    for path in artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime_keys.append(payload["scientific_discovery_runtime"])
    for field in BANK_FIELDS:
        if all(runtime.get(field) for runtime in runtime_keys):
            return field
    raise ValueError("arms expose no common scorable bank field; refusing to compare")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child-timeout", type=float, default=DEFAULT_CHILD_TIMEOUT_S)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    root = args.output_root
    if root.exists() and not args.resume:
        raise ValueError("ablation output root already exists")
    (root / "children").mkdir(parents=True, exist_ok=args.resume)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    coordinates = [(family, task, seed)
                   for family, tasks in sorted(freeze["selected_tasks"].items())
                   for task in sorted(tasks)
                   for seed in sorted(freeze["seeds"])]

    pending = []
    for family, task, seed in coordinates:
        item = {"family": family, "task": task, "seed": seed}
        base = root / "children" / family / task / str(seed)
        # The child contract requires its own --output-dir to not exist yet
        # (one creation attempt per coordinate), so only the coordinate parent
        # is created here and the arm directory is left to the child.
        base.mkdir(parents=True, exist_ok=True)
        for arm in ARM_ORDER:
            out = base / arm
            if _prepare_arm_dir(out):
                pending.append((family, task, seed, arm, out,
                                _child_command(freeze, item, arm, out)))

    env = _child_env(root, args.workers)
    timing = {}

    def run(work):
        family, task, seed, arm, out, command = work
        log_path = out.parent / f"{arm}.child.log"
        code, elapsed = _run_child(command, log_path, args.child_timeout, env)
        record = {"return_code": code, "elapsed_seconds": elapsed,
                  "timed_out": code is None,
                  "log": str(log_path.relative_to(root))}
        return (family, task, seed, arm), record

    done = 0
    if args.workers == 1:
        for work in pending:
            key, record = run(work)
            timing["/".join(map(str, key))] = record
            done += 1
            _write_progress(root, done, len(pending))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, work) for work in pending]
            for future in as_completed(futures):
                key, record = future.result()
                timing["/".join(map(str, key))] = record
                done += 1
                _write_progress(root, done, len(pending))
    (root / "CHILD_TIMING.json").write_text(
        json.dumps({"schema": "ablation-child-timing-v1",
                    "child_timeout_s": args.child_timeout,
                    "workers": args.workers,
                    "completed_child_executions": len(timing),
                    "per_child": dict(sorted(timing.items()))}, indent=2) + "\n",
        encoding="utf-8")

    failures = [f"{key}: {'timeout' if rec['timed_out'] else 'exit ' + str(rec['return_code'])}"
                for key, rec in sorted(timing.items())
                if rec["timed_out"] or rec["return_code"] != 0]
    if failures:
        raise RuntimeError("ablation children failed; nothing was summarised and the "
                           "completed coordinates stay on disk for --resume: "
                           + "; ".join(failures))

    rows, usage_rows = [], []
    for family, task, seed in coordinates:
        base = root / "children" / family / task / str(seed)
        artifacts, per_arm_usage = {}, {}
        for arm in ARM_ORDER:
            out = base / arm
            artifacts[arm] = out / "pcpi_artifacts" / f"{task}.json"
            per_arm_usage[arm] = _usage(out / RESULT_NAME)
        field = _common_bank_field(list(artifacts.values()))
        scores = {arm: score_arm(arm_artifact=artifacts[arm],
                                 hdf5=Path(freeze["hdf5"]["path"]), family=family,
                                 task=task, seed=seed, bank_field=field)
                  for arm in artifacts}
        rows.append({"family": family, "task": task, "seed": seed,
                     "scores": scores, "bank_field": field})
        usage_rows.append({"family": family, "task": task, "seed": seed,
                           "usage": per_arm_usage})
        (root / "PROGRESS.json").write_text(json.dumps(
            {"completed": len(rows), "of": len(coordinates)}, indent=2) + "\n",
            encoding="utf-8")

    summary = summarize_ablation(freeze, rows)
    placebo = placebo_halves(freeze, rows, freeze["statistical_analysis"])
    budget = {
        "schema": "ablation-compute-budget-comparison-v1",
        "note": ("Measured, not assumed. Any material asymmetry must be reported and bars "
                 "attributing the difference to the mechanism alone."),
        "per_arm_totals": {
            arm: {
                "provider_requests": sum(r["usage"][arm]["provider_requests"]
                                         for r in usage_rows),
                "prompt_tokens": sum(r["usage"][arm]["prompt_tokens"] for r in usage_rows),
                "completion_tokens": sum(r["usage"][arm]["completion_tokens"]
                                         for r in usage_rows),
                "wall_time_seconds": sum(r["usage"][arm]["wall_time_seconds"] or 0.0
                                         for r in usage_rows),
            } for arm in ARM_ORDER},
        "decoded_response_roles_identical": len({
            json.dumps(r["usage"][arm]["decoded_response_roles"], sort_keys=True)
            for r in usage_rows for arm in r["usage"]}) == 1,
        "per_coordinate": usage_rows,
    }
    (root / "MATRIX_RESULT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "PLACEBO.json").write_text(
        json.dumps(placebo, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "COMPUTE_BUDGET_COMPARISON.json").write_text(
        json.dumps(budget, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"component_conclusion": summary["component_conclusion"],
                      "mean": summary["task_mean_effects"],
                      "ci95": summary["task_bootstrap"]["intervals"],
                      "bank_fields": sorted({r["bank_field"] for r in rows})}, indent=2))
    return 0


def _write_progress(root, done, total):
    (root / "PROGRESS.json").write_text(json.dumps(
        {"completed_child_executions": done, "of_child_executions": total},
        indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
