"""Run the frozen fresh-task MatSci decision-risk acquisition confirmation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.discovery.realized_drr import run_realized_drr_trajectory
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(freeze: dict) -> None:
    if (
        freeze.get("schema")
        != "scientific-aistats-matsci-acquisition-confirmation-freeze-v1"
        or freeze.get("condition") != "no_llm_v6"
        or freeze.get("policies") != ["decision_risk", "random"]
        or freeze.get("test_or_ood_accessed") is not False
        or freeze.get("heldout_opened") is not False
    ):
        raise ValueError("invalid MatSci confirmation freeze")
    for path, expected in freeze["files"].items():
        if _sha(Path(path)) != expected:
            raise ValueError(f"MatSci confirmation frozen file changed: {path}")
    for path, expected in freeze["exclusion_freezes"].items():
        if _sha(Path(path)) != expected:
            raise ValueError(f"MatSci exclusion freeze changed: {path}")
    if _sha(Path(freeze["hdf5"]["path"])) != freeze["hdf5"]["sha256"]:
        raise ValueError("MatSci confirmation HDF5 changed")
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("MatSci confirmation benchmark source changed")
    if verify_clean_git_source(
        ROOT.parent / "hypothesis_mvp"
    ) != freeze["mainline_source"]:
        raise ValueError("MatSci confirmation mainline source changed")


def _ordered_roles(handle, task: str, seed: int):
    dataset = handle[f"lsr_synth/matsci/{task}/train"]
    order = drr_adapter._ordered_indices(len(dataset), task, seed)
    cuts = (
        int(round(0.50 * len(dataset))),
        int(round(0.70 * len(dataset))),
        int(round(0.90 * len(dataset))),
    )
    initial_indices = tuple(order[cuts[1] : cuts[2]])[:32]
    action_indices = tuple(order[cuts[2] :])
    initial_sorted = sorted(initial_indices)
    action_sorted = sorted(action_indices)
    initial_rows = np.asarray(dataset[initial_sorted, :])
    action_rows = np.asarray(dataset[action_sorted, :])
    initial_lookup = dict(zip(initial_sorted, initial_rows, strict=True))
    action_lookup = dict(zip(action_sorted, action_rows, strict=True))
    initial = np.asarray([initial_lookup[index] for index in initial_indices])
    actions = np.asarray([action_lookup[index] for index in action_indices])
    return initial[:, 1:], initial[:, 0], actions[:, 1:], actions[:, 0]


def _discovery_command(freeze: dict, task: str, seed: int, run_dir: Path):
    return [
        sys.executable,
        str(ROOT / "eval.py"),
        "--dataset",
        "matsci",
        "--searcher_config",
        str(ROOT / "configs/aistats_drr_no_llm_v6.yaml"),
        "--problem_name",
        task,
        "--ds_root_folder",
        freeze["benchmark_data_root"],
        "--resume_from",
        str(run_dir),
        "--seed",
        str(seed),
        "--budget",
        str(freeze["discovery_budget"]),
    ]


def _run_discovery(freeze: dict, task: str, seed: int, run_dir: Path) -> dict:
    ledger_path = run_dir / "CONFIRMATION_DISCOVERY.json"
    artifact_path = run_dir / "pcpi_artifacts" / f"{task}.json"
    if ledger_path.is_file():
        ledger = _read(ledger_path)
        if (
            ledger.get("task") != task
            or ledger.get("seed") != seed
            or ledger.get("condition") != "no_llm_v6"
            or ledger.get("artifact_sha256") != _sha(artifact_path)
        ):
            raise ValueError("MatSci discovery resume identity changed")
        return _read(artifact_path)["scientific_discovery_runtime"]

    run_dir.mkdir(parents=True, exist_ok=False)
    cache = Path(freeze["benchmark_cache_root"])
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(seed),
            "HYPOTHESIS_MVP_ROOT": freeze["mainline_root"],
            "PCPI_DRR_CONDITION": "no_llm_v6",
            "PCPI_DISABLE_HELDOUT_PROMPT_DATA": "1",
            "PCPI_STRICT_TRAIN_VAL_ONLY": "1",
            "HF_HOME": str(cache / "huggingface"),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
        }
    )
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout:
        try:
            process = subprocess.run(
                _discovery_command(freeze, task, seed, run_dir),
                cwd=ROOT,
                env=env,
                check=False,
                timeout=float(freeze["wall_time_seconds_per_discovery"]),
                stdout=stdout,
                stderr=subprocess.STDOUT,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
    if returncode != 0 or not artifact_path.is_file():
        raise RuntimeError(f"MatSci discovery failed: task={task} seed={seed}")
    report = _read(artifact_path)["scientific_discovery_runtime"]
    if (
        int(report.get("scientist_agent_logical_llm_call_count") or 0) != 0
        or report.get("candidate_response_accessed") is not False
        or report.get("heldout_opened") is not False
    ):
        raise RuntimeError("MatSci no-LLM discovery contract violated")
    ledger = {
        "schema": "scientific-aistats-matsci-confirmation-discovery-v1",
        "task": task,
        "seed": seed,
        "condition": "no_llm_v6",
        "returncode": returncode,
        "artifact_sha256": _sha(artifact_path),
        "logical_llm_call_count": 0,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _candidate_rows(report: dict, n_features: int):
    if report.get("drr_candidate_rows"):
        return tuple(dict(row) for row in report["drr_candidate_rows"])
    protected = [
        candidate
        for row in report.get("scientist_agent_counterfactual_backbones", ())
        for candidate in row.get("backbone_candidates", ())
    ]
    return drr_adapter.candidate_rows(
        report, report["best_expression"], n_features, protected
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    freeze = _read(args.freeze)
    _verify(freeze)
    if args.output_dir.exists() and not args.resume:
        raise ValueError("MatSci confirmation output must be new")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    plan = [
        {
            "task": task,
            "seed": seed,
            "condition": "no_llm_v6",
            "policies": ["decision_risk", "random"],
            "run_dir": str(
                (
                    args.output_dir
                    / "runs"
                    / task
                    / f"seed{seed}"
                    / "no_llm_v6"
                ).resolve()
            ),
        }
        for task in freeze["selected_tasks"]["matsci"]
        for seed in freeze["seeds"]
    ]
    plan_path = args.output_dir / "PLAN.json"
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if plan_path.is_file() and plan_path.read_text(encoding="utf-8") != text:
        raise ValueError("MatSci confirmation plan identity changed")
    plan_path.write_text(text, encoding="utf-8")
    if args.preflight_only:
        result = {
            "schema": "scientific-aistats-matsci-acquisition-preflight-v1",
            "passed": True,
            "planned_discovery_runs": len(plan),
            "planned_trajectories": 2 * len(plan),
            "task_arrays_accessed": False,
            "candidate_response_accessed": False,
            "llm_called": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
            "execution_started": False,
        }
        (args.output_dir / "PREFLIGHT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    rows_path = args.output_dir / "CONFIRMATION_ROWS.json"
    rows = _read(rows_path) if rows_path.is_file() else []
    completed = {
        (row["task"], row["seed"], row["policy"]) for row in rows
    }
    with h5py.File(freeze["hdf5"]["path"], "r") as handle:
        for index, item in enumerate(plan, start=1):
            task, seed = item["task"], int(item["seed"])
            print(
                f"[{index}/{len(plan)}] no_llm_v6 matsci/{task} seed={seed}",
                flush=True,
            )
            report = _run_discovery(
                freeze, task, seed, Path(item["run_dir"])
            )
            X0, y0, action_X, action_y = _ordered_roles(handle, task, seed)
            candidates = _candidate_rows(report, X0.shape[1])
            for policy in freeze["policies"]:
                if (task, seed, policy) in completed:
                    continue
                identity = sha256(
                    f"matsci-acquisition-confirmation-v1:{task}:{seed}".encode()
                ).hexdigest()
                try:
                    trajectory = run_realized_drr_trajectory(
                        candidates,
                        X0,
                        y0,
                        action_X,
                        action_y,
                        condition="no_llm_v6",
                        exploration_identity=identity,
                        policy=policy,
                        random_seed=seed,
                    )
                    failure = ""
                except Exception as error:
                    trajectory = {}
                    failure = f"{type(error).__name__}:{error}"
                rows.append(
                    {
                        "family": "matsci",
                        "task": task,
                        "seed": seed,
                        "condition": "no_llm_v6",
                        "policy": policy,
                        "symmetric_realized_risk_aulc": float(
                            trajectory.get("symmetric_realized_risk_aulc", 0.0)
                        ),
                        "absolute_realized_risk_aulc": float(
                            trajectory.get("absolute_realized_risk_aulc", 0.0)
                        ),
                        "status": "success" if not failure else "failure",
                        "failure": failure,
                        "trajectory": trajectory,
                    }
                )
                rows_path.write_text(
                    json.dumps(rows, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    by = {
        (row["task"], row["seed"], row["policy"]): row for row in rows
    }
    effects = {}
    for task in freeze["selected_tasks"]["matsci"]:
        effects[task] = float(
            np.mean(
                [
                    by[(task, seed, "decision_risk")][
                        "symmetric_realized_risk_aulc"
                    ]
                    - by[(task, seed, "random")][
                        "symmetric_realized_risk_aulc"
                    ]
                    for seed in freeze["seeds"]
                ]
            )
        )
    tolerance = freeze["confirmation_decision"]["numerical_tolerance"]
    global_mean = float(np.mean(list(effects.values())))
    decisions = {
        "all_16_trajectories_accounted_for": len(rows) == 16,
        "zero_failures": all(row["status"] == "success" for row in rows),
        "all_tasks_have_nonnegative_effect": all(
            value >= -tolerance for value in effects.values()
        ),
        "at_least_three_strictly_positive_tasks": sum(
            value > tolerance for value in effects.values()
        )
        >= 3,
        "global_mean_strictly_positive": global_mean > tolerance,
        "no_llm_calls": True,
    }
    result = {
        "schema": "scientific-aistats-matsci-acquisition-confirmation-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "task_effects": effects,
        "mean_targeted_minus_random_effect": global_mean,
        "failure_count": sum(row["status"] != "success" for row in rows),
        "row_count": len(rows),
        "rows": rows,
        "candidate_response_accessed": True,
        "action_response_accessed": True,
        "llm_called": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": all(decisions.values()),
        "claim_boundary": (
            "Fresh-task MatSci development confirmation of decision-risk "
            "acquisition versus matched random only; no LLM synthesis, "
            "test/OOD, held-out, or universal superiority claim."
        ),
    }
    path = args.output_dir / "AISTATS_MATSCI_ACQUISITION_CONFIRMATION.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
