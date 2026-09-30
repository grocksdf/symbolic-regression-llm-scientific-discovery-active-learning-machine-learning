"""Run the frozen fresh-task PCPI versus LLM-SR external comparison."""

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

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from scripts.run_aistats_drr_benchmark import _sha


DATASET_MAP = {
    "bio_pop_growth": "bio_pop_growth",
    "chem_react": "chem_react",
    "lsr_transform": "lsrtransform",
    "matsci": "matsci",
}
CONFIGS = {
    "full_scientist_v6": "configs/aistats_drr_full_v6.yaml",
    "no_llm_v6": "configs/aistats_drr_no_llm_v6.yaml",
    "external_llmsr_glm53": "configs/aistats_external_llmsr_glm53.yaml",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _provider_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    raise ValueError("registered provider env has no OPENAI_API_KEY")


def _verify(freeze: dict) -> None:
    if (
        freeze.get("schema")
        != "scientific-aistats-external-llmsr-freeze-v1"
        or freeze.get("execution_authorized") is not True
    ):
        raise ValueError("invalid external LLM-SR freeze")
    for path, expected in {
        **freeze["files"], **freeze["exclusion_freezes"]
    }.items():
        if _sha(path) != expected:
            raise ValueError(f"external LLM-SR frozen file changed: {path}")
    if _sha(freeze["hdf5"]["path"]) != freeze["hdf5"]["sha256"]:
        raise ValueError("external LLM-SR HDF5 changed")
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("external LLM-SR benchmark source changed")
    if verify_clean_git_source(
        ROOT.parent / "hypothesis_mvp"
    ) != freeze["mainline_source"]:
        raise ValueError("external LLM-SR mainline source changed")


def _command(freeze: dict, family: str, task: str, seed: int,
             condition: str, run_dir: Path) -> list[str]:
    budget = freeze["condition_budgets"][condition].get(
        "candidate_evaluation_limit",
        freeze["condition_budgets"][condition].get("sample_limit"),
    )
    return [
        sys.executable,
        str(ROOT / "eval.py"),
        "--dataset",
        DATASET_MAP[family],
        "--searcher_config",
        str(ROOT / CONFIGS[condition]),
        "--problem_name",
        task,
        "--ds_root_folder",
        freeze["benchmark_data_root"],
        "--resume_from",
        str(run_dir),
        "--seed",
        str(seed),
        "--budget",
        str(budget),
    ]


def _result_row(family: str, task: str, seed: int, condition: str,
                run_dir: Path, returncode: int, failure: str) -> dict:
    result_path = run_dir / "results.jsonl"
    metrics = {
        "id_nmse": 100.0,
        "ood_nmse": 100.0,
        "id_acc_0_1": 0.0,
        "ood_acc_0_1": 0.0,
        "search_time_seconds": 0.0,
    }
    if not failure and result_path.is_file():
        try:
            rows = [
                json.loads(line)
                for line in result_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != 1 or len(rows[0].get("eval_results", ())) != 1:
                raise ValueError("expected exactly one evaluation result")
            result = rows[0]["eval_results"][0]
            metrics = {
                "id_nmse": float(result.get("id_nmse", 100.0)),
                "ood_nmse": float(
                    100.0 if result.get("ood_nmse") is None
                    else result["ood_nmse"]),
                "id_acc_0_1": float(result.get("id_acc@0.1", 0.0)),
                "ood_acc_0_1": float(
                    0.0 if result.get("ood_acc@0.1") is None
                    else result["ood_acc@0.1"]),
                "search_time_seconds": float(result["search_time"]),
            }
        except Exception as error:
            failure = f"result-invalid:{type(error).__name__}:{error}"
    elif not failure:
        failure = "result-missing"
    return {
        "family": family,
        "task": task,
        "seed": seed,
        "condition": condition,
        "status": "success" if not failure else "failure",
        "failure": failure,
        "returncode": returncode,
        "run_dir": str(run_dir),
        **metrics,
    }


def _aggregate(rows: list[dict]) -> dict:
    output = {}
    for condition in CONFIGS:
        selected = [row for row in rows if row["condition"] == condition]
        output[condition] = {
            "run_count": len(selected),
            "failure_count": sum(row["status"] != "success" for row in selected),
            "mean_id_nmse": float(np.mean([row["id_nmse"] for row in selected])),
            "mean_ood_nmse": float(np.mean([row["ood_nmse"] for row in selected])),
            "mean_id_acc_0_1": float(
                np.mean([row["id_acc_0_1"] for row in selected])),
            "mean_ood_acc_0_1": float(
                np.mean([row["ood_acc_0_1"] for row in selected])),
            "mean_search_time_seconds": float(
                np.mean([row["search_time_seconds"] for row in selected])),
        }
    return output


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
        raise ValueError("external LLM-SR output must be new")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    plan = [
        {
            "family": family,
            "task": task,
            "seed": seed,
            "condition": condition,
            "run_dir": str(
                (
                    args.output_dir / "runs" / family / task /
                    f"seed{seed}" / condition
                ).resolve()
            ),
        }
        for family, task in freeze["selected_tasks"].items()
        for seed in freeze["seeds"]
        for condition in freeze["conditions"]
    ]
    plan_path = args.output_dir / "PLAN.json"
    plan_text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if plan_path.is_file() and plan_path.read_text(encoding="utf-8") != plan_text:
        raise ValueError("external LLM-SR plan identity changed")
    plan_path.write_text(plan_text, encoding="utf-8")
    if args.preflight_only:
        result = {
            "schema": "scientific-aistats-external-llmsr-preflight-v1",
            "passed": True,
            "planned_child_runs": len(plan),
            "task_arrays_accessed": False,
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

    rows_path = args.output_dir / "EXTERNAL_ROWS.json"
    rows = _read(rows_path) if rows_path.is_file() else []
    completed = {
        (row["family"], row["task"], row["seed"], row["condition"])
        for row in rows
    }
    provider_key = _provider_key(Path(freeze["provider_env"]))
    cache = Path(freeze["benchmark_cache_root"])
    cache.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(plan, start=1):
        key = (
            item["family"], item["task"],
            int(item["seed"]), item["condition"],
        )
        if key in completed:
            continue
        print(
            f"[{index}/{len(plan)}] {item['condition']} "
            f"{item['family']}/{item['task']} seed={item['seed']}",
            flush=True,
        )
        run_dir = Path(item["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update({
            "PYTHONHASHSEED": str(item["seed"]),
            "HYPOTHESIS_MVP_ROOT": freeze["mainline_root"],
            "OPENAI_API_KEY": provider_key,
            "LLMSR_PROVIDER_MAX_ATTEMPTS": "3",
            "PYTHONPATH": os.pathsep.join([
                freeze["external_site"], str(ROOT),
                freeze["mainline_root"],
                env.get("PYTHONPATH", ""),
            ]),
            "HF_HOME": str(cache / "huggingface"),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "PCPI_DISABLE_HELDOUT_PROMPT_DATA": "1",
            "PCPI_STRICT_TRAIN_VAL_ONLY": "1",
        })
        timeout = float(
            freeze["condition_budgets"][item["condition"]][
                "wall_time_seconds"])
        failure = ""
        try:
            with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout:
                process = subprocess.run(
                    _command(
                        freeze, item["family"], item["task"],
                        int(item["seed"]), item["condition"], run_dir),
                    cwd=ROOT,
                    env=env,
                    check=False,
                    timeout=timeout,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                )
            returncode = process.returncode
            if returncode != 0:
                failure = f"child-returncode-{returncode}"
        except subprocess.TimeoutExpired:
            returncode, failure = 124, "wall-time-timeout"
        rows.append(_result_row(
            item["family"], item["task"], int(item["seed"]),
            item["condition"], run_dir, returncode, failure))
        rows_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    aggregate = _aggregate(rows)
    result = {
        "schema": "scientific-aistats-external-llmsr-result-v1",
        "protocol_complete": len(rows) == len(plan),
        "row_count": len(rows),
        "failure_count": sum(row["status"] != "success" for row in rows),
        "condition_summaries": aggregate,
        "rows": rows,
        "task_selection_unchanged": True,
        "failures_included_as_worst_score": True,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": True,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Frozen fresh-task external comparison against LLM-SR with "
            "failure-inclusive ID/OOD reporting; not held-out confirmation "
            "or universal superiority."
        ),
    }
    path = args.output_dir / "AISTATS_EXTERNAL_LLMSR_RESULT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
