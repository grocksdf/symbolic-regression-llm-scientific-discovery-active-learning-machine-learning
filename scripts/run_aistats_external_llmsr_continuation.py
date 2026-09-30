"""Run only the infrastructure-failed LLM-SR rows and preserve source rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from scripts.run_aistats_drr_benchmark import _sha
from scripts.run_aistats_external_llmsr_benchmark import (
    _aggregate,
    _command,
    _provider_key,
    _result_row,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(continuation: dict) -> None:
    if (
        continuation.get("schema")
        != "scientific-aistats-external-llmsr-continuation-v1"
        or continuation.get("rerun_condition") != "external_llmsr_glm53"
        or continuation.get("rerun_count") != 8
    ):
        raise ValueError("invalid external LLM-SR continuation")
    for path, expected in continuation["files"].items():
        if _sha(path) != expected:
            raise ValueError(f"external continuation frozen file changed: {path}")
    if _sha(continuation["source_freeze"]) != continuation[
            "source_freeze_sha256"]:
        raise ValueError("external continuation source freeze changed")
    if _sha(continuation["source_result"]) != continuation[
            "source_result_sha256"]:
        raise ValueError("external continuation source result changed")
    if verify_clean_git_source(ROOT) != continuation["benchmark_source"]:
        raise ValueError("external continuation benchmark source changed")
    if verify_clean_git_source(
        ROOT.parent / "hypothesis_mvp"
    ) != continuation["mainline_source"]:
        raise ValueError("external continuation mainline source changed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    continuation = _read(args.continuation)
    _verify(continuation)
    if args.output_dir.exists() and not args.resume:
        raise ValueError("external continuation output must be new")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    source_freeze = _read(Path(continuation["source_freeze"]))
    source_result = _read(Path(continuation["source_result"]))
    preserved = [
        dict(row) for row in source_result["rows"]
        if row["condition"] != "external_llmsr_glm53"]
    rows_path = args.output_dir / "EXTERNAL_ROWS.json"
    rows = _read(rows_path) if rows_path.is_file() else list(preserved)
    if rows[:len(preserved)] != preserved:
        raise ValueError("preserved external comparison rows changed")
    completed = {
        (row["family"], row["task"], row["seed"], row["condition"])
        for row in rows
    }
    provider_key = _provider_key(Path(continuation["provider_env"]))
    cache = Path(continuation["benchmark_cache_root"])
    cache.mkdir(parents=True, exist_ok=True)
    planned = [
        (family, task, seed)
        for family, task in continuation["selected_tasks"].items()
        for seed in continuation["seeds"]
    ]
    for index, (family, task, seed) in enumerate(planned, start=1):
        key = (family, task, seed, "external_llmsr_glm53")
        if key in completed:
            continue
        print(
            f"[{index}/{len(planned)}] external_llmsr_glm53 "
            f"{family}/{task} seed={seed}", flush=True)
        run_dir = (
            args.output_dir / "runs" / family / task /
            f"seed{seed}" / "external_llmsr_glm53")
        run_dir.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update({
            "PYTHONHASHSEED": str(seed),
            "HYPOTHESIS_MVP_ROOT": continuation["mainline_root"],
            "OPENAI_API_KEY": provider_key,
            "LLMSR_PROVIDER_MAX_ATTEMPTS": "3",
            "PYTHONPATH": os.pathsep.join([
                continuation["external_site"], str(ROOT),
                continuation["mainline_root"],
                env.get("PYTHONPATH", ""),
            ]),
            "HF_HOME": str(cache / "huggingface"),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "PCPI_DISABLE_HELDOUT_PROMPT_DATA": "1",
            "PCPI_STRICT_TRAIN_VAL_ONLY": "1",
        })
        timeout = float(
            continuation["condition_budgets"][
                "external_llmsr_glm53"]["wall_time_seconds"])
        failure = ""
        try:
            with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout:
                process = subprocess.run(
                    _command(
                        source_freeze, family, task, seed,
                        "external_llmsr_glm53", run_dir),
                    cwd=ROOT, env=env, check=False, timeout=timeout,
                    stdout=stdout, stderr=subprocess.STDOUT)
            returncode = process.returncode
            if returncode != 0:
                failure = f"child-returncode-{returncode}"
        except subprocess.TimeoutExpired:
            returncode, failure = 124, "wall-time-timeout"
        rows.append(_result_row(
            family, task, seed, "external_llmsr_glm53",
            run_dir, returncode, failure))
        rows_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    result = {
        "schema": "scientific-aistats-external-llmsr-result-v1",
        "protocol_complete": len(rows) == 24,
        "continuation_of": continuation["source_result"],
        "preserved_internal_row_count": len(preserved),
        "rerun_external_row_count": len(rows) - len(preserved),
        "row_count": len(rows),
        "failure_count": sum(row["status"] != "success" for row in rows),
        "condition_summaries": _aggregate(rows),
        "rows": rows,
        "task_selection_unchanged": True,
        "failures_included_as_worst_score": True,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": True,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Infrastructure-corrected frozen external LLM-SR comparison; "
            "internal rows and failures preserved, no task/seed replacement, "
            "not held-out confirmation or universal superiority."),
    }
    path = args.output_dir / "AISTATS_EXTERNAL_LLMSR_RESULT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
