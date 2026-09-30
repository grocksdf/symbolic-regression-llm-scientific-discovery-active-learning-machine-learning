"""Freeze an infrastructure-only continuation for failed LLM-SR rows."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--spawn-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("external LLM-SR continuation freeze must be new")
    source_freeze = _read(args.source_freeze)
    source_result = _read(args.source_result)
    spawn_gate = _read(args.spawn_gate)
    if (
        source_freeze.get("schema")
        != "scientific-aistats-external-llmsr-freeze-v1"
        or source_result.get("schema")
        != "scientific-aistats-external-llmsr-result-v1"
        or source_result.get("protocol_complete") is not True
        or spawn_gate.get("passed") is not True
        or spawn_gate.get("real_data_accessed") is not False
    ):
        raise ValueError("invalid external LLM-SR continuation inputs")
    external = [
        row for row in source_result["rows"]
        if row["condition"] == "external_llmsr_glm53"]
    if (
        len(external) != 8
        or any(row["failure"] != "wall-time-timeout" for row in external)
    ):
        raise ValueError("continuation requires eight timeout-only LLM-SR rows")
    logs = []
    for row in external:
        path = Path(row["run_dir"]) / "stdout.log"
        text = path.read_text(encoding="utf-8", errors="replace")
        if "An attempt has been made to start a new process" not in text:
            raise ValueError("LLM-SR timeout lacks the registered spawn failure")
        logs.append({"path": str(path.resolve()), "sha256": _sha(path)})

    files = [
        ROOT / "eval.py",
        ROOT / "scripts/run_aistats_external_llmsr_continuation.py",
        ROOT / "scripts/run_aistats_external_llmsr_benchmark.py",
        args.source_freeze,
        args.source_result,
        args.spawn_gate,
    ]
    result = {
        "schema": "scientific-aistats-external-llmsr-continuation-v1",
        "source_freeze": str(args.source_freeze.resolve()),
        "source_freeze_sha256": _sha(args.source_freeze),
        "source_result": str(args.source_result.resolve()),
        "source_result_sha256": _sha(args.source_result),
        "source_output": str(args.source_output.resolve()),
        "selected_tasks": source_freeze["selected_tasks"],
        "seeds": source_freeze["seeds"],
        "conditions": source_freeze["conditions"],
        "condition_budgets": source_freeze["condition_budgets"],
        "rerun_condition": "external_llmsr_glm53",
        "rerun_count": 8,
        "preserved_condition_rows": 16,
        "failure_policy": (
            "preserve all Full/No-LLM rows including failures; replace only "
            "the eight LLM-SR spawn-infrastructure timeouts"),
        "spawn_failure_logs": logs,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(path.resolve()): _sha(path) for path in files},
        "provider_env": source_freeze["provider_env"],
        "external_site": source_freeze["external_site"],
        "benchmark_data_root": source_freeze["benchmark_data_root"],
        "benchmark_cache_root": source_freeze["benchmark_cache_root"],
        "mainline_root": source_freeze["mainline_root"],
        "test_or_ood_already_opened_in_source": True,
        "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Infrastructure-only continuation of the frozen external LLM-SR "
            "comparison; no task, seed, budget, internal row, or analysis "
            "replacement."
        ),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_EXTERNAL_LLMSR_CONTINUATION.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
