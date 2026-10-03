"""Tiny opened-development smoke for prospective formula materialization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from formula_bank_materialization_v2 import (
    expanded_bank_rows, proposal_stage_audit, read_generator_artifact,
)
from provider_health_contract import require_healthy_generation


CASES = (
    ("bio_pop_growth", "BPG3", 501,
     "lsr_synth_bio_pop_growth-00000-of-00001.parquet"),
    ("phys_osc", "PO25", 501,
     "lsr_synth_phys_osc-00000-of-00001.parquet"),
)
MINIMUM_MATERIALIZATION_RATE = 0.5


def _symbols(path, task):
    table = pq.read_table(path, columns=["name", "symbols"])
    names = [str(value) for value in table.column("name").to_pylist()]
    if names.count(task) != 1:
        raise ValueError("smoke task metadata identity invalid")
    symbols = [str(value) for value in
               table.column("symbols")[names.index(task)].as_py()]
    if len(symbols) < 2:
        raise ValueError("smoke task has no registered inputs")
    return symbols[1:]


def _one(args, family, task, seed, metadata_name):
    run_dir = args.output_dir / "runs" / family / task / f"seed{seed}"
    metadata = args.data_root / "data" / metadata_name
    command = [
        sys.executable,
        str(ROOT / "scripts/run_aistats_three_arm_formula_child.py"),
        "--family", family, "--task", task, "--seed", str(seed),
        "--condition", "three_arm_l_gap_v1",
        "--config",
        str(ROOT / "configs/aistats_three_arm_formula_prospective.yaml"),
        "--hdf5", str(args.data_root / "lsr_bench_data.hdf5"),
        "--provider-env", str(args.provider_env),
        "--metadata-parquet", str(metadata),
        "--input-symbols-json", json.dumps(_symbols(metadata, task)),
        "--output-dir", str(run_dir),
    ]
    env = os.environ.copy()
    env["THREE_ARM_PROMPT_BYTES"] = "65536"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(run_dir) + ".stdout.log").open(
            "w", encoding="utf-8") as stdout:
        process = subprocess.run(
            command, cwd=ROOT, env=env, stdout=stdout,
            stderr=subprocess.STDOUT, check=False, timeout=900)
    completed = run_dir / "THREE_ARM_CHILD_RESULT.json"
    if process.returncode or not completed.is_file():
        failure = run_dir / "THREE_ARM_CHILD_FAILURE.json"
        payload = json.loads(failure.read_text(
            encoding="utf-8")) if failure.is_file() else {}
        return {
            "family": family, "task": task, "seed": seed,
            "status": "failure", "child_returncode": process.returncode,
            "provider_cost": payload.get("provider_cost", []),
            "typed_directive_count": 0,
            "optional_materialized_count": 0,
        }
    child = json.loads(completed.read_text(encoding="utf-8"))
    health = require_healthy_generation(child["provider_cost"])
    report = read_generator_artifact(run_dir, task)
    n_features = len(_symbols(metadata, task))
    _, _, materialization = expanded_bank_rows(report, n_features, 7)
    stages = proposal_stage_audit(
        report, child["provider_cost"], materialization)
    return {
        "family": family, "task": task, "seed": seed,
        "status": "success", "child_returncode": 0,
        "provider_health": health,
        "provider_cost": child["provider_cost"],
        "typed_directive_count": stages["typed_directive_count"],
        "typed_compiled_count": stages["typed_compiled_count"],
        "optional_materialized_count":
            stages["optional_materialized_count"],
        "proposal_rejected_by_grammar":
            stages["proposal_rejected_by_grammar"],
        "proposal_not_novel": stages["proposal_not_novel"],
        "ground_truth_expression_opened": False,
    }


def summarize(rows):
    eligible = [
        row for row in rows
        if row.get("status") == "success"
        and row.get("provider_health", {}).get("provider_attempted") is True
        and row.get("provider_health", {}).get("transport_valid") is True
    ]
    materialized = sum(
        row.get("optional_materialized_count", 0) > 0 for row in eligible)
    rate = materialized / len(eligible) if eligible else 0.0
    provider_2xx = sum(
        sum(1 for attempt in row.get("provider_cost", [])
            if 200 <= attempt.get("http_status", 0) < 300)
        for row in rows)
    directives = sum(row.get("typed_directive_count", 0) for row in rows)
    candidates = sum(
        row.get("optional_materialized_count", 0) for row in rows)
    decisions = {
        "all_smoke_runs_completed": all(
            row.get("status") == "success" for row in rows),
        "provider_2xx_observed": provider_2xx > 0,
        "typed_directives_observed": directives > 0,
        "novel_materialized_candidates_observed": candidates > 0,
        "conditional_materialization_rate_at_least_half":
            rate >= MINIMUM_MATERIALIZATION_RATE,
    }
    return {
        "eligible_successful_llm_runs": len(eligible),
        "runs_with_materialized_novel_candidate": materialized,
        "materialized_novel_candidate_given_successful_llm_response_run":
            rate,
        "provider_2xx_count": provider_2xx,
        "typed_directive_count": directives,
        "optional_materialized_count": candidates,
        "decisions": decisions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("tiny development smoke output must be new")
    args.output_dir.mkdir(parents=True)
    rows = [_one(args, *case) for case in CASES]
    summary = summarize(rows)
    result = {
        "schema": "expanded-formula-v2-tiny-development-smoke-v1",
        "passed": all(summary["decisions"].values()),
        **summary, "rows": rows,
        "minimum_registered_materialization_rate":
            MINIMUM_MATERIALIZATION_RATE,
        "task_role": "previously-opened-development-only",
        "ground_truth_expression_opened": False,
        "test_or_od_accessed": False, "heldout_opened": False,
        "confirmation_accessed": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Tiny development transport-to-materialization diagnostic only; "
            "not efficacy, superiority, task selection or confirmation."),
    }
    (args.output_dir / "GATE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
