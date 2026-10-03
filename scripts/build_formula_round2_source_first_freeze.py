"""Build the immutable executable freeze without opening formula truth."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


FILES = {
    "chem_react": "lsr_synth_chem_react-00000-of-00001.parquet",
    "lsr_transform": "lsr_transform-00000-of-00001.parquet",
    "phys_osc": "lsr_synth_phys_osc-00000-of-00001.parquet",
}


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--gap-routing-gate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--engine-checkpoint-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("round-two executable freeze output must be new")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    gate = json.loads(args.gap_routing_gate.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (preflight.get("execution_authorized") is not False
            or gate.get("passed") is not True
            or gate.get("execution_authorized") is not False):
        raise ValueError("round-two prerequisite identity invalid")
    inputs, metadata = {}, {}
    for family, tasks in preflight["development_tasks"].items():
        path = args.data_root / "data" / FILES[family]
        table = pq.read_table(path, columns=["name", "symbols"])
        names = [str(value) for value in table.column("name").to_pylist()]
        for task in tasks:
            if names.count(task) != 1:
                raise ValueError("frozen task metadata missing")
            symbols = [
                str(value) for value in
                table.column("symbols")[names.index(task)].as_py()]
            if len(symbols) < 2 or len(set(symbols[1:])) != len(symbols) - 1:
                raise ValueError("invalid frozen input symbols")
            inputs[f"{family}/{task}"] = symbols[1:]
        metadata[family] = {"path": str(path.resolve()), "sha256": _sha(path)}
    plan = [{
        "family": family, "task": task, "seed": seed,
    } for family, tasks in preflight["development_tasks"].items()
      for task in tasks for seed in preflight["seeds"]]
    if len(plan) != preflight["coordinate_count"]:
        raise ValueError("round-two coordinate count changed")
    checkpoint_source = None
    if args.engine_checkpoint_source is not None:
        checkpoints = {}
        for item in plan:
            path = (args.engine_checkpoint_source / "children"
                    / item["family"] / item["task"]
                    / f"seed{item['seed']}" / "ENGINE_STAGE.json")
            if not path.is_file():
                continue
            stage = json.loads(path.read_text(encoding="utf-8"))
            if (stage.get("family"), stage.get("task"),
                    stage.get("seed")) != (
                    item["family"], item["task"], item["seed"]):
                raise ValueError("engine checkpoint coordinate changed")
            if (stage.get("provider_called") is not False
                    or stage.get("ground_truth_expression_opened") is not False
                    or stage.get("engine_executions_per_coordinate") != 1):
                raise ValueError("engine checkpoint boundary invalid")
            checkpoints[
                f"{item['family']}/{item['task']}/{item['seed']}"] = {
                    "path": str(path.resolve()), "sha256": _sha(path)}
        checkpoint_source = {
            "root": str(args.engine_checkpoint_source.resolve()),
            "checkpoints": checkpoints,
            "checkpoint_count": len(checkpoints),
            "reuse_scope": "engine-evidence-and-response-free-gap-only",
            "composition_and_admission_must_rerun": True,
        }
    result = {
        "schema": "formula-round2-source-first-freeze-v1",
        "method": "source-first-diagnose-compose-admit-generation-round-v1",
        "plan": plan,
        "coordinate_count": len(plan),
        "development_tasks": preflight["development_tasks"],
        "seeds": preflight["seeds"],
        "input_symbols_by_task": inputs,
        "metadata_files": metadata,
        "hdf5": {"path": str(args.hdf5.resolve()),
                 "sha256": _sha(args.hdf5)},
        "config": {"path": str(args.config.resolve()),
                   "sha256": _sha(args.config),
                   "values": config},
        "provider": {
            "env_path": str(args.provider_env.resolve()),
            "base_url": config["llm_api_url"],
            "model": config["llm_model"],
            "secret_value_frozen_or_published": False,
        },
        "engine_checkpoint_source": checkpoint_source,
        "preflight": {"path": str(args.preflight.resolve()),
                      "sha256": _sha(args.preflight)},
        "gap_routing_gate": {
            "path": str(args.gap_routing_gate.resolve()),
            "sha256": _sha(args.gap_routing_gate)},
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "execution": {
            "engine_executions_per_coordinate": 1,
            "engine_workers_within_coordinate": 4,
            "coordinate_workers": 1,
            "provider_concurrency": 1,
            "resume_completed_children_only": True,
            "resume_engine_checkpoint_after_transport_failure": True,
            "transport_continuation_may_not_rerun_engines": True,
        },
        "truth_boundary": {
            "all_child_banks_freeze_before_truth": True,
            "all_admissions_freeze_before_truth": True,
            "development_truth_only": True,
            "test_or_ood_closed": True,
            "untouched_confirmation_closed": True,
        },
        "efficacy_gate": {
            "primary_metric": "structural topology support recall",
            "gap_pre_minus_blind_pre_strictly_positive": True,
            "gap_pre_minus_multi_engine_strictly_positive": True,
            "minimum_positive_gap_minus_blind_families": 2,
            "all_coordinates_included": True,
            "failures_score_as_no_recovery": True,
            "admission_and_map_metrics_are_secondary": True
        },
        "ground_truth_expression_values_opened": False,
        "provider_called": False,
        "engine_called": False,
        "test_or_ood_accessed": False,
        "confirmation_accessed": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Fresh development Formula Generation round two. Truth may open "
            "only after all source-first child banks and independent "
            "admissions are frozen. Untouched confirmation remains closed."),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "FORMULA_ROUND2_SOURCE_FIRST_FREEZE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "coordinate_count": len(plan),
        "execution_authorized": True,
        "ground_truth_expression_values_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
