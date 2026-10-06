"""Freeze fresh tasks and complete execution identity for matched attribution."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import h5py
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.iterative_matrix_contract import (
    FAMILY_PATHS, digest, read, source_identity, validate_config,
)
from scripts.effective_structural_contract import validate_registration


SCHEMA = "matched-structural-attribution-freeze-v1"


def _metadata(rows, families):
    result = {}
    for row in rows:
        family, separator, path = row.partition("=")
        if not separator or family not in families or family in result:
            raise ValueError("matched metadata requires unique FAMILY=PARQUET")
        result[family] = Path(path).resolve(strict=True)
    if set(result) != set(families):
        raise ValueError("matched metadata missing registered family")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--provider-gate", type=Path, required=True)
    parser.add_argument("--generator-gate", type=Path, required=True)
    parser.add_argument("--composition-gate", type=Path, required=True)
    parser.add_argument("--correctness-gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--metadata", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("matched attribution freeze output exists")
    mainline = args.mainline_root.resolve(strict=True)
    hdf5 = args.hdf5.resolve(strict=True)
    provider_env = args.provider_env.resolve(strict=True)
    provider_gate = args.provider_gate.resolve(strict=True)
    generator_gate = args.generator_gate.resolve(strict=True)
    composition_gate = args.composition_gate.resolve(strict=True)
    correctness_gate = args.correctness_gate.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    exclusion_path = args.exclusion_manifest.resolve(strict=True)
    provider = read(provider_gate)
    generator = read(generator_gate)
    composition = read(composition_gate)
    correctness = read(correctness_gate)
    protocol = read(protocol_path)
    families = tuple(protocol["data_design"]["families"])
    tasks_per_family = int(protocol["data_design"]["tasks_per_family"])
    seeds = [int(value) for value in protocol["data_design"]["seeds"]]
    namespace = str(
        protocol["data_design"]["task_selection_namespace"])
    if (not families or len(set(families)) != len(families)
            or any(family not in FAMILY_PATHS for family in families)
            or tasks_per_family < 1 or not seeds
            or len(seeds) != len(set(seeds)) or not namespace):
        raise ValueError("matched protocol family capacity invalid")
    metadata = _metadata(args.metadata, families)
    exclusions = read(exclusion_path)
    config_path = ROOT / (
        "configs/aistats_three_arm_formula_expanded_candidate.yaml")
    config = validate_config(config_path)
    benchmark_source, mainline_source = source_identity(ROOT, mainline)
    s_policy, s_contract = validate_registration(
        protocol, (generator, composition, correctness),
        benchmark_source, mainline_source)
    if (provider.get("classification") != "transport_2xx"
            or provider.get("http_status") != 200
            or provider.get("openai_completion_contract_valid") is not True
            or provider.get("model") != config["llm_model"]
            or provider.get("provider_env_path") != str(provider_env)
            or generator.get("passed") is not True
            or composition.get("passed") is not True
            or correctness.get("passed") is not True
            or correctness.get("benchmark_source") != benchmark_source
            or correctness.get("mainline_source") != mainline_source
            or correctness.get("config_sha256") != digest(config_path)
            or protocol.get("execution_authorized") is not False):
        raise ValueError("matched prerequisite identity failed")
    selected, dimensions = {}, {}
    excluded = {
        family: set(exclusions["selected_tasks"][family])
        for family in families}
    with h5py.File(hdf5, "r") as handle:
        for family in families:
            group = handle[FAMILY_PATHS[family]]
            available = sorted(
                (task for task in group.keys()
                 if task not in excluded[family]),
                key=lambda task: sha256(
                    f"{namespace}:{family}:{task}".encode()).digest())
            if len(available) < tasks_per_family:
                raise ValueError(
                    f"insufficient fresh matched tasks in {family}")
            selected[family] = available[:tasks_per_family]
            for task in selected[family]:
                dimensions[f"{family}/{task}"] = int(
                    group[task]["train"].shape[1]) - 1
    columns = ["name", "symbols", "symbol_descs", "symbol_properties"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sliced, symbols = {}, {}
    for family in families:
        table = pq.read_table(metadata[family], columns=columns)
        names = list(map(str, table.column("name").to_pylist()))
        indices = [names.index(task) for task in selected[family]]
        subset = table.take(pa.array(indices))
        path = args.output_dir / f"{family}_symbols_only.parquet"
        pq.write_table(subset, path)
        sliced[family] = {"path": str(path.resolve()),
                          "sha256": digest(path)}
        for task, index in zip(selected[family], indices, strict=True):
            values = list(map(str, table.column("symbols")[index].as_py()))
            if (len(values) != dimensions[f"{family}/{task}"] + 1
                    or len(set(values[1:])) != len(values) - 1):
                raise ValueError("matched input symbol mapping changed")
            symbols[f"{family}/{task}"] = values[1:]
    tracked = [
        ROOT / "scripts/matched_gap_structural_generator.py",
        ROOT / "scripts/effective_structural_generator.py",
        ROOT / "scripts/effective_structural_contract.py",
        ROOT / "scripts/matched_three_arm_projection.py",
        ROOT / "scripts/run_matched_s_branch.py",
        ROOT / "scripts/audit_matched_three_arm_reporting.py",
        ROOT / "scripts/run_matched_attribution_matrix.py",
        ROOT / "scripts/run_aistats_three_arm_formula_child.py",
        ROOT / "scripts/iterative_matrix_contract.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_searcher.py",
        config_path, protocol_path, exclusion_path,
        provider_gate, generator_gate, composition_gate, correctness_gate,
        *metadata.values(),
    ]
    freeze = {
        "schema": SCHEMA,
        "selected_tasks": selected,
        "registered_families": list(families),
        "tasks_per_family": tasks_per_family,
        "excluded_tasks": {
            family: sorted(excluded[family]) for family in families},
        "task_selection_rule":
            f"SHA-256 sorted {namespace}:{{family}}:{{task}}",
        "seeds": seeds,
        "coordinate_count": (
            len(families) * tasks_per_family * len(seeds)),
        "arms": ["E", "E_plus_S_gap_matched", "E_plus_L_gap"],
        "matched_non_llm_skeleton_attempts": True,
        "s_generator_policy": s_policy,
        "s_generator_contract": s_contract,
        "historical_no_go_preserved": protocol.get("historical_no_go_preserved", True),
        "formula_recovery_attribution_authorized": False,
        "measured_action_authorized": False,
        "durable_resume_authorized": bool(
            protocol.get("durable_interruption_recovery", {}).get(
                "authorized")),
        "maximum_recovery_attempts_per_coordinate": int(
            protocol.get("durable_interruption_recovery", {}).get(
                "maximum_recovery_attempts_per_coordinate", 0)),
        "input_symbols_by_task": symbols,
        "development_metadata_files": sliced,
        "hdf5": {"path": str(hdf5), "sha256": digest(hdf5)},
        "provider_env": str(provider_env),
        "provider_env_sha256": digest(provider_env),
        "mainline_root": str(mainline),
        "benchmark_source": benchmark_source,
        "mainline_source": mainline_source,
        "config_sha256": digest(config_path),
        "interpreter": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "dependency_versions": correctness["dependencies"],
        "julia_runtime": correctness["julia_runtime"],
        "files": {
            str(path.resolve()): digest(path) for path in tracked},
        "exclusion_sources": {
            str(exclusion_path): digest(exclusion_path)},
        "gates": {
            "provider": str(provider_gate),
            "generator": str(generator_gate),
            "composition": str(composition_gate),
            "end_to_end": str(correctness_gate),
        },
        "primary_estimand":
            "task_mean_L_gap_minus_S_gap_matched_reporting_log_score",
        "primary_pass_conditions": {
            "all_coordinates_included": True,
            "task_mean_L_minus_S_strictly_positive": True,
            "task_mean_L_minus_E_strictly_positive": True,
            "at_least_one_verified_L_feedback_pair": True,
        },
        "ground_truth_expression_opened": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "execution_authorized": True,
        "claim_boundary": protocol["claim_boundary"],
    }
    path = args.output_dir / "MATCHED_ATTRIBUTION_FREEZE.json"
    path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "path": str(path.resolve()), "selected_tasks": selected,
        "coordinate_count": freeze["coordinate_count"],
        "execution_authorized": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
