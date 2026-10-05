"""Freeze fresh tasks and execution identity for proposal mechanism diagnostic v2."""
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


SCHEMA = "llm-proposal-mechanism-diagnostic-freeze-v2"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--provider-gate", type=Path, required=True)
    parser.add_argument("--correctness-gate", type=Path, required=True)
    parser.add_argument("--compiler-gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--metadata", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("proposal diagnostic freeze output exists")
    protocol_path = args.protocol.resolve(strict=True)
    protocol = read(protocol_path)
    families = tuple(protocol["data_design"]["families"])
    tasks_per_family = int(protocol["data_design"]["tasks_per_family"])
    seeds = [int(value) for value in protocol["data_design"]["seeds"]]
    namespace = protocol["data_design"]["task_selection_namespace"]
    metadata = {}
    for row in args.metadata:
        family, separator, path = row.partition("=")
        if not separator or family not in families or family in metadata:
            raise ValueError("invalid proposal diagnostic metadata")
        metadata[family] = Path(path).resolve(strict=True)
    if set(metadata) != set(families):
        raise ValueError("proposal diagnostic metadata missing")
    mainline = args.mainline_root.resolve(strict=True)
    hdf5 = args.hdf5.resolve(strict=True)
    provider_env = args.provider_env.resolve(strict=True)
    provider_gate = args.provider_gate.resolve(strict=True)
    correctness_gate = args.correctness_gate.resolve(strict=True)
    compiler_gate = args.compiler_gate.resolve(strict=True)
    exclusion = args.exclusion_manifest.resolve(strict=True)
    provider, correctness, compiler = (
        read(provider_gate), read(correctness_gate), read(compiler_gate))
    exclusions = read(exclusion)
    config_path = ROOT / (
        "configs/aistats_three_arm_formula_expanded_candidate.yaml")
    config = validate_config(config_path)
    benchmark_source, mainline_source = source_identity(ROOT, mainline)
    if (provider.get("classification") != "transport_2xx"
            or provider.get("http_status") != 200
            or provider.get("model") != config["llm_model"]
            or provider.get("provider_env_path") != str(provider_env)
            or correctness.get("passed") is not True
            or correctness.get("benchmark_source") != benchmark_source
            or correctness.get("mainline_source") != mainline_source
            or compiler.get("passed") is not True
            or compiler.get("source", {}).get("source_git_commit")
            != mainline_source["source_git_commit"]
            or protocol.get("execution_authorized") is not False):
        raise ValueError("proposal diagnostic prerequisite failed")
    excluded = {
        family: set(exclusions["selected_tasks"][family])
        for family in families}
    selected, dimensions = {}, {}
    with h5py.File(hdf5, "r") as handle:
        for family in families:
            group = handle[FAMILY_PATHS[family]]
            available = sorted(
                (task for task in group if task not in excluded[family]),
                key=lambda task: sha256(
                    f"{namespace}:{family}:{task}".encode()).digest())
            if len(available) < tasks_per_family:
                raise ValueError(
                    f"insufficient fresh proposal diagnostic tasks: {family}")
            selected[family] = available[:tasks_per_family]
            for task in selected[family]:
                dimensions[f"{family}/{task}"] = (
                    int(group[task]["train"].shape[1]) - 1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    columns = ["name", "symbols", "symbol_descs", "symbol_properties"]
    sliced, symbols = {}, {}
    for family in families:
        table = pq.read_table(metadata[family], columns=columns)
        names = list(map(str, table.column("name").to_pylist()))
        indices = [names.index(task) for task in selected[family]]
        path = args.output_dir / f"{family}_symbols_only.parquet"
        pq.write_table(table.take(pa.array(indices)), path)
        sliced[family] = {"path": str(path.resolve()),
                          "sha256": digest(path)}
        for task, index in zip(selected[family], indices, strict=True):
            values = list(map(str, table.column("symbols")[index].as_py()))
            if len(values) != dimensions[f"{family}/{task}"] + 1:
                raise ValueError("proposal diagnostic symbol mapping changed")
            symbols[f"{family}/{task}"] = values[1:]
    tracked = [
        ROOT / "scripts/run_llm_proposal_diagnostic_matrix.py",
        ROOT / "scripts/audit_llm_proposal_mechanisms.py",
        ROOT / "scripts/run_aistats_three_arm_formula_child.py",
        ROOT / "scripts/iterative_matrix_contract.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_searcher.py",
        config_path, protocol_path, exclusion, provider_gate,
        correctness_gate, compiler_gate, *metadata.values(),
    ]
    freeze = {
        "schema": SCHEMA,
        "selected_tasks": selected,
        "seeds": seeds,
        "coordinate_count": (
            len(families) * tasks_per_family * len(seeds)),
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
        "exclusion_sources": {str(exclusion): digest(exclusion)},
        "provider_gate": str(provider_gate),
        "gates": {
            "correctness": str(correctness_gate),
            "compiler": str(compiler_gate),
        },
        "durable_resume_authorized": True,
        "maximum_recovery_attempts_per_coordinate": 2,
        "predictive_superiority_claim_authorized": False,
        "formula_recovery_attribution_authorized": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "execution_authorized": True,
        "claim_boundary": protocol["claim_boundary"],
    }
    path = args.output_dir / "PROPOSAL_DIAGNOSTIC_FREEZE.json"
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
