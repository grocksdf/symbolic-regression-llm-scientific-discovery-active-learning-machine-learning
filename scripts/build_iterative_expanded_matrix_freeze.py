"""Freeze fresh train-only tasks and source for an iterative predictive matrix.

This reads task names, shapes and input-symbol columns, never response arrays
or formula truth.  Confirmation, test and OOD remain sealed.
"""

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
from scripts.iterative_matrix_contract import (  # noqa: E402
    CONDITION, FAMILY_PATHS, SCHEMA, digest, read, selected_tasks,
    source_identity, validate_config,
)


def _metadata_arguments(rows: list[str]) -> dict[str, Path]:
    files = {}
    for row in rows:
        family, sep, path = row.partition("=")
        if not sep or family not in FAMILY_PATHS or family in files:
            raise ValueError("--metadata requires one unique FAMILY=PARQUET")
        files[family] = Path(path).resolve(strict=True)
    return files


def _pick_tasks(hdf5: Path, families: list[str], excluded: dict,
                task_count: int):
    selected, dimensions = {}, {}
    with h5py.File(hdf5, "r") as handle:
        for family in families:
            group = handle[FAMILY_PATHS[family]]
            available = sorted((task for task in group.keys()
                                if task not in excluded[family]),
                               key=lambda task: sha256(
                                   f"iterative-expanded-matrix-v1:{family}:{task}"
                                   .encode()).digest())
            if len(available) < task_count:
                raise ValueError(f"insufficient fresh tasks in {family}")
            selected[family] = available[:task_count]
            for task in selected[family]:
                dimensions[f"{family}/{task}"] = int(
                    group[task]["train"].shape[1]) - 1
    return selected, dimensions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mainline-root", type=Path, required=True)
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--provider-env", type=Path, required=True)
    p.add_argument("--provider-gate", type=Path, required=True)
    p.add_argument("--correctness-gate", type=Path, required=True)
    p.add_argument("--exclude-freeze", type=Path, action="append", required=True)
    p.add_argument("--metadata", action="append", required=True,
                   help="FAMILY=PARQUET; only names and variable columns are read")
    p.add_argument("--family", action="append", choices=tuple(FAMILY_PATHS),
                   required=True)
    p.add_argument("--tasks-per-family", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[901, 902])
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output_dir.exists():
        raise ValueError("iterative matrix freeze output already exists")
    if (len(set(a.family)) != len(a.family)
            or a.tasks_per_family < 1
            or not a.seeds or len(a.seeds) != len(set(a.seeds))
            or any(seed < 0 for seed in a.seeds)):
        raise ValueError("invalid task/seed registration")
    metadata = _metadata_arguments(a.metadata)
    if set(metadata) != set(a.family):
        raise ValueError("exactly one metadata source required for every family")
    hdf5 = a.hdf5.resolve(strict=True)
    provider_env = a.provider_env.resolve(strict=True)
    provider_gate = a.provider_gate.resolve(strict=True)
    correctness_gate = a.correctness_gate.resolve(strict=True)
    transport = read(provider_gate)
    correctness = read(correctness_gate)
    if (transport.get("schema") != "response-free-provider-transport-preflight-v2"
            or transport.get("classification") != "transport_2xx"
            or transport.get("http_status") != 200
            or transport.get("openai_completion_contract_valid") is not True
            or transport.get("scientific_data_accessed") is not False
            or correctness.get("schema") != "iterative-matrix-response-free-gate-v1"
            or correctness.get("passed") is not True
            or correctness.get("benchmark_arrays_opened") is not False):
        raise ValueError("transport or response-free correctness Gate failed")
    config_path = ROOT / "configs/aistats_three_arm_formula_expanded_candidate.yaml"
    config = validate_config(config_path)
    if (transport.get("base_url") != config["llm_api_url"].rstrip("/")
            or transport.get("model") != config["llm_model"]
            or transport.get("provider_env_path") != str(provider_env)):
        raise ValueError("provider preflight targets another model or credential file")
    mainline_root = a.mainline_root.resolve(strict=True)
    benchmark_source, mainline_source = source_identity(ROOT, mainline_root)
    if (correctness.get("benchmark_source") != benchmark_source
            or correctness.get("mainline_source") != mainline_source
            or correctness.get("config_sha256") != digest(config_path)):
        raise ValueError("correctness Gate is not bound to current source/config")

    excluded = {family: set() for family in a.family}
    freeze_hashes = {}
    for raw in a.exclude_freeze:
        path = raw.resolve(strict=True)
        previous = read(path)
        for family in a.family:
            excluded[family].update(selected_tasks(previous, family))
        freeze_hashes[str(path)] = digest(path)
    if not any(excluded.values()):
        raise ValueError("exclusion freezes name no prior tasks")
    selected, dimensions = _pick_tasks(hdf5, a.family, excluded,
                                       a.tasks_per_family)
    rows_by_family, symbols = {}, {}
    columns = ["name", "symbols", "symbol_descs", "symbol_properties"]
    for family in a.family:
        table = pq.read_table(metadata[family], columns=columns)
        names = list(map(str, table.column("name").to_pylist()))
        if any(names.count(task) != 1 for task in selected[family]):
            raise ValueError("selected task metadata missing or duplicated")
        rows_by_family[family] = table.take(pa.array(
            [names.index(task) for task in selected[family]]))
        for task in selected[family]:
            values = list(map(str, table.column("symbols")[names.index(task)].as_py()))
            size = dimensions[f"{family}/{task}"]
            if len(values) != size + 1 or len(set(values[1:])) != size:
                raise ValueError("input-symbol mapping does not match task features")
            symbols[f"{family}/{task}"] = values[1:]

    a.output_dir.mkdir(parents=True, exist_ok=False)
    sliced_metadata = {}
    for family, table in rows_by_family.items():
        path = a.output_dir / f"{family}_symbols_only.parquet"
        pq.write_table(table, path)
        sliced_metadata[family] = {"path": str(path.resolve()),
                                   "sha256": digest(path)}
    tracked = [
        ROOT / "scripts/iterative_matrix_contract.py",
        ROOT / "scripts/build_iterative_expanded_matrix_freeze.py",
        ROOT / "scripts/run_iterative_expanded_matrix.py",
        ROOT / "scripts/run_aistats_three_arm_formula_child.py",
        ROOT / "scripts/audit_iterative_bank_reporting.py",
        ROOT / "scripts/provider_transport_preflight.py",
        ROOT / "scripts/check_iterative_matrix_correctness.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_searcher.py",
        config_path, provider_gate, correctness_gate, *metadata.values(),
    ]
    contract = {
        "schema": SCHEMA,
        "condition": CONDITION,
        "selected_tasks": selected,
        "excluded_tasks": {k: sorted(v) for k, v in excluded.items()},
        "exclusion_freezes": freeze_hashes,
        "selection_rule": "SHA-256 sorted fresh task names after supplied exclusions",
        "seeds": a.seeds,
        "pair_count": sum(map(len, selected.values())) * len(a.seeds),
        "engine_schedule": {key: config["agent_config"][key]
                            for key in ("engines", "engine_budget",
                                        "engine_repeats", "engine_workers",
                                        "engine_timeout_s", "engine_retries", "cycles")},
        "candidate_search_budget": {key: config["agent_config"][key]
                                    for key in ("discovery_budget", "discovery_rounds",
                                                "candidates_per_island", "new_skeleton_quota",
                                                "llm_evaluation_reserve",
                                                "synthesis_evaluation_reserve")},
        "bank_contract": {
            "engine_jobs_shared_by_construction": True,
            "initial_data_action_domain_seed_shared": True,
            "full_candidate_attempt_cap_frozen": True,
            "matched_non_llm_skeleton_attempts": False,
            "formula_recovery_attribution_authorized": False,
            "full_no_llm_frozen_engine_bank_predictive_contrast": True,
        },
        "primary_metric": "task_mean_full_minus_no_llm_log_score_on_shared_train_reporting",
        "decision": {"strictly_positive_task_mean_log_score_gain": True,
                     "zero_missing_pairs": True},
        "claim_boundary": ("Fresh-task iterative bank-conditional predictive contrast; "
                           "no attributable formula recovery, common-class "
                           "decision, measured acquisition, test/OOD or "
                           "untouched confirmation claim."),
        "input_symbols_by_task": symbols,
        "development_metadata_files": sliced_metadata,
        "hdf5": {"path": str(hdf5), "sha256": digest(hdf5),
                 "size_bytes": hdf5.stat().st_size},
        "provider_env": str(provider_env),
        "provider_env_sha256": digest(provider_env),
        "interpreter": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "dependency_versions": correctness["dependencies"],
        "mainline_root": str(mainline_root),
        "benchmark_source": benchmark_source,
        "mainline_source": mainline_source,
        "config_sha256": digest(config_path),
        "files": {str(path.resolve()): digest(path) for path in tracked},
        "gates": {"provider": str(provider_gate),
                  "correctness": str(correctness_gate)},
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": True,
    }
    path = a.output_dir / "ITERATIVE_MATRIX_FREEZE.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(str(path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
