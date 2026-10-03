"""Build the new prospective Formula Discovery v2 development freeze."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "hypothesis_mvp"))

from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from run_aistats_drr_benchmark import _sha


FILES = {
    "bio_pop_growth":
        "lsr_synth_bio_pop_growth-00000-of-00001.parquet",
    "chem_react": "lsr_synth_chem_react-00000-of-00001.parquet",
    "lsr_transform": "lsr_transform-00000-of-00001.parquet",
    "matsci": "lsr_synth_matsci-00000-of-00001.parquet",
    "phys_osc": "lsr_synth_phys_osc-00000-of-00001.parquet",
}
ACTIVE_FAMILIES = (
    "bio_pop_growth", "chem_react", "lsr_transform", "phys_osc")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _collect(value, output):
    if isinstance(value, dict):
        for field in (
                "selected_tasks", "development_tasks",
                "untouched_confirmation_tasks"):
            selected = value.get(field)
            if isinstance(selected, dict):
                for family, tasks in selected.items():
                    values = [tasks] if isinstance(tasks, str) else (
                        tasks if isinstance(tasks, list) else [])
                    output.setdefault(str(family), set()).update(
                        str(task) for task in values)
        for child in value.values():
            _collect(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect(child, output)


def _exclusions(roots):
    output = {}
    for path in sorted({
            path.resolve() for root in roots for path in root.rglob("*.json")}):
        try:
            _collect(_read(path), output)
        except Exception:
            continue
    return output


def _key(family, task):
    return sha256(
        f"expanded-formula-v2-fresh:{family}:{task}".encode()).digest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--exclusion-root", type=Path, action="append",
                        required=True)
    parser.add_argument("--previous-preflight", type=Path, required=True)
    parser.add_argument("--provider-gate", type=Path, required=True)
    parser.add_argument("--synthetic-gate", type=Path, required=True)
    parser.add_argument("--tiny-smoke", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("Formula Discovery v2 freeze output must be new")
    previous = _read(args.previous_preflight)
    transport = _read(args.provider_gate)
    synthetic = _read(args.synthetic_gate)
    smoke = _read(args.tiny_smoke)
    if (transport.get("classification") != "transport_2xx"
            or transport.get("http_status") != 200
            or synthetic.get("passed") is not True
            or smoke.get("passed") is not True
            or smoke.get(
                "materialized_novel_candidate_given_successful_llm_response_run",
                0.0) < 0.5):
        raise ValueError("Formula Discovery v2 prerequisite Gate failed")
    excluded = _exclusions(args.exclusion_root)
    _collect(previous, excluded)
    selected, input_symbols, source_metadata = {}, {}, {}
    args.output_dir.mkdir(parents=True)
    metadata_root = args.output_dir / "development_metadata"
    metadata_root.mkdir()
    inventory = {}
    for family, filename in FILES.items():
        source = args.data_root / "data" / filename
        table = pq.read_table(source)
        names = [str(value) for value in table.column("name").to_pylist()]
        fresh = sorted(
            (task for task in names
             if task not in excluded.get(family, set())),
            key=lambda task: _key(family, task))
        inventory[family] = {
            "source_total": len(names),
            "excluded_count": len(set(names) & excluded.get(family, set())),
            "fresh_remaining_count": len(fresh),
            "active_development_family": family in ACTIVE_FAMILIES,
        }
        if family not in ACTIVE_FAMILIES:
            if fresh:
                raise ValueError(
                    "inactive MatSci family unexpectedly has fresh tasks")
            continue
        if len(fresh) < 4:
            raise ValueError(f"insufficient new v2 tasks: {family}")
        tasks = fresh[:4]
        selected[family] = tasks
        # Construct the mask without exposing equation values in output/logs.
        mask = pc.is_in(
            table.column("name"),
            value_set=__import__("pyarrow").array(tasks))
        subset = table.filter(mask)
        if subset.num_rows != len(tasks):
            raise ValueError("development metadata subset identity failed")
        output = metadata_root / f"{family}.parquet"
        pq.write_table(subset, output)
        source_metadata[family] = str(output.resolve())
        subset_names = [
            str(value) for value in subset.column("name").to_pylist()]
        symbols = subset.column("symbols")
        for task in tasks:
            index = subset_names.index(task)
            values = [str(value) for value in symbols[index].as_py()]
            if len(values) < 2:
                raise ValueError("registered task has no input symbols")
            input_symbols[f"{family}/{task}"] = values[1:]
    confirmation = previous["untouched_confirmation_tasks"]
    files = [
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py",
        ROOT / "scripts/run_aistats_three_arm_formula_child.py",
        ROOT / "scripts/expanded_formula_admission_v2.py",
        ROOT / "scripts/formula_bank_materialization_v2.py",
        ROOT / "scripts/formula_recovery_contract.py",
        ROOT / "scripts/provider_health_contract.py",
        ROOT / "configs/aistats_three_arm_formula_prospective.yaml",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/"
            "expanded_formula_synthesis.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/"
            "formula_recovery.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/"
            "pcpi_adapter.py",
        args.previous_preflight, args.provider_gate,
        args.synthetic_gate, args.tiny_smoke,
        *map(Path, source_metadata.values()),
    ]
    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    count = sum(len(tasks) for tasks in selected.values())
    result = {
        "schema": "scientific-expanded-formula-discovery-freeze-v2",
        "method": "diagnose-compose-admit-v2",
        "development_tasks": selected,
        "untouched_confirmation_tasks": confirmation,
        "development_task_count": count,
        "confirmation_task_count": sum(
            len(tasks) for tasks in confirmation.values()),
        "seeds": [701, 702],
        "gap_generation_run_count": count * 2,
        "admission_arms": [
            "independent_protected", "same_data", "accept_all"],
        "admission_row_count": count * 2 * 3,
        "minimum_paired_bank_contrasts": 3,
        "llm_materialization_limit": 7,
        "llm_user_prompt_utf8_bytes": 65536,
        "wall_time_seconds_per_run": 900,
        "maximum_llm_calls_per_run": 12,
        "maximum_completion_tokens_per_call": 1800,
        "primary_endpoint":
            "posterior-map-structural-topology-recovery",
        "secondary_endpoints": [
            "literal-exact-recovery-when-applicable",
            "bank-structural-topology-recall",
            "bank-size", "provider-calls-and-token-use"],
        "metadata_scope": "physically-development-only-v1",
        "development_metadata_files": source_metadata,
        "input_symbols_by_task": input_symbols,
        "inventory": inventory,
        "hdf5": {"path": str(hdf5.resolve()), "sha256": _sha(hdf5),
                 "size_bytes": hdf5.stat().st_size},
        "provider_env": str(args.provider_env.resolve()),
        "mainline_root": str((ROOT.parent / "hypothesis_mvp").resolve()),
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(Path(path).resolve()): _sha(path) for path in files},
        "gates": {
            "provider_transport_sha256": _sha(args.provider_gate),
            "synthetic_materialization_sha256": _sha(args.synthetic_gate),
            "tiny_development_smoke_sha256": _sha(args.tiny_smoke),
        },
        "ground_truth_expression_values_printed": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "confirmation_results_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Fresh Formula Discovery v2 development experiment on entirely "
            "new tasks. One expanded Gap bank is projected through three "
            "admission policies. Ground truth opens only after a preregistered "
            "bank-contrast Gate. Test/OOD and prior untouched confirmation "
            "remain closed."),
    }
    path = args.output_dir / "EXPANDED_FORMULA_DISCOVERY_V2_FREEZE.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in {"input_symbols_by_task", "files"}
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
