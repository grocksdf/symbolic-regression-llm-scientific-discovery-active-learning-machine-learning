"""Build a response-free Expanded Formula Discovery task registration."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.expanded_formula_discovery_harness import ADMISSION_ARMS
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


FILES = {
    "bio_pop_growth":
        "lsr_synth_bio_pop_growth-00000-of-00001.parquet",
    "chem_react": "lsr_synth_chem_react-00000-of-00001.parquet",
    "matsci": "lsr_synth_matsci-00000-of-00001.parquet",
    "phys_osc": "lsr_synth_phys_osc-00000-of-00001.parquet",
    "lsr_transform": "lsr_transform-00000-of-00001.parquet",
}


def _sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_selected(value, output):
    if isinstance(value, dict):
        selected = value.get("selected_tasks")
        if isinstance(selected, dict):
            for family, tasks in selected.items():
                values = [tasks] if isinstance(tasks, str) else (
                    tasks if isinstance(tasks, list) else [])
                output.setdefault(str(family), set()).update(
                    str(task) for task in values)
        for child in value.values():
            _collect_selected(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_selected(child, output)


def _registered_tasks(roots):
    selected, files = {}, {}
    paths = sorted({
        path.resolve() for root in roots for path in root.rglob("*.json")
    })
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        before = sum(len(value) for value in selected.values())
        _collect_selected(payload, selected)
        after = sum(len(value) for value in selected.values())
        if after > before:
            files[str(path.resolve())] = _sha(path)
    return selected, files


def _key(namespace, family, task):
    return sha256(f"{namespace}:{family}:{task}".encode()).digest()


def _metadata(path):
    parquet = pq.ParquetFile(path)
    names = parquet.schema.names
    if "name" not in names or "expression" not in names:
        raise ValueError("equation metadata schema is incomplete")
    tasks = [str(value) for value in
             pq.read_table(path, columns=["name"]).column("name").to_pylist()]
    expression_index = names.index("expression")
    null_counts = []
    for group_index in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(group_index).column(
            expression_index).statistics
        null_counts.append(None if stats is None else stats.null_count)
    return tasks, {
        "row_count": parquet.metadata.num_rows,
        "columns": names,
        "ground_truth_expression_column_registered": True,
        "ground_truth_values_opened": False,
        "expression_null_count_from_file_metadata": (
            None if any(value is None for value in null_counts)
            else int(sum(null_counts))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exclusion-root", type=Path, action="append",
                        required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-per-family", type=int, default=4)
    parser.add_argument("--confirmation-per-family", type=int, default=2)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("formula-discovery preflight output must be new")
    if args.development_per_family < 1 or args.confirmation_per_family < 1:
        raise ValueError("positive task counts required")
    used, exclusion_files = _registered_tasks(args.exclusion_root)
    inventory, development, confirmation = {}, {}, {}
    metadata_files = {}
    for family, filename in FILES.items():
        path = args.data_root / "data" / filename
        tasks, audit = _metadata(path)
        excluded = set(used.get(family, ()))
        fresh = sorted(
            (task for task in tasks if task not in excluded),
            key=lambda task: _key(
                "expanded-formula-discovery-v1", family, task))
        development_count = min(
            args.development_per_family,
            len(fresh) - args.confirmation_per_family)
        if development_count < 2:
            raise ValueError(f"insufficient fresh equation tasks: {family}")
        needed = development_count + args.confirmation_per_family
        development[family] = fresh[:development_count]
        confirmation[family] = fresh[
            development_count:needed]
        inventory[family] = {
            **audit, "registered_before_count": len(excluded & set(tasks)),
            "fresh_available_count": len(fresh),
            "requested_development_count":
                args.development_per_family,
            "development_count": len(development[family]),
            "confirmation_count": len(confirmation[family]),
        }
        metadata_files[str(path.resolve())] = _sha(path)
    result = {
        "schema":
            "scientific-expanded-formula-discovery-preflight-v1",
        "method": "diagnose-compose-admit-v1",
        "admission_arms": list(ADMISSION_ARMS),
        "admission_definitions": {
            "independent_protected": (
                "complete frozen engine/core bank plus LLM candidates that "
                "pass candidatewise fold-safe Bayesian log-score admission "
                "on a gap-disjoint admission split and a second disjoint "
                "selector-update split"),
            "same_data": (
                "the identical frozen engine/core and LLM candidate bank "
                "screened by the identical rule while reusing the gap-"
                "diagnosis responses for both admission looks"),
            "accept_all": (
                "the identical frozen engine/core bank plus every structurally "
                "valid, support-novel LLM candidate; no admission response is "
                "accessed"),
        },
        "candidate_bank_contract": {
            "one_gap_generated_bank_per_task_seed": True,
            "same_frozen_candidates_across_admission_arms": True,
            "complete_engine_core_protected": True,
            "llm_has_generative_authority": True,
            "llm_has_epistemic_authority": False,
        },
        "evaluators": {
            "exact_formula_recovery": (
                "after mapping registered ordered input symbols to x0..xd-1, "
                "SymPy together/cancel and simplify must prove candidate minus "
                "ground truth equals exactly zero; unresolved is false"),
            "structural_formula_recovery": (
                "ordered-symbol-normalized operator/variable topology must "
                "match after abstracting fitted multiplicative amplitudes; "
                "function identities and numeric exponents remain structural"),
            "ground_truth_access": (
                "development ground truth is opened only by an evaluator after "
                "all candidate banks and admissions are frozen; confirmation "
                "ground truth remains unopened until one-time confirmation"),
        },
        "development_tasks": development,
        "untouched_confirmation_tasks": confirmation,
        "inventory": inventory,
        "metadata_files": metadata_files,
        "exclusion_registration_files": exclusion_files,
        "benchmark_source": verify_clean_git_source(ROOT),
        "statistical_core_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "candidate_responses_accessed": False,
        "ground_truth_expression_values_opened": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "confirmation_results_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Task-identity, evaluator, admission-arm and leakage preflight "
            "only. No candidate generation, equation value evaluation, "
            "efficacy, test/OOD or confirmation access."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "EXPANDED_FORMULA_DISCOVERY_PREFLIGHT.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
