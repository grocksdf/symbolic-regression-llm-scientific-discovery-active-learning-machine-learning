#!/usr/bin/env python3
"""Run frozen-budget RD1.1 real-only development and paired ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_real_discovery.py"
DEFAULT_CONTRACT = PROJECT_ROOT / "contracts" / "rd1_1_real_development_contract.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def _seeds(value: str, defaults: Sequence[int]) -> tuple[int, ...]:
    if not value.strip():
        return tuple(int(seed) for seed in defaults)
    return tuple(dict.fromkeys(int(item) for item in _csv_values(value)))


def _contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "rd1.1-real-development-contract-v1":
        raise ValueError("unsupported RD1.1 contract schema")
    if payload.get("heldout_policy", {}).get("opened") is not False:
        raise ValueError("RD1.1 contract does not seal held-out")
    return payload


def _confirmed_library(library: Path | None, manifest: Path | None) -> dict[str, Any] | None:
    if library is None and manifest is None:
        return None
    if library is None or manifest is None or not library.is_file() or not manifest.is_file():
        raise ValueError("confirmed knowledge requires both library and manifest files")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != "confirmed-source-knowledge-v1":
        raise ValueError("unsupported confirmed knowledge manifest")
    if payload.get("source_only") is not True or payload.get("contains_target_heldout_rows") is not False:
        raise ValueError("knowledge manifest does not establish source-only isolation")
    confirmations = payload.get("independent_confirmations")
    if not isinstance(confirmations, list) or not confirmations:
        raise ValueError("knowledge manifest contains no independent confirmations")
    if not all(isinstance(row, Mapping) and row.get("passed") is True for row in confirmations):
        raise ValueError("knowledge manifest contains a non-passing confirmation")
    sources = payload.get("source_datasets")
    if not isinstance(sources, list) or not all(isinstance(value, str) for value in sources):
        raise ValueError("knowledge manifest does not enumerate source datasets")
    digest = _sha256_file(library)
    if digest != payload.get("library_sha256"):
        raise ValueError("confirmed knowledge library hash mismatch")
    if not any(line.strip() for line in library.read_text(encoding="utf-8").splitlines()):
        raise ValueError("confirmed knowledge library is empty")
    return {"library": library, "manifest": payload, "sha256": digest}


def _variants(phase: str, knowledge: Mapping[str, Any] | None) -> tuple[str, ...]:
    if phase == "development":
        return ("full",)
    values = ["full", "no_llm", "single_engine"]
    if knowledge is not None:
        values.extend(("knowledge_on", "knowledge_off"))
    return tuple(values)


def _variant_args(name: str, budget: Mapping[str, Any]) -> list[str]:
    common = [
        "--engines", ",".join(budget["engines"]),
        "--engine-repeats", str(budget["engine_repeats"]),
        "--engine-budget", str(budget["engine_evaluation_budget_per_cycle"]),
        "--engine-workers", str(budget["engine_workers"]),
        "--engine-timeout-s", str(budget["engine_timeout_s"]),
        "--search-iterations", str(budget["search_iterations"]),
        "--discovery-budget", str(budget["discovery_evaluation_budget_per_cycle"]),
        "--cycles", str(budget["cycles"]),
    ]
    if name == "no_llm":
        return [*common, "--no-llm"]
    if name == "single_engine":
        index = common.index("--engines")
        common[index + 1] = "polynomial_lasso"
        index = common.index("--engine-repeats")
        common[index + 1] = str(budget["engine_jobs_per_cycle"])
    if name == "no_acquisition":
        raise ValueError("legacy acquisition ablation was removed; use canonical P3B")
    if name == "knowledge_on":
        common.append("--use-knowledge")
    return common


def _seed_knowledge(
    knowledge_dir: Path, task_name: str, confirmed: Mapping[str, Any] | None,
    variant: str,
) -> None:
    if variant not in {"knowledge_on", "knowledge_off"}:
        return
    if confirmed is None:
        raise RuntimeError("knowledge ablation requested without confirmed source knowledge")
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "knowledge-namespace-v1",
        "namespace": task_name.lower(),
        "task_name": task_name,
        "source_library_sha256": confirmed["sha256"],
        "source_confirmation_manifest": confirmed["manifest"],
    }
    (knowledge_dir / "knowledge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    shutil.copy2(confirmed["library"], knowledge_dir / "structure_library.jsonl")


def _command(
    *, dataset: str, seed: int, variant: str, data_root: Path,
    output: Path, config: Path, budget: Mapping[str, Any],
) -> list[str]:
    task_name = f"rd1_1_{dataset}_real_development"
    return [
        sys.executable, "-B", str(RUNNER),
        "--dataset-id", dataset, "--data-root", str(data_root),
        "--task-name", task_name,
        "--task-description", "discover a falsifiable mechanism from real measured observations",
        "--output-dir", str(output), "--knowledge-dir", str(output / "knowledge"),
        "--llm-config", str(config), "--seed", str(seed),
        *_variant_args(variant, budget),
    ]


def _summary_row(result_path: Path, dataset: str, seed: int, variant: str) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    report = result["discovery_report"]
    return {
        "dataset": dataset, "seed": seed, "variant": variant,
        "status": result.get("status"), "hypothesis_id": result.get("hypothesis_id"),
        "expression_sha256": hashlib.sha256(str(result.get("expression")).encode()).hexdigest(),
        "best_val_nmse": report.get("best_val_nmse"),
        "best_val_relative_error_p99": report.get("best_val_relative_error_p99"),
        "best_val_strict_max_relative_error": report.get("best_val_strict_max_relative_error"),
        "best_complexity": report.get("best_complexity"),
        "evaluation_budget_used": report.get("evaluation_budget_used"),
        "llm_call_count": report.get("llm_call_count"),
        "provider_attempt_count": report.get("provider_attempt_count"),
        "cycle_provider_calls": [row.get("provider_calls") for row in result.get("cycles", [])],
        "selection_used_heldout": result.get("selection_used_heldout"),
        "untouched_heldout_opened": result.get("untouched_heldout_opened"),
        "knowledge_stage_status": report.get("knowledge_stage_status"),
        "result_path": str(result_path),
    }


def _paired_effects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["dataset"], row["seed"], row["variant"]): row for row in rows}
    effects: list[dict[str, Any]] = []
    for (dataset, seed, variant), row in sorted(indexed.items()):
        if variant in {"full", "knowledge_on"}:
            continue
        reference_variant = "knowledge_on" if variant == "knowledge_off" else "full"
        reference = indexed.get((dataset, seed, reference_variant))
        if reference is None:
            continue
        effect = float(row["best_val_nmse"]) - float(reference["best_val_nmse"])
        effects.append({
            "dataset": dataset, "seed": seed, "ablation": variant,
            "reference_variant": reference_variant,
            "ablation_minus_reference_val_nmse": effect,
            "supports_full_on_primary_metric": effect > 0.0,
        })
    return effects


def _write_summary(path: Path, contract: Path, rows: Sequence[Mapping[str, Any]], blocked: Sequence[str]) -> None:
    payload = {
        "schema": "rd1.1-real-development-results-v1",
        "contract_sha256": _sha256_file(contract),
        "runs": list(rows), "paired_effects": _paired_effects(rows),
        "blocked_components": list(blocked),
        "heldout_opened": False, "selection_used_heldout": False,
        "claim_boundary": "development_and_ablation_only_not_independent_confirmation",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="RD1.1 real-only development and ablation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase", choices=("development", "ablation", "all"), default="development")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--llm-config", default=str(PROJECT_ROOT / "config" / "bigmodel_glm_5_2.json"))
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--confirmed-knowledge-library")
    parser.add_argument("--confirmed-knowledge-manifest")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    contract = _contract(contract_path)
    datasets = _csv_values(args.datasets) or tuple(contract["datasets"])
    unknown = sorted(set(datasets) - set(contract["datasets"]))
    if unknown:
        raise ValueError("datasets outside RD1.1 contract: " + ", ".join(unknown))
    default_seeds = (
        contract["development_seeds"]
        if args.phase == "development" else contract["paired_ablation_seeds"]
    )
    seeds = _seeds(args.seeds, default_seeds)
    confirmed = _confirmed_library(
        Path(args.confirmed_knowledge_library) if args.confirmed_knowledge_library else None,
        Path(args.confirmed_knowledge_manifest) if args.confirmed_knowledge_manifest else None,
    )
    base_variants = _variants(args.phase, confirmed)
    blocked = [] if confirmed is not None else [
        "knowledge_reuse_ablation:requires_independently_confirmed_source_only_library"
    ]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary_path = output_root / "rd1_1_summary.json"
    for dataset in datasets:
        variants = tuple(
            variant for variant in base_variants
            if variant not in {"knowledge_on", "knowledge_off"}
            or dataset not in set(confirmed["manifest"]["source_datasets"])
        )
        if confirmed is not None and len(variants) != len(base_variants):
            blocked.append(
                f"knowledge_reuse_ablation:{dataset}:target_is_present_in_source_datasets"
            )
        for seed in seeds:
            for variant in variants:
                output = output_root / dataset / f"seed_{seed}" / variant
                result_path = output / "discovery_result.json"
                if result_path.exists():
                    if not args.resume:
                        raise FileExistsError(f"result already exists: {result_path}")
                else:
                    _seed_knowledge(
                        output / "knowledge", f"rd1_1_{dataset}_real_development",
                        confirmed, variant,
                    )
                    command = _command(
                        dataset=dataset, seed=seed, variant=variant,
                        data_root=Path(args.data_root), output=output,
                        config=Path(args.llm_config), budget=contract["full_system_budget"],
                    )
                    completed = subprocess.run(command, check=False)
                    if completed.returncode != 0:
                        _write_summary(summary_path, contract_path, rows, blocked)
                        raise RuntimeError(
                            f"RD1.1 run failed: dataset={dataset} seed={seed} "
                            f"variant={variant} exit={completed.returncode}"
                        )
                row = _summary_row(result_path, dataset, seed, variant)
                if row["selection_used_heldout"] is not False or row["untouched_heldout_opened"] is not False:
                    raise RuntimeError("held-out isolation violation")
                rows.append(row)
                _write_summary(summary_path, contract_path, rows, blocked)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
