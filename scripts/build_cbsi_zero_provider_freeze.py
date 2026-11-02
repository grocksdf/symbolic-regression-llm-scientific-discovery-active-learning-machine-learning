"""Build a response-free CB-SI zero-provider development freeze."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MAINLINE = WORKSPACE / "hypothesis_mvp"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        encoding="utf-8", errors="strict", stdout=subprocess.PIPE
    ).stdout.strip()


def source(root: Path) -> dict:
    return {
        "root": str(root.resolve()),
        "commit": git(root, "rev-parse", "HEAD"),
        "tree": git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(git(
            root, "status", "--short", "--untracked-files=no")),
    }


def write_new(path: Path, value) -> None:
    if path.exists():
        raise ValueError("CB-SI freeze already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args(argv)
    if args.task_count < 4:
        raise ValueError("CB-SI development freeze needs at least four tasks")
    parent = read(args.source_freeze)
    if parent.get("schema") != "stacking-gain-attribution-freeze-v1":
        raise ValueError("unexpected frozen candidate-pool source")
    mainline_source = source(MAINLINE)
    benchmark_source = source(ROOT)
    if mainline_source["tracked_dirty"] or benchmark_source["tracked_dirty"]:
        raise ValueError("CB-SI freeze requires clean tracked source")

    by_task = {}
    for row in parent["coordinates"]:
        if row.get("coordinate_failure"):
            continue
        by_task.setdefault(str(row["task"]), []).append(row)
    selected = []
    task_order = sorted(
        by_task,
        key=lambda task: sha256(
            f"cbsi-v1:{args.seed}:{task}".encode()).hexdigest())
    for task in task_order[:args.task_count]:
        coordinates = sorted(
            by_task[task],
            key=lambda row: sha256(json.dumps(
                {key: row[key] for key in ("family", "task", "seed")},
                sort_keys=True).encode()).hexdigest())
        selected.append(dict(coordinates[0]))
    if len(selected) != args.task_count:
        raise ValueError("insufficient frozen development tasks")

    code_paths = [
        MAINLINE / "hypothesis_mvp/discovery/"
        "counterfactual_bank_symbolic_invention.py",
        MAINLINE / "hypothesis_mvp/discovery/equation_runtime.py",
        MAINLINE / "hypothesis_mvp/discovery/"
        "joint_evidence_gated_expansion.py",
        ROOT / "scripts/run_cbsi_zero_provider_experiment.py",
        ROOT / "scripts/stacking_diagnostics.py",
    ]
    missing = [str(path) for path in code_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing CB-SI source: " + ", ".join(missing))

    artifact_manifest = []
    for coordinate in selected:
        root = Path(coordinate["coordinate_root"])
        paths = [root / "SHARED_BLOCK.json"]
        paths.extend(
            root / condition / "GENERATION_RESULT.json"
            for condition in parent["candidate_conditions"])
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            artifact_manifest.append({
                "path": str(path.resolve()), "sha256": digest(path)})

    freeze = {
        "schema": "cbsi-zero-provider-development-freeze-v1",
        "created_date": "2026-11-01",
        "execution_authorized": True,
        "provider_calls_authorized": False,
        "expensive_engine_runs_authorized": False,
        "post_reporting_development_only": True,
        "independent_confirmation_claim_authorized": False,
        "source_freeze": str(args.source_freeze.resolve()),
        "source_freeze_sha256": digest(args.source_freeze),
        "source_output": parent["source_output"],
        "hdf5": parent["hdf5"],
        "hdf5_sha256": parent["hdf5_sha256"],
        "candidate_conditions": list(parent["candidate_conditions"]),
        "coordinates": selected,
        "coordinate_count": len(selected),
        "selection_rule": (
            "SHA-256 order over task names, then SHA-256 order over frozen "
            "successful coordinates; no outcomes or formula truth used"),
        "arms": [
            "blind_random",
            "single_fitness_proxy",
            "residual_proxy",
            "influence_proxy",
            "bank_greedy",
            "standard_thompson",
            "cbsi_full",
        ],
        "operators": [
            "subtree_replacement",
            "subtree_composition",
            "operator_substitution",
            "factor_or_expand",
            "interaction_insertion",
            "correction_term",
            "structural_simplification",
        ],
        "budget": {
            "edit_evaluations_per_arm": 10,
            "matched_block_size": 2,
            "selected_candidates_per_arm": 1,
            "provider_calls": 0,
            "pysr_calls": 0,
        },
        "roles": {
            "fit": "discovery_development",
            "crossfit_credit": [
                "discovery_validation", "gap_audit"],
            "independent_development_evaluation": "gap_admission",
            "forbidden": [
                "reporting", "test", "ood", "untouched_confirmation"],
        },
        "analysis": {
            "bootstrap_seed": args.seed,
            "bootstrap_resamples": 5000,
            "primary_contrast": (
                "cbsi_full minus strongest non-CBSI arm on independent "
                "development evaluation"),
            "mechanism_gate": (
                "cbsi_full must exceed single_fitness_proxy and "
                "residual_proxy in task-mean evaluation gain"),
            "cost_gate": (
                "all arms have identical edit-evaluation budget and zero "
                "provider/PySR calls"),
        },
        "artifact_manifest": artifact_manifest,
        "file_hashes": {
            str(path.resolve()): digest(path) for path in code_paths},
        "source": {
            "mainline": mainline_source,
            "benchmark": benchmark_source,
        },
        "claim_boundary": (
            "Provider-free post-reporting development mechanism experiment. "
            "RESTART/IGSR arms are explicitly simplified proxies. It cannot "
            "establish independent efficacy, formula truth recovery, or "
            "AISTATS-ready novelty."),
    }
    write_new(args.output, freeze)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "coordinate_count": len(selected),
        "tasks": [row["task"] for row in selected],
        "provider_calls_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

