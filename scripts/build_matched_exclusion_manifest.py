"""Freeze every historically allocated task identity without reading responses."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.iterative_matrix_contract import (
    FAMILY_PATHS, digest, selected_tasks,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registrations-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("matched exclusion manifest output exists")
    registrations = args.registrations_root.resolve(strict=True)
    outputs = args.outputs_root.resolve(strict=True)
    hdf5 = args.hdf5.resolve(strict=True)
    excluded = {family: set() for family in FAMILY_PATHS}
    registration_files = {}
    parse_failures = []
    for path in sorted(registrations.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as error:
            parse_failures.append({
                "path": str(path),
                "error_type": type(error).__name__})
            continue
        used = {family: selected_tasks(value, family)
                for family in FAMILY_PATHS}
        if any(used.values()):
            registration_files[str(path)] = digest(path)
            for family in FAMILY_PATHS:
                excluded[family].update(used[family])
    if parse_failures:
        raise ValueError(
            f"historical registration parse failures: {parse_failures}")
    output_paths = {family: {} for family in FAMILY_PATHS}
    with h5py.File(hdf5, "r") as handle:
        known = {
            family: set(handle[path].keys())
            for family, path in FAMILY_PATHS.items()}
    for path in outputs.rglob("*"):
        if not path.is_dir():
            continue
        for family, tasks in known.items():
            if path.name in tasks:
                excluded[family].add(path.name)
                output_paths[family].setdefault(
                    path.name, []).append(str(path))
    result = {
        "schema": "matched-attribution-comprehensive-exclusion-v1",
        "selection_rule":
            "union of registration allocations and historical output task directory identities",
        "selected_tasks": {
            family: sorted(excluded[family] & known[family])
            for family in FAMILY_PATHS},
        "fresh_tasks": {
            family: sorted(known[family] - excluded[family])
            for family in FAMILY_PATHS},
        "registration_allocation_files": registration_files,
        "historical_output_task_paths": output_paths,
        "hdf5": {"path": str(hdf5), "sha256": digest(hdf5)},
        "ground_truth_expression_opened": False,
        "response_arrays_opened": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "execution_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "excluded_counts": {
            family: len(rows)
            for family, rows in result["selected_tasks"].items()},
        "fresh_counts": {
            family: len(rows)
            for family, rows in result["fresh_tasks"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
