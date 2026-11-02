"""Materialize the hash-selected PMLB confirmation suite without analysis."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pmlb import fetch_data


def digest(path: Path):
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("PMLB confirmation data output already exists")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    tasks = [str(value) for value in preflight["selected_tasks"]]
    if len(tasks) != 8 or len(set(tasks)) != 8:
        raise ValueError("expected eight unique hash-selected tasks")
    args.output_dir.mkdir(parents=True)
    hdf5 = args.output_dir / "pmlb_confirmation_v1.hdf5"
    metadata = args.output_dir / "pmlb_symbols_only.parquet"
    manifest = []
    metadata_rows = []
    with h5py.File(hdf5, "w") as handle:
        base = handle.require_group("lsr_synth/matsci")
        for task in tasks:
            X, y = fetch_data(
                task, return_X_y=True,
                local_cache_dir=str(args.cache_dir))
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float).reshape(-1)
            if (X.ndim != 2 or len(X) != len(y) or len(X) < 160
                    or not 1 <= X.shape[1] <= 8
                    or not np.all(np.isfinite(X))
                    or not np.all(np.isfinite(y))):
                raise ValueError(f"selected PMLB task became infeasible: {task}")
            raw_count = int(len(X))
            complete = np.column_stack((y, X))
            complete = np.unique(complete, axis=0)
            y = complete[:, 0]
            X = complete[:, 1:]
            if len(X) < 160:
                raise ValueError(
                    f"selected PMLB task has fewer than 160 unique rows: {task}")
            # Covariate-only scaling is frozen before any role responses are
            # opened. The target is copied unchanged and is never summarized.
            mean = np.mean(X, axis=0)
            scale = np.std(X, axis=0)
            scale = np.where(scale > 0, scale, 1.0)
            X = (X - mean) / scale
            values = np.column_stack((y, X))
            group = base.require_group(task)
            group.create_dataset(
                "train", data=values, compression="gzip",
                compression_opts=4, shuffle=True)
            manifest.append({
                "task": task, "raw_row_count": raw_count,
                "row_count": int(len(X)),
                "duplicate_complete_rows_removed": raw_count - int(len(X)),
                "feature_count": int(X.shape[1]),
                "covariate_scaling": "global-covariate-only-zscore",
                "target_transformation": "none"})
            symbols = ["y", *[f"x{i}" for i in range(X.shape[1])]]
            metadata_rows.append({
                "name": task,
                "symbols": symbols,
                "symbol_descs": [
                    "measured PMLB regression response",
                    *[f"standardized registered covariate {i}"
                      for i in range(X.shape[1])]],
                "symbol_properties": [
                    "target", *(["dimensionless-covariate"] * X.shape[1])],
            })
    pq.write_table(pa.Table.from_pylist(metadata_rows), metadata)
    result = {
        "schema": "pmlb-external-confirmation-data-v1",
        "selection_preflight": str(args.preflight.resolve()),
        "selection_preflight_sha256": digest(args.preflight),
        "tasks": tasks,
        "manifest": manifest,
        "hdf5": str(hdf5.resolve()),
        "hdf5_sha256": digest(hdf5),
        "metadata": str(metadata.resolve()),
        "metadata_sha256": digest(metadata),
        "response_statistics_computed": False,
        "response_values_published": False,
        "response_values_serialized_not_analyzed": True,
        "model_or_method_results_accessed": False,
        "test_or_ood_accessed": False,
    }
    (args.output_dir / "DATA_MANIFEST.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "tasks": tasks, "hdf5_sha256": result["hdf5_sha256"],
        "metadata_sha256": result["metadata_sha256"],
        "response_statistics_computed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
