"""Real-measurement loading and role-preserving splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_pool: np.ndarray | None = None
    y_pool: np.ndarray | None = None
    X_heldout: np.ndarray | None = None
    y_heldout: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_frame(path: Path, delimiter: str, header: int | None) -> pd.DataFrame:
    readers = {
        ".csv": lambda: pd.read_csv(path, delimiter=delimiter, header=header),
        ".tsv": lambda: pd.read_csv(path, delimiter="\t", header=header),
        ".xlsx": lambda: pd.read_excel(path, header=header),
        ".xls": lambda: pd.read_excel(path, header=header),
    }
    if path.suffix.lower() not in readers:
        raise ValueError("supported real-data formats are csv, tsv, xlsx and xls")
    return readers[path.suffix.lower()]()


def _numeric_arrays(
    frame: pd.DataFrame,
    target: str,
    features: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if target not in frame.columns:
        raise ValueError(f"target column not found: {target}")
    names = tuple(features or (column for column in frame.columns if column != target))
    missing = [name for name in names if name not in frame.columns]
    if not names or missing:
        raise ValueError(f"invalid feature columns: {missing or 'empty selection'}")
    values = frame[list(names) + [target]].apply(pd.to_numeric, errors="coerce")
    values = values[np.isfinite(values.to_numpy(dtype=float)).all(axis=1)]
    if len(values) < 8:
        raise ValueError("at least eight finite measured rows are required")
    return (
        values[list(names)].to_numpy(dtype=float),
        values[target].to_numpy(dtype=float),
        names,
    )


def _selection_score(X: np.ndarray, strategy: str) -> np.ndarray:
    normalized = X - np.mean(X, axis=0, keepdims=True)
    scale = np.std(normalized, axis=0, keepdims=True)
    normalized /= np.where(scale > 0.0, scale, 1.0)
    if strategy == "feature0":
        return X[:, 0]
    if strategy == "pca1":
        _, _, vectors = np.linalg.svd(normalized, full_matrices=False)
        return np.abs(normalized @ vectors[0])
    if strategy == "mahalanobis":
        covariance = np.atleast_2d(np.cov(normalized, rowvar=False))
        inverse = np.linalg.pinv(covariance + 1.0e-6 * np.eye(X.shape[1]))
        return np.sum((normalized @ inverse) * normalized, axis=1)
    if strategy == "critical_point":
        order = np.argsort(X[:, 0])
        score = np.empty(len(X), dtype=float)
        score[order] = np.linspace(0.0, 1.0, len(X))
        return np.abs(score - 0.5)
    raise ValueError("split strategy must be pca1, mahalanobis, feature0 or critical_point")


def _split_indices(
    X: np.ndarray,
    train_ratio: float,
    pool_ratio: float,
    heldout_ratio: float,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < train_ratio < 1.0 or pool_ratio < 0.0 or heldout_ratio <= 0.0:
        raise ValueError("invalid train, acquisition-pool or untouched-heldout ratio")
    extra_ratio = pool_ratio + heldout_ratio
    if extra_ratio >= 0.8:
        raise ValueError("pool_ratio + heldout_ratio must be below 0.8")
    extra_count = min(max(2, int(round(len(X) * extra_ratio))), len(X) - 4)
    score = _selection_score(X, strategy)
    extra = np.argsort(score)[-extra_count:]
    remaining = np.setdiff1d(np.arange(len(X)), extra, assume_unique=True)
    rng = np.random.default_rng(seed)
    remaining, extra = rng.permutation(remaining), rng.permutation(extra)
    pool_count = int(round(extra_count * pool_ratio / extra_ratio))
    pool_count = min(max(pool_count, 0), extra_count - 1)
    train_count = min(max(2, int(round(len(remaining) * train_ratio))), len(remaining) - 2)
    return remaining[:train_count], remaining[train_count:], extra[:pool_count], extra[pool_count:]


def split_real_arrays(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_ratio: float,
    pool_ratio: float,
    heldout_ratio: float,
    strategy: str,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> DatasetBundle:
    features = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1)
    if features.ndim != 2 or len(features) != len(target):
        raise ValueError("real inputs and targets must be aligned")
    labelled = np.column_stack((features, target))
    _, first_indices = np.unique(labelled, axis=0, return_index=True)
    if len(first_indices) < len(labelled):
        keep = np.sort(first_indices)
        features, target = features[keep], target[keep]
        metadata = {
            **dict(metadata or {}),
            "exact_duplicate_labelled_rows_removed": int(len(labelled) - len(keep)),
        }
    train, validation, pool, heldout = _split_indices(
        features, train_ratio, pool_ratio, heldout_ratio, strategy, seed
    )
    return DatasetBundle(
        X_train=features[train], y_train=target[train],
        X_val=features[validation], y_val=target[validation],
        X_pool=features[pool] if len(pool) else None,
        y_pool=target[pool] if len(pool) else None,
        X_heldout=features[heldout], y_heldout=target[heldout],
        metadata=dict(metadata or {}),
    )


def load_real_data_from_file(
    file_path: str | Path,
    *,
    target_column: str,
    feature_columns: Sequence[str] | None = None,
    delimiter: str = ",",
    header: int | None = 0,
    train_ratio: float = 0.70,
    pool_ratio: float = 0.15,
    heldout_ratio: float = 0.15,
    split_strategy: str = "pca1",
    random_state: int = 42,
) -> DatasetBundle:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"real-data file not found: {path}")
    X, y, names = _numeric_arrays(
        _read_frame(path, delimiter, header), target_column, feature_columns
    )
    return split_real_arrays(
        X, y, train_ratio=train_ratio, pool_ratio=pool_ratio,
        heldout_ratio=heldout_ratio, strategy=split_strategy,
        seed=random_state,
        metadata={
            "source": str(path.resolve()),
            "feature_names": list(names),
            "target_name": target_column,
            "synthetic": False,
            "noise_added_by_loader": False,
        },
    )


__all__ = ["DatasetBundle", "load_real_data_from_file", "split_real_arrays"]
