"""Label access restricted to an explicitly measured acquisition pool."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoolOracle:
    X_pool: np.ndarray
    y_pool: np.ndarray

    def __post_init__(self) -> None:
        X = np.asarray(self.X_pool, dtype=float)
        y = np.asarray(self.y_pool, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or not len(X) or len(X) != len(y):
            raise ValueError("measured acquisition inputs and labels must be aligned")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("measured acquisition inputs and labels must be finite")
        object.__setattr__(self, "X_pool", np.ascontiguousarray(X))
        object.__setattr__(self, "y_pool", np.ascontiguousarray(y))

    def acquire_nearest(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(query, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        if points.ndim != 2 or points.shape[1] != self.X_pool.shape[1]:
            raise ValueError("query and acquisition pool feature dimensions differ")
        distances = np.sum((points[:, None, :] - self.X_pool[None, :, :]) ** 2, axis=2)
        selected: list[int] = []
        used: set[int] = set()
        for row in distances:
            available = next((int(index) for index in np.argsort(row) if int(index) not in used), None)
            if available is not None:
                selected.append(available)
                used.add(available)
        indices = np.asarray(selected, dtype=int)
        return self.X_pool[indices], self.y_pool[indices], indices

    def acquire_indices(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reveal labels only after a policy has selected visible pool indices."""

        selected = np.asarray(indices, dtype=int).reshape(-1)
        if not len(selected) or len(selected) != len(set(selected.tolist())):
            raise ValueError("acquisition indices must be non-empty and unique")
        if np.any(selected < 0) or np.any(selected >= len(self.X_pool)):
            raise IndexError("acquisition index lies outside the measured pool")
        return self.X_pool[selected], self.y_pool[selected], selected


__all__ = ["PoolOracle"]
