"""Deterministic closed-basis engines with complementary inductive biases."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hypothesis_mvp.config import SymbolicConfig


@dataclass(frozen=True)
class _Basis:
    label: str
    values: np.ndarray


def _mixed_library(X: np.ndarray) -> tuple[_Basis, ...]:
    X = np.asarray(X, dtype=float)
    rows: list[_Basis] = []
    for index in range(X.shape[1]):
        column = X[:, index]
        rows.extend((
            _Basis(f"x{index}", column),
            _Basis(f"x{index}**2", column ** 2),
            _Basis(f"x{index}**3", column ** 3),
            _Basis(f"sin(x{index})", np.sin(column)),
            _Basis(f"cos(x{index})", np.cos(column)),
            _Basis(f"tanh(x{index})", np.tanh(column)),
        ))
    for left in range(X.shape[1]):
        for right in range(left + 1, X.shape[1]):
            rows.append(_Basis(
                f"x{left}*x{right}", X[:, left] * X[:, right]))
    return tuple(rows)


def _additive_library(X: np.ndarray) -> tuple[_Basis, ...]:
    return tuple(row for row in _mixed_library(X)
                 if "*x" not in row.label)


def _design(library: tuple[_Basis, ...]) -> np.ndarray:
    return np.column_stack([row.values for row in library])


def _render(intercept: float, labels: tuple[str, ...],
            coefficients: np.ndarray) -> str:
    terms = [f"{float(intercept):.17g}"]
    terms.extend(
        f"({float(value):.17g})*({label})"
        for label, value in zip(labels, coefficients, strict=True)
        if abs(float(value)) > 1e-14)
    return " + ".join(terms)


def _fit_columns(matrix: np.ndarray, target: np.ndarray,
                 active: np.ndarray) -> tuple[float, np.ndarray, float]:
    values = np.asarray(matrix[:, active], dtype=float)
    design = np.column_stack((np.ones(len(values)), values))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    mse = max(float(np.mean((target - prediction) ** 2)),
              np.finfo(float).tiny)
    return float(coefficients[0]), np.asarray(coefficients[1:]), mse


def _bic(mse: float, count: int, observations: int) -> float:
    return float(observations * np.log(mse)
                 + max(1, count + 1) * np.log(max(observations, 2)))


def _closed_frontier(expressions, n_features: int, limit: int) -> tuple[str, ...]:
    from hypothesis_mvp.discovery.pcpi_adapter import structural_terms
    output = []
    for expression in expressions:
        try:
            structural_terms(expression, n_features)
        except ValueError:
            continue
        if expression not in output:
            output.append(expression)
        if len(output) >= limit:
            break
    if not output:
        raise ValueError("engine produced no adapter-compatible support")
    return tuple(output)


class SparseLibraryRegressor:
    """Sequential thresholded least squares on a frozen mixed library."""

    def __init__(self, config: SymbolicConfig) -> None:
        self.config = config
        self._library: tuple[_Basis, ...] = ()
        self._active = np.empty(0, dtype=int)
        self._intercept = 0.0
        self._coefficients = np.empty(0)
        self._frontier: tuple[str, ...] = ()
        self._means = np.empty(0)
        self._scales = np.empty(0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SparseLibraryRegressor":
        self._library = _mixed_library(np.asarray(X, dtype=float))
        matrix, target = _design(self._library), np.asarray(y, dtype=float)
        self._means = np.mean(matrix, axis=0)
        self._scales = np.std(matrix, axis=0)
        self._scales[self._scales <= np.finfo(float).eps] = 1.0
        scaled = (matrix - self._means) / self._scales
        centered = target - np.mean(target)
        candidates = []
        for threshold in self.config.sparse_library_thresholds:
            active = np.arange(scaled.shape[1])
            for _ in range(self.config.sparse_library_iterations):
                if not len(active):
                    break
                coefficients, *_ = np.linalg.lstsq(
                    scaled[:, active], centered, rcond=None)
                order = np.argsort(-np.abs(coefficients), kind="stable")
                retained = order[
                    np.abs(coefficients[order]) >= float(threshold)]
                retained = retained[:self.config.sparse_library_max_terms]
                updated = np.sort(active[retained])
                if np.array_equal(updated, active):
                    break
                active = updated
            if not len(active):
                continue
            intercept, coefficients, mse = _fit_columns(
                matrix, target, active)
            expression = _render(intercept,
                tuple(self._library[i].label for i in active), coefficients)
            candidates.append((
                _bic(mse, len(active), len(target)), expression,
                active, intercept, coefficients, mse))
        if not candidates:
            raise ValueError("sparse library identified no stable support")
        candidates.sort(key=lambda row: (row[0], len(row[2]), row[1]))
        best = candidates[0]
        self._active, self._intercept = best[2], best[3]
        self._coefficients = best[4]
        self._frontier = _closed_frontier(
            (row[1] for row in candidates), X.shape[1],
            self.config.sparse_library_frontier_size)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not len(self._active):
            raise RuntimeError("sparse library backend has not been fitted")
        matrix = _design(_mixed_library(np.asarray(X, dtype=float)))
        values = (self._intercept
                  + matrix[:, self._active] @ self._coefficients)
        return np.asarray(values).reshape(-1, 1)

    def best_expression(self) -> str:
        if not self._frontier:
            raise RuntimeError("sparse library backend has not been fitted")
        return self._frontier[0]

    def candidate_expressions(self) -> tuple[str, ...]:
        return self._frontier

    def info(self) -> dict:
        return {"engine": "sparse_library",
            "method": "sequential-thresholded-least-squares-v1",
            "library_size": len(self._library),
            "selected_feature_count": len(self._active),
            "thresholds": list(self.config.sparse_library_thresholds),
            "frontier_size_limit": self.config.sparse_library_frontier_size,
            "expression_contract": self.config.expression_contract}


class AdditiveMechanismRegressor:
    """Residual-forward selection restricted to additive mechanisms."""

    def __init__(self, config: SymbolicConfig) -> None:
        self.config = config
        self._active = np.empty(0, dtype=int)
        self._intercept = 0.0
        self._coefficients = np.empty(0)
        self._frontier: tuple[str, ...] = ()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdditiveMechanismRegressor":
        library = _additive_library(np.asarray(X, dtype=float))
        matrix, target = _design(library), np.asarray(y, dtype=float)
        linear = np.asarray([
            index for index, row in enumerate(library)
            if row.label.startswith("x") and "**" not in row.label],
            dtype=int)
        active, available = list(linear), set(range(len(library))) - set(linear)
        intercept, coefficients, mse = _fit_columns(
            matrix, target, np.asarray(active))
        frontier = [(_bic(mse, len(active), len(target)),
                     _render(intercept, tuple(library[i].label for i in active),
                             coefficients),
                     tuple(active), intercept, coefficients, mse)]
        for _ in range(self.config.additive_mechanism_max_terms):
            proposals = []
            for index in sorted(available):
                trial = np.asarray([*active, index])
                trial_intercept, trial_coefficients, trial_mse = _fit_columns(
                    matrix, target, trial)
                proposals.append((
                    _bic(trial_mse, len(trial), len(target)), index,
                    trial_intercept, trial_coefficients, trial_mse))
            if not proposals:
                break
            proposals.sort(key=lambda row: (row[0], row[1]))
            candidate = proposals[0]
            if candidate[0] >= frontier[-1][0] - 1e-12:
                break
            active.append(candidate[1]); available.remove(candidate[1])
            expression = _render(
                candidate[2], tuple(library[i].label for i in active),
                candidate[3])
            frontier.append((candidate[0], expression, tuple(active),
                             candidate[2], candidate[3], candidate[4]))
        frontier.sort(key=lambda row: (row[0], len(row[2]), row[1]))
        best = frontier[0]
        self._active, self._intercept = np.asarray(best[2]), best[3]
        self._coefficients = np.asarray(best[4])
        self._frontier = _closed_frontier(
            (row[1] for row in frontier), X.shape[1],
            self.config.additive_mechanism_frontier_size)
        self._library_labels = tuple(row.label for row in library)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not len(self._active):
            raise RuntimeError("additive mechanism backend has not been fitted")
        matrix = _design(_additive_library(np.asarray(X, dtype=float)))
        values = (self._intercept
                  + matrix[:, self._active] @ self._coefficients)
        return np.asarray(values).reshape(-1, 1)

    def best_expression(self) -> str:
        if not self._frontier:
            raise RuntimeError("additive mechanism backend has not been fitted")
        return self._frontier[0]

    def candidate_expressions(self) -> tuple[str, ...]:
        return self._frontier

    def info(self) -> dict:
        return {"engine": "additive_mechanisms",
            "method": "residual-forward-additive-mechanism-selection-v1",
            "selected_feature_count": len(self._active),
            "frontier_size_limit": self.config.additive_mechanism_frontier_size,
            "expression_contract": self.config.expression_contract}


__all__ = ["AdditiveMechanismRegressor", "SparseLibraryRegressor"]
