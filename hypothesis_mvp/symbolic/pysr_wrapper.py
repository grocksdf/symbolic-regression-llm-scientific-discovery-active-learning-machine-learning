"""Adapters for the production symbolic backends."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from hypothesis_mvp.config import SymbolicConfig

from .base import SymbolicRegressor
from .mcts_agent import MCTSSymbolicAgent


class PySRSymbolicRegressor(SymbolicRegressor):
    def __init__(self, config: SymbolicConfig) -> None:
        try:
            from pysr import PySRRegressor
        except Exception as error:
            raise ImportError(
                "the requested pysr backend is unavailable; install the symbolic extra"
            ) from error
        self.config = config
        self._regressor_type = PySRRegressor
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PySRSymbolicRegressor":
        options: dict[str, Any] = {
            "niterations": self.config.niterations,
            "population_size": self.config.population_size,
            "loss": self.config.loss,
            "binary_operators": self.config.binary_operators,
            "unary_operators": self.config.unary_operators,
            "model_selection": self.config.pysr_model_selection,
            "maxsize": self.config.maxsize,
            "complexity_of_constants": self.config.complexity_of_constants,
            "constraints": {"^": (-1, 1), "pow": (-1, 1)},
        }
        if self.config.complexity_of_operators:
            options["complexity_of_operators"] = self.config.complexity_of_operators
        self._model = self._regressor_type(**options)
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PySR backend has not been fitted")
        return np.asarray(self._model.predict(X), dtype=float).reshape(-1, 1)

    def best_expression(self) -> str:
        if self._model is None:
            raise RuntimeError("PySR backend has not been fitted")
        return str(self._model.sympy())

    def info(self) -> dict[str, Any]:
        return {"engine": "pysr", **asdict(self.config)}


class PolynomialLassoRegressor(SymbolicRegressor):
    def __init__(self, degree: int = 4, alpha: float = 1.0e-3) -> None:
        try:
            from sklearn.linear_model import Lasso
            from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        except Exception as error:
            raise ImportError("the polynomial_lasso backend requires scikit-learn") from error
        self.degree = int(degree)
        self.alpha = float(alpha)
        self._poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self._scaler = StandardScaler()
        self._model = Lasso(alpha=self.alpha, max_iter=200000, tol=1.0e-4)
        self._feature_names: np.ndarray | None = None
        self._target_mean = 0.0
        self._target_scale = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialLassoRegressor":
        transformed = self._poly.fit_transform(np.asarray(X, dtype=float))
        scaled = self._scaler.fit_transform(transformed)
        target = np.asarray(y, dtype=float).reshape(-1)
        self._target_mean = float(np.mean(target))
        self._target_scale = max(float(np.std(target)), np.finfo(float).eps)
        normalized_target = (target - self._target_mean) / self._target_scale
        self._model.fit(scaled, normalized_target)
        self._feature_names = self._poly.get_feature_names_out()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        transformed = self._poly.transform(np.asarray(X, dtype=float))
        scaled = self._scaler.transform(transformed)
        normalized = np.asarray(self._model.predict(scaled), dtype=float)
        return (self._target_scale * normalized + self._target_mean).reshape(-1, 1)

    def best_expression(self) -> str:
        if self._feature_names is None:
            raise RuntimeError("polynomial_lasso backend has not been fitted")
        coefficients = self._target_scale * self._model.coef_ / self._scaler.scale_
        intercept = float(
            self._target_mean
            + self._target_scale * self._model.intercept_
            - np.dot(coefficients, self._scaler.mean_)
        )
        terms = [
            f"{coefficient:.17g}*{feature.replace(' ', '*')}"
            for coefficient, feature in zip(coefficients, self._feature_names, strict=True)
            if float(coefficient) != 0.0
        ]
        return f"{intercept:.17g} + {' + '.join(terms) if terms else '0'}"

    def info(self) -> dict[str, Any]:
        return {
            "engine": "polynomial_lasso", "degree": self.degree,
            "alpha": self.alpha, "feature_scaling": "standard",
            "target_scaling": "standard",
            "iterations": int(getattr(self._model, "n_iter_", 0)),
            "max_iterations": int(self._model.max_iter),
        }


def get_symbolic_regressor(config: SymbolicConfig) -> SymbolicRegressor:
    factories = {
        "pysr": lambda: PySRSymbolicRegressor(config),
        "polynomial_lasso": lambda: PolynomialLassoRegressor(
            degree=config.polynomial_degree, alpha=config.polynomial_alpha
        ),
        "mcts": lambda: MCTSSymbolicAgent(
            config, seed_expressions=list(config.seed_expressions)
        ),
    }
    if config.engine not in factories:
        raise ValueError(f"unsupported symbolic engine: {config.engine}")
    return factories[config.engine]()


__all__ = ["PolynomialLassoRegressor", "PySRSymbolicRegressor", "get_symbolic_regressor"]
