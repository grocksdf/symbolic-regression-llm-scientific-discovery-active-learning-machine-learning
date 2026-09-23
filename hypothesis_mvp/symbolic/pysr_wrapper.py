"""Adapters for the production symbolic backends."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from hypothesis_mvp.config import SymbolicConfig

from .base import SymbolicRegressor
from .library_engines import (
    AdditiveMechanismRegressor, SparseLibraryRegressor,
)
from .mcts_agent import MCTSSymbolicAgent
from .registry import engine_spec, registered_engine_names


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
    def __init__(self, degree: int = 4, alpha: float = 1.0e-3,
                 expression_contract: str = "unrestricted",
                 skill_controls: tuple[str, ...] = ()) -> None:
        try:
            from sklearn.linear_model import Lasso
            from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        except Exception as error:
            raise ImportError("the polynomial_lasso backend requires scikit-learn") from error
        controls = tuple(dict.fromkeys(str(value) for value in skill_controls))
        allowed = set(engine_spec("polynomial_lasso").capabilities)
        if set(controls) - allowed:
            raise ValueError("unsupported polynomial_lasso skill control")
        degree_controls = {"linear": 1, "quadratic": 2, "cubic": 3,
                           "quartic": 4}
        selected_degrees = [degree_controls[value] for value in controls
                            if value in degree_controls]
        self.degree = max(selected_degrees) if selected_degrees else int(degree)
        self.skill_controls = controls
        self.allow_interactions = not controls or "interactions" in controls
        self.expression_contract = expression_contract
        if expression_contract not in {"unrestricted", "pcpi-closed-basis-v1"}:
            raise ValueError("unknown symbolic expression contract")
        if expression_contract == "pcpi-closed-basis-v1" and not 1 <= self.degree <= 4:
            raise ValueError("closed symbolic contract requires polynomial degree one to four")
        self.alpha = float(alpha)
        self._poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self._scaler = StandardScaler()
        self._model = Lasso(alpha=self.alpha, max_iter=200000, tol=1.0e-4)
        self._feature_names: np.ndarray | None = None
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._selected_features: np.ndarray | None = None

    def _admit_training_library(self, scaled: np.ndarray, target: np.ndarray, n_features: int) -> np.ndarray:
        """Train-only correlation screening under the exact adapter AST cap.

        Select the feature library before the single Lasso fit. No fitted
        coefficients are discarded afterwards. Worst-case signed scalar
        literals reserve sufficient syntax for rendering fitted amplitudes.
        """
        from hypothesis_mvp.discovery.pcpi_adapter import structural_terms
        names = self._poly.get_feature_names_out()
        scores = np.abs(scaled.T @ target)
        if not np.all(np.isfinite(scores)):
            raise ValueError("nonfinite closed polynomial feature screening")
        order = sorted(range(len(names)), key=lambda i: (-scores[i], i))
        selected, pieces = [], ["-1.2345678901234567e-308"]
        for index in order:
            feature = names[index].replace(" ", "*").replace("^", "**")
            variables = {
                value for value in names[index].replace("^", " ").split()
                if value.startswith("x")}
            if not self.allow_interactions and len(variables) > 1:
                continue
            proposal = [*pieces, f"(-1.2345678901234567e-308)*{feature}"]
            try:
                structural_terms(" + ".join(proposal), n_features)
            except ValueError:
                continue
            selected.append(index)
            pieces = proposal
        if not selected:
            raise ValueError("no feature fits registered closed expression contract")
        return np.asarray(selected, dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialLassoRegressor":
        transformed = self._poly.fit_transform(np.asarray(X, dtype=float))
        scaled = self._scaler.fit_transform(transformed)
        target = np.asarray(y, dtype=float).reshape(-1)
        self._target_mean = float(np.mean(target))
        self._target_scale = max(float(np.std(target)), np.finfo(float).eps)
        normalized_target = (target - self._target_mean) / self._target_scale
        self._selected_features = (self._admit_training_library(scaled, normalized_target, X.shape[1])
            if self.expression_contract == "pcpi-closed-basis-v1" else np.arange(scaled.shape[1]))
        scaled = scaled[:, self._selected_features]
        self._model.fit(scaled, normalized_target)
        self._feature_names = self._poly.get_feature_names_out()[self._selected_features]
        if self.expression_contract == "pcpi-closed-basis-v1":
            import sympy as sp
            from hypothesis_mvp.discovery.pcpi_adapter import structural_terms
            structural_terms(str(sp.sympify(self.best_expression())), X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        transformed = self._poly.transform(np.asarray(X, dtype=float))
        scaled = self._scaler.transform(transformed)
        if self._selected_features is not None:
            scaled = scaled[:, self._selected_features]
        normalized = np.asarray(self._model.predict(scaled), dtype=float)
        return (self._target_scale * normalized + self._target_mean).reshape(-1, 1)

    def best_expression(self) -> str:
        if self._feature_names is None:
            raise RuntimeError("polynomial_lasso backend has not been fitted")
        coefficients = self._target_scale * self._model.coef_ / self._scaler.scale_[self._selected_features]
        intercept = float(
            self._target_mean
            + self._target_scale * self._model.intercept_
            - np.dot(coefficients, self._scaler.mean_[self._selected_features])
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
            "expression_contract": self.expression_contract,
            "library_selection": ("train-correlation-pre-fit-ast-cap" if
                self.expression_contract == "pcpi-closed-basis-v1" else "all-polynomial-features"),
            "selected_feature_count": None if self._selected_features is None else len(self._selected_features),
            "lasso_fit_count": 1 if self._feature_names is not None else 0,
            "skill_controls": list(self.skill_controls),
        }


def get_symbolic_regressor(config: SymbolicConfig) -> SymbolicRegressor:
    if config.expression_contract not in {"unrestricted", "pcpi-closed-basis-v1"}:
        raise ValueError("unknown symbolic expression contract")
    if (config.expression_contract != "unrestricted"
            and config.engine not in set(registered_engine_names())):
        raise ValueError("backend does not implement closed expression contract")
    factories = {
        "pysr": lambda: PySRSymbolicRegressor(config),
        "polynomial_lasso": lambda: PolynomialLassoRegressor(
            degree=config.polynomial_degree, alpha=config.polynomial_alpha,
            expression_contract=config.expression_contract,
            skill_controls=tuple(config.skill_controls)
        ),
        "mcts": lambda: MCTSSymbolicAgent(
            config, seed_expressions=list(config.seed_expressions)
        ),
        "sparse_library": lambda: SparseLibraryRegressor(config),
        "additive_mechanisms": lambda: AdditiveMechanismRegressor(config),
    }
    if config.engine not in factories:
        raise ValueError(f"unsupported symbolic engine: {config.engine}")
    return factories[config.engine]()


__all__ = [
    "AdditiveMechanismRegressor", "PolynomialLassoRegressor",
    "PySRSymbolicRegressor", "SparseLibraryRegressor",
    "get_symbolic_regressor",
]
