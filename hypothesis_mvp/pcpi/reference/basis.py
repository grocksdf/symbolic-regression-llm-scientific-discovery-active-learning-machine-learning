"""Closed, non-evaluating basis library for finite PCPI universes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import re

import numpy as np


BasisFunction = Callable[[np.ndarray], np.ndarray]
DESIGN_PRECONDITIONING_METHOD = "termwise-center-scale"
DESIGN_PRECONDITIONING_ROLE = "initial-development-covariates-only"


@dataclass(frozen=True)
class DesignPreconditioner:
    """Frozen x-only centering and scaling for a closed basis dictionary."""

    terms: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.terms or len(self.terms) != len(set(self.terms)):
            raise ValueError("preconditioner terms must be non-empty and unique")
        if len(self.means) != len(self.terms) or len(self.scales) != len(self.terms):
            raise ValueError("preconditioner statistics must align with its terms")
        if not np.all(np.isfinite(self.means)) or not np.all(np.isfinite(self.scales)):
            raise ValueError("preconditioner statistics must be finite")
        if any(scale <= 0.0 for scale in self.scales):
            raise ValueError("preconditioner scales must be positive")

    @classmethod
    def fit(cls, x: np.ndarray, terms: Sequence[str]) -> "DesignPreconditioner":
        ordered = tuple(dict.fromkeys(str(term) for term in terms))
        if not ordered:
            raise ValueError("cannot fit an empty basis preconditioner")
        matrix = design_matrix(x, ordered)
        means: list[float] = []
        scales: list[float] = []
        for index, term in enumerate(ordered):
            if term in {"intercept", "1"}:
                means.append(0.0)
                scales.append(1.0)
                continue
            mean = float(np.mean(matrix[:, index]))
            scale = float(np.std(matrix[:, index], ddof=0))
            floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(matrix[:, index]))))
            if not np.isfinite(scale) or scale <= floor:
                raise ValueError(f"basis term {term!r} has no development variation")
            means.append(mean)
            scales.append(scale)
        return cls(ordered, tuple(means), tuple(scales))

    def transform(self, x: np.ndarray, terms: Sequence[str]) -> np.ndarray:
        requested = tuple(str(term) for term in terms)
        positions = {term: index for index, term in enumerate(self.terms)}
        if any(term not in positions for term in requested):
            missing = sorted(set(requested) - set(positions))
            raise ValueError(f"preconditioner does not cover basis terms: {missing}")
        raw = design_matrix(x, requested)
        means = np.asarray([self.means[positions[term]] for term in requested])
        scales = np.asarray([self.scales[positions[term]] for term in requested])
        return np.ascontiguousarray((raw - means) / scales)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "pcpi-design-preconditioner-v1",
            "method": DESIGN_PRECONDITIONING_METHOD,
            "fit_role": DESIGN_PRECONDITIONING_ROLE,
            "terms": list(self.terms),
            "means": list(self.means),
            "scales": list(self.scales),
        }

    @property
    def stable_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def _ones(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _square(x: np.ndarray) -> np.ndarray:
    return np.square(x)


def _cube(x: np.ndarray) -> np.ndarray:
    return np.power(x, 3)


def _reciprocal_quadratic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.square(x))


BASIS_FUNCTIONS: dict[str, BasisFunction] = {
    "1": _ones,
    "x": _identity,
    "x_alias": _identity,
    "x2": _square,
    "x3": _cube,
    "sin_x": np.sin,
    "cos_x": np.cos,
    "exp_neg_x2": lambda x: np.exp(-np.square(x)),
    "reciprocal_1_plus_x2": _reciprocal_quadratic,
}


_FEATURE = re.compile(r"^x([0-9]+)$")
_SQUARE = re.compile(r"^x([0-9]+)_sq$")
_CUBIC = re.compile(r"^x([0-9]+)_cube$")
_INTERACTION = re.compile(r"^x([0-9]+)_x([0-9]+)$")


def _inputs(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("reference inputs must be a non-empty finite matrix")
    return np.ascontiguousarray(array)


def _generic_column(values: np.ndarray, term: str) -> np.ndarray:
    match = _FEATURE.fullmatch(term)
    if match:
        index = int(match.group(1))
        if index >= values.shape[1]:
            raise ValueError(f"basis term {term!r} exceeds the feature dimension")
        return values[:, index]
    match = _SQUARE.fullmatch(term)
    if match:
        index = int(match.group(1))
        if index >= values.shape[1]:
            raise ValueError(f"basis term {term!r} exceeds the feature dimension")
        return np.square(values[:, index])
    match = _CUBIC.fullmatch(term)
    if match:
        index = int(match.group(1))
        if index >= values.shape[1]:
            raise ValueError(f"basis term {term!r} exceeds the feature dimension")
        return np.power(values[:, index], 3)
    match = _INTERACTION.fullmatch(term)
    if match:
        left, right = (int(item) for item in match.groups())
        if left >= values.shape[1] or right >= values.shape[1] or left >= right:
            raise ValueError(f"invalid interaction basis term: {term!r}")
        return values[:, left] * values[:, right]
    raise ValueError(f"unknown reference basis term: {term!r}")


def design_matrix(x: np.ndarray, terms: Sequence[str]) -> np.ndarray:
    """Build a finite matrix without parsing or executing expression text."""

    values = _inputs(x)
    columns: list[np.ndarray] = []
    for term in terms:
        if term in BASIS_FUNCTIONS and values.shape[1] == 1:
            columns.append(BASIS_FUNCTIONS[term](values[:, 0]))
        elif term == "intercept":
            columns.append(np.ones(len(values), dtype=float))
        elif term in BASIS_FUNCTIONS and _FEATURE.fullmatch(term) is None:
            raise ValueError(f"legacy basis term {term!r} requires one feature")
        else:
            columns.append(_generic_column(values, term))
    matrix = np.column_stack(columns)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("reference design matrix contains non-finite values")
    return np.ascontiguousarray(matrix, dtype=float)
