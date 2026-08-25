"""Prequential semiparametric reconstruction of a predictive residual law.

The scientific engine supplies a continuous base forecast ``F_t``.  P3H
models the raw probability residual ``U_t = F_t(Y_t)`` with a dyadic Pólya
tree whose split probabilities have independent Beta(1/2, 1/2) priors.  At
round ``t`` the tree contains only earlier raw PIT values.  Its depth grows as
``floor(log2(max(n, 1)) / 2)``, so the active leaf count is at most
``sqrt(n)`` and no response-dependent bandwidth or stopping choice exists.

The resulting forecast is ``G_t(F_t(y))`` with density
``g_t(F_t(y)) f_t(y)``.  This is an online semiparametric predictive
composition, not a relabeling of the base scientific posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np

from .structurewise_discrepancy import (
    RegisteredStructurewiseDiscrepancyEngine,
    SequentialStructurewiseDiscrepancyState,
    StructurewisePredictiveLaw,
)


P3H_RESIDUAL_METHOD = "prequential-kt-dyadic-polya-tree-residual-law-v1"
P3H_DEPTH_SCHEDULE = "floor(log2(max(history-count,1))/2)"
P3H_SPLIT_PRIOR = "independent-beta-one-half-one-half"


def _readonly(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("semiparametric residual arrays must be finite")
    array.setflags(write=False)
    return array


def _validate_unit_interval(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must lie in the closed unit interval")
    return array


def universal_dyadic_depth(history_count: int) -> int:
    """Return the response-value-independent P3H sieve depth."""

    if isinstance(history_count, bool) or int(history_count) != history_count:
        raise ValueError("residual history count must be an integer")
    count = int(history_count)
    if count < 0:
        raise ValueError("residual history count cannot be negative")
    return int(math.floor(math.log2(max(count, 1)) / 2.0))


@dataclass(frozen=True)
class DyadicPolyaTreeState:
    """Immutable raw-PIT history; no current or future response is stored."""

    raw_pits: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        values = _validate_unit_interval(
            np.asarray(self.raw_pits, dtype=float), name="raw predictive residuals"
        )
        object.__setattr__(self, "raw_pits", tuple(float(value) for value in values))

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(P3H_RESIDUAL_METHOD.encode("ascii"))
        digest.update(np.asarray(self.raw_pits, dtype=np.float64).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class DyadicPolyaTreePredictiveLaw:
    """Continuous piecewise-uniform posterior predictive law on ``[0, 1]``."""

    depth: int
    leaf_probabilities: np.ndarray
    history_count: int

    def __post_init__(self) -> None:
        probabilities = _readonly(self.leaf_probabilities).reshape(-1)
        if (
            isinstance(self.depth, bool)
            or self.depth < 0
            or len(probabilities) != 2**self.depth
            or isinstance(self.history_count, bool)
            or self.history_count < 0
            or np.any(probabilities <= 0.0)
            or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=2e-15)
        ):
            raise ValueError("dyadic Pólya-tree predictive law is invalid")
        object.__setattr__(self, "leaf_probabilities", probabilities)

    @property
    def leaf_width(self) -> float:
        return math.ldexp(1.0, -self.depth)

    def density(self, raw_pits: np.ndarray) -> np.ndarray:
        values = _validate_unit_interval(
            np.asarray(raw_pits, dtype=float), name="density coordinates"
        )
        indices = np.minimum(
            np.floor(values * len(self.leaf_probabilities)).astype(int),
            len(self.leaf_probabilities) - 1,
        )
        return self.leaf_probabilities[indices] / self.leaf_width

    def log_density(self, raw_pits: np.ndarray) -> np.ndarray:
        return np.log(self.density(raw_pits))

    def cdf(self, raw_pits: np.ndarray) -> np.ndarray:
        values = _validate_unit_interval(
            np.asarray(raw_pits, dtype=float), name="CDF coordinates"
        )
        leaf_count = len(self.leaf_probabilities)
        scaled = values * leaf_count
        indices = np.minimum(np.floor(scaled).astype(int), leaf_count - 1)
        cumulative = np.concatenate(([0.0], np.cumsum(self.leaf_probabilities)))
        within = np.where(values == 1.0, 1.0, scaled - indices)
        result = cumulative[indices] + self.leaf_probabilities[indices] * within
        return np.where(values == 1.0, 1.0, result)

    def inverse_cdf(self, probabilities: np.ndarray) -> np.ndarray:
        values = _validate_unit_interval(
            np.asarray(probabilities, dtype=float), name="inverse-CDF probabilities"
        )
        cumulative = np.concatenate(([0.0], np.cumsum(self.leaf_probabilities)))
        indices = np.searchsorted(cumulative, values, side="right") - 1
        indices = np.clip(indices, 0, len(self.leaf_probabilities) - 1)
        within = (values - cumulative[indices]) / self.leaf_probabilities[indices]
        result = (indices + within) * self.leaf_width
        return np.where(values == 1.0, 1.0, result)


class DyadicPolyaTreeResidualModel:
    """Exact conjugate split-count updates for the registered dyadic tree."""

    method = P3H_RESIDUAL_METHOD
    split_prior = P3H_SPLIT_PRIOR
    depth_schedule = P3H_DEPTH_SCHEDULE

    @staticmethod
    def prior_state() -> DyadicPolyaTreeState:
        return DyadicPolyaTreeState()

    @staticmethod
    def update(state: DyadicPolyaTreeState, raw_pit: float) -> DyadicPolyaTreeState:
        value = float(raw_pit)
        _validate_unit_interval(np.asarray([value]), name="observed raw PIT")
        return DyadicPolyaTreeState(state.raw_pits + (value,))

    @staticmethod
    def predictive_law(state: DyadicPolyaTreeState) -> DyadicPolyaTreePredictiveLaw:
        depth = universal_dyadic_depth(len(state.raw_pits))
        masses = np.ones(1, dtype=float)
        if depth:
            values = np.asarray(state.raw_pits, dtype=float)
            for level in range(depth):
                node_count = 2**level
                child_indices = np.minimum(
                    np.floor(values * (2 * node_count)).astype(int),
                    2 * node_count - 1,
                )
                child_counts = np.bincount(
                    child_indices, minlength=2 * node_count
                ).reshape(node_count, 2)
                parent_counts = np.sum(child_counts, axis=1)
                left = (child_counts[:, 0] + 0.5) / (parent_counts + 1.0)
                split = np.column_stack((left, 1.0 - left))
                masses = (masses[:, None] * split).reshape(-1)
        masses /= float(np.sum(masses))
        return DyadicPolyaTreePredictiveLaw(depth, masses, len(state.raw_pits))

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        for value in (self.method, self.split_prior, self.depth_schedule):
            digest.update(value.encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class SemiparametricPredictiveLaw:
    """Composition of one scientific base forecast and one residual law."""

    base_law: StructurewisePredictiveLaw
    residual_law: DyadicPolyaTreePredictiveLaw

    def cdf(self, targets: np.ndarray) -> np.ndarray:
        return self.residual_law.cdf(self.base_law.cdf(targets))

    def logpdf(self, targets: np.ndarray) -> np.ndarray:
        base_cdf = self.base_law.cdf(targets)
        return self.base_law.logpdf(targets) + self.residual_law.log_density(base_cdf)


@dataclass(frozen=True)
class SequentialSemiparametricResidualState:
    """Joint predictable state of the base learner and residual correction."""

    base_state: SequentialStructurewiseDiscrepancyState
    residual_state: DyadicPolyaTreeState

    @property
    def observation_indices(self) -> tuple[int, ...]:
        return self.base_state.observation_indices

    @property
    def probabilities(self) -> tuple[float, ...]:
        return self.base_state.probabilities


class SequentialSemiparametricResidualEngine:
    """Leakage-safe composition around the registered scientific engine."""

    method = P3H_RESIDUAL_METHOD

    def __init__(
        self,
        base_engine: RegisteredStructurewiseDiscrepancyEngine,
        residual_model: DyadicPolyaTreeResidualModel | None = None,
    ) -> None:
        if not isinstance(base_engine, RegisteredStructurewiseDiscrepancyEngine):
            raise TypeError("P3H requires the registered structure-wise base engine")
        self.base_engine = base_engine
        self.residual_model = residual_model or DyadicPolyaTreeResidualModel()

    @property
    def bank(self):
        return self.base_engine.bank

    @property
    def records(self):
        return self.base_engine.records

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.method.encode("ascii"))
        digest.update(self.base_engine.stable_hash.encode("ascii"))
        digest.update(self.residual_model.stable_hash.encode("ascii"))
        return digest.hexdigest()

    def prior_state(self) -> SequentialSemiparametricResidualState:
        return SequentialSemiparametricResidualState(
            self.base_engine.prior_state(), self.residual_model.prior_state()
        )

    def sequential_predictive_law(
        self,
        state: SequentialSemiparametricResidualState,
        row_indices: tuple[int, ...],
    ) -> SemiparametricPredictiveLaw:
        if not isinstance(state, SequentialSemiparametricResidualState):
            raise TypeError("P3H predictive state is invalid")
        base_law = self.base_engine.sequential_predictive_law(
            state.base_state, row_indices
        )
        residual_law = self.residual_model.predictive_law(state.residual_state)
        return SemiparametricPredictiveLaw(base_law, residual_law)

    def update(
        self,
        state: SequentialSemiparametricResidualState,
        row_index: int,
        target: float,
    ) -> SequentialSemiparametricResidualState:
        """Observe one response after computing its leave-future-out raw PIT."""

        if not isinstance(state, SequentialSemiparametricResidualState):
            raise TypeError("P3H update state is invalid")
        base_law = self.base_engine.sequential_predictive_law(
            state.base_state, (int(row_index),)
        )
        raw_pit = float(base_law.cdf(np.asarray([float(target)]))[0])
        next_residual = self.residual_model.update(state.residual_state, raw_pit)
        next_base = self.base_engine.update(state.base_state, row_index, target)
        return SequentialSemiparametricResidualState(next_base, next_residual)
