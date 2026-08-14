"""Dataset-agnostic finite bank and development-only standardization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from .basis import DesignPreconditioner
from .models import NormalInverseGammaPrior, ReferenceBank, ReferenceStructure


@dataclass(frozen=True)
class DevelopmentStandardizer:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "DevelopmentStandardizer":
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(targets) or not len(features):
            raise ValueError("standardizer requires aligned development observations")
        feature_scale = np.std(features, axis=0, ddof=0)
        target_scale = float(np.std(targets, ddof=0))
        if np.any(feature_scale <= 0.0) or target_scale <= 0.0:
            raise ValueError("development features and target must have positive scale")
        return cls(np.mean(features, axis=0), feature_scale, float(np.mean(targets)), target_scale)

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("feature matrix does not match the frozen standardizer")
        return np.ascontiguousarray((values - self.feature_mean) / self.feature_scale)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            (np.asarray(y, dtype=float).reshape(-1) - self.target_mean) / self.target_scale
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "fit_role": "development",
        }

    @property
    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def _structure(
    identifier: str,
    expression: str,
    terms: tuple[str, ...],
    probability: float,
) -> ReferenceStructure:
    return ReferenceStructure(identifier, expression, terms, probability)


def generic_real_bank(feature_count: int) -> ReferenceBank:
    """Build one column-order-invariant hierarchy for every input dimension."""

    if feature_count < 1:
        raise ValueError("real reference bank requires at least one feature")
    linear = tuple(f"x{index}" for index in range(feature_count))
    squares = tuple(f"x{index}_sq" for index in range(feature_count))
    cubics = tuple(f"x{index}_cube" for index in range(feature_count))
    interactions = tuple(
        f"x{left}_x{right}"
        for left in range(feature_count)
        for right in range(left + 1, feature_count)
    )
    probability = 1.0 / 6.0
    structures = (
        _structure("intercept", "beta0", ("intercept",), probability),
        _structure(
            "linear",
            "beta0 + sum_j beta_j*x_j",
            ("intercept",) + linear,
            probability,
        ),
        _structure(
            "additive_quadratic",
            "beta0 + sum_j (beta_j*x_j + gamma_j*x_j^2)",
            ("intercept",) + linear + squares,
            probability,
        ),
        _structure(
            "full_quadratic",
            "beta0 + linear + diagonal_quadratic + all_pairwise_interactions",
            ("intercept",) + linear + squares + interactions,
            probability,
        ),
        _structure(
            "additive_cubic",
            "beta0 + linear + diagonal_quadratic + diagonal_cubic",
            ("intercept",) + linear + squares + cubics,
            probability,
        ),
        _structure(
            "full_quadratic_additive_cubic",
            "beta0 + full_quadratic + diagonal_cubic",
            ("intercept",) + linear + squares + interactions + cubics,
            probability,
        ),
    )
    prior = NormalInverseGammaPrior(
        coefficient_mean=0.0,
        coefficient_precision=1.0,
        noise_shape=3.0,
        noise_scale=1.0,
    )
    return ReferenceBank(structures, prior)


def fit_bank_preconditioner(
    bank: ReferenceBank, development_X: np.ndarray
) -> DesignPreconditioner:
    """Fit one x-only transform shared by every structure in a finite bank."""

    terms = tuple(
        dict.fromkeys(
            term
            for structure in bank.structures
            for term in structure.basis_terms
        )
    )
    return DesignPreconditioner.fit(development_X, terms)


def stable_budget_indices(row_ids: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Choose rows without using covariates or targets."""

    identifiers = np.asarray(row_ids, dtype=object).reshape(-1)
    if budget < 1 or budget > len(identifiers):
        raise ValueError("observation budget must lie within the visible role")
    keys = np.asarray(
        [sha256(f"{seed}:{item}".encode("utf-8")).digest() for item in identifiers],
        dtype="|S32",
    )
    return np.argsort(keys, kind="stable")[:budget]


__all__ = [
    "DevelopmentStandardizer",
    "fit_bank_preconditioner",
    "generic_real_bank",
    "stable_budget_indices",
]
