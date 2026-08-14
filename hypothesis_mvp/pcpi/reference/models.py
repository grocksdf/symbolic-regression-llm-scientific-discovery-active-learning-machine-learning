"""Immutable contracts for the finite P1 symbolic universe."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class NormalInverseGammaPrior:
    """Shared conjugate parameter/noise prior for every bank member."""

    coefficient_mean: float = 0.0
    coefficient_precision: float = 1.0
    noise_shape: float = 2.5
    noise_scale: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.coefficient_mean,
            self.coefficient_precision,
            self.noise_shape,
            self.noise_scale,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("reference prior values must be finite")
        if min(self.coefficient_precision, self.noise_shape, self.noise_scale) <= 0:
            raise ValueError("reference precision, shape, and scale must be positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "coefficient_mean": self.coefficient_mean,
            "coefficient_precision": self.coefficient_precision,
            "noise_shape": self.noise_shape,
            "noise_scale": self.noise_scale,
        }


@dataclass(frozen=True)
class ReferenceStructure:
    """A symbolic structure represented by a closed tuple of basis terms."""

    structure_id: str
    expression: str
    basis_terms: tuple[str, ...]
    prior_probability: float

    def __post_init__(self) -> None:
        if not self.structure_id or not self.expression or not self.basis_terms:
            raise ValueError("reference structures require id, expression, and terms")
        if len(set(self.basis_terms)) != len(self.basis_terms):
            raise ValueError("reference basis terms must be unique within a structure")
        if not math.isfinite(self.prior_probability) or self.prior_probability <= 0:
            raise ValueError("structure prior probability must be positive and finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "structure_id": self.structure_id,
            "expression": self.expression,
            "basis_terms": list(self.basis_terms),
            "prior_probability": self.prior_probability,
        }


@dataclass(frozen=True)
class ReferenceBank:
    """Finite normalized structure prior for exact-reference comparison."""

    structures: tuple[ReferenceStructure, ...]
    prior: NormalInverseGammaPrior

    def __post_init__(self) -> None:
        if len(self.structures) < 2:
            raise ValueError("reference bank requires at least two structures")
        identifiers = [item.structure_id for item in self.structures]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("reference structure identifiers must be unique")
        probability_sum = sum(item.prior_probability for item in self.structures)
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("structure prior probabilities must sum to one")

    @property
    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "pcpi-reference-bank-v1",
            "prior": self.prior.to_dict(),
            "structures": [item.to_dict() for item in self.structures],
        }
