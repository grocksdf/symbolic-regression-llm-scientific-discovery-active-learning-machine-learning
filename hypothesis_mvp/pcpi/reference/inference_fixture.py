"""Exactly enumerable inference-correctness fixture shared across PCPI stages.

The real-data discovery and acquisition runtimes never import this module.
Its deterministic observations test numerical inference only and are not
scientific-discovery efficacy evidence.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from .models import NormalInverseGammaPrior, ReferenceBank, ReferenceStructure


FIXTURE_ROLE = "inference_correctness_diagnostic_fixture"


def correctness_diagnostic_bank() -> ReferenceBank:
    definitions = (
        ("constant", "b0", ("1",)),
        ("linear", "b0 + b1*x", ("1", "x")),
        ("linear_alias", "b0 + b1*x_alias", ("1", "x_alias")),
        ("quadratic", "b0 + b1*x + b2*x^2", ("1", "x", "x2")),
        ("cubic", "b0 + b1*x + b2*x^2 + b3*x^3", ("1", "x", "x2", "x3")),
        ("sinusoid", "b0 + b1*sin(x)", ("1", "sin_x")),
        ("reciprocal", "b0 + b1/(1+x^2)", ("1", "reciprocal_1_plus_x2")),
    )
    probability = 1.0 / len(definitions)
    structures = tuple(
        ReferenceStructure(identifier, expression, terms, probability)
        for identifier, expression, terms in definitions
    )
    return ReferenceBank(
        structures,
        NormalInverseGammaPrior(
            coefficient_mean=0.0,
            coefficient_precision=0.5,
            noise_shape=3.0,
            noise_scale=0.05,
        ),
    )


def correctness_diagnostic_observations(
    seed: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if seed < 0 or count < 4:
        raise ValueError("diagnostic seed and observation count must be valid")
    rng = np.random.default_rng(seed)
    actions = np.linspace(-1.5, 1.5, count)
    targets = (
        0.75
        - 1.2 * actions
        + 0.55 * np.square(actions)
        + rng.normal(0.0, 0.08, count)
    )
    return np.ascontiguousarray(actions), np.ascontiguousarray(targets)


def correctness_fixture_hash(*arrays: np.ndarray) -> str:
    if not arrays:
        raise ValueError("fixture hash requires at least one numerical array")
    digest = sha256()
    for values in arrays:
        array = np.ascontiguousarray(values, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("fixture arrays must be finite")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


__all__ = [
    "FIXTURE_ROLE",
    "correctness_diagnostic_bank",
    "correctness_diagnostic_observations",
    "correctness_fixture_hash",
]
