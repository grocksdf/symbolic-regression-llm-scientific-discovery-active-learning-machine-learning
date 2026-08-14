"""Deterministic unit-test inputs; never written to formal evidence outputs."""

from __future__ import annotations

import numpy as np

from hypothesis_mvp.pcpi.reference.models import (
    NormalInverseGammaPrior,
    ReferenceBank,
    ReferenceStructure,
)


def unit_bank() -> ReferenceBank:
    identifiers = (
        ("constant", "b0", ("1",)),
        ("linear", "b0 + b1*x", ("1", "x")),
        ("linear_alias", "b0 + b1*x_alias", ("1", "x_alias")),
        ("quadratic", "b0 + b1*x + b2*x^2", ("1", "x", "x2")),
        ("cubic", "b0 + b1*x + b2*x^2 + b3*x^3", ("1", "x", "x2", "x3")),
        ("sinusoid", "b0 + b1*sin(x)", ("1", "sin_x")),
        ("reciprocal", "b0 + b1/(1+x^2)", ("1", "reciprocal_1_plus_x2")),
    )
    probability = 1.0 / len(identifiers)
    return ReferenceBank(
        tuple(
            ReferenceStructure(identifier, expression, terms, probability)
            for identifier, expression, terms in identifiers
        ),
        NormalInverseGammaPrior(0.0, 0.5, 3.0, 0.05),
    )


def unit_observations(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.5, 1.5, count)
    y = 0.75 - 1.2 * x + 0.55 * np.square(x) + rng.normal(0.0, 0.08, count)
    return x, y
