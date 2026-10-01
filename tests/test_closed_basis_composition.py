"""Algebraic compiler correctness fixtures, not efficacy experiments."""

import numpy as np
import pytest

from hypothesis_mvp.discovery.closed_basis_composition import (
    materialize_closed_basis_composition,
)
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.pcpi_adapter import structural_terms


@pytest.mark.parametrize(("correction", "incumbent", "expected"), [
    ("y_hat*x0", "x0+x1", ("x0_sq", "x0_x1")),
    ("y_hat**2", "x0+x1", ("x0_sq", "x0_x1", "x1_sq")),
    ("tanh(y_hat)", "x0", ("tanh_x0",)),
    ("y_hat + x1**2", "sin(x0)", ("sin_x0", "x1_sq")),
])
def test_materialized_correction_is_adaptable_and_exact_on_points(
    correction, incumbent, expected,
):
    materialized, terms = materialize_closed_basis_composition(
        correction, incumbent, 2)
    assert terms == expected
    assert structural_terms(materialized, 2) == terms
    runtime = EquationRuntime(2)
    x = np.array([[-.7, 1.2], [0., .1], [1.3, -1.], [2., 3.]])
    composed = correction.replace("y_hat", "(" + incumbent + ")")
    np.testing.assert_allclose(runtime.predict(materialized, x),
                               runtime.predict(composed, x), atol=1e-12)


@pytest.mark.parametrize(("correction", "incumbent"), [
    ("sin(y_hat)", "x0+x1"),  # not in registered closed basis
    ("y_hat/(1+x0)", "x1"),
    ("y_hat*x0**4", "x0"),  # degree five
    ("y_hat+x0", "x0/x0"),  # invalid frozen incumbent
    ("__import__('os')+y_hat", "x0"),
    ("x0+x1", "x0"),  # not a correction of the incumbent
])
def test_unsupported_or_unbound_composition_fails_closed(correction, incumbent):
    with pytest.raises((ValueError, SyntaxError)):
        materialize_closed_basis_composition(correction, incumbent, 2)


def test_polynomial_cross_product_explosion_is_bounded_before_materialization():
    with pytest.raises(ValueError, match="budget"):
        materialize_closed_basis_composition(
            "y_hat**4", "x0+x1+x2+x3", 4)
