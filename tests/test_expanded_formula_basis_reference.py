"""Inference correctness diagnostic fixtures, never formula efficacy data."""

import numpy as np
import pytest

from hypothesis_mvp.pcpi.reference import (
    NormalInverseGammaPrior, ReferenceBank, ReferenceStructure,
    SequentialReferencePosterior,
)
from hypothesis_mvp.pcpi.reference.basis import design_matrix
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    compile_fixed_formula_support, compile_fixed_formula_term,
    evaluate_fixed_formula_term,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    freeze_discovery_model, freeze_expanded_formula_model,
)


def test_bounded_formula_terms_share_one_design_interpreter():
    x = np.array([[0.5, 1.0], [1.0, 2.0], [1.5, 3.0]])
    examples = {
        "2*x0*exp(-Abs(x0))": x[:, 0] * np.exp(-np.abs(x[:, 0])),
        "log(Abs(x1)+1)": np.log(np.abs(x[:, 1]) + 1),
        "x0/(1+x1**2)": x[:, 0] / (1 + x[:, 1] ** 2),
        "sqrt(x0)": np.sqrt(x[:, 0]),
        "sin(1.7*x1)": np.sin(1.7 * x[:, 1]),
        "x0**(1/2)": np.sqrt(x[:, 0]),
    }
    for expression, expected in examples.items():
        token = compile_fixed_formula_term(expression, 2)
        assert np.allclose(design_matrix(x, (token,))[:, 0], expected)
        assert np.array_equal(evaluate_fixed_formula_term(token, x),
                              design_matrix(x, (token,))[:, 0])


def test_compound_formula_uses_existing_finite_conjugate_posterior():
    x = np.linspace(.3, 1.5, 8)[:, None]
    y = 1.3 * x[:, 0] * np.exp(-np.abs(x[:, 0]))
    token = compile_fixed_formula_term("x0*exp(-Abs(x0))", 1)
    bank = ReferenceBank((
        ReferenceStructure("compound", "b0*compound", (token,), .5),
        ReferenceStructure("linear", "b0*x0", ("x0",), .5)),
        NormalInverseGammaPrior())
    engine = SequentialReferencePosterior(bank)
    batch = engine.fit_batch(x, y)
    sequential = engine.fit_sequential(x, y)
    assert batch.bank_hash == sequential.bank_hash == bank.stable_hash
    assert np.allclose([member.probability for member in batch.members],
                       [member.probability for member in sequential.members])
    updated = engine.update_one(batch, np.array([1.7]), .2)
    assert updated.probability_sum == pytest.approx(1.)


def test_additive_formula_compiles_to_one_frozen_support():
    x = np.array([[.4, .8], [1., 1.2]])
    support = compile_fixed_formula_support(
        "3 + 2*x0*exp(-Abs(x0)) - .7*log(1+x1**2)", 2)
    assert "intercept" in support
    assert len(support) == 3
    matrix = design_matrix(x, support)
    assert matrix.shape == (2, 3)
    assert np.all(np.isfinite(matrix))
    assert support == compile_fixed_formula_support(
        "(1+2) - .7*log(1+x1**2) + 2*x0*exp(-Abs(x0))", 2)


def test_additive_support_refuses_duplicate_structures():
    with pytest.raises(ValueError, match="duplicate"):
        compile_fixed_formula_support("x0 - 2*x0", 1)


def test_expanded_bank_is_opt_in_and_has_distinct_model_identity():
    rows = [{"expression": "x0", "source": "engine:a"},
            {"expression": "x0*exp(-Abs(x0))", "source": "llm",
             "origin": "llm"}]
    with pytest.raises(ValueError):
        freeze_discovery_model(
            rows, n_features=1, prior=NormalInverseGammaPrior(),
            exploration_identity="a" * 64,
            coefficient_policy="discard-fitted-coefficients-refit-closed-basis")
    expanded = freeze_expanded_formula_model(
        rows, n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64,
        coefficient_policy="discard-outer-amplitudes-freeze-inner-parameters-v1")
    assert expanded.refit_policy == (
        "discard-outer-amplitudes-freeze-inner-parameters-v1")
    assert expanded.engine(expanded.stable_hash).fit_batch(
        np.array([[.5], [1.], [1.5]]), np.array([.1, .2, .3])
    ).probability_sum == pytest.approx(1.)
    with pytest.raises(ValueError, match="contract"):
        freeze_expanded_formula_model(
            rows, n_features=1, prior=NormalInverseGammaPrior(),
            exploration_identity="a" * 64,
            coefficient_policy="discard-fitted-coefficients-refit-closed-basis")


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo unsafe')", "x0.__class__", "x2",
    "1/(x0-x0)", "x0**x0", "x0**5",
])
def test_undefined_or_unsupported_formula_fails_closed(expression):
    x = np.array([[.5], [1.]])
    try:
        token = compile_fixed_formula_term(expression, 1)
    except (ValueError, SyntaxError):
        return
    with pytest.raises(ValueError, match="undefined"):
        design_matrix(x, (token,))
