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
    decode_fixed_formula_term,
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


# A real polynomial-lasso output from the gain-localization E arm: 17 additive
# terms (16 structural + intercept) and 204 AST nodes.  The original 128-node /
# 16-term caps sat INSIDE the measured distribution of legitimate engine output
# and clipped this arm while leaving the L arms, which top out at exactly 128
# nodes, untouched.  Raising the caps is a de-biasing change, not a relaxation.
_LASSO_POLYNOMIAL = (
    "-1.307267890584689e-6*x0**4 - 3.548166629345148e-5*x0**3*x3 "
    "- 4.454720938192162e-5*x0**3*x5 + 0.0004273778638153389*x0**3 "
    "- 0.00019734481625690898*x0**2*x3*x5 + 0.004486835591360316*x0**2*x3 "
    "+ 0.005138465255818977*x0**2*x5 - 0.03652511136297161*x0**2 "
    "- 0.128409515976008*x0*x3 - 0.1391245724923443*x0*x5 "
    "+ 0.59236597285685425*x0 - 0.25530951489304367*x1*x2**2 "
    "+ 1.253743311622216*x1*x2 - 0.25597619159114993*x2**2*x4 "
    "- 0.47555786117753443*x2**2 + 1.257201452937371*x2*x4 "
    "+ 9.71313266644076*x2 - 32.28560888301153")


def test_finite_budget_admits_legitimate_polynomial_lasso_output():
    import ast as _ast
    assert len(list(_ast.walk(_ast.parse(_LASSO_POLYNOMIAL, mode="eval")))) > 128
    support = compile_fixed_formula_support(_LASSO_POLYNOMIAL, 6)
    assert len(support) == 18
    assert len(set(support)) == len(support)
    assert "intercept" in support


def test_finite_budget_still_rejects_pathological_expression():
    # 300 structurally repeated summands: well under the 4096-character input
    # bound but far past the node budget, so the guard must still fire.
    deep = " + ".join(f"{i}*x0" for i in range(300))
    assert len(deep) <= 4096
    with pytest.raises(ValueError, match="finite budget"):
        compile_fixed_formula_support(deep, 6)


@pytest.mark.parametrize(("proposal", "refitted"), [
    (
        "(x0+x1)*log(1+Abs(x2))/x2",
        "-0.76985240254872178*(x0+x1)*log(1+Abs(x2))/x2",
    ),
    (
        "x0/(1+x1**2)",
        "0.125*x0/(1+x1**2)",
    ),
    (
        "x0/(2*x1)",
        "-3*x0/(2*x1)",
    ),
    (
        "sin(1.7*x0)",
        "-4.2*sin(1.7*x0)",
    ),
])
def test_outer_amplitude_factorization_preserves_fixed_inner_support(
        proposal, refitted):
    assert compile_fixed_formula_support(
        proposal, 3) == compile_fixed_formula_support(refitted, 3)


def test_outer_amplitude_factorization_keeps_inner_literals_frozen():
    assert compile_fixed_formula_support(
        "2*sin(1.7*x0)", 1) != compile_fixed_formula_support(
        "2*sin(1.8*x0)", 1)


def test_commutative_syntax_order_has_one_fixed_inner_identity():
    assert compile_fixed_formula_support(
        "(x0+x1)*log(1+Abs(x2))/x2", 3
    ) == compile_fixed_formula_support(
        "(x1+x0)*log(Abs(x2)+1)/x2", 3)


def test_unary_wrapped_addition_remains_outer_amplitude_terms():
    assert compile_fixed_formula_support(
        "-(2*x0 + .5*sin(x1))", 2
    ) == compile_fixed_formula_support(
        "-(7*x0 + 3.1*sin(x1))", 2)


def test_common_scale_inside_structural_numerator_is_quotiented():
    assert compile_fixed_formula_support(
        "(-.25*x0 + .5*sin(x1))/(1+Abs(x1))", 2
    ) == compile_fixed_formula_support(
        "(-1.5*x0 + 3*sin(x1))/(1+Abs(x1))", 2)


def test_changed_inner_relative_ratio_remains_a_distinct_support():
    assert compile_fixed_formula_support(
        "(-.25*x0 + .5*sin(x1))/(1+Abs(x1))", 2
    ) != compile_fixed_formula_support(
        "(-.25*x0 + .7*sin(x1))/(1+Abs(x1))", 2)


@pytest.mark.parametrize("expression", [
    "(sin(x0)-2)/x0",
    "(-.25*x0 + .5*sin(x1))/(1+Abs(x1))",
    "-(2*x0 + .5*sin(x1))",
])
def test_fixed_formula_token_canonicalization_is_idempotent(expression):
    token = compile_fixed_formula_term(expression, 2)
    decoded = decode_fixed_formula_term(token, 2)
    assert compile_fixed_formula_term(decoded, 2) == token


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
