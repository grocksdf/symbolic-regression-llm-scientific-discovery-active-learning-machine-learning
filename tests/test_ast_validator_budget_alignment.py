"""No-response safety checks for the registered 512-node AST boundary."""
from __future__ import annotations

import ast

import pytest

from hypothesis_mvp.hypotheses.ast_validator import (
    ExpressionValidationError,
    SafeExpressionValidator,
)
from hypothesis_mvp.hypotheses.expression_limits import AST_NODE_BUDGET
from hypothesis_mvp.discovery.candidate_validity import _tree
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    compile_fixed_formula_support,
)

ORIGINAL_R_V2_AST_FAILURE_265 = (
    "1441.6626651917502*x0**3 + 298.01729030420222*x0**2 - "
    "6.5599165231756755*x0*x3**2 + 31.701220328890418*x0*x3 + "
    "257.71427096930512*x0/(Abs(x0) + 1.0) + "
    "0.04555523082133893*x1**3 - 0.12005773755633344*x1**2 - "
    "0.022072502387102452*x1*x3 - 0.52368574045991556*x1 + "
    "0.21964389162179832*x2**3 - 1.0807663224411761*x2**2 + "
    "0.57519783850309436*x2 - 24.431110607667645*x3**3 + "
    "218.91507083566017*x3**2 - 1570.4243350611737*x3 + "
    "5351.7019496964776*x3/(Abs(x3) + 1.0) - "
    "45.29060484687087*log(Abs(x0) + 1) + "
    "2220.6044528965367*log(Abs(x3) + 1) - "
    "80.168226891618389*sin(x3) - 0.32926706255159299*cos(x1) - "
    "1.4405289030629151*cos(x2) + 232.26329593867848*cos(x3) - "
    "279.87468230978413*tanh(x0) - "
    "2369.7397257772377*tanh(x3) - 1087.7968765982935"
)


def _balanced(leaves, operator="+"):
    values = list(leaves)
    while len(values) > 1:
        values = [
            f"({values[index]} {operator} {values[index + 1]})"
            for index in range(0, len(values), 2)
        ]
    return values[0]


def _expression_with_nodes(count, operator="+"):
    """Construct the exact registered boundary counts at shallow depth."""
    if count in {255, 256, 257, 265}:
        leaves = ["x0"] * 64
        if count == 256:
            leaves[0] = "-1"
        expression = _balanced(leaves, operator)
        if count in {257, 265}:
            wraps = 1 if count == 257 else 5
            for _ in range(wraps):
                expression = f"-({expression})"
    elif count in {512, 513}:
        leaves = ["x0"] * 128
        if count == 512:
            leaves[0] = "-1"
        expression = _balanced(leaves, operator)
        if count == 513:
            expression = f"-({expression})"
    else:
        raise ValueError("unregistered test count")
    assert len(list(ast.walk(ast.parse(expression, mode="eval")))) == count
    return expression


@pytest.mark.parametrize("count", [255, 256, 257, 265, 512])
def test_registered_ast_budget_accepts_every_count_through_512(count):
    result = SafeExpressionValidator().validate(
        _expression_with_nodes(count), allowed_variables={"x0"})
    assert result.node_count == count
    assert result.maximum_depth <= 32


def test_registered_ast_budget_rejects_513_nodes():
    with pytest.raises(ExpressionValidationError, match="limit is 512"):
        SafeExpressionValidator().validate(
            _expression_with_nodes(513), allowed_variables={"x0"})


@pytest.mark.parametrize("count", [255, 256, 257, 265, 512])
def test_all_three_formula_entry_points_share_the_complete_ast_count(count):
    expression = _expression_with_nodes(count, operator="*")
    assert AST_NODE_BUDGET == 512
    assert _tree(expression, 1) is not None
    assert compile_fixed_formula_support(expression, 1)


def test_all_three_formula_entry_points_reject_513():
    expression = _expression_with_nodes(513, operator="*")
    with pytest.raises(ValueError, match="finite budget"):
        _tree(expression, 1)
    with pytest.raises(ValueError, match="finite budget"):
        compile_fixed_formula_support(expression, 1)


def test_original_r_v2_265_node_failure_now_passes_all_syntax_gates():
    expression = ORIGINAL_R_V2_AST_FAILURE_265
    assert len(list(ast.walk(ast.parse(expression, mode="eval")))) == 265
    result = SafeExpressionValidator().validate(
        expression, allowed_variables={"x0", "x1", "x2", "x3"})
    assert result.node_count == 265
    assert _tree(expression, 4) is not None
    assert compile_fixed_formula_support(expression, 4)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo unsafe')",
    "(lambda: 1)()",
    "x0.__class__",
    "[x0 for x0 in (1,)]",
    "open('forbidden')",
])
def test_larger_budget_does_not_expand_the_safe_grammar(expression):
    with pytest.raises(ExpressionValidationError):
        SafeExpressionValidator().validate(
            expression, allowed_variables={"x0"})


def test_larger_budget_does_not_expand_depth_limit():
    expression = "x0"
    for _ in range(33):
        expression = f"-({expression})"
    with pytest.raises(ExpressionValidationError, match="depth"):
        SafeExpressionValidator().validate(
            expression, allowed_variables={"x0"})
