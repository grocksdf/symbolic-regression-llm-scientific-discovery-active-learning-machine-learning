"""Bounded materialization of g(y_hat, x) in the registered closed basis.

Research-only compiler reference. No production proposal or adapter path
imports this module. Materialize the frozen incumbent before checking novelty;
an unsupported composite is rejected rather than approximated or evaluated.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import math

from .pcpi_adapter import structural_terms

MAX_SOURCE_NODES = 128
MAX_EXPANDED_TERMS = 64
MAX_OUTPUT_NODES = 256
MAX_EXPRESSION_LENGTH = 4096


class _SubstitutePrediction(ast.NodeTransformer):
    def __init__(self, incumbent: ast.AST) -> None:
        self.incumbent = incumbent
        self.used = False

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "y_hat":
            self.used = True
            return deepcopy(self.incumbent)
        return node


def _bounded_terms(node: ast.AST) -> list[ast.AST]:
    """Expand only addition under polynomial multiplication/powers.

    Unsupported transforms remain as ASTs and are checked against the
    registered basis after this exact algebraic rewrite.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = _bounded_terms(node.left), _bounded_terms(node.right)
        if len(left) + len(right) > MAX_EXPANDED_TERMS:
            raise ValueError("closed-basis composition term budget exceeded")
        if isinstance(node.op, ast.Sub):
            right = [ast.UnaryOp(op=ast.USub(), operand=value) for value in right]
        return left + right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        terms = _bounded_terms(node.operand)
        return [ast.UnaryOp(op=deepcopy(node.op), operand=value) for value in terms]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = _bounded_terms(node.left), _bounded_terms(node.right)
        if len(left) * len(right) > MAX_EXPANDED_TERMS:
            raise ValueError("closed-basis composition term budget exceeded")
        return [ast.BinOp(left=deepcopy(a), op=ast.Mult(), right=deepcopy(b))
                for a in left for b in right]
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Constant)
            and type(node.right.value) is int and 1 <= node.right.value <= 4):
        base = _bounded_terms(node.left)
        if len(base) ** node.right.value > MAX_EXPANDED_TERMS:
            raise ValueError("closed-basis composition term budget exceeded")
        terms: list[ast.AST] = [ast.Constant(value=1)]
        for _ in range(node.right.value):
            terms = [ast.BinOp(left=a, op=ast.Mult(), right=deepcopy(b))
                     for a in terms for b in base]
        return terms
    return [node]


def _amplitude_and_basis(node: ast.AST) -> tuple[float, ast.AST]:
    """Pull numerical coefficients out of a single expanded basis term."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value), ast.Constant(value=1)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        coefficient, basis = _amplitude_and_basis(node.operand)
        return (-coefficient if isinstance(node.op, ast.USub) else coefficient), basis
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, a = _amplitude_and_basis(node.left)
        right, b = _amplitude_and_basis(node.right)
        if isinstance(a, ast.Constant) and a.value == 1:
            return left * right, b
        if isinstance(b, ast.Constant) and b.value == 1:
            return left * right, a
        return left * right, ast.BinOp(left=a, op=ast.Mult(), right=b)
    return 1., node


def materialize_closed_basis_composition(
    correction: str, incumbent: str, n_features: int,
) -> tuple[str, tuple[str, ...]]:
    """Return exact expression and distinct basis supports after substitution.

    This performs no coefficient fitting, posterior update, provider call, or
    validity check on measured outcomes. Novelty is relative to a frozen bank
    and must be checked by its caller *after* this materialization.
    """
    if (type(n_features) is not int or n_features < 1
            or not isinstance(correction, str) or not correction
            or not isinstance(incumbent, str) or not incumbent
            or len(correction) > MAX_EXPRESSION_LENGTH
            or len(incumbent) > MAX_EXPRESSION_LENGTH):
        raise ValueError("invalid composition controls")
    proposed = ast.parse(correction, mode="eval")
    current = ast.parse(incumbent, mode="eval")
    if (len(list(ast.walk(proposed))) > MAX_SOURCE_NODES
            or len(list(ast.walk(current))) > MAX_SOURCE_NODES):
        raise ValueError("closed-basis composition source budget exceeded")
    # The incumbent must already be a valid closed-basis symbolic hypothesis.
    structural_terms(incumbent, n_features)
    substitution = _SubstitutePrediction(current.body)
    expanded = substitution.visit(proposed.body)
    if not substitution.used:
        raise ValueError("composition did not reference current prediction")
    terms = _bounded_terms(expanded)
    if not terms or len(terms) > MAX_EXPANDED_TERMS:
        raise ValueError("closed-basis composition term budget exceeded")
    coefficients: dict[str, tuple[float, ast.AST]] = {}
    for term in terms:
        amplitude, basis = _amplitude_and_basis(term)
        if not math.isfinite(amplitude):
            raise ValueError("nonfinite closed-basis amplitude")
        text = ast.unparse(ast.fix_missing_locations(basis))
        support, = structural_terms(text, n_features)
        old, _ = coefficients.get(support, (0., basis))
        value = old + amplitude
        if not math.isfinite(value):
            raise ValueError("nonfinite closed-basis amplitude")
        coefficients[support] = (value, basis)
    rendered = [ast.BinOp(left=ast.Constant(value=value), op=ast.Mult(),
                          right=basis)
                for _, (value, basis) in sorted(coefficients.items()) if value != 0.]
    if not rendered:
        raise ValueError("composition cancels to zero")
    output = rendered[0]
    for term in rendered[1:]:
        output = ast.BinOp(left=output, op=ast.Add(), right=term)
    if len(list(ast.walk(output))) > MAX_OUTPUT_NODES:
        raise ValueError("closed-basis composition output budget exceeded")
    expression = ast.unparse(ast.fix_missing_locations(output))
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("closed-basis composition expression budget exceeded")
    return expression, structural_terms(expression, n_features)
