"""Prospective, parameter-aware formula-recovery diagnostic.

The historical strict exact and structural scores remain immutable.  This
module deliberately returns *unresolved* for exact equality when metadata
contains unbound physical parameters.  It never estimates them from ground
truth or reporting responses.
"""
from __future__ import annotations

import ast
import math
import re
from collections.abc import Sequence

import sympy as sp
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    compile_fixed_formula_support,
)


_FUNCTIONS = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
              "tanh": sp.tanh, "exp": sp.exp, "log": sp.log,
              "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs}
_INPUT = re.compile(r"x(?:0|[1-9][0-9]*)\Z")


class FormulaNotEvaluable(ValueError):
    """A malformed or unsupported formula is a separate audit outcome."""


def _parse(expression: str, input_symbols: Sequence[str], *, truth: bool):
    text = str(expression)
    if len(text) > 4096 or not text or not input_symbols:
        raise FormulaNotEvaluable("invalid-expression-budget-or-inputs")
    names = tuple(str(value) for value in input_symbols)
    if len(set(names)) != len(names) or any(not name.isidentifier() for name in names):
        raise FormulaNotEvaluable("ambiguous-input-symbols")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise FormulaNotEvaluable("invalid-expression-syntax") from error
    if len(list(ast.walk(tree))) > 256:
        raise FormulaNotEvaluable("expression-ast-budget")
    inputs = {name: sp.Symbol(f"x{i}") for i, name in enumerate(names)}
    inputs.update({f"x{i}": sp.Symbol(f"x{i}") for i in range(len(names))})
    parameters = set()

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value):
                raise FormulaNotEvaluable("nonfinite-literal")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            if node.id in inputs:
                return inputs[node.id]
            if not truth or not node.id.isidentifier() or node.id in _FUNCTIONS:
                raise FormulaNotEvaluable("undeclared-candidate-symbol")
            parameters.add(node.id)
            return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow):
                if not right.is_Rational or abs(right) > 8:
                    raise FormulaNotEvaluable("unbounded-or-symbolic-power")
                return left ** right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCTIONS and len(node.args) == 1
                and not node.keywords):
            return _FUNCTIONS[node.func.id](visit(node.args[0]))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in inputs and len(node.args) == 1
                and not node.keywords):
            # Benchmark metadata may write one measured state coordinate as
            # P(t), A(t), x(t), or v(t), while the sample column is registered
            # as P, A, x, or v. The argument is observation context, not an
            # additional formula input.
            visit(node.args[0])
            return inputs[node.func.id]
        raise FormulaNotEvaluable("unsupported-expression-node")

    try:
        result = visit(tree.body)
    except (ZeroDivisionError, OverflowError, TypeError, ValueError) as error:
        if isinstance(error, FormulaNotEvaluable):
            raise
        raise FormulaNotEvaluable("invalid-expression-domain") from error
    if result in (sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise FormulaNotEvaluable("nonfinite-expression")
    return result, tuple(sorted(parameters))


def _shape(node, inputs: frozenset[sp.Symbol]):
    if isinstance(node, sp.Symbol):
        return ("variable", str(node)) if node in inputs else ("constant", "free")
    if node.is_Number:
        return ("constant", "negative" if node < 0 else "free")
    if isinstance(node, sp.Add):
        return ("sum", tuple(sorted((_shape(term, inputs)
                for term in node.args), key=repr)))
    if isinstance(node, sp.Mul):
        scalars = [factor for factor in node.args
                   if not factor.free_symbols & inputs]
        factors = [factor for factor in node.args
                   if factor.free_symbols & inputs]
        # A fitted internal constant remains a factor, even though its exact
        # value is unknown. Collapse purely scalar products to one slot.
        marker = (("constant", "negative" if any(
            factor.is_negative is True for factor in scalars)
            else "free"),) if scalars else ()
        return ("product", tuple(sorted(
            (*marker, *(_shape(factor, inputs) for factor in factors)),
            key=repr)))
    if isinstance(node, sp.Pow):
        exponent = ("power", str(node.exp)) if node.exp.is_Number else (
            "power-expression", _shape(node.exp, inputs))
        return ("pow", _shape(node.base, inputs), exponent)
    if isinstance(node, sp.Function):
        return (node.func.__name__, tuple(_shape(arg, inputs) for arg in node.args))
    raise FormulaNotEvaluable("unsupported-symbolic-shape")


def _term_shape(term, inputs):
    """Only response-scale factors outside the mechanism are amplitudes."""
    if isinstance(term, sp.Mul):
        factors = [factor for factor in term.args if factor.free_symbols & inputs]
        if not factors:
            return ("offset",)
        return _shape(sp.Mul(*factors), inputs)
    if not (term.free_symbols & inputs):
        return ("offset",)
    return _shape(term, inputs)


def _topology(expression, inputs):
    terms = expression.args if isinstance(expression, sp.Add) else (expression,)
    return tuple(sorted((_term_shape(term, inputs) for term in terms), key=repr))


def assess_formula_recovery(candidate: str, truth: str,
                            input_symbols: Sequence[str], *,
                            parameter_bindings: dict[str, float] | None = None) -> dict:
    """Separate literal, bound-parameter exact, and topology outcomes.

    A benchmark must register `input_symbols` and any true physical parameter
    bindings independently of the proposal. No response-based fitting happens
    here; numerical tolerance equivalence is deliberately outside this exact
    algebraic contract.
    """
    try:
        # The evaluator accepts exactly the candidate language that can enter
        # the opt-in finite bank. Truth may have unbound physical parameters.
        compile_fixed_formula_support(candidate, len(input_symbols))
        reference, parameters = _parse(truth, input_symbols, truth=True)
        proposal, _ = _parse(candidate, input_symbols, truth=False)
        inputs = frozenset(sp.Symbol(f"x{i}") for i in range(len(input_symbols)))
        structural = _topology(reference, inputs) == _topology(proposal, inputs)
        def same(target):
            # "Literal exact" is equality of SymPy's canonical construction,
            # not an unbounded theorem-proving request. Global simplify,
            # trigsimp and factor can have superlinear memory growth on a
            # finite but large candidate bank and are deliberately forbidden.
            return bool(proposal == target or proposal - target == 0)
        exact = None if parameters else same(reference)
        bound_exact = exact if not parameters else None
        if parameter_bindings is not None:
            if (not isinstance(parameter_bindings, dict)
                    or set(parameter_bindings) != set(parameters)
                    or any(type(value) not in (int, float)
                           or not math.isfinite(value)
                           for value in parameter_bindings.values())):
                raise FormulaNotEvaluable("missing-or-invalid-parameter-binding")
            if parameters:
                instantiated = reference.subs({sp.Symbol(name): sp.Rational(
                    str(parameter_bindings[name])) for name in parameters})
                bound_exact = same(instantiated)
        return {"schema": "prospective-formula-recovery-assessment-v1",
                "candidate_grammar": "expanded-formula-ast-v1",
                "applicability": "unbound-parameters" if parameters else "evaluated",
                "unbound_truth_parameters": list(parameters),
                "literal_exact": exact,
                "structural_topology": structural,
                "parameter_instantiated_exact": bound_exact,
                "parameter_bindings_registered": parameter_bindings is not None,
                "recovery_success_claimed": False}
    except (FormulaNotEvaluable, SyntaxError, ValueError) as error:
        return {"schema": "prospective-formula-recovery-assessment-v1",
                "candidate_grammar": "expanded-formula-ast-v1",
                "applicability": "not-evaluable", "reason": str(error),
                "literal_exact": None, "structural_topology": None,
                "parameter_instantiated_exact": None,
                "parameter_bindings_registered": parameter_bindings is not None,
                "recovery_success_claimed": False}
