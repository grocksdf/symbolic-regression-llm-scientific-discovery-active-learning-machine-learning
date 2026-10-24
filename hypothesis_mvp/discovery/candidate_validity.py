"""Response-free definedness on registered covariate points, not a proof on a box.

Keep the unsimplified hypothesis. No clipping, epsilon, absolute value or retry
is inserted. Dangerous operators remain in the language with explicit domain
obligations. Failures precede expensive fitting and retain their exact stage.
"""
from __future__ import annotations

import ast
from decimal import Decimal
from hashlib import sha256

import numpy as np

from hypothesis_mvp.hypotheses.expression_limits import AST_NODE_BUDGET
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    _check, _constant, _FUNCTIONS,
)


class CandidateValidityError(ValueError):
    def __init__(self, reason, detail, node):
        self.reason = reason
        self.detail = detail
        self.node = ast.unparse(node)
        super().__init__(f"{reason}: {detail}: {self.node}")


def _tree(expression, n_features):
    if not isinstance(expression, str) or not 0 < len(expression) <= 4096:
        raise ValueError("invalid validity expression budget")
    parsed = ast.parse(expression, mode="eval")
    if len(list(ast.walk(parsed))) > AST_NODE_BUDGET:
        raise ValueError("validity AST exceeds finite budget")
    root = parsed.body
    for node in ast.walk(root):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                finite = np.isfinite(float(node.value))
            except OverflowError:
                finite = False
            if not finite:
                raise CandidateValidityError("numerical_nonfinite", "nonfinite literal", node)
            if (type(node.value) is float and node.value == 0
                    and Decimal(ast.get_source_segment(expression, node).replace("_", "")) != 0):
                raise CandidateValidityError("numerical_nonfinite", "literal underflow", node)
    _check(root, n_features)
    return root


def domain_obligations(expression, n_features):
    """Return obligations without asserting that syntax discharges them."""
    result = []
    for node in ast.walk(_tree(expression, n_features)):
        condition = None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            condition = "denominator != 0"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = _constant(node.right)
            parts = []
            if not exponent.is_integer():
                parts.append("base >= 0 (real floating power)")
            if exponent < 0:
                parts.append("base != 0")
            condition = "; ".join(parts) or None
        elif isinstance(node, ast.Call):
            condition = {"log": "argument > 0", "sqrt": "argument >= 0",
                         "exp": "finite representable exponential"}.get(node.func.id)
        if condition:
            result.append({"expression": ast.unparse(node), "requires": condition})
    return tuple(result)


def _require(condition, node, detail):
    if not np.all(condition):
        raise CandidateValidityError("domain_undefined", detail, node)


def _evaluate(node, values):
    if isinstance(node, ast.Constant):
        return np.float64(_constant(node))
    if isinstance(node, ast.Name):
        return values[:, int(node.id[1:])]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, values)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left, values), _evaluate(node.right, values)
        if isinstance(node.op, ast.Div):
            _require(right != 0, node, "zero denominator")
            return np.divide(left, right)
        if isinstance(node.op, ast.Pow):
            exponent = _constant(node.right)
            if not exponent.is_integer():
                _require(left >= 0, node, "negative fractional-power base")
            if exponent < 0:
                _require(left != 0, node, "zero negative-power base")
            return np.power(left, right)
        operation = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply}
        return operation[type(node.op)](left, right)
    if isinstance(node, ast.Call):
        argument = _evaluate(node.args[0], values)
        if node.func.id == "log":
            _require(argument > 0, node, "nonpositive logarithm argument")
        if node.func.id == "sqrt":
            _require(argument >= 0, node, "negative square-root argument")
        return _FUNCTIONS[node.func.id](argument)
    raise ValueError("unsupported validity node")


def numerical_definedness_preflight(expression, covariates):
    """Check every supplied legal point, including intermediate float errors.

    The caller supplies development/selection covariates and registered action
    points only. Admission/reporting/heldout rows and responses are excluded.
    This finite-point check is deliberately not universal domain certification.
    """
    values = np.asarray(covariates, dtype=float)
    if (values.ndim != 2 or not len(values) or not values.shape[1]
            or not np.all(np.isfinite(values))):
        raise ValueError("invalid registered preflight covariates")
    root = _tree(expression, values.shape[1])
    try:
        with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
            result = np.broadcast_to(_evaluate(root, values), (len(values),))
        if not np.all(np.isfinite(result)):
            raise CandidateValidityError("numerical_nonfinite", "nonfinite output", root)
    except (FloatingPointError, OverflowError) as error:
        raise CandidateValidityError("numerical_nonfinite", str(error), root) from error
    return {"schema": "candidate-numerical-definedness-v1", "passed": True,
            "point_count": len(values), "domain_obligations": list(
                domain_obligations(expression, values.shape[1])),
            "covariate_identity": sha256(values.tobytes()).hexdigest(),
            "scope": "supplied-registered-covariate-points-only",
            "response_accessed": False, "semantics_repaired": False}


def proposal_domain_context(covariates):
    values = np.asarray(covariates, dtype=float)
    if (values.ndim != 2 or not len(values) or not values.shape[1]
            or not np.all(np.isfinite(values))):
        raise ValueError("invalid proposal covariates")
    return {"schema": "proposal-domain-context-v1",
            "scope": "registered-covariate-points-not-cartesian-box",
            "variables": [{"symbol": f"x{i}", "type": "real",
                           "min": float(column.min()), "max": float(column.max()),
                           "includes_zero": bool(np.any(column == 0))}
                          for i, column in enumerate(values.T)],
            "operator_obligations": {"div": "denominator != 0", "log": "argument > 0",
                "sqrt": "argument >= 0", "fractional_power": "base >= 0",
                "negative_power": "base != 0",
                "all": "finite evaluation, including intermediate overflow/underflow"},
            "implicit_semantic_repairs_allowed": False, "response_accessed": False}
