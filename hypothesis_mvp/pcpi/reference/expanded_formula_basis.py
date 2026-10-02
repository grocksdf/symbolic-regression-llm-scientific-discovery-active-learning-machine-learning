"""Bounded fixed-inner-parameter formula basis for finite reference banks.

An outer linear coefficient remains conjugate. Every constant *inside* a
nonlinear transform is fixed as part of the bank member and its identity.
This is a correctness primitive, not a search over unknown continuous
parameters, nor authorization for the historical measured executor.
"""
from __future__ import annotations

import ast
import base64
import math
import re

import numpy as np


PREFIX = "formula_ast_v1:"
_VARIABLE = re.compile(r"x(?:0|[1-9][0-9]*)\Z")
_FUNCTIONS = {"sin": np.sin, "cos": np.cos, "tanh": np.tanh,
              "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
              "Abs": np.abs, "abs": np.abs}


def _constant(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        try:
            value = float(node.value)
        except OverflowError as error:
            raise ValueError("nonfinite formula constant") from error
        if not math.isfinite(value):
            raise ValueError("nonfinite formula constant")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        value = _constant(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = _constant(node.left), _constant(node.right)
        if right == 0:
            raise ValueError("zero formula constant denominator")
        return left / right
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left, right = _constant(node.left), _constant(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        else:
            result = left * right
        if not math.isfinite(result):
            raise ValueError("nonfinite formula scalar")
        return result
    raise ValueError("formula exponent or scalar is not fixed")


def _check(node, n_features: int):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        _constant(node)
        return
    if isinstance(node, ast.Name) and _VARIABLE.fullmatch(node.id):
        if int(node.id[1:]) >= n_features:
            raise ValueError("formula feature exceeds frozen dimension")
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        _check(node.operand, n_features)
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
            raise ValueError("formula operator outside bounded grammar")
        _check(node.left, n_features)
        if isinstance(node.op, ast.Pow):
            exponent = _constant(node.right)
            if not math.isfinite(exponent) or abs(exponent) > 4:
                raise ValueError("formula exponent outside bounded grammar")
        else:
            _check(node.right, n_features)
        return
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and len(node.args) == 1
            and not node.keywords):
        _check(node.args[0], n_features)
        return
    raise ValueError("formula node outside bounded grammar")


def _parse(expression, n_features):
    if (type(n_features) is not int or n_features < 1
            or not isinstance(expression, str) or not 0 < len(expression) <= 4096):
        raise ValueError("invalid fixed-formula contract")
    root = ast.parse(expression, mode="eval")
    if len(list(ast.walk(root))) > 128:
        raise ValueError("formula AST exceeds finite budget")
    _check(root.body, n_features)
    if not any(isinstance(node, ast.Name)
               and _VARIABLE.fullmatch(node.id) for node in ast.walk(root)):
        raise ValueError("fixed formula has no registered covariate")
    return root.body


def compile_fixed_formula_term(expression: str, n_features: int) -> str:
    """Compile one basis term; strip only its outer fitted numeric amplitude."""
    node = _parse(expression, n_features)
    while isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
        node = node.operand
    def multiply_factors(value):
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
            return multiply_factors(value.left) + multiply_factors(value.right)
        return [value]
    factors = multiply_factors(node)
    retained = []
    for factor in factors:
        try:
            amplitude = _constant(factor)
        except ValueError:
            retained.append(factor)
        else:
            if amplitude == 0:
                raise ValueError("zero outer formula amplitude")
    if len(retained) < len(factors):
        if not retained:
            raise ValueError("fixed formula has no structural covariate")
        node = retained[0]
        for factor in retained[1:]:
            node = ast.BinOp(left=node, op=ast.Mult(), right=factor)
    canonical = ast.unparse(node)
    _parse(canonical, n_features)
    encoded = base64.urlsafe_b64encode(canonical.encode("utf-8")).decode("ascii")
    return PREFIX + encoded.rstrip("=")


def compile_fixed_formula_support(expression: str, n_features: int
                                  ) -> tuple[str, ...]:
    """Compile additive external-amplitude terms into one finite support."""
    if not isinstance(expression, str) or not 0 < len(expression) <= 4096:
        raise ValueError("invalid bounded formula expression")
    root = ast.parse(expression, mode="eval")
    if len(list(ast.walk(root))) > 128:
        raise ValueError("formula AST exceeds finite budget")
    def summands(node):
        if not any(isinstance(child, ast.Name) and _VARIABLE.fullmatch(child.id)
                   for child in ast.walk(node)):
            return [node]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return summands(node.left) + summands(node.right)
        return [node]
    terms = []
    for node in summands(root.body):
        if not any(isinstance(child, ast.Name) and _VARIABLE.fullmatch(child.id)
                   for child in ast.walk(node)):
            if _constant(node) != 0:
                terms.append("intercept")
        else:
            terms.append(compile_fixed_formula_term(ast.unparse(node), n_features))
    if len(terms) > 16:
        raise ValueError("fixed formula support exceeds finite budget")
    if not terms:
        raise ValueError("empty fixed formula support")
    if len(set(terms)) != len(terms):
        raise ValueError("duplicate or cancelling fixed formula terms")
    return tuple(sorted(terms))


def _run(node, x):
    if isinstance(node, ast.Constant):
        return _constant(node)
    if isinstance(node, ast.Name):
        return x[:, int(node.id[1:])]
    if isinstance(node, ast.UnaryOp):
        value = _run(node.operand, x)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _run(node.left, x)
        if isinstance(node.op, ast.Pow):
            return np.power(left, _constant(node.right))
        right = _run(node.right, x)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
    if isinstance(node, ast.Call):
        return _FUNCTIONS[node.func.id](_run(node.args[0], x))
    raise ValueError("invalid compiled formula node")


def decode_fixed_formula_term(token: str, n_features: int) -> str:
    """Return the canonical expression only after bounded token validation."""
    if not isinstance(token, str) or not token.startswith(PREFIX):
        raise ValueError("not a fixed formula basis token")
    data = token[len(PREFIX):]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,6000}", data):
        raise ValueError("invalid formula token encoding")
    try:
        expression = base64.b64decode(data + "=" * (-len(data) % 4),
                                       altchars=b"-_", validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid formula token encoding") from error
    _parse(expression, n_features)
    if compile_fixed_formula_term(expression, n_features) != token:
        raise ValueError("noncanonical fixed formula token")
    return expression


def evaluate_fixed_formula_term(token: str, x: np.ndarray) -> np.ndarray:
    """Decode, revalidate, and evaluate one declared finite basis term."""
    values = np.asarray(x, dtype=float)
    if (values.ndim != 2 or not len(values) or not np.all(np.isfinite(values))):
        raise ValueError("invalid formula input domain")
    expression = decode_fixed_formula_term(token, values.shape[1])
    node = _parse(expression, values.shape[1])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            result = np.broadcast_to(_run(node, values), (len(values),))
        except (FloatingPointError, ValueError, OverflowError) as error:
            raise ValueError("fixed formula undefined on registered domain") from error
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite fixed formula design column")
    return np.asarray(result, dtype=float)
