"""Construct mathematical SymPy trees without evaluating source strings."""
import ast
import math

import sympy as sp


def parse_math_expression(text, symbols, functions, *, evaluate=False):
    tree = ast.parse(text, mode="eval")
    if len(list(ast.walk(tree))) > 4096:
        raise ValueError("expression_AST_budget_exceeded")

    def build(node, depth=0):
        if depth > 64:
            raise ValueError("expression_AST_depth_exceeded")
        descend = lambda child: build(child, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value):
                raise ValueError("nonfinite_numeric_literal")
            return sp.Integer(node.value) if type(node.value) is int else sp.Float(node.value)
        if isinstance(node, ast.Name):
            if node.id in symbols:
                return symbols[node.id]
            if node.id in {"pi", "E"}:
                return {"pi": sp.pi, "E": sp.E}[node.id]
            raise ValueError("unknown_symbols:" + node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = descend(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            return -value if value.is_Number else sp.Mul(-1, value, evaluate=evaluate)
        if isinstance(node, ast.BinOp):
            if type(node.op) not in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
                raise ValueError("unsupported_mathematical_operator")
            left, right = descend(node.left), descend(node.right)
            if isinstance(node.op, ast.Add):
                return sp.Add(left, right, evaluate=evaluate)
            if isinstance(node.op, ast.Sub):
                return sp.Add(left, sp.Mul(-1, right, evaluate=evaluate), evaluate=evaluate)
            if isinstance(node.op, ast.Mult):
                return sp.Mul(left, right, evaluate=evaluate)
            if isinstance(node.op, ast.Div):
                return sp.Mul(left, sp.Pow(right, -1, evaluate=evaluate), evaluate=evaluate)
            if right.is_number and (not right.is_finite or abs(float(right)) > 64):
                raise ValueError("literal_exponent_budget_exceeded")
            return sp.Pow(left, right, evaluate=evaluate)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and not node.keywords):
            arity = 2 if node.func.id in {"abspow", "signpow"} else 1
            if len(node.args) != arity:
                raise ValueError("invalid_mathematical_function_arity")
            return functions[node.func.id](*(descend(arg) for arg in node.args),
                                           evaluate=evaluate)
        raise ValueError("unsafe_expression_AST:" + type(node).__name__)

    return build(tree.body)
