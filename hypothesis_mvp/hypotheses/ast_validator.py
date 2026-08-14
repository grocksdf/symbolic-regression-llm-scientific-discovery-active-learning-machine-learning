"""Safe, evaluation-free validation for symbolic hypothesis expressions.

The validator accepts a deliberately small mathematical expression language.
It parses Python expression syntax, walks the resulting AST, and never calls
``eval`` or imports expression-selected code.  The canonical expression and
its digest can therefore be stored as auditable hypothesis evidence.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Collection, Mapping


DEFAULT_FUNCTION_ARITIES: Mapping[str, tuple[int, ...]] = {
    "abs": (1,),
    "sqrt": (1,),
    "exp": (1,),
    "log": (1,),
    "log10": (1,),
    "sin": (1,),
    "cos": (1,),
    "tan": (1,),
    "asin": (1,),
    "acos": (1,),
    "atan": (1,),
    "sinh": (1,),
    "cosh": (1,),
    "tanh": (1,),
    "minimum": (2,),
    "maximum": (2,),
}
DEFAULT_CONSTANTS = frozenset({"pi", "e"})


class ExpressionValidationError(ValueError):
    """Raised when an expression falls outside the safe hypothesis grammar."""


@dataclass(frozen=True)
class ASTValidationResult:
    canonical_expression: str
    expression_sha256: str
    variables_used: tuple[str, ...]
    functions_used: tuple[str, ...]
    node_count: int
    maximum_depth: int


class SafeExpressionValidator:
    """Validate a mathematical RHS against an explicit AST allowlist."""

    def __init__(
        self,
        *,
        function_arities: Mapping[str, tuple[int, ...]] = DEFAULT_FUNCTION_ARITIES,
        constants: Collection[str] = DEFAULT_CONSTANTS,
        maximum_nodes: int = 256,
        maximum_depth: int = 32,
        maximum_absolute_constant: float = 1.0e12,
        maximum_absolute_literal_exponent: float = 12.0,
    ) -> None:
        self.function_arities = {
            str(name): tuple(int(value) for value in arities)
            for name, arities in function_arities.items()
        }
        self.constants = frozenset(str(value) for value in constants)
        self.maximum_nodes = int(maximum_nodes)
        self.maximum_depth = int(maximum_depth)
        self.maximum_absolute_constant = float(maximum_absolute_constant)
        self.maximum_absolute_literal_exponent = float(
            maximum_absolute_literal_exponent
        )
        if self.maximum_nodes < 1 or self.maximum_depth < 1:
            raise ValueError("AST limits must be positive")

    def validate(
        self,
        expression: str,
        *,
        allowed_variables: Collection[str],
    ) -> ASTValidationResult:
        if not isinstance(expression, str) or not expression.strip():
            raise ExpressionValidationError("expression must be a non-empty string")
        variables = frozenset(str(value) for value in allowed_variables)
        if any(not name.isidentifier() for name in variables):
            raise ValueError("allowed variable names must be Python identifiers")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as error:
            raise ExpressionValidationError(
                f"expression is not valid expression syntax: {error.msg}"
            ) from error

        nodes = list(ast.walk(tree))
        if len(nodes) > self.maximum_nodes:
            raise ExpressionValidationError(
                f"expression has {len(nodes)} AST nodes; limit is {self.maximum_nodes}"
            )
        depth = self._depth(tree)
        if depth > self.maximum_depth:
            raise ExpressionValidationError(
                f"expression depth is {depth}; limit is {self.maximum_depth}"
            )

        variables_used: set[str] = set()
        functions_used: set[str] = set()
        self._validate_node(
            tree,
            variables=variables,
            variables_used=variables_used,
            functions_used=functions_used,
        )
        canonical = ast.unparse(tree.body)
        return ASTValidationResult(
            canonical_expression=canonical,
            expression_sha256=sha256(canonical.encode("utf-8")).hexdigest(),
            variables_used=tuple(sorted(variables_used)),
            functions_used=tuple(sorted(functions_used)),
            node_count=len(nodes),
            maximum_depth=depth,
        )

    def _depth(self, node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(self._depth(child) for child in children)

    def _validate_node(
        self,
        node: ast.AST,
        *,
        variables: frozenset[str],
        variables_used: set[str],
        functions_used: set[str],
    ) -> None:
        if isinstance(node, ast.Expression):
            self._visit(node.body, variables, variables_used, functions_used)
            return
        if isinstance(node, ast.BinOp):
            self._validate_binary(node, variables, variables_used, functions_used)
            return
        if isinstance(node, ast.UnaryOp):
            self._validate_unary(node, variables, variables_used, functions_used)
            return
        if isinstance(node, ast.Call):
            self._validate_call(node, variables, variables_used, functions_used)
            return
        if self._validate_leaf(node, variables, variables_used):
            return
        raise ExpressionValidationError(
            f"AST node {type(node).__name__} is not allowed"
        )

    def _visit(
        self, node: ast.AST, variables: frozenset[str],
        variables_used: set[str], functions_used: set[str],
    ) -> None:
        self._validate_node(
            node, variables=variables, variables_used=variables_used,
            functions_used=functions_used,
        )

    def _validate_binary(
        self, node: ast.BinOp, variables: frozenset[str],
        variables_used: set[str], functions_used: set[str],
    ) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise ExpressionValidationError(f"binary operator {type(node.op).__name__} is not allowed")
        if isinstance(node.op, ast.Pow):
            literal = self._signed_numeric_literal(node.right)
            if literal is not None and abs(literal) > self.maximum_absolute_literal_exponent:
                raise ExpressionValidationError("literal exponent exceeds the configured safety limit")
        if isinstance(node.op, ast.Div) and self._signed_numeric_literal(node.right) == 0.0:
            raise ExpressionValidationError("literal division by zero")
        self._visit(node.left, variables, variables_used, functions_used)
        self._visit(node.right, variables, variables_used, functions_used)

    def _validate_unary(
        self, node: ast.UnaryOp, variables: frozenset[str],
        variables_used: set[str], functions_used: set[str],
    ) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ExpressionValidationError(f"unary operator {type(node.op).__name__} is not allowed")
        self._visit(node.operand, variables, variables_used, functions_used)

    def _validate_call(
        self, node: ast.Call, variables: frozenset[str],
        variables_used: set[str], functions_used: set[str],
    ) -> None:
        if not isinstance(node.func, ast.Name):
            raise ExpressionValidationError("only direct calls to allowlisted mathematical functions are allowed")
        function = node.func.id
        if function not in self.function_arities:
            raise ExpressionValidationError(f"function {function!r} is not allowlisted")
        if node.keywords:
            raise ExpressionValidationError("keyword arguments are not allowed")
        if len(node.args) not in self.function_arities[function]:
            expected = ", ".join(str(value) for value in self.function_arities[function])
            raise ExpressionValidationError(f"function {function!r} expects arity {expected}")
        functions_used.add(function)
        for argument in node.args:
            self._visit(argument, variables, variables_used, functions_used)

    def _validate_leaf(
        self, node: ast.AST, variables: frozenset[str], variables_used: set[str],
    ) -> bool:
        if isinstance(node, ast.Name):
            if node.id in variables:
                variables_used.add(node.id)
                return True
            if node.id in self.constants:
                return True
            raise ExpressionValidationError(f"name {node.id!r} is not declared")
        if not isinstance(node, ast.Constant):
            return False
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionValidationError("only real numeric constants are allowed")
        value = float(node.value)
        if not math.isfinite(value):
            raise ExpressionValidationError("numeric constants must be finite")
        if abs(value) > self.maximum_absolute_constant:
            raise ExpressionValidationError("numeric constant exceeds the configured safety limit")
        return True

    @staticmethod
    def _signed_numeric_literal(node: ast.AST) -> float | None:
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            if isinstance(node.value, (int, float)):
                return float(node.value)
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and not isinstance(node.operand.value, bool)
            and isinstance(node.operand.value, (int, float))
        ):
            value = float(node.operand.value)
            return -value if isinstance(node.op, ast.USub) else value
        return None
