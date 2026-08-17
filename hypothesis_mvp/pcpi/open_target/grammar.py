"""A proper countably-open typed expression prior and exact finite slices.

The target prior is not depth truncated.  It first draws a positive AST node
count from a geometric distribution and is then uniform over the finite set
of well-typed raw ASTs at that size.  ``enumerate_slice`` is an exact-reference
operation conditioned on a registered maximum size; it reports, rather than
hides, the omitted prior tail mass.

P3F.2 initially registers only dimensionless real-valued expressions with
``one``, variables, unary negation, addition, and multiplication.  This small
language is sufficient to prove normalization, semantic class aggregation,
and trans-dimensional inference identities without claiming search efficacy.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Literal

import numpy as np


P3F2_GRAMMAR_SCHEMA = "pcpi-p3f2-countably-open-typed-grammar-v1"
P3F2_EXPRESSION_TYPE = "dimensionless-real"
ExpressionOperator = Literal["one", "variable", "neg", "add", "mul"]
PolynomialKey = tuple[tuple[tuple[int, ...], int], ...]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class TypedExpression:
    operator: ExpressionOperator
    children: tuple["TypedExpression", ...] = ()
    variable_index: int | None = None
    expression_type: str = P3F2_EXPRESSION_TYPE

    def __post_init__(self) -> None:
        arity = {"one": 0, "variable": 0, "neg": 1, "add": 2, "mul": 2}
        if self.operator not in arity or len(self.children) != arity[self.operator]:
            raise ValueError("typed expression operator has invalid arity")
        if self.expression_type != P3F2_EXPRESSION_TYPE:
            raise ValueError("P3F.2 registers dimensionless-real expressions only")
        if self.operator == "variable":
            if self.variable_index is None or self.variable_index < 0:
                raise ValueError("variable expressions require a non-negative index")
        elif self.variable_index is not None:
            raise ValueError("only variable expressions may carry a variable index")
        if any(child.expression_type != self.expression_type for child in self.children):
            raise ValueError("typed operator children must share the registered type")

    @property
    def node_count(self) -> int:
        return 1 + sum(child.node_count for child in self.children)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": self.operator,
            "type": self.expression_type,
            "children": [child.to_dict() for child in self.children],
        }
        if self.variable_index is not None:
            result["variable_index"] = self.variable_index
        return result

    @property
    def raw_ast_id(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_string(self) -> str:
        if self.operator == "one":
            return "1"
        if self.operator == "variable":
            return f"x{self.variable_index}"
        if self.operator == "neg":
            return f"(-{self.children[0].to_string()})"
        symbol = "+" if self.operator == "add" else "*"
        return f"({self.children[0].to_string()}{symbol}{self.children[1].to_string()})"


def one() -> TypedExpression:
    return TypedExpression("one")


def variable(index: int) -> TypedExpression:
    return TypedExpression("variable", variable_index=index)


def neg(child: TypedExpression) -> TypedExpression:
    return TypedExpression("neg", (child,))


def add(left: TypedExpression, right: TypedExpression) -> TypedExpression:
    return TypedExpression("add", (left, right))


def mul(left: TypedExpression, right: TypedExpression) -> TypedExpression:
    return TypedExpression("mul", (left, right))


def _add_polynomials(
    left: dict[tuple[int, ...], int],
    right: dict[tuple[int, ...], int],
) -> dict[tuple[int, ...], int]:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, 0) + coefficient
        if result[monomial] == 0:
            del result[monomial]
    return result


def _multiply_polynomials(
    left: dict[tuple[int, ...], int],
    right: dict[tuple[int, ...], int],
) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = {}
    for left_power, left_coefficient in left.items():
        for right_power, right_coefficient in right.items():
            power = tuple(a + b for a, b in zip(left_power, right_power, strict=True))
            result[power] = result.get(power, 0) + left_coefficient * right_coefficient
    return {power: coefficient for power, coefficient in result.items() if coefficient}


def polynomial_key(expression: TypedExpression, feature_count: int) -> PolynomialKey:
    """Return an exact semantic key for the registered algebraic language."""

    if feature_count < 1:
        raise ValueError("feature count must be positive")

    def visit(node: TypedExpression) -> dict[tuple[int, ...], int]:
        if node.operator == "one":
            return {(0,) * feature_count: 1}
        if node.operator == "variable":
            assert node.variable_index is not None
            if node.variable_index >= feature_count:
                raise ValueError("expression variable exceeds the registered feature count")
            powers = [0] * feature_count
            powers[node.variable_index] = 1
            return {tuple(powers): 1}
        if node.operator == "neg":
            return {power: -coefficient for power, coefficient in visit(node.children[0]).items()}
        if node.operator == "add":
            return _add_polynomials(visit(node.children[0]), visit(node.children[1]))
        if node.operator == "mul":
            return _multiply_polynomials(visit(node.children[0]), visit(node.children[1]))
        raise AssertionError(node.operator)

    polynomial = visit(expression)
    return tuple(sorted(polynomial.items()))


def equivalence_class_id(expression: TypedExpression, feature_count: int) -> str:
    payload = {
        "schema": "pcpi-p3f2-exact-polynomial-equivalence-v1",
        "feature_count": feature_count,
        "polynomial": [
            {"powers": list(powers), "coefficient": coefficient}
            for powers, coefficient in polynomial_key(expression, feature_count)
        ],
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def evaluate_expression(expression: TypedExpression, actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("expression actions must be a non-empty finite matrix")

    def visit(node: TypedExpression) -> np.ndarray:
        if node.operator == "one":
            return np.ones(len(values), dtype=float)
        if node.operator == "variable":
            assert node.variable_index is not None
            if node.variable_index >= values.shape[1]:
                raise ValueError("expression variable exceeds action dimension")
            return values[:, node.variable_index]
        if node.operator == "neg":
            return -visit(node.children[0])
        if node.operator == "add":
            return visit(node.children[0]) + visit(node.children[1])
        if node.operator == "mul":
            return visit(node.children[0]) * visit(node.children[1])
        raise AssertionError(node.operator)

    result = np.asarray(visit(expression), dtype=float)
    if result.shape != (len(values),) or not np.all(np.isfinite(result)):
        raise ValueError("expression evaluation is not a finite response vector")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class PriorNormalizationCertificate:
    maximum_nodes: int
    enumerated_expression_count: int
    enumerated_prior_mass: float
    analytic_slice_mass: float
    omitted_tail_mass: float
    maximum_absolute_error: float


class CountablyOpenTypedGrammar:
    """Proper response-independent prior over a countably infinite AST set."""

    def __init__(self, feature_count: int, continuation_probability: float = 0.45) -> None:
        if feature_count < 1:
            raise ValueError("feature count must be positive")
        if not 0.0 < continuation_probability < 1.0:
            raise ValueError("continuation probability must lie strictly inside (0, 1)")
        self.feature_count = int(feature_count)
        self.continuation_probability = float(continuation_probability)
        self._count_cache: dict[int, int] = {1: self.feature_count + 1}
        self._enumeration_cache: dict[int, tuple[TypedExpression, ...]] = {
            1: (one(),) + tuple(variable(index) for index in range(self.feature_count))
        }

    @property
    def stable_hash(self) -> str:
        return sha256(
            _canonical_json(
                {
                    "schema": P3F2_GRAMMAR_SCHEMA,
                    "expression_type": P3F2_EXPRESSION_TYPE,
                    "feature_count": self.feature_count,
                    "continuation_probability": self.continuation_probability,
                    "terminals": ["one", "variable"],
                    "unary_operators": ["neg"],
                    "binary_operators": ["add", "mul"],
                    "size_prior": "geometric-on-positive-node-count",
                }
            ).encode("utf-8")
        ).hexdigest()

    def expression_count(self, node_count: int) -> int:
        if node_count < 1:
            raise ValueError("node count must be positive")
        for size in range(2, node_count + 1):
            if size in self._count_cache:
                continue
            unary = self._count_cache[size - 1]
            binary = 2 * sum(
                self._count_cache[left] * self._count_cache[size - 1 - left]
                for left in range(1, size - 1)
            )
            self._count_cache[size] = unary + binary
        return self._count_cache[node_count]

    def expressions_of_size(self, node_count: int) -> tuple[TypedExpression, ...]:
        self.expression_count(node_count)
        for size in range(2, node_count + 1):
            if size in self._enumeration_cache:
                continue
            expressions: list[TypedExpression] = [
                neg(child) for child in self._enumeration_cache[size - 1]
            ]
            for left_size in range(1, size - 1):
                right_size = size - 1 - left_size
                for left in self._enumeration_cache[left_size]:
                    for right in self._enumeration_cache[right_size]:
                        expressions.append(add(left, right))
                        expressions.append(mul(left, right))
            result = tuple(expressions)
            if len(result) != self._count_cache[size] or len(set(result)) != len(result):
                raise AssertionError("typed grammar enumeration and counting disagree")
            self._enumeration_cache[size] = result
        return self._enumeration_cache[node_count]

    def enumerate_slice(self, maximum_nodes: int) -> tuple[TypedExpression, ...]:
        if maximum_nodes < 1:
            raise ValueError("maximum nodes must be positive")
        return tuple(
            expression
            for size in range(1, maximum_nodes + 1)
            for expression in self.expressions_of_size(size)
        )

    def size_probability(self, node_count: int) -> float:
        if node_count < 1:
            raise ValueError("node count must be positive")
        rho = self.continuation_probability
        return (1.0 - rho) * rho ** (node_count - 1)

    def prior_probability(self, expression: TypedExpression) -> float:
        polynomial_key(expression, self.feature_count)
        return self.size_probability(expression.node_count) / self.expression_count(
            expression.node_count
        )

    def slice_mass(self, maximum_nodes: int) -> float:
        if maximum_nodes < 1:
            raise ValueError("maximum nodes must be positive")
        return 1.0 - self.continuation_probability ** maximum_nodes

    def tail_mass(self, maximum_nodes: int) -> float:
        if maximum_nodes < 1:
            raise ValueError("maximum nodes must be positive")
        return self.continuation_probability ** maximum_nodes

    def normalization_certificate(
        self, maximum_nodes: int
    ) -> PriorNormalizationCertificate:
        expressions = self.enumerate_slice(maximum_nodes)
        enumerated = math.fsum(self.prior_probability(item) for item in expressions)
        analytic = self.slice_mass(maximum_nodes)
        return PriorNormalizationCertificate(
            maximum_nodes=maximum_nodes,
            enumerated_expression_count=len(expressions),
            enumerated_prior_mass=enumerated,
            analytic_slice_mass=analytic,
            omitted_tail_mass=self.tail_mass(maximum_nodes),
            maximum_absolute_error=max(
                abs(enumerated - analytic),
                abs(enumerated + self.tail_mass(maximum_nodes) - 1.0),
            ),
        )


def aggregate_equivalence_mass(
    expressions: tuple[TypedExpression, ...],
    probabilities: np.ndarray,
    feature_count: int,
) -> dict[str, float]:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(expressions) != len(values) or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("expressions and finite non-negative masses must align")
    result: dict[str, float] = {}
    for expression, probability in zip(expressions, values, strict=True):
        identifier = equivalence_class_id(expression, feature_count)
        result[identifier] = result.get(identifier, 0.0) + float(probability)
    return result
