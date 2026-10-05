"""Opt-in typed AST composition of witnessed terms, without response access.

References resolve to frozen structural terms, never to their fitted outer
amplitudes. A proposal carries every inner literal in its support identity.
"""
from __future__ import annotations

import json
import math
from hashlib import sha256
from collections.abc import Mapping, Sequence

import numpy as np

from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    compile_fixed_formula_support, decode_fixed_formula_term,
)
from hypothesis_mvp.pcpi.reference.basis import design_matrix


_OPERATORS = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}
_FUNCTIONS = {"sin", "cos", "tanh", "exp", "log", "sqrt", "Abs"}
_TOTAL_OPERATORS = {"safe_div", "signed_log", "safe_sqrt", "decay_exp"}


class ExpandedFormulaDomainError(ValueError):
    """A compiled support is undefined on one frozen covariate domain."""

    def __init__(self, domain_name: str) -> None:
        super().__init__(
            f"expanded formula undefined on registered domain: {domain_name}")
        self.domain_name = str(domain_name)


def certify_expanded_formula_domains(
    expression: str,
    n_features: int,
    domains: Sequence[tuple[str, np.ndarray]],
) -> dict[str, object]:
    """Certify finite design columns on response-free registered X domains."""
    support = compile_fixed_formula_support(expression, n_features)
    rows = []
    for name, raw in domains:
        values = np.ascontiguousarray(raw, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != n_features
                or not len(values) or not np.all(np.isfinite(values))):
            raise ValueError("invalid registered formula covariate domain")
        try:
            matrix = design_matrix(values, support)
        except (ArithmeticError, FloatingPointError, OverflowError, ValueError):
            raise ExpandedFormulaDomainError(str(name)) from None
        if (matrix.shape != (len(values), len(support))
                or not np.all(np.isfinite(matrix))):
            raise ExpandedFormulaDomainError(str(name))
        rows.append({
            "name": str(name),
            "row_count": len(values),
            "covariate_sha256": sha256(
                str(values.shape).encode("ascii") + values.tobytes()
            ).hexdigest(),
        })
    return {
        "schema": "expanded-formula-domain-certificate-v1",
        "response_accessed": False,
        "support": list(support),
        "domains": rows,
        "passed": True,
    }


def witness_formula_terms(expression: str, n_features: int) -> tuple[str, ...]:
    # The support parser is the authoritative gate for all source equations.
    support = compile_fixed_formula_support(expression, n_features)
    return tuple(decode_fixed_formula_term(token, n_features)
                 if token != "intercept" else "1"
                 for token in support)


def compile_typed_formula_ast(
    tree: Mapping, lineage_ids: Sequence[str], engine_evidence: Sequence[Mapping],
    n_features: int,
) -> tuple[str, tuple[str, ...]]:
    """Compile one lineage-bound tree with generic, bounded algebraic nodes."""
    if (not isinstance(tree, Mapping) or type(n_features) is not int
            or n_features < 1 or len(lineage_ids) < 2
            or len(set(lineage_ids)) != len(lineage_ids)
            or len(json.dumps(tree, sort_keys=True, default=str)) > 4096):
        raise ValueError("invalid expanded typed AST contract")
    witnesses = {str(row.get("lineage_id") or ""): row
                 for row in engine_evidence}
    if any(lineage not in witnesses for lineage in lineage_ids):
        raise ValueError("unknown witnessed lineage")
    terms = {lineage: witness_formula_terms(
        str(witnesses[lineage]["expression"]), n_features)
        for lineage in lineage_ids}
    used: set[str] = set()
    budget = 0

    def render(node, depth):
        nonlocal budget
        budget += 1
        if budget > 128 or depth > 20 or not isinstance(node, Mapping):
            raise ValueError("expanded typed AST budget exceeded")
        if set(node) == {"ref"}:
            ref = node["ref"]
            if not isinstance(ref, Mapping) or set(ref) != {"lineage_id", "term_index"}:
                raise ValueError("malformed witness reference")
            lineage, index = ref["lineage_id"], ref["term_index"]
            if (lineage not in terms or type(index) is not int
                    or index < 0 or index >= len(terms[lineage])):
                raise ValueError("unregistered witness term")
            used.add(lineage)
            return f"({terms[lineage][index]})"
        if set(node) == {"const"}:
            value = node["const"]
            if type(value) not in (int, float):
                raise ValueError("typed constant must be numeric")
            try:
                finite = math.isfinite(float(value))
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("nonfinite typed constant")
            return repr(value)
        if set(node) != {"op", "args"} or not isinstance(node["args"], list):
            raise ValueError("invalid typed formula node")
        op, children = node["op"], node["args"]
        if op in _OPERATORS and len(children) == 2:
            left, right = (render(child, depth + 1) for child in children)
            return f"({left}{_OPERATORS[op]}{right})"
        if op == "safe_div" and len(children) == 2:
            left, right = (render(child, depth + 1) for child in children)
            return f"({left}/(1+Abs({right})))"
        if op == "neg" and len(children) == 1:
            return f"(-{render(children[0], depth + 1)})"
        if op == "signed_log" and len(children) == 1:
            return f"log(1+Abs({render(children[0], depth + 1)}))"
        if op == "safe_sqrt" and len(children) == 1:
            return f"sqrt(Abs({render(children[0], depth + 1)}))"
        if op == "decay_exp" and len(children) == 1:
            return f"exp(-Abs({render(children[0], depth + 1)}))"
        if op in _FUNCTIONS and len(children) == 1:
            return f"{op}({render(children[0], depth + 1)})"
        raise ValueError("operator outside expanded typed grammar")

    expression = render(tree, 0)
    if used != set(lineage_ids):
        raise ValueError("every declared parent lineage must contribute")
    support = compile_fixed_formula_support(expression, n_features)
    parent_supports = {compile_fixed_formula_support(
        str(witnesses[lineage]["expression"]), n_features)
        for lineage in lineage_ids}
    if support in parent_supports:
        raise ValueError("expanded composition has no structural novelty")
    return expression, support
