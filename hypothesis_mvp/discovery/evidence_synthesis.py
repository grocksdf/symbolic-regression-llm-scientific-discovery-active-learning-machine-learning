"""Compile typed Scientist evidence directives into closed-basis hypotheses."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from .pcpi_adapter import additive_closed_form, structural_terms
from .scientist_policy import SynthesisDirective


def _summands(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_summands(node.left), *_summands(node.right)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return [*_summands(node.left),
                *[ast.UnaryOp(ast.USub(), row) for row in _summands(node.right)]]
    return [node]


def _support_map(expression: str, n_features: int) -> dict[str, str]:
    root = ast.parse(
        additive_closed_form(str(expression), n_features), mode="eval").body
    result = {}
    for node in _summands(root):
        text = ast.unparse(ast.fix_missing_locations(node))
        terms = structural_terms(text, n_features)
        if len(terms) != 1:
            raise ValueError("synthesis summand must map to one closed support")
        result.setdefault(terms[0], text)
    return result


def _compile_expression(
    operation: str, expressions: Sequence[str], n_features: int,
) -> tuple[str, tuple[str, ...]]:
    maps = [_support_map(expression, n_features) for expression in expressions]
    if operation in {"UNION_SUPPORTS", "AUGMENT_BASE"}:
        supports = set().union(*(set(row) for row in maps))
    elif operation == "INTERSECTION_SUPPORTS":
        supports = set(maps[0]).intersection(*(set(row) for row in maps[1:]))
    else:
        raise ValueError("unknown evidence synthesis operation")
    if "intercept" not in supports:
        supports.add("intercept")
    if len(supports) < 2:
        raise ValueError("evidence synthesis produced fewer than two supports")
    ordered = sorted(supports)
    pieces = []
    for support in ordered:
        source = next((row[support] for row in maps if support in row), None)
        if support == "intercept" and source is None:
            source = "1"
        if source is None:
            raise RuntimeError("missing compiled support source")
        pieces.append(f"({source})")
    expression = " + ".join(pieces)
    if tuple(sorted(structural_terms(expression, n_features))) != tuple(ordered):
        raise ValueError("compiled evidence support identity changed")
    return expression, tuple(ordered)


def compile_evidence_synthesis(
    directives: Sequence[SynthesisDirective],
    engine_evidence: Sequence[Mapping[str, Any]],
    n_features: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compile only lineage-bound engine evidence; never parse free-form text."""
    evidence = {
        str(row.get("lineage_id") or ""): dict(row)
        for row in engine_evidence if str(row.get("lineage_id") or "")}
    candidates, records, seen = [], [], set()
    for directive in directives:
        missing = [value for value in directive.lineage_ids
                   if value not in evidence]
        if missing:
            raise ValueError("synthesis directive references unknown lineage")
        rows = [evidence[value] for value in directive.lineage_ids]
        expression, supports = _compile_expression(
            directive.operation,
            [str(row["expression"]) for row in rows], n_features)
        source_supports = {
            tuple(structural_terms(str(row["expression"]), n_features))
            for row in rows}
        support = tuple(structural_terms(expression, n_features))
        if support in source_supports:
            raise ValueError("synthesis directive produced no structural novelty")
        identity_payload = {
            "operation": directive.operation,
            "lineage_ids": list(directive.lineage_ids),
            "support": list(supports)}
        identity = sha256(json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if support in seen:
            continue
        seen.add(support)
        candidates.append({
            "expression": expression,
            "source": "llm_evidence_synthesis",
            "origin": "llm",
            "lineage_id": identity,
            "synthesis_operation": directive.operation,
            "parent_lineage_ids": list(directive.lineage_ids),
            "rationale": directive.rationale,
        })
        records.append({
            **identity_payload, "candidate_identity": identity,
            "expression": expression,
            "candidate_response_accessed": False,
            "heldout_opened": False})
    return candidates, {
        "schema": "scientific-evidence-conditioned-synthesis-v1",
        "directive_count": len(directives),
        "compiled_candidate_count": len(candidates),
        "records": records,
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }


__all__ = ["compile_evidence_synthesis"]
