"""Artifact-only replay of the typed Scientist synthesis compiler."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any

from .evidence_synthesis import compile_evidence_synthesis
from .pcpi_adapter import (EXPANDED_FORMULA_POLICY,
                           support_parser_for_policy)
from .scientist_policy import SYNTHESIS_OPERATIONS, SynthesisDirective

_CLOSED_BASIS_POLICY = "discard-fitted-coefficients-refit-closed-basis"


def _directive_payload(operation, left, right):
    """Return the policy and directive extras one registered operation needs.

    `COMPOSE_FORMULA_AST` is only well formed under the opt-in expanded
    grammar and must reference witnessed terms by lineage and index.
    """
    lineage_ids = (left["lineage_id"], right["lineage_id"])
    if operation != "COMPOSE_FORMULA_AST":
        return _CLOSED_BASIS_POLICY, lineage_ids, {}
    tree = {"op": "mul", "args": [
        {"ref": {"lineage_id": left["lineage_id"], "term_index": 0}},
        {"ref": {"lineage_id": right["lineage_id"], "term_index": 0}}]}
    return EXPANDED_FORMULA_POLICY, lineage_ids, {"formula_ast": tree}


def _full_result(source_output: Path) -> tuple[Path, dict[str, Any]]:
    matches = sorted(source_output.glob("*/[0-9]*/exploration/full/RESULT.json"))
    if len(matches) != 1:
        raise ValueError("typed synthesis replay requires one completed full result")
    return matches[0], json.loads(matches[0].read_text(encoding="utf-8"))


def audit_typed_synthesis_replay(
    source_output: str | Path,
) -> dict[str, Any]:
    root = Path(source_output).resolve()
    result_path, result = _full_result(root)
    rows = [
        dict(row) for row in result["hypothesis_provenance"][
            "raw_engine_candidates"]
        if str(row.get("engine") or "")
        and str(row.get("expression") or "")
        and str(row.get("lineage_id") or "")
    ]
    feature_names = json.loads(
        next(root.glob("*/[0-9]*/DATA_MANIFEST.json")).read_text(
            encoding="utf-8"))["scientific_context"]["feature_names"]
    n_features = len(feature_names)
    attempts, candidates, seen = [], [], set()
    for left, right in combinations(rows, 2):
        if left["engine"] == right["engine"]:
            continue
        evidence = (
            {"engine": left["engine"], "expression": left["expression"],
             "lineage_id": left["lineage_id"]},
            {"engine": right["engine"], "expression": right["expression"],
             "lineage_id": right["lineage_id"]},
        )
        for operation in SYNTHESIS_OPERATIONS:
            policy, lineage_ids, extras = _directive_payload(
                operation, left, right)
            directive = SynthesisDirective(
                operation, lineage_ids,
                "artifact-only exhaustive registered-operation replay",
                **extras)
            try:
                compiled, audit = compile_evidence_synthesis(
                    (directive,), evidence, n_features,
                    coefficient_policy=policy)
            except Exception as error:
                attempts.append({
                    "operation": operation,
                    "engines": [left["engine"], right["engine"]],
                    "passed": False, "reason": str(error)})
                continue
            if not compiled:
                # A recorded rejection is part of the replayed contract: a
                # registered operation the compiler refuses is an admissible
                # outcome, not a harness failure.
                attempts.append({
                    "operation": operation,
                    "engines": [left["engine"], right["engine"]],
                    "passed": False,
                    "reason": "; ".join(
                        str(row.get("reason") or "") for row in
                        (audit.get("rejections") or [])) or
                    "compiler produced no candidate"})
                continue
            candidate = compiled[0]
            support = tuple(support_parser_for_policy(policy)(
                candidate["expression"], n_features))
            attempts.append({
                "operation": operation,
                "engines": [left["engine"], right["engine"]],
                "passed": True, "support": list(support),
                "candidate_identity": candidate["lineage_id"]})
            if support not in seen:
                seen.add(support)
                candidates.append({
                    "operation": operation,
                    "engines": [left["engine"], right["engine"]],
                    "support": list(support),
                    "candidate_identity": candidate["lineage_id"],
                    "compiler_audit": audit})
    checks = {
        "source_protocol_was_response_closed": (
            result.get("heldout_opened") is False
            and result.get("selection_used_heldout") is False),
        "multiple_engine_lineages_available": len({
            row["engine"] for row in rows}) >= 2,
        "registered_operations_exhaustively_replayed": bool(attempts),
        "novel_pcpi_compatible_synthesis_exists": bool(candidates),
    }
    return {
        "schema": "scientific-typed-synthesis-artifact-replay-v1",
        "source_output": str(root),
        "source_result": str(result_path),
        "checks": checks,
        "passed": all(checks.values()),
        "attempt_count": len(attempts),
        "passed_attempt_count": sum(row["passed"] for row in attempts),
        "distinct_synthesized_support_count": len(candidates),
        "candidates": candidates,
        "candidate_response_accessed": False,
        "reporting_validation_response_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "artifact-only structural compiler replay; no predictive, "
            "decision-risk or superiority claim"),
    }


__all__ = ["audit_typed_synthesis_replay"]
