"""Deterministic non-LLM structural proposals for the matched attribution arm."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, support_parser_for_policy,
)


ISLANDS = ("low_complexity", "nmse", "novelty")
OPERATIONS = (
    "subtree_insert", "unary_wrap", "binary_compose",
    "safe_ratio", "bounded_power", "cross_support_interaction",
)
DYADIC_CONSTANTS = (0.5, 1.0, 2.0)


def _hash(value) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, default=str,
        separators=(",", ":")).encode()).hexdigest()


def _ticket(value) -> int:
    return int(_hash(value)[:16], 16)


def _parents(
        engine_rows: Sequence[Mapping],
        parent_dispersion: Mapping[str, float],
        gap_identity: str,
) -> list[dict]:
    rows = []
    for raw in engine_rows:
        lineage = str(raw.get("lineage_id") or "")
        expression = str(raw.get("expression") or "").strip()
        if not lineage or not expression:
            continue
        dispersion = float(parent_dispersion.get(lineage, float("-inf")))
        if not math.isfinite(dispersion):
            continue
        rows.append({
            **dict(raw), "lineage_id": lineage,
            "expression": expression, "gap_region_dispersion": dispersion,
        })
    return sorted(rows, key=lambda row: (
        -row["gap_region_dispersion"],
        _hash([gap_identity, row["lineage_id"]])))


def _expression(operation: str, first: str, second: str | None,
                feature: str, constant: float, unary: str) -> str:
    if operation == "subtree_insert":
        return f"({first}) + ({constant})*({feature})"
    if operation == "unary_wrap":
        if unary == "log":
            return f"log(1 + Abs({first}))"
        if unary == "sqrt":
            return f"sqrt(0.5 + Abs({first}))"
        return f"{unary}({first})"
    if second is None:
        raise ValueError("binary matched operation requires two parents")
    if operation == "binary_compose":
        return f"({first}) + ({second})"
    if operation == "safe_ratio":
        return f"({first})/(1 + Abs({second}))"
    if operation == "bounded_power":
        return f"(0.5 + Abs({first}))**2"
    if operation == "cross_support_interaction":
        return f"({first})*({second})"
    raise ValueError("unknown matched structural operation")


def generate_matched_slot(
        *, engine_rows: Sequence[Mapping], gap_brief: Mapping,
        parent_dispersion: Mapping[str, float], n_features: int,
        task_identity: str, seed: int, cycle: int, island: str,
        slot: int = 0) -> dict:
    """Issue exactly one response-free proposal attempt for one frozen slot."""
    if (type(seed) is not int or type(cycle) is not int or cycle not in (0, 1)
            or island not in ISLANDS or slot != 0 or n_features < 1):
        raise ValueError("invalid matched structural slot")
    if (gap_brief.get("candidate_response_accessed") is not False
            or gap_brief.get("heldout_opened") is not False
            or not gap_brief.get("region_identity")):
        raise ValueError("matched structural generator requires response-free gap")
    gap_identity = str(gap_brief["region_identity"])
    slot_identity = {
        "schema": "matched-gap-structural-slot-v1",
        "task_identity": task_identity, "seed": seed, "cycle": cycle,
        "island": island, "slot": slot, "gap_identity": gap_identity,
    }
    ticket = _ticket(slot_identity)
    operation = OPERATIONS[ticket % len(OPERATIONS)]
    constant = DYADIC_CONSTANTS[
        (ticket // len(OPERATIONS)) % len(DYADIC_CONSTANTS)]
    unary = ("sin", "cos", "tanh", "exp", "log", "sqrt")[
        (ticket // (len(OPERATIONS) * len(DYADIC_CONSTANTS))) % 6]
    feature = f"x{(ticket // 97) % n_features}"
    parents = _parents(engine_rows, parent_dispersion, gap_identity)
    audit = {
        **slot_identity,
        "operation": operation, "constant": constant,
        "feature": feature, "attempt_count": 1, "retry_count": 0,
        "parent_selection":
            "max-gap-region-dispersion-with-hash-stable-ties",
        "candidate_response_accessed": False, "heldout_opened": False,
    }
    if gap_brief.get("propose_allowed") is False:
        return {
            **audit, "status": "abstained",
            "reason": "independent-gap-did-not-authorize-proposal",
            "attempt_count": 0,
        }
    if not parents:
        return {**audit, "status": "rejected",
                "reason": "no-finite-dispersion-parent"}
    first = parents[0]
    second = parents[1] if len(parents) > 1 else None
    try:
        expression = _expression(
            operation, first["expression"],
            None if second is None else second["expression"],
            feature, constant, unary)
        parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
        support = parser(expression, n_features)
        existing = {
            parser(str(row["expression"]), n_features)
            for row in engine_rows if str(row.get("expression") or "").strip()}
        if support in existing:
            raise ValueError("matched proposal duplicates frozen bank")
    except (SyntaxError, TypeError, ValueError) as error:
        return {**audit, "status": "rejected",
                "reason": str(error), "error_type": type(error).__name__}
    candidate_identity = _hash({
        **slot_identity, "operation": operation,
        "parent_lineage_ids": [
            first["lineage_id"],
            *([second["lineage_id"]] if second else [])],
        "support": list(support),
    })
    return {
        **audit, "status": "materialized",
        "candidate": {
            "expression": expression,
            "source": "matched_gap_structural_search",
            "origin": "non_llm_matched",
            "lineage_id": candidate_identity,
            "synthesis_operation": operation,
            "parent_lineage_ids": [
                first["lineage_id"],
                *([second["lineage_id"]] if second else [])],
            "support": list(support),
            "slot_identity": _hash(slot_identity),
        },
    }


def generate_matched_schedule(
        *, engine_rows_by_cycle: Sequence[Sequence[Mapping]],
        gap_briefs: Sequence[Mapping],
        parent_dispersion_by_cycle: Sequence[Mapping[str, float]],
        n_features: int, task_identity: str, seed: int) -> dict:
    if not (len(engine_rows_by_cycle) == len(gap_briefs)
            == len(parent_dispersion_by_cycle) == 2):
        raise ValueError("matched schedule requires exactly two cycles")
    slots = [
        generate_matched_slot(
            engine_rows=engine_rows_by_cycle[cycle],
            gap_brief=gap_briefs[cycle],
            parent_dispersion=parent_dispersion_by_cycle[cycle],
            n_features=n_features, task_identity=task_identity,
            seed=seed, cycle=cycle, island=island)
        for cycle in range(2) for island in ISLANDS
    ]
    return {
        "schema": "matched-gap-structural-schedule-v1",
        "registered_slot_count": 6,
        "activated_slot_count": sum(
            row["attempt_count"] for row in slots),
        "attempt_count": sum(row["attempt_count"] for row in slots),
        "retry_count": sum(row["retry_count"] for row in slots),
        "materialized_count": sum(
            row["status"] == "materialized" for row in slots),
        "slots": slots,
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }


__all__ = [
    "DYADIC_CONSTANTS", "ISLANDS", "OPERATIONS",
    "generate_matched_schedule", "generate_matched_slot",
]
