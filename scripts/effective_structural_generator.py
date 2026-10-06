"""Response-free S planning; the historical slot generator stays frozen.

Plan legal parent/operator combinations before spending a structural attempt.
Neither fit outcomes nor admission evidence can trigger replacement candidates.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import numpy as np

from hypothesis_mvp.discovery.candidate_validity import (
    CandidateValidityError, numerical_definedness_preflight,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, support_parser_for_policy,
)
from scripts.matched_gap_structural_generator import (
    DYADIC_CONSTANTS, ISLANDS, OPERATIONS,
)

POLICY = "effective-response-free-v1"
LEGACY_POLICY = "historical-slots-v1"
ARITY = {operation: 1 if operation in (
    "subtree_insert", "unary_wrap", "bounded_power") else 2
    for operation in OPERATIONS}
UNARIES = ("sin", "cos", "tanh", "exp", "log", "sqrt")


@dataclass(frozen=True)
class PlanningBudget:
    ast_nodes: int = 128
    expression_chars: int = 4096
    parent_limit: int = 32
    checks_per_slot: int = 4096

    def __post_init__(self):
        limits = {"ast_nodes": 128, "expression_chars": 4096,
                  "parent_limit": 32, "checks_per_slot": 4096}
        if any(type(value) is not int or not 1 <= value <= limits[key]
               for key, value in asdict(self).items()):
            raise ValueError("invalid effective planning budget")


def policy_contract():
    return {"policy": POLICY, "budget": asdict(PlanningBudget()),
            "arity": dict(ARITY), "operations": list(OPERATIONS),
            "unary_operations": list(UNARIES),
            "matching_basis": "nominal-slots-with-response-free-feasibility",
            "alternative_policy": "cyclic-operations-features-unaries-ranked-distinct-parents",
            "post_fit_or_admission_top_up": False,
            "numerical_scope": "registered-covariate-points-only",
            "predictive_effect_or_equal_evaluable_yield_claimed": False}


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                            separators=(",", ":")).encode()).hexdigest()


def _rotate(values, ticket):
    index = ticket % len(values)
    return (*values[index:], *values[:index])


def _formula(operation, first, second, feature, constant, unary):
    if operation == "subtree_insert":
        return f"({first}) + ({constant})*({feature})"
    if operation == "unary_wrap":
        if unary == "log":
            return f"log(1 + Abs({first}))"
        if unary == "sqrt":
            return f"sqrt(0.5 + Abs({first}))"
        return f"{unary}({first})"
    if operation == "bounded_power":
        return f"(0.5 + Abs({first}))**2"
    if second is None:
        raise ValueError("binary operation requires two distinct parents")
    if operation == "binary_compose":
        return f"({first}) + ({second})"
    if operation == "safe_ratio":
        return f"({first})/(1 + Abs({second}))"
    if operation == "cross_support_interaction":
        return f"({first})*({second})"
    raise ValueError("unknown structural operation")


def _bank(engine_rows, dispersion, gap_identity, n_features, parser, budget):
    """Project only expression/lineage; scores and other evidence are unread."""
    parents, supports, invalid, lineages, compile_checks = [], set(), [], {}, 0
    for raw in engine_rows:
        expression = str(raw.get("expression") or "").strip()
        lineage = str(raw.get("lineage_id") or "")
        if lineage in lineages and lineages[lineage] != expression:
            raise ValueError("conflicting parent lineage")
        if lineage:
            lineages[lineage] = expression
        try:
            compile_checks += 1
            support = parser(expression, n_features)
            body_nodes = len(list(ast.walk(ast.parse(expression, mode="eval").body)))
        except (SyntaxError, TypeError, ValueError, RecursionError) as error:
            invalid.append({"lineage_id": lineage, "reason": str(error)})
            continue
        supports.add(support)
        score = float(dispersion.get(lineage, float("-inf")))
        if lineage and math.isfinite(score):
            parents.append({"expression": expression, "lineage_id": lineage,
                            "body_nodes": body_nodes, "support": support,
                            "dispersion": score})
    parents.sort(key=lambda row: (-row["dispersion"], _hash([
        gap_identity, row["lineage_id"], row["expression"]])))
    unique, seen = [], set()
    for row in parents:
        if row["support"] not in seen:
            unique.append(row)
            seen.add(row["support"])
    return unique[:budget.parent_limit], supports, {
        "bank_compile_checks": compile_checks,
        "invalid_parents": invalid, "eligible_unique_parents": len(unique),
        "parents_excluded_by_limit": max(0, len(unique) - budget.parent_limit)}


def _options(parents, ticket, n_features):
    """A fixed finite alternative policy, evaluated before candidate fitting."""
    feature = f"x{(ticket // 97) % n_features}"
    constant = DYADIC_CONSTANTS[(ticket // 6) % len(DYADIC_CONSTANTS)]
    for operation in _rotate(OPERATIONS, ticket):
        if ARITY[operation] == 2 and len(parents) < 2:
            yield operation, None, None, feature, constant, "sin"
            continue
        for first in parents:
            features = tuple(f"x{index}" for index in _rotate(
                tuple(range(n_features)), ticket // 97)) if (
                    operation == "subtree_insert") else (feature,)
            seconds = [row for row in parents if row is not first] if (
                ARITY[operation] == 2) else [None]
            unaries = _rotate(UNARIES, ticket // 18) if (
                operation == "unary_wrap") else ("sin",)
            for second in seconds:
                for chosen_feature in features:
                    for unary in unaries:
                        yield operation, first, second, chosen_feature, constant, unary


def _choose(parents, seen, ticket, points, parser, budget):
    reasons, checks = Counter(), 0
    costs = {"compiler_calls": 0, "numerical_preflight_calls": 0,
             "preflight_point_evaluations": 0}
    for operation, first, second, feature, constant, unary in _options(
            parents, ticket, points.shape[1]):
        if checks == budget.checks_per_slot:
            return None, checks, dict(reasons), "planning-budget-exhausted", costs
        checks += 1
        if first is None:
            reasons["parent_unavailable"] += 1
            continue
        template = _formula(operation, "P0", "P1", feature, constant, unary)
        count = len(list(ast.walk(ast.parse(template, mode="eval"))))
        count += first["body_nodes"] - 2
        chars = len(template) + len(first["expression"]) - 2
        if second is not None:
            count += second["body_nodes"] - 2
            chars += len(second["expression"]) - 2
        if count > budget.ast_nodes or chars > budget.expression_chars:
            reasons["ast_or_source_budget"] += 1
            continue
        expression = _formula(operation, first["expression"],
                              None if second is None else second["expression"],
                              feature, constant, unary)
        try:
            costs["compiler_calls"] += 1
            support = parser(expression, points.shape[1])
            if support in seen:
                reasons["duplicate_support"] += 1
                continue
            costs["numerical_preflight_calls"] += 1
            costs["preflight_point_evaluations"] += len(points)
            preflight = numerical_definedness_preflight(expression, points)
        except CandidateValidityError as error:
            reasons[error.reason] += 1
            continue
        except (SyntaxError, TypeError, ValueError, RecursionError):
            reasons["compiler_incompatible"] += 1
            continue
        return {"expression": expression, "support": list(support),
                "synthesis_operation": operation, "unary": unary,
                "feature": feature, "constant": constant,
                "parent_lineage_ids": [first["lineage_id"], *(
                    [second["lineage_id"]] if second is not None else [])],
                "ast_node_count": count, "expression_char_count": chars,
                "definedness_preflight": preflight}, checks, dict(reasons), "", costs
    return None, checks, dict(reasons), "no-feasible-structural-candidate", costs


def generate_effective_cycle(*, engine_rows, gap_brief, parent_dispersion,
                             covariates, task_identity, seed, cycle,
                             budget=PlanningBudget()):
    """Freeze three distinct proposals using legal covariates only, no y."""
    points = np.asarray(covariates, dtype=float)
    if (type(seed) is not int or type(cycle) is not int or cycle not in (0, 1)
            or not isinstance(budget, PlanningBudget)
            or not isinstance(task_identity, str) or not task_identity
            or points.ndim != 2 or not points.size
            or not np.all(np.isfinite(points))):
        raise ValueError("invalid effective structural coordinate")
    if (gap_brief.get("candidate_response_accessed") is not False
            or gap_brief.get("heldout_opened") is not False
            or not gap_brief.get("region_identity")
            or type(gap_brief.get("propose_allowed")) is not bool):
        raise ValueError("explicit response-free gap authorization required")
    parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    gap_identity = str(gap_brief["region_identity"])
    allowed = gap_brief["propose_allowed"]
    parents, seen, parent_audit = _bank(
        engine_rows, parent_dispersion, gap_identity, points.shape[1], parser,
        budget) if allowed else ([], set(), {})
    bank_identity = _hash(sorted(seen))
    slots = []
    for island in ISLANDS:
        identity = {"policy": POLICY, "task_identity": task_identity,
                    "seed": seed, "cycle": cycle, "island": island,
                    "gap_identity": gap_identity, "bank_identity": bank_identity,
                    "budget": asdict(budget)}
        ticket = int(_hash(identity)[:16], 16)
        candidate, checks, reasons, failure, costs = _choose(
            parents, seen, ticket, points, parser, budget) if allowed else (
                None, 0, {}, "independent-gap-did-not-authorize-proposal", {})
        audit = {**identity, "slot_identity": _hash(identity),
                 "preferred_operation": OPERATIONS[ticket % len(OPERATIONS)],
                 "gap_activated": allowed, "attempt_count": int(candidate is not None),
                 "effective_opportunity_count": int(candidate is not None),
                 "retry_count": 0, "planning_checks": checks,
                 "planning_rejections": reasons,
                 "planning_costs": costs,
                 "candidate_response_accessed": False, "heldout_opened": False}
        if candidate is None:
            slots.append({**audit, "status": "abstained", "reason": failure})
            continue
        seen.add(tuple(candidate["support"]))
        candidate = {**candidate, "source": "matched_gap_structural_search",
                     "origin": "non_llm_matched", "generator_policy": POLICY,
                     "slot_identity": audit["slot_identity"],
                     "lineage_id": _hash({**identity, "candidate": candidate})}
        slots.append({**audit, "status": "materialized",
                      "operation": candidate["synthesis_operation"],
                      "candidate": candidate})
    counts = {"registered_slots": len(slots),
              "gap_activated_slots": sum(row["gap_activated"] for row in slots),
              "effective_opportunities": sum(row["attempt_count"] for row in slots),
              "planning_checks": sum(row["planning_checks"] for row in slots),
              "bank_compile_checks": parent_audit.get("bank_compile_checks", 0),
              "compiler_calls": sum(row["planning_costs"].get("compiler_calls", 0)
                                    for row in slots),
              "numerical_preflight_calls": sum(row["planning_costs"].get(
                  "numerical_preflight_calls", 0) for row in slots),
              "preflight_point_evaluations": sum(row["planning_costs"].get(
                  "preflight_point_evaluations", 0) for row in slots),
              "retry_count": 0}
    return {"schema": "effective-structural-cycle-v1", "policy": POLICY,
            "budget": asdict(budget), "counts": counts, "slots": slots,
            "parent_audit": parent_audit,
            "structural_operation_count": len({row["operation"] for row in slots
                                                if row["status"] == "materialized"}),
            "covariate_identity": _hash([points.shape, sha256(points.tobytes()).hexdigest()]),
            "response_accessed": False, "heldout_opened": False,
            "quality_or_admission_guaranteed": False}
