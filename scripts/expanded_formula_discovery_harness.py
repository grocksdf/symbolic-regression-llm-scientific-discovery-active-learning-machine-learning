"""Experiment-only admission arms and symbolic recovery evaluators.

This module does not generate candidates and does not modify the PCPI
statistical core.  It projects one frozen core-plus-LLM candidate bank through
three registered admission policies.
"""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Mapping, Sequence

import sympy

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.pcpi_adapter import structural_terms
from hypothesis_mvp.discovery.source_stacking import (
    filter_fold_safe_source_candidates,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


ADMISSION_ARMS = (
    "independent_protected",
    "same_data",
    "accept_all",
)


def _protected(rows):
    return [{
        **dict(row),
        "source": (
            "protected_counterfactual_backbone:"
            f"{row.get('source') or 'engine:unknown'}"),
    } for row in rows]


def _optional_distinct(core, optional, n_features):
    occupied = {structural_terms(str(row["expression"]), n_features)
                for row in core}
    selected, seen = [], set()
    for row in sorted(optional, key=lambda value: sha256(json.dumps(
            value, sort_keys=True, default=str).encode()).hexdigest()):
        support = structural_terms(str(row["expression"]), n_features)
        if support in occupied or support in seen:
            continue
        seen.add(support)
        selected.append(dict(row))
    return selected


def project_admission_arms(
    core: Sequence[Mapping[str, str]],
    optional: Sequence[Mapping[str, str]],
    fit: RoleDataset,
    gap_diagnosis: RoleDataset,
    independent_admission: RoleDataset,
    selector_update: RoleDataset,
    action_domain,
    *,
    n_features: int,
    identity: str,
):
    """Project one frozen candidate bank through three admission policies."""
    roles = (fit, gap_diagnosis, independent_admission, selector_update)
    if (fit.role is not DataRole.DEVELOPMENT
            or any(role.role is not DataRole.VALIDATION
                   for role in roles[1:])
            or any(a.row_fingerprints & b.row_fingerprints
                   for index, a in enumerate(roles)
                   for b in roles[index + 1:])):
        raise ValueError("formula admission roles must be disjoint")
    optional_rows = _optional_distinct(core, optional, n_features)
    protected = _protected(core)
    kwargs = {
        "n_features": n_features,
        "prior": NormalInverseGammaPrior(),
        "exploration_identity": identity,
        "coefficient_policy":
            "discard-fitted-coefficients-refit-closed-basis",
        "measurement_budget": 2,
        "action_domain": action_domain,
    }

    def screened(first_role, second_role):
        first, first_report = filter_fold_safe_source_candidates(
            [*protected, *optional_rows], fit, first_role, **kwargs)
        admitted = [dict(row) for row in first
                    if row.get("origin") == "llm"]
        second, second_report = filter_fold_safe_source_candidates(
            [*protected, *admitted], fit, second_role, **kwargs)
        final = [dict(row) for row in second
                 if row.get("origin") == "llm"]
        return [*[dict(row) for row in core], *final], {
            "first": first_report, "second": second_report,
            "admitted_llm_count": len(final),
        }

    independent, independent_audit = screened(
        independent_admission, selector_update)
    same_data, same_data_audit = screened(
        gap_diagnosis, gap_diagnosis)
    accept_all = [*[dict(row) for row in core], *optional_rows]
    candidate_identity = sha256(json.dumps(
        {"core": [dict(row) for row in core],
         "optional": optional_rows},
        sort_keys=True, default=str).encode()).hexdigest()
    return {
        "candidate_bank_identity": candidate_identity,
        "arms": {
            "independent_protected": independent,
            "same_data": same_data,
            "accept_all": accept_all,
        },
        "audits": {
            "independent_protected": independent_audit,
            "same_data": same_data_audit,
            "accept_all": {
                "response_accessed": False,
                "admitted_llm_count": len(optional_rows),
            },
        },
        "generative_authority_equals_epistemic_authority": False,
    }


def _parse(expression: str, input_symbols: Sequence[str]):
    locals_map = {
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt,
        "tanh": sympy.tanh, "Abs": sympy.Abs, "abs": sympy.Abs,
    }
    text = str(expression)
    replacements = sorted(
        ((str(name), f"x{index}") for index, name in enumerate(input_symbols)),
        key=lambda item: (-len(item[0]), item[0]))
    for source, target in replacements:
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])",
            target, text)
    return sympy.sympify(text, locals=locals_map)


def exact_formula_recovery(
        candidate: str, truth: str, input_symbols: Sequence[str]) -> bool:
    """Exact symbolic equality; an unresolved simplification is not recovery."""
    try:
        difference = sympy.cancel(
            sympy.together(_parse(candidate, input_symbols)
                           - _parse(truth, input_symbols)))
        return difference == 0 or sympy.simplify(difference) == 0
    except Exception:
        return False


def _signature(node):
    if isinstance(node, sympy.Symbol):
        return ("symbol", str(node))
    if node.is_Number:
        return ("constant",)
    if isinstance(node, sympy.Add):
        return ("add", tuple(sorted((_signature(arg) for arg in node.args),
                                    key=repr)))
    if isinstance(node, sympy.Mul):
        children = [_signature(arg) for arg in node.args if not arg.is_Number]
        return ("constant",) if not children else (
            "mul", tuple(sorted(children, key=repr)))
    if isinstance(node, sympy.Pow):
        exponent = (str(node.exp) if node.exp.is_Number
                    else _signature(node.exp))
        return ("pow", _signature(node.base), exponent)
    if isinstance(node, sympy.Function):
        return (node.func.__name__, tuple(_signature(arg)
                                         for arg in node.args))
    return (type(node).__name__, tuple(_signature(arg) for arg in node.args))


def structural_formula_recovery(
        candidate: str, truth: str, input_symbols: Sequence[str]) -> bool:
    """Operator/variable topology equality with fitted amplitudes abstracted."""
    try:
        return _signature(_parse(candidate, input_symbols)) == _signature(
            _parse(truth, input_symbols))
    except Exception:
        return False


__all__ = [
    "ADMISSION_ARMS", "exact_formula_recovery",
    "project_admission_arms", "structural_formula_recovery",
]
