"""Prospective three-arm projection using the one expanded support parser."""
from __future__ import annotations

from hashlib import sha256
import json

from hypothesis_mvp.data.roles import DataRole
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, support_parser_for_policy,
)
from hypothesis_mvp.discovery.source_stacking import (
    filter_fold_safe_source_candidates,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.reference.basis import design_matrix


ADMISSION_ARMS = (
    "independent_protected", "same_data", "accept_all")


def _finite_on_registered_domains(row, parser, n_features, domains):
    support = parser(str(row["expression"]), n_features)
    for domain in domains:
        design_matrix(domain, support)
    return support


def project_expanded_admission_arms(
    core, optional, fit, gap_diagnosis, independent_admission,
    selector_update, action_domain, *, n_features, identity,
    requested_arms=ADMISSION_ARMS,
):
    roles = (fit, gap_diagnosis, independent_admission, selector_update)
    if (fit.role is not DataRole.DEVELOPMENT
            or any(role.role is not DataRole.VALIDATION for role in roles[1:])
            or any(a.row_fingerprints & b.row_fingerprints
                   for index, a in enumerate(roles)
                   for b in roles[index + 1:])):
        raise ValueError("expanded formula admission roles must be disjoint")
    parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    requested = tuple(dict.fromkeys(str(value) for value in requested_arms))
    if not requested or any(value not in ADMISSION_ARMS
                            for value in requested):
        raise ValueError("unknown expanded formula admission arm")
    domains = tuple(role.X for role in roles) + (action_domain,)
    seen = set()
    eligible_core = []
    core_rejections = []
    for row in core:
        try:
            support = _finite_on_registered_domains(
                row, parser, n_features, domains)
        except (FloatingPointError, SyntaxError, TypeError, ValueError) as error:
            core_rejections.append({
                "expression_sha256": sha256(
                    str(row.get("expression") or "").encode()).hexdigest(),
                "source": str(row.get("source") or "engine:unknown"),
                "reason": "undefined-on-registered-domain",
                "error_type": type(error).__name__,
                "candidate_response_accessed": False,
                "heldout_opened": False,
            })
            continue
        if support not in seen:
            seen.add(support)
            eligible_core.append(dict(row))
    if len(eligible_core) < 2:
        raise ValueError(
            "fewer than two operationally eligible engine supports")
    optional_rows = []
    optional_rejections = []
    for row in sorted(optional, key=lambda value: sha256(json.dumps(
            value, sort_keys=True, default=str).encode()).hexdigest()):
        try:
            support = _finite_on_registered_domains(
                row, parser, n_features, domains)
        except (FloatingPointError, SyntaxError, TypeError, ValueError) as error:
            optional_rejections.append({
                "expression_sha256": sha256(
                    str(row.get("expression") or "").encode()).hexdigest(),
                "reason": "undefined-on-registered-domain",
                "error_type": type(error).__name__,
                "candidate_response_accessed": False,
                "heldout_opened": False,
            })
            continue
        if support not in seen:
            seen.add(support)
            optional_rows.append(dict(row))
    protected = [{**dict(row), "source": "protected_counterfactual_backbone:"
                  + str(row.get("source") or "engine:unknown")}
                 for row in eligible_core]
    kwargs = dict(n_features=n_features, prior=NormalInverseGammaPrior(),
                  exploration_identity=identity,
                  coefficient_policy=EXPANDED_FORMULA_POLICY,
                  measurement_budget=2, action_domain=action_domain)

    def screened(first_role, second_role):
        first, first_report = filter_fold_safe_source_candidates(
            [*protected, *optional_rows], fit, first_role, **kwargs)
        admitted = [dict(row) for row in first if row.get("origin") == "llm"]
        second, second_report = filter_fold_safe_source_candidates(
            [*protected, *admitted], fit, second_role, **kwargs)
        final = [dict(row) for row in second if row.get("origin") == "llm"]
        return [*eligible_core, *final], {
            "first": first_report, "second": second_report,
            "admitted_llm_count": len(final)}

    identity_material = {"grammar": "expanded-formula-ast-v1",
                         "core": eligible_core,
                         "optional": optional_rows}
    arms, audits = {}, {}
    if "independent_protected" in requested:
        independent, independent_audit = screened(
            independent_admission, selector_update)
        arms["independent_protected"] = independent
        audits["independent_protected"] = independent_audit
    if "same_data" in requested:
        same_data, same_data_audit = screened(
            gap_diagnosis, gap_diagnosis)
        arms["same_data"] = same_data
        audits["same_data"] = same_data_audit
    if "accept_all" in requested:
        arms["accept_all"] = [
            *eligible_core, *optional_rows]
        audits["accept_all"] = {
            "response_accessed": False,
            "admitted_llm_count": len(optional_rows)}
    return {
        "candidate_bank_identity": sha256(json.dumps(
            identity_material, sort_keys=True, default=str).encode()).hexdigest(),
        "arms": arms,
        "audits": audits,
        "requested_arms": list(requested),
        "core_input_count": len(core),
        "eligible_core_count": len(eligible_core),
        "core_domain_rejections": core_rejections,
        "optional_domain_rejections": optional_rejections,
        "coefficient_policy": EXPANDED_FORMULA_POLICY,
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }
