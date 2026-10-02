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


def project_expanded_admission_arms(
    core, optional, fit, gap_diagnosis, independent_admission,
    selector_update, action_domain, *, n_features, identity,
):
    roles = (fit, gap_diagnosis, independent_admission, selector_update)
    if (fit.role is not DataRole.DEVELOPMENT
            or any(role.role is not DataRole.VALIDATION for role in roles[1:])
            or any(a.row_fingerprints & b.row_fingerprints
                   for index, a in enumerate(roles)
                   for b in roles[index + 1:])):
        raise ValueError("expanded formula admission roles must be disjoint")
    parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    seen = {parser(str(row["expression"]), n_features) for row in core}
    optional_rows = []
    for row in sorted(optional, key=lambda value: sha256(json.dumps(
            value, sort_keys=True, default=str).encode()).hexdigest()):
        support = parser(str(row["expression"]), n_features)
        if support not in seen:
            seen.add(support)
            optional_rows.append(dict(row))
    protected = [{**dict(row), "source": "protected_counterfactual_backbone:"
                  + str(row.get("source") or "engine:unknown")}
                 for row in core]
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
        return [*[dict(row) for row in core], *final], {
            "first": first_report, "second": second_report,
            "admitted_llm_count": len(final)}

    independent, independent_audit = screened(
        independent_admission, selector_update)
    same_data, same_data_audit = screened(gap_diagnosis, gap_diagnosis)
    identity_material = {"grammar": "expanded-formula-ast-v1",
                         "core": [dict(row) for row in core],
                         "optional": optional_rows}
    return {
        "candidate_bank_identity": sha256(json.dumps(
            identity_material, sort_keys=True, default=str).encode()).hexdigest(),
        "arms": {"independent_protected": independent,
                 "same_data": same_data,
                 "accept_all": [*[dict(row) for row in core], *optional_rows]},
        "audits": {"independent_protected": independent_audit,
                   "same_data": same_data_audit,
                   "accept_all": {"response_accessed": False,
                                  "admitted_llm_count": len(optional_rows)}},
        "coefficient_policy": EXPANDED_FORMULA_POLICY,
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }
