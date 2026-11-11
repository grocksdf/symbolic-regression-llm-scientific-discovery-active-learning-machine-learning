"""Synthetic correctness fixtures only; no provider or benchmark access."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.proposal_runtime import ProposalBatch, ProposalCandidate
from hypothesis_mvp.discovery.structural_plan_pool import (
    StructuralPlan, candidate_domain_status, freeze_plan_pool, plans_from_proposal_batch,
)


def fixture():
    x = np.column_stack((np.linspace(.2, 2., 12), np.linspace(.2, 2., 12) ** 2))
    fit = RoleDataset(DataRole.DEVELOPMENT, x, x[:, 0] ** 2)
    ids = tuple(f"fit:{i}" for i in range(len(x)))
    feedback = SimpleNamespace(search_data_identity=fit.fingerprint, search_row_ids=ids,
        bank_identity="frozen-bank", identity="feedback", location_score=tuple(x[:, 0] ** 2 - x[:, 0]))
    return dict(fit=fit, fit_row_ids=ids, forbidden_row_ids=("never:report",), feedback=feedback,
        bank_locations=x[:, :1], core_rows=[{"expression": "x0"}],
        regions=[{"id": "upper", "axis": 0, "side": "gt", "cut": 1.}], task_identity="correctness")


def plan(expression="x0*x1", **kwargs):
    return StructuralPlan(expression, "x0", "upper", "compose", "program", "p", **kwargs)


def batch_fixture():
    c = ProposalCandidate("c", "island", "sin(x0)", "PROPOSE_NEW_SKELETON", "test", "parent",
                          lineage_id="lineage", prompt_hash="prompt", response_hash="response")
    request = {"provider_all_attempts_preserved": True, "provider_outcomes": [
        {"provider_http_status": 200, "provider_prompt_tokens": 10, "provider_completion_tokens": 20,
         "provider_elapsed_s": .1}]}
    return ProposalBatch((c,), True, "ok", "prompt", "response", {"provider_requests": [request]})


def test_llm_source_adapter_preserves_complete_equation_and_transport():
    plans, audit = plans_from_proposal_batch(batch_fixture(), parent_expressions={"parent": "x0"}, region_id="upper")
    assert plans[0].source == "llm" and plans[0].expression == "sin(x0)"
    assert plans[0].site_path is None  # Never invent an LLM-selected subtree.
    assert audit["provider_attempts"] == 1 and audit["completion_tokens"] == 20
    assert audit["plan_ids"] == [plans[0].identity]


@pytest.mark.parametrize("edit", [
    {"reason": "provider_or_protocol_failure"}, {"telemetry": {}},
    {"telemetry": {"provider_requests": [{"provider_all_attempts_preserved": True,
                                          "provider_outcomes": [{"provider_http_status": 429}]}]}},
    {"response_hash": "wrong"},
])
def test_provider_failures_and_misbound_responses_cannot_be_empty_pool(edit):
    with pytest.raises(ValueError):
        plans_from_proposal_batch(replace(batch_fixture(), **edit), parent_expressions={"parent": "x0"}, region_id="upper")


def test_fixed_source_blind_features_and_explicit_support_deduplication():
    p = plan()
    llm = replace(p, source="llm", lineage_id="l", prompt_hash="prompt", response_hash="response")
    a = freeze_plan_pool([p], **fixture())
    b = freeze_plan_pool([llm], **fixture())
    assert a["rows"][0]["features"] == b["rows"][0]["features"]
    assert len(a["rows"][0]["features"]) == 8
    pool = freeze_plan_pool([p, llm, plan("x0")], **fixture())
    assert len(pool["rows"]) == 2 and len(pool["rejections"]) == 1
    assert pool["rejections"][0]["stage"] == "alias"
    assert any(r["core_duplicate"] for r in pool["rows"])


def test_undefined_behavior_flag_is_distinct_from_observed_zero_utility():
    pool = freeze_plan_pool([plan("log(x0-1)")], **fixture())
    assert pool["rows"][0]["features"][-1] == 0
    valid, reason = candidate_domain_status("log(x0-1)", n_features=2, domains=[fixture()["fit"].X])
    assert not valid and "undefined" in reason
    assert "fold_gains" not in pool["rows"][0]


def test_very_large_finite_features_remain_bounded_without_changing_formula():
    pool = freeze_plan_pool([plan("exp(300*x0)")], **fixture())
    row = pool["rows"][0]
    assert np.all(np.isfinite(row["features"])) and max(map(abs, row["features"])) <= 1
    assert row["features"][-1] == 1 and row["plan"]["expression"] == "exp(300*x0)"


def test_syntax_rejections_preserved_and_no_execution_of_unsafe_expression():
    pool = freeze_plan_pool([plan("__import__('os').getcwd()")], **fixture())
    assert not pool["rows"] and pool["rejections"][0]["stage"] == "compile"


def test_forbidden_roles_and_invalid_context_rejected():
    kw = fixture()
    with pytest.raises(ValueError, match="isolated"):
        freeze_plan_pool([plan()], **{**kw, "forbidden_row_ids": kw["fit_row_ids"]})
    with pytest.raises(ValueError, match="unknown region"):
        freeze_plan_pool([replace(plan(), region_id="unknown")], **kw)
    with pytest.raises(ValueError, match="invalid registered"):
        candidate_domain_status("x0", n_features=2, domains=[np.empty((0, 2))])


def test_unknown_numeric_contract_failure_is_not_candidate_invalid(monkeypatch):
    import hypothesis_mvp.discovery.structural_plan_pool as module
    def broken(*a, **kw): raise ValueError("broken fitter contract")
    monkeypatch.setattr(module, "design_matrix", broken)
    with pytest.raises(ValueError, match="broken fitter"):
        candidate_domain_status("x0", n_features=2, domains=[fixture()["fit"].X])
