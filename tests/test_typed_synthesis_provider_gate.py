"""Synthetic-only typed synthesis provider gate fixtures."""

import pytest
import requests

from hypothesis_mvp.discovery.proposal_runtime import (
    ProposalRuntime, ProviderInfrastructureError,
    ProviderRoute, ProviderSettings,
)
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.typed_synthesis_provider_gate import (
    run_typed_synthesis_provider_gate,
)


def test_provider_gate_compiles_typed_lineage_directive(monkeypatch):
    settings = ProviderSettings(routes=(
        ProviderRoute("https://fixture.invalid", "fixture", "key"),))
    runtime = ProposalRuntime(
        EquationRuntime(3, refit_policy="pcpi-closed-basis-amplitudes"),
        3, settings, 1)

    responses = iter([
        ({
            "protocol_id": "scientific-research-plan-v1",
            "mechanisms": ["compare registered mechanisms"],
            "engine_calls": [
                {"engine": "polynomial_lasso", "jobs": 2,
                 "objective": "polynomial evidence",
                 "expected_evidence": "validated polynomial structure",
                 "requested_operations": ["linear", "quadratic"]},
                {"engine": "mcts", "jobs": 1,
                 "objective": "tree evidence",
                 "expected_evidence": "validated nonlinear structure",
                 "requested_operations": ["trigonometric"]},
                {"engine": "sparse_library", "jobs": 2,
                 "objective": "sparse evidence",
                 "expected_evidence": "validated sparse structure",
                 "requested_operations": ["monomials", "trigonometric"]},
                {"engine": "additive_mechanisms", "jobs": 1,
                 "objective": "additive evidence",
                 "expected_evidence": "validated additive structure",
                 "requested_operations": ["linear"]}],
            "comparison_questions": ["Which supports agree?"],
            "synthesis_goal": "compile a typed candidate",
            "stop_conditions": ["registered budget exhausted"]},
            {"provider_all_attempts_preserved": True}),
        ({
            "protocol_id": "scientific-engine-evidence-review-v1",
            "supported_mechanisms": ["complementary evidence"],
            "contradicted_mechanisms": [],
            "cross_engine_conflicts": [],
            "synthesis_instructions": ["compile support union"],
            "synthesis_directives": [{
                "operation": "UNION_SUPPORTS",
                "lineage_ids": ["linear", "nonlinear"],
                "rationale": "combine complementary mechanisms"}],
            "stop": False, "stop_reason": "one synthesis remains"},
            {"provider_all_attempts_preserved": True}),
    ])

    def complete_json(**kwargs):
        return next(responses)

    monkeypatch.setattr(runtime, "complete_json", complete_json)
    result = run_typed_synthesis_provider_gate(
        settings, runtime=runtime)
    assert result["passed"] is True
    assert result["real_data_accessed"] is False
    assert result["compiled_candidate_count"] == 1
    assert result["checks"]["provider_returned_valid_research_plan"] is True


def test_provider_exhaustion_exports_only_sanitized_diagnostic(monkeypatch):
    settings = ProviderSettings(routes=(
        ProviderRoute("https://fixture.invalid", "fixture", "key"),),
        attempts=2, retry_backoff_s=0)
    runtime = ProposalRuntime(EquationRuntime(1), 1, settings, 1)
    monkeypatch.setattr(
        runtime, "_post",
        lambda *args: (_ for _ in ()).throw(
            requests.ReadTimeout("secret upstream detail")))
    with pytest.raises(
            ProviderInfrastructureError,
            match="provider-infrastructure-failure:timeout:attempts=2"):
        runtime._request(({"role": "user", "content": "{}"},), "hash")
