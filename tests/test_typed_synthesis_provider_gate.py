"""Synthetic-only typed synthesis provider gate fixtures."""

from hypothesis_mvp.discovery.proposal_runtime import (
    ProposalRuntime, ProviderRoute, ProviderSettings,
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

    def complete_json(**kwargs):
        return ({
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
            {"provider_all_attempts_preserved": True})

    monkeypatch.setattr(runtime, "complete_json", complete_json)
    result = run_typed_synthesis_provider_gate(
        settings, runtime=runtime)
    assert result["passed"] is True
    assert result["real_data_accessed"] is False
    assert result["compiled_candidate_count"] == 1
