"""Response-free correctness checks for separately budgeted proposal stages."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.discovery.agent import (
    DiscoveryAgent, DiscoveryAgentConfig, _bounded_seed_bank,
)
from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.evaluation_runtime import EvaluationBudget
from hypothesis_mvp.discovery.factory import build_scientific_discovery_runtime
from hypothesis_mvp.discovery.proposal_runtime import ProviderRoute, ProviderSettings


def _settings():
    return ProviderSettings(routes=(ProviderRoute(
        base_url="https://example.invalid", model="fixture", api_key="fixture"),))


def _selection():
    x = np.column_stack((np.linspace(-1., 1., 32),
                         np.linspace(-1., 1., 32) ** 2))
    return SimpleNamespace(development=SimpleNamespace(X=x, y=x[:, 0] + x[:, 1]))


def test_augmentation_routes_inner_provider_without_squeezing_engine_seeds(
        monkeypatch, tmp_path):
    settings = _settings()
    engine = SimpleNamespace(all_results=tuple(SimpleNamespace(
        expression=f"1+x0+{index}*x1", engine="mcts", lineage_id=str(index))
        for index in range(10)))
    config = DiscoveryAgentConfig(
        scientist_orchestration=True, typed_evidence_synthesis=True,
        typed_inner_augmentation=True, synthesis_evaluation_reserve=3,
        llm_evaluation_reserve=2, discovery_budget=16,
        discovery_islands=("balanced",))
    selection = _selection()
    synthesis = [{"expression": "1+x0+x1+x0*x1",
                  "source": "llm_evidence_synthesis", "lineage_id": "synthetic"}]
    seeds, audit = _bounded_seed_bank(engine, (), selection, config, synthesis)
    assert len(seeds) <= 10  # 16 - 3 - 2 - 1
    assert all(row["source"] != "llm_evidence_synthesis" for row in seeds)
    reference_seeds, _ = _bounded_seed_bank(
        engine, (), selection, DiscoveryAgentConfig(
            discovery_budget=11, discovery_islands=("balanced",)), ())
    assert seeds == reference_seeds
    captured = {}
    def fake_discover(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(report={})
    monkeypatch.setattr("hypothesis_mvp.discovery.agent.discover_from_selection",
                        fake_discover)
    monkeypatch.setattr("hypothesis_mvp.discovery.agent._engine_payload",
                        lambda result: {})
    monkeypatch.setattr("hypothesis_mvp.discovery.agent.attach_system_evidence",
                        lambda *args: None)
    DiscoveryAgent(config, settings)._discover(
        selection, engine, (), "fixture", "fixture", tmp_path, tmp_path, {},
        synthesized_candidates=synthesis)
    assert captured["provider_settings"] is settings
    assert captured["supplemental_candidates"] == synthesis
    assert captured["config"].synthesis_evaluation_reserve == 3
    assert captured["config"].llm_evaluation_reserve == 2
    assert captured["orchestration_context"]["seed_bank"] == audit
    old = DiscoveryAgentConfig(
        scientist_orchestration=True, typed_evidence_synthesis=True,
        llm_evaluation_reserve=2, discovery_budget=16,
        discovery_islands=("balanced",))
    DiscoveryAgent(old, settings)._discover(
        selection, engine, (), "fixture", "fixture", tmp_path, tmp_path, {},
        synthesized_candidates=synthesis)
    assert captured["provider_settings"] is None
    assert captured["supplemental_candidates"] == ()
    with pytest.raises(ValueError, match="separate positive"):
        DiscoveryAgent(DiscoveryAgentConfig(
            scientist_orchestration=True, typed_evidence_synthesis=True,
            typed_inner_augmentation=True, discovery_budget=16,
            llm_evaluation_reserve=2, synthesis_evaluation_reserve=0), settings)
    with pytest.raises(ValueError, match="provider"):
        DiscoveryAgent(config, ProviderSettings())


def test_evaluation_stages_have_independent_caps():
    budget = EvaluationBudget(12)
    budget.configure_llm_reserve(5)
    for _ in range(7):
        assert budget.consume("anchor_candidate")
    assert not budget.consume("anchor_candidate")
    budget.begin_llm_phase()
    budget.limit_phase_work(3)
    for _ in range(3):
        assert budget.consume("llm_candidate")
    assert not budget.consume("llm_candidate")
    budget.limit_phase_work(2)
    assert budget.consume("llm_candidate")
    assert budget.consume("llm_candidate")
    assert not budget.consume("llm_candidate")


def test_runtime_validates_supplement_before_inner_phase(monkeypatch, tmp_path):
    config = DiscoveryConfig.from_mapping({
        "evaluation_budget": 12, "llm_evaluation_reserve": 2,
        "synthesis_evaluation_reserve": 2,
        "refit_policy": "pcpi-closed-basis-amplitudes",
        "islands": ["balanced"], "max_rounds": 1})
    runtime = build_scientific_discovery_runtime(
        n_features=2, config=config, library_path=tmp_path / "library.jsonl",
        ledger_path=tmp_path / "ledger.jsonl", provider_settings=_settings())
    assert runtime.proposal.enabled
    monkeypatch.setattr(runtime, "_deterministic_search",
                        lambda state, arrays: (state, [], []))
    reached = []
    def fake_transport(messages, prompt_hash):
        request = json.loads(messages[1]["content"])
        return json.dumps({
            "protocol_id": "hypothesis-proposal-v1",
            "round_id": request["round_id"], "island": request["island"],
            "candidates": [{"candidate_id": "fixture1",
                "parent_hash": request["parent_hash"], "action": "ADD",
                "equation": "1+x0+x0*x1", "rationale": "registered interaction"}]}), {
                    "fixture_only": True, "provider_all_attempts_preserved": True}
    monkeypatch.setattr(runtime.proposal, "_request", fake_transport)
    def mock_inner(state, arrays):
        reached.append((runtime.proposal.enabled,
                        runtime.evaluation.budget.phase_ceiling,
                        runtime.evaluation.budget.used))
        batch = runtime.proposal.propose(
            task_name="fixture", task_desc="fixture", round_id=1,
            island="balanced",
            parent_hash=state.deterministic_reference.dag.canonical_hash,
            island_context={}, library_rows=[], ephemeral_refinements=[])
        assert batch.protocol_valid and len(batch.candidates) == 1
        return state, [], []
    monkeypatch.setattr(runtime, "_llm_search", mock_inner)
    x = _selection().development.X
    expression, report = runtime.run(
        X_train=x, y_train=x[:, 0] + x[:, 1],
        X_val=x, y_val=x[:, 0] + x[:, 1],
        base_candidates=[{"expression": "1+x0", "source": "engine:mcts"}],
        supplemental_candidates=[{"expression": "1+x0+x1+x0*x1",
                                  "source": "llm_evidence_synthesis",
                                  "lineage_id": "s"}])
    assert reached and reached[0][0]
    assert reached[0][1] - reached[0][2] == 2
    assert report["supplemental_synthesis_validated"] == 1
    assert any(row["source"] == "llm_evidence_synthesis"
               for row in report["evaluated_hypothesis_bank"])
    assert report["inner_llm_enabled"] is True
    assert report["llm_call_count"] == 1
    assert expression
