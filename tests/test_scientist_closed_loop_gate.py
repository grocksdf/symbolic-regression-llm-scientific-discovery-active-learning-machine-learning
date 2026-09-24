"""No-data correctness fixtures for the task-local Scientist closed loop."""

from dataclasses import asdict

from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig
from hypothesis_mvp.discovery.scientist_closed_loop_gate import (
    run_scientist_closed_loop_gate,
)
from tests.test_system_executor import _config


def _closed_loop_config():
    config = _config()
    config["agent"] = asdict(DiscoveryAgentConfig(
        engines=(
            "polynomial_lasso", "mcts", "sparse_library",
            "additive_mechanisms"),
        engine_repeats=1, engine_budget=4, engine_workers=1,
        engine_retries=0, cycles=2, discovery_budget=24,
        acquisition_enabled=False, use_knowledge=False,
        task_local_memory=True, discovery_rounds=2,
        candidates_per_island=2, task_local_memory_topk=8,
        llm_evaluation_reserve=4,
        discovery_islands=("low_complexity", "nmse", "novelty"),
        scientist_orchestration=True,
        require_explicit_skill_controls=True,
        typed_evidence_synthesis=True,
    ))
    config["single_engine"] = "polynomial_lasso"
    gate = config["marginal_influence_gate"]
    gate["schema"] = "scientific-policy-and-source-influence-gate-v6"
    gate["required_contributions"] = [
        "scientist_policy", "engine:mcts", "engine:sparse_library",
        "engine:additive_mechanisms"]
    gate["required_active_contributions"] = ["scientist_policy"]
    gate["rejectable_contributions"] = [
        "engine:mcts", "engine:sparse_library",
        "engine:additive_mechanisms"]
    gate["scientist_policy_ablation"] = (
        "full-vs-no_llm-matched-budget-v1")
    return config


def test_closed_loop_gate_passes_without_data_or_confirmation():
    result = run_scientist_closed_loop_gate(_closed_loop_config())
    assert result["passed"] is True
    assert result["candidate_response_accessed"] is False
    assert result["heldout_opened"] is False
    assert result["real_data_accessed"] is False


def test_closed_loop_gate_rejects_single_cycle_configuration():
    config = _closed_loop_config()
    config["agent"]["cycles"] = 1
    result = run_scientist_closed_loop_gate(config)
    assert result["passed"] is False
    assert result["checks"]["multiple_outer_cycles"] is False
