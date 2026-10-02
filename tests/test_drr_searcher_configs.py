"""Static honesty checks for the four frozen DRR conditions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _config(name):
    return (ROOT / "configs" / name).read_text(encoding="utf-8")


def test_drr_conditions_use_true_agent_variants():
    full = _config("aistats_drr_full_v6.yaml")
    no_llm = _config("aistats_drr_no_llm_v6.yaml")
    single = _config("aistats_drr_single_engine_v6.yaml")
    entropy = _config("aistats_drr_full_v5.yaml")
    assert "class_name: DRRBenchmarkSearcher" in full
    assert "scientist_orchestration: true" in full
    assert "typed_evidence_synthesis: true" in full
    assert "engines: [polynomial_lasso, mcts, sparse_library, additive_mechanisms]" in full
    assert "use_llm: false" in no_llm
    assert "engines: [polynomial_lasso, mcts, sparse_library, additive_mechanisms]" in no_llm
    assert "engines: [polynomial_lasso]" in single
    assert "portfolio_method: v5" in entropy
    assert "portfolio_method: v6" in full


def test_three_arm_configs_match_engine_and_llm_budgets():
    engine = _config("aistats_three_arm_e_v1.yaml")
    blind = _config("aistats_three_arm_l_blind_v1.yaml")
    gap = _config("aistats_three_arm_l_gap_v1.yaml")
    shared = [
        "engines: [polynomial_lasso, mcts, sparse_library, additive_mechanisms]",
        "engine_budget: 4",
        "cycles: 2",
        "task_local_memory: false",
        "use_knowledge: false",
    ]
    assert all(token in engine and token in blind and token in gap
               for token in shared)
    assert "discovery_budget: 55" in engine
    assert "discovery_budget: 55" in blind
    assert "discovery_budget: 55" in gap
    assert "synthesis_evaluation_reserve: 3" in blind
    assert "synthesis_evaluation_reserve: 3" in gap
    assert "llm_evaluation_reserve: 4" in blind
    assert "llm_evaluation_reserve: 4" in gap
    assert "scientist_orchestration: true" in blind
    assert "scientist_orchestration: true" in gap
    assert "typed_evidence_synthesis: true" in blind
    assert "typed_evidence_synthesis: true" in gap
    assert "posterior_gap_directed: false" in blind
    assert "posterior_gap_directed: true" in gap
