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
