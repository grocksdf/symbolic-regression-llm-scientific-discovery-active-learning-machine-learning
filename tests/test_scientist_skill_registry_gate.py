"""No-data correctness tests for heterogeneous Scientist skills."""

from hypothesis_mvp.discovery.skill_registry_gate import (
    run_skill_registry_correctness_gate,
)


def test_heterogeneous_skill_registry_gate_passes_without_real_data():
    result = run_skill_registry_correctness_gate()
    assert result["passed"], result["decisions"]
    assert result["registered_engines"] == [
        "polynomial_lasso", "mcts", "sparse_library",
        "additive_mechanisms"]
    assert result["real_data_accessed"] is False
    assert result["candidate_response_accessed"] is False
    assert result["heldout_opened"] is False
    assert result["efficacy_demonstrated"] is False
