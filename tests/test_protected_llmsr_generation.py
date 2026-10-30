"""Synthetic proposal-loop tests for protected LLM-SR search."""
from hashlib import sha256
import json

import numpy as np

from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.factory import (
    build_scientific_discovery_runtime,
)
from hypothesis_mvp.discovery.proposal_runtime import (
    ProposalCandidate, ProposalRuntime, _ValidatedResponse,
)
from methods.hypothesis_mvp_pcpi.protected_llmsr_generation import (
    run_search_rich_proposals,
)


def test_search_rich_loop_uses_equal_fixed_budget_and_development_scores(
        tmp_path, monkeypatch):
    seen = []

    def response(self, payload, context):
        seen.append(payload)
        rows = []
        for index in range(context.max_candidates):
            expression = f"x0 + {round(.01 * (len(seen) + index + 1), 3)}*sin(x0)"
            digest = sha256(
                f"{len(seen)}:{index}:{expression}".encode()).hexdigest()
            rows.append(ProposalCandidate(
                f"c{len(seen)}-{index}", context.island,
                expression, "ADD", "synthetic search fixture",
                context.parent_hash, lineage_id=digest,
                prompt_hash=digest, response_hash=digest))
        return _ValidatedResponse(
            tuple(rows), (), (), "prompt", "response", {})

    monkeypatch.setattr(ProposalRuntime, "_request_validated", response)
    runtime = build_scientific_discovery_runtime(
        n_features=1,
        config=DiscoveryConfig(
            evaluation_budget=40, llm_evaluation_reserve=16,
            candidates_per_island=4, max_rounds=4,
            refit_policy="pcpi-expanded-fixed-inner-v1",
            islands=("nmse",)),
        library_path=tmp_path / "knowledge.jsonl",
        ledger_path=tmp_path / "ledger.jsonl")
    X = np.linspace(-1., 1., 48)[:, None]
    y = X[:, 0]
    parent, _ = runtime.evaluation.seed_survivor(
        [{"expression": "x0", "source": "engine:fixture"}],
        X[:32], y[:32], X[32:], y[32:])
    runtime.evaluation.reset()
    states, audit = run_search_rich_proposals(
        runtime, parent=parent,
        arrays=(X[:32], y[:32], X[32:], y[32:]),
        task_name="fixture", task_desc="fixture",
        island_context={
            "current_expression": parent.dag.expression,
            "new_skeleton_quota": 0},
        condition="B0", rounds=4, candidates_per_prompt=4,
        num_islands=4, functions_per_prompt=2,
        temperature=.1, seed=17)
    assert len(seen) == 4
    assert audit["maximum_candidate_opportunities"] == 16
    assert audit["buffer"]["num_islands"] == 4
    assert audit["admission_response_accessed"] is False
    assert all(
        "search_rich_generation" in payload["scientific_context"]
        for payload in seen)
    assert states
