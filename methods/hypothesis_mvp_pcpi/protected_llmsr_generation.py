"""Search-rich proposal loop backed by the protected LLM-SR buffer."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping

from hypothesis_mvp.discovery.contracts import json_safe
from hypothesis_mvp.discovery.proposal_runtime import ProposalBatch

from .protected_llmsr_search import (
    ProtectedLLMSRBuffer, ProtectedSearchCandidate,
)


def _candidate_from_state(state, condition, round_id):
    metrics = state.metrics
    return ProtectedSearchCandidate(
        expression=state.dag.expression,
        lineage_id=state.lineage_id or state.dag.canonical_hash,
        score=-float(metrics.val_nmse),
        signature=(
            -round(float(metrics.val_nmse), 10),
            -round(float(metrics.val_p99), 10),
            -round(float(metrics.stress_nmse), 10),
            -round(float(metrics.complexity), 6)),
        source_condition=str(condition),
        generation_round=int(round_id))


def run_search_rich_proposals(
    runtime, *, parent, arrays, task_name, task_desc,
    island_context: Mapping, condition: str,
    rounds: int = 8, candidates_per_prompt: int = 4,
    num_islands: int = 8, functions_per_prompt: int = 2,
    temperature: float = .1, seed: int = 0,
    candidate_filter: Callable[[ProposalBatch, int, int],
                               tuple[ProposalBatch, list[dict]]] | None = None,
):
    if (type(rounds) is not int or rounds < 1
            or type(candidates_per_prompt) is not int
            or candidates_per_prompt < 1
            or runtime.proposal.candidates_per_island
               < candidates_per_prompt):
        raise ValueError("invalid search-rich proposal budget")
    buffer = ProtectedLLMSRBuffer(
        num_islands=num_islands,
        functions_per_prompt=functions_per_prompt,
        temperature=temperature, seed=seed)
    buffer.register(
        _candidate_from_state(parent, condition, 0), island=None)
    all_states, records = [], []
    for round_id in range(1, rounds + 1):
        island, history = buffer.sample_context()
        batch = runtime.proposal.propose(
            task_name=task_name, task_desc=task_desc,
            round_id=round_id, island="nmse",
            parent_hash=parent.dag.canonical_hash,
            island_context={
                **dict(island_context),
                "search_rich_generation": {
                    "schema": "protected-llmsr-search-context-v1",
                    "round": round_id, "island": island,
                    "maximum_rounds": rounds,
                    "candidates_per_prompt": candidates_per_prompt,
                    "experience_rows": len(history),
                }},
            library_rows=(), ephemeral_refinements=history,
            max_candidates=candidates_per_prompt,
            allow_protocol_repair=False)
        filter_audit = []
        if candidate_filter is not None:
            batch, filter_audit = candidate_filter(
                batch, round_id, island)
        _, states, validity = runtime._evaluate_batch(
            batch, parent, arrays, "nmse", round_id)
        for state in states:
            buffer.register(
                _candidate_from_state(
                    state, condition, round_id),
                island=island)
        all_states.extend(states)
        records.append({
            "round": round_id, "island": island,
            "experience_context": history,
            "provider_protocol_valid": batch.protocol_valid,
            "provider_candidate_count": len(batch.candidates),
            "materialized_candidate_count": len(states),
            "validity": json_safe(validity),
            "filter_audit": json_safe(filter_audit),
            "buffer_after_round": buffer.audit(),
        })
    distinct = {}
    for state in all_states:
        distinct.setdefault(state.dag.canonical_hash, state)
    return tuple(distinct.values()), {
        "schema": "protected-llmsr-generation-v1",
        "rounds": rounds,
        "candidates_per_prompt": candidates_per_prompt,
        "maximum_candidate_opportunities":
            rounds * candidates_per_prompt,
        "materialized_candidate_count": len(all_states),
        "distinct_materialized_candidate_count": len(distinct),
        "records": records, "buffer": buffer.audit(),
        "admission_response_accessed": False,
        "posterior_update_response_accessed": False,
        "reporting_response_accessed": False,
        "heldout_opened": False,
    }


__all__ = ["run_search_rich_proposals"]
