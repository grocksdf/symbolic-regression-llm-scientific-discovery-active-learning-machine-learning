"""Response-free checks for the protected official-LLM-SR adapter."""
import numpy as np
import pytest

from methods.hypothesis_mvp_pcpi.protected_llmsr_search import (
    ProtectedLLMSRBuffer, ProtectedSearchCandidate,
)


def candidate(number, score, signature):
    return ProtectedSearchCandidate(
        expression=f"x0 + {number}", lineage_id=f"lineage-{number}",
        score=float(score), signature=tuple(signature),
        source_condition="fixture", generation_round=number)


def test_multi_island_buffer_reuses_official_softmax_without_responses():
    buffer = ProtectedLLMSRBuffer(
        num_islands=4, functions_per_prompt=2,
        temperature=.1, seed=17)
    for index in range(4):
        buffer.register(
            candidate(index, index, (index % 2, index)), island=index)
    island, rows = buffer.sample_context(island=3)
    assert island == 3 and len(rows) == 1
    assert rows[0]["response_values_retained"] is False
    audit = buffer.audit()
    assert audit["num_islands"] == 4
    assert audit["admission_response_accessed"] is False
    assert audit["reporting_response_accessed"] is False


def test_signature_clusters_keep_best_and_identity_deduplicates():
    buffer = ProtectedLLMSRBuffer(
        num_islands=2, functions_per_prompt=2,
        temperature=.2, seed=1)
    weak = candidate(1, 1., (0., 1.))
    strong = candidate(2, 2., (0., 1.))
    assert buffer.register(weak, island=0)
    assert not buffer.register(weak, island=0)
    assert buffer.register(strong, island=0)
    _, rows = buffer.sample_context(island=0)
    assert len(rows) == 1
    assert rows[0]["lineage_id"] == strong.lineage_id


def test_weak_island_reset_uses_strong_founder():
    buffer = ProtectedLLMSRBuffer(
        num_islands=4, functions_per_prompt=1,
        temperature=.1, seed=5)
    for island, score in enumerate((1., 2., 10., 9.)):
        buffer.register(
            candidate(island, score, (float(island),)), island=island)
    resets = buffer.reset_weak_half()
    assert {row["reset_island"] for row in resets} == {0, 1}
    assert buffer.audit()["cluster_count_by_island"] == [1, 1, 1, 1]


@pytest.mark.parametrize("kwargs", [
    {"num_islands": 1, "functions_per_prompt": 1, "temperature": .1},
    {"num_islands": 2, "functions_per_prompt": 0, "temperature": .1},
    {"num_islands": 2, "functions_per_prompt": 1, "temperature": 0.},
])
def test_invalid_buffer_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        ProtectedLLMSRBuffer(seed=1, **kwargs)
