import json

from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime


def _content(equations, *, parent="parent", island="nmse", round_id=1):
    candidates = [
        {
            "candidate_id": f"candidate_{index}",
            "parent_hash": parent,
            "action": "REPLACE",
            "equation": equation,
            "rationale": "generic structural revision",
        }
        for index, equation in enumerate(equations)
    ]
    return json.dumps({
        "protocol_id": "hypothesis-proposal-v1",
        "runtime_id": "canonical-real-only-discovery",
        "round_id": round_id,
        "island": island,
        "candidates": candidates,
    })


def _propose(runtime):
    return runtime.propose(
        task_name="opaque_structure_search", task_desc="generic measured system",
        round_id=1, island="nmse", parent_hash="parent",
        island_context={"current_equation_state": {"expression": "x0"}},
        library_rows=[], ephemeral_refinements=[],
    )


def test_y_assignment_is_normalized_to_rhs(monkeypatch) -> None:
    runtime = ProposalRuntime(EquationRuntime(3), 3, None, candidates_per_island=4)
    monkeypatch.setattr(
        runtime, "_request",
        lambda messages, prompt_hash: (_content(["y = x0 + x1"]), {"actual_provider": "test"}),
    )
    batch = _propose(runtime)
    assert batch.protocol_valid
    assert batch.candidates[0].equation == "x0 + x1"
    assert batch.telemetry["candidate_normalizations"][0]["assignment_removed"] is True


def test_invalid_y_hat_candidate_does_not_discard_valid_peer(monkeypatch) -> None:
    runtime = ProposalRuntime(EquationRuntime(3), 3, None, candidates_per_island=4)
    malformed = (
        "y = 0.0000907729682815634*x0*x2 "
        "+ 24.97398749762013*x2/(1+Abs(x0)) "
        "- 0.001570584913198859*y_hat**3 "
        "- 10000.0*y_hat/(1+Abs(y_hat)) + 988283.4879732945"
    )
    monkeypatch.setattr(
        runtime, "_request",
        lambda messages, prompt_hash: (
            _content([malformed, "x0 + x2"]), {"actual_provider": "test"}
        ),
    )
    batch = _propose(runtime)
    assert not batch.protocol_valid
    assert batch.reason == "partial_candidates_rejected"
    assert [candidate.equation for candidate in batch.candidates] == ["x0 + x2"]
    assert "forbidden_output_symbol:y_hat" in batch.telemetry["candidate_validation_rejections"][0]["error"]


def test_all_invalid_candidates_trigger_one_protocol_repair(monkeypatch) -> None:
    runtime = ProposalRuntime(EquationRuntime(3), 3, None, candidates_per_island=4)
    responses = iter([
        _content(["y = y_hat + x0"]),
        _content(["x0 + x2"]),
    ])
    monkeypatch.setattr(
        runtime, "_request",
        lambda messages, prompt_hash: (next(responses), {"actual_provider": "test"}),
    )
    batch = _propose(runtime)
    assert batch.protocol_valid
    assert batch.reason == "ok_after_protocol_repair"
    assert batch.candidates[0].equation == "x0 + x2"
    assert batch.telemetry["protocol_repair_attempted"] is True
    assert runtime.call_count == 2


def test_provider_failure_is_explicit_without_aborting_search(monkeypatch) -> None:
    runtime = ProposalRuntime(EquationRuntime(3), 3, None, candidates_per_island=4)

    def fail(messages, prompt_hash):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(runtime, "_request", fail)
    batch = _propose(runtime)
    assert not batch.protocol_valid
    assert not batch.candidates
    assert batch.reason == "provider_or_protocol_failure"
    assert "provider unavailable" in batch.error
