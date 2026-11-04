"""Protocol correctness only; no network or experimental observations."""
import json
import pytest
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime, ProtocolError


def run(monkeypatch, content, *, envelope_policy="request-bound-semantic-v1", **overrides):
    runtime = ProposalRuntime(EquationRuntime(1, refit_policy="pcpi-expanded-fixed-inner-v1"), 1, None, 1)
    requests = []
    def request(messages, prompt_hash):
        requests.append(messages)
        return content, {}
    monkeypatch.setattr(runtime, "_request", request)
    args = dict(task_name="fixture", task_desc="synthetic correctness", round_id=3, island="composition",
        parent_hash="fixture-parent", island_context={"expanded_formula_discovery":True,"new_skeleton_quota":1},
        library_rows=(), ephemeral_refinements=(), max_candidates=1, allow_protocol_repair=False,
        envelope_policy=envelope_policy)
    args.update(overrides)
    return runtime.propose(**args), requests


def semantic():
    return {"candidates":[{"candidate_id":"a", "parent_hash":"fixture-parent",
        "action":"PROPOSE_NEW_SKELETON", "equation":"x0**2", "rationale":"synthetic fixture"}]}


def test_semantic_response_binds_to_synchronous_request_once(monkeypatch):
    batch, requests = run(monkeypatch, json.dumps(semantic()))
    assert batch.protocol_valid and len(requests) == 1
    audit = batch.telemetry["provider_requests"][0]["request_binding"]
    assert audit["round_id"] == 3 and audit["island"] == "composition"
    assert audit["response_hash"] == batch.response_hash and audit["prompt_hash"] == batch.prompt_hash
    assert not audit["provider_freshness_attested"]
    assert "omit those administrative fields" in requests[0][0]["content"]
    legacy, _ = run(monkeypatch, json.dumps(semantic()), envelope_policy="legacy")
    assert not legacy.protocol_valid


@pytest.mark.parametrize("field,value", [("round_id",4),("round_id",True),("island","other"),
    ("protocol_id","wrong"),("runtime_id","wrong"),("parent_hash","other")])
def test_conflicting_metadata_is_rejected_not_repaired(monkeypatch, field, value):
    payload = semantic(); payload[field] = value
    batch, requests = run(monkeypatch, json.dumps(payload))
    assert not batch.candidates and len(requests) == 1
    assert batch.telemetry["candidate_validation_rejections"][0]["error"] == "request_binding_conflict:"+field


def test_candidate_parent_and_duplicate_keys_remain_strict(monkeypatch):
    payload = semantic(); payload["candidates"][0]["parent_hash"] = "other"
    batch, _ = run(monkeypatch, json.dumps(payload))
    assert not batch.candidates
    batch, _ = run(monkeypatch, '{"candidates": [], "candidates": []}')
    assert not batch.candidates


def test_no_implicit_repairs_or_localized_rewrite(monkeypatch):
    for overrides in ({"allow_protocol_repair":True}, {"island_context":{"localized_refinement_contract":True}}):
        with pytest.raises(ProtocolError, match="no_repair"):
            run(monkeypatch, json.dumps(semantic()), **overrides)
