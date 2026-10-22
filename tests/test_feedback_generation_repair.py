"""Synthetic, response-free contracts. No provider or scientific data access."""
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig
from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.factory import build_scientific_discovery_runtime
from hypothesis_mvp.discovery.feedback_generation import (
    POLICY, blind_context, initialize_streams, slot_plan, validate_policy,
    verify_slot_audits,
)
from hypothesis_mvp.discovery.proposal_runtime import (
    ProposalCandidate, ProposalRuntime, ProviderRoute, ProviderSettings,
    _ValidatedResponse,
)


def _runtime(tmp_path, allowed, monkeypatch):
    calls = []
    def request(self, payload, context):
        calls.append(json.loads(json.dumps(payload)))
        assert payload["contract"]["max_candidates"] == 1
        slot = payload["scientific_context"]["generation_slot"]
        equation = "x0**2" if slot == 1 else "sin(x0)"
        digest = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        row = ProposalCandidate("fixture", context.island, equation,
            "PROPOSE_NEW_SKELETON" if context.new_skeleton_quota else "REPLACE",
            "synthetic correctness", context.parent_hash,
            lineage_id=digest, prompt_hash=digest, response_hash="fixture-response")
        return _ValidatedResponse((row,), (), (), digest, "fixture-response", {})
    monkeypatch.setattr(ProposalRuntime, "_request_validated", request)
    runtime = build_scientific_discovery_runtime(n_features=1,
        config=DiscoveryConfig(evaluation_budget=24, llm_evaluation_reserve=4,
            max_rounds=2, islands=("nmse",), candidates_per_island=2,
            new_skeleton_quota=1, refit_policy="pcpi-expanded-fixed-inner-v1",
            feedback_generation_policy=POLICY),
        library_path=tmp_path / "library.jsonl", ledger_path=tmp_path / "ledger.jsonl",
        provider_settings=ProviderSettings(routes=(ProviderRoute(
            "https://fixture.invalid", "fixture", "fixture"),)),
        orchestration_context={"posterior_gap_brief": {
            "propose_allowed": allowed, "regions": [{"region": 0}], "secret_gap": 42},
            "posterior_gap_existing_supports": [["intercept"], ["x0"]],
            "scientist_review": {"gap_bearing_text": "must never reach blind"},
            "development_validity_history": [{"origin": "llm", "rejection_reason": "syntax_invalid"}]})
    return runtime, calls


@pytest.mark.parametrize("allowed", [False, True])
def test_actual_inner_runtime_protects_blind_and_never_expands_slots(tmp_path, monkeypatch, allowed):
    runtime, calls = _runtime(tmp_path, allowed, monkeypatch)
    x = np.linspace(-1., 1., 16)[:, None]
    _, report = runtime.run(X_train=x[:12], y_train=x[:12, 0] ** 2,
        X_val=x[12:], y_val=x[12:, 0] ** 2,
        base_candidates=[{"expression": "x0", "source": "engine:polynomial_lasso"},
                         {"expression": "1", "source": "engine:pysr"}])
    assert len(calls) == 4  # no transition-based early stop, two fixed rounds
    rounds = report["llm_rounds"]
    for row in rounds:
        memory = row["islands"][0]["memory"]
        verify_slot_audits([memory], runtime.orchestration_context["posterior_gap_brief"])
        assert sum(slot["candidate_count"] for slot in memory["generation_slots"]) <= 2
        assert sum(slot["new_skeleton_quota"] for slot in memory["generation_slots"]) == 1
    for call in calls:
        ctx = call["scientific_context"]
        if ctx["generation_slot"] == 1 or not allowed:
            text = json.dumps(ctx)
            assert "posterior_gap" not in text and "must never reach blind" not in text
            assert "secret_gap" not in text and "syntax_invalid" not in text
            assert call["verified_structure_library"] == call["accepted_refinements"] == []
    lineages = [slot["candidate_lineages"][0] for row in rounds
                for slot in row["islands"][0]["memory"]["generation_slots"]]
    assert len(lineages) == len(set(lineages))


def test_blind_payload_is_invariant_to_gap_and_other_slot_state(tmp_path, monkeypatch):
    first, _ = _runtime(tmp_path / "one", True, monkeypatch)
    second, calls = _runtime(tmp_path / "two", False, monkeypatch)
    x = np.linspace(-1., 1., 12)[:, None]
    seed = {"expression": "x0**2", "source": "engine:polynomial_lasso"}
    arrays = (x, x[:, 0] ** 2, x + .01, (x[:, 0] + .01) ** 2)
    for runtime in (first, second):
        parent, seeds = runtime.evaluation.seed_survivor([seed], *arrays)
        initialize_streams(runtime, seeds, {"nmse": parent})
        exploration = runtime._explore(parent, *arrays[:2], "nmse")
        ctx = blind_context(runtime, parent, exploration, "nmse", 1)
        if runtime is first:
            original = ctx
            runtime._protected_gap_parents["nmse"] = replace(parent, source="gap-leak")
            assert blind_context(runtime, parent, exploration, "nmse", 1) == original
        else:
            assert ctx == original
    from hypothesis_mvp.discovery.feedback_generation import _request_slot
    for runtime in (first, second):
        _request_slot(runtime, "nmse", 1, "blind", 1, 1, arrays)
    assert len(calls) == 2 and calls[0] == calls[1]


def _raw_protocol_runtime(tmp_path, monkeypatch, allowed):
    # Restore the actual protocol parser; replace only the network transport.
    with monkeypatch.context() as setup:
        runtime, _ = _runtime(tmp_path, allowed, setup)
    x = np.linspace(-1., 1., 16)[:, None]
    arrays = (x[:12], x[:12, 0] ** 2, x[12:], x[12:, 0] ** 2)
    parent, seeds = runtime.evaluation.seed_survivor(
        [{"expression": "x0**2", "source": "engine:polynomial_lasso"}], *arrays)
    initialize_streams(runtime, seeds, {"nmse": parent})
    calls = []
    def transport(messages, prompt_hash):
        payload = json.loads(messages[-1]["content"])
        calls.append(payload)
        slot = payload["scientific_context"]["generation_slot"]
        expression = payload["scientific_context"]["current_equation_state"]["expression"]
        if not getattr(runtime, "fixture_return_parent", False):
            expression = "sin(x0)" if slot == 1 else "cos(x0)"
        return json.dumps({"protocol_id": "hypothesis-proposal-v1",
            "runtime_id": "canonical-real-only-discovery", "round_id": payload["round_id"],
            "island": payload["island"], "candidates": [{"candidate_id": "raw-fixture",
                "parent_hash": payload["parent_hash"], "equation": expression,
                "action": "PROPOSE_NEW_SKELETON" if slot == 1 else "REPLACE",
                "rationale": "synthetic protocol correctness fixture"}]}), {"fixture": True}
    monkeypatch.setattr(runtime.proposal, "_request", transport)
    return runtime, arrays, calls


def test_raw_protocol_blind_new_skeleton_rejects_its_own_parent_without_retry(tmp_path, monkeypatch):
    from hypothesis_mvp.discovery.feedback_generation import _request_slot
    runtime, arrays, calls = _raw_protocol_runtime(tmp_path, monkeypatch, False)
    runtime.fixture_return_parent = True
    _, _, batch, audit = _request_slot(runtime, "nmse", 1, "blind", 1, 1, arrays)
    assert not batch.candidates and audit["candidate_count"] == 0
    assert len(calls) == runtime.proposal.call_count == 1
    assert not batch.telemetry["protocol_repair_attempted"]


@pytest.mark.parametrize("allowed", [False, True])
def test_raw_protocol_accepts_both_registered_slots_with_real_lineages(tmp_path, monkeypatch, allowed):
    from hypothesis_mvp.discovery.feedback_generation import _request_slot
    runtime, arrays, calls = _raw_protocol_runtime(tmp_path, monkeypatch, allowed)
    audits = []
    for stream, slot, quota in slot_plan(allowed):
        _, _, batch, audit = _request_slot(runtime, "nmse", slot, stream, quota, 1, arrays)
        assert batch.protocol_valid and len(batch.candidates) == 1
        assert batch.candidates[0].generation_stream == stream
        assert batch.candidates[0].lineage_id in audit["candidate_lineages"]
        audits.append(audit)
    verify_slot_audits([{"generation_slots": audits}], runtime.orchestration_context["posterior_gap_brief"])
    assert len(calls) == runtime.proposal.call_count == 2


def _audited_pair():
    return {"generation_slots": [{"stream": "blind", "slot": number,
        "context_identity": str(number) * 64, "parent_hash": "engine-parent",
        "cycle": 0, "round_id": 1, "island": "nmse", "posterior_gap_identity": "",
        "candidate_count": 1, "new_skeleton_quota": 1 if number == 1 else 0,
        "candidate_lineages": [f"lineage-{number}"], "fallback": number == 2}
        for number in (1, 2)]}


@pytest.mark.parametrize("field,value", [("candidate_count", -1), ("candidate_count", True),
    ("candidate_lineages", []), ("round_id", 2), ("fallback", True), ("new_skeleton_quota", True)])
def test_slot_audit_rejects_invalid_budget_and_batch_bindings(field, value):
    pair = _audited_pair()
    verify_slot_audits([pair], {"propose_allowed": False})
    pair["generation_slots"][0][field] = value
    with pytest.raises(ValueError):
        verify_slot_audits([pair], {"propose_allowed": False})


def test_slot_audit_rejects_empty_repeated_batches_and_reused_lineages():
    brief = {"propose_allowed": False}
    with pytest.raises(ValueError):
        verify_slot_audits([], brief)
    pair = _audited_pair()
    with pytest.raises(ValueError):
        verify_slot_audits([pair, pair], brief)
    pair["generation_slots"][1]["candidate_lineages"] = ["lineage-1"]
    with pytest.raises(ValueError):
        verify_slot_audits([pair], brief)


def test_policy_only_accepts_registered_two_slot_design():
    config = DiscoveryAgentConfig(feedback_generation_policy=POLICY,
        iterative_posterior_refinement=True, posterior_gap_directed=True,
        expanded_formula_synthesis=True, candidates_per_island=2, new_skeleton_quota=1)
    validate_policy(config)
    for changed in ({"candidates_per_island": 3}, {"new_skeleton_quota": 2},
                    {"mask_posterior_gap_from_generation": True}, {"task_local_memory": True}):
        with pytest.raises(ValueError):
            validate_policy(replace(config, **changed))
    assert slot_plan(False) == (("blind", 1, 1), ("blind", 2, 0))
    assert slot_plan(True) == (("blind", 1, 1), ("gap", 2, 0))


def test_invalid_second_slot_does_not_trigger_provider_repair(tmp_path, monkeypatch):
    runtime, calls = _runtime(tmp_path, False, monkeypatch)
    count = []
    def empty(self, payload, context):
        count.append(payload)
        return _ValidatedResponse((), ({"error": "fixture-invalid"},), (), "p", "r", {})
    monkeypatch.setattr(ProposalRuntime, "_request_validated", empty)
    batch = runtime.proposal.propose(task_name="fixture", task_desc="fixture",
        round_id=1, island="nmse", parent_hash="p", island_context={"new_skeleton_quota": 0},
        library_rows=(), ephemeral_refinements=(), max_candidates=1, allow_protocol_repair=False)
    assert not batch.candidates and len(count) == 1


def test_no_winner_does_not_stop_protected_round_schedule(tmp_path, monkeypatch):
    runtime, _ = _runtime(tmp_path, False, monkeypatch)
    calls = []
    def empty(self, payload, context):
        calls.append(payload)
        return _ValidatedResponse((), ({"error": "fixture-invalid"},), (), "p", "r", {})
    monkeypatch.setattr(ProposalRuntime, "_request_validated", empty)
    x = np.linspace(-1., 1., 16)[:, None]
    _, report = runtime.run(X_train=x[:12], y_train=x[:12, 0] ** 2,
        X_val=x[12:], y_val=x[12:, 0] ** 2,
        base_candidates=[{"expression": "x0**2", "source": "engine:polynomial_lasso"}])
    assert len(calls) == 4 and len(report["llm_rounds"]) == 2
    assert all(row["accepted_transition_count"] == 0 for row in report["llm_rounds"])


def test_outer_agent_routes_abstention_to_actual_protected_runtime(tmp_path, monkeypatch):
    from hypothesis_mvp.discovery.agent import DiscoveryAgent
    from hypothesis_mvp.discovery.scientist_policy import ScientistState, deterministic_plan
    from hypothesis_mvp.discovery.scientist_policy import ScientistReview
    from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
    from tests.test_iterative_posterior_feedback import _selection_and_roles
    selection, roles = _selection_and_roles()
    _, calls = _runtime(tmp_path / "setup", False, monkeypatch)
    config = DiscoveryAgentConfig(engines=("polynomial_lasso", "mcts"), engine_budget=2,
        engine_repeats=1, engine_workers=1, cycles=2, discovery_budget=55,
        scientist_orchestration=True, typed_evidence_synthesis=True,
        typed_inner_augmentation=True, posterior_gap_directed=True,
        iterative_posterior_refinement=True, expanded_formula_synthesis=True,
        refit_policy="pcpi-expanded-fixed-inner-v1", discovery_islands=("nmse",),
        discovery_rounds=2, candidates_per_island=2, new_skeleton_quota=1,
        synthesis_evaluation_reserve=3, llm_evaluation_reserve=6,
        feedback_generation_policy=POLICY)
    agent = DiscoveryAgent(config, ProviderSettings(routes=(ProviderRoute(
        "https://fixture.invalid", "fixture", "fixture"),)))
    rows = tuple(SimpleNamespace(engine=e, expression=x, mse_val=1.,
        complexity=2, score=1., lineage_id=e, diagnostics={})
        for e, x in (("polynomial_lasso", "x0**2"), ("mcts", "sin(x0)")))
    monkeypatch.setattr(agent, "_run_engines", lambda *args: SimpleNamespace(all_results=rows))
    monkeypatch.setattr(agent, "_resolve_plan", lambda *args:
        (deterministic_plan(config.engines, config.engine_budget), {}, None, 0))
    review = ScientistReview(("fixture",), (), (), ("fixture",), False, "fixture")
    monkeypatch.setattr(agent, "_resolve_review", lambda *args, **kwargs: (review, {}, None, 0))
    monkeypatch.setattr(ScientistState, "advance", lambda self, **kwargs: self)
    monkeypatch.setattr(agent, "_compile_cycle_synthesis", lambda *args: ([], {}))
    monkeypatch.setattr(agent, "_cycle_record", lambda *args: SimpleNamespace(posterior_gap_brief={}))
    import hypothesis_mvp.discovery.agent as module
    monkeypatch.setattr(module, "system_evaluation", lambda *args: {})
    original = agent._independent_gap_brief
    def abstain(*args, **kwargs):
        gap = original(*args, **kwargs)
        gap["prompt"]["propose_allowed"] = False
        return gap
    monkeypatch.setattr(agent, "_independent_gap_brief", abstain)
    monkeypatch.setattr(module, "attach_system_evidence", lambda *args: None)
    monkeypatch.setattr(module, "_engine_payload", lambda *args: {})
    result = agent.run(selection=selection, task_name="fixture", task_description="fixture",
        output_dir=tmp_path / "run", knowledge_dir=tmp_path / "knowledge", variable_metadata={},
        gap_prior=NormalInverseGammaPrior(), gap_measurement_budget=4, gap_cycle_roles=roles)
    trace = result.system_evaluation["iterative_feedback_trace"]
    assert len(calls) == 8 and len(trace) == 2
    assert all(len(row["generation_slots"]) == 4 for row in trace)
    assert all(slot["stream"] == "blind" for row in trace for slot in row["generation_slots"])
    assert all(not row["admitted_supports"] for row in trace)
    assert trace[0]["bank_after"] == trace[1]["gap_bank"]
