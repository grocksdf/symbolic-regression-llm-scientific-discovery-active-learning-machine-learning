"""Response-free checks for topology capacity and loop claim boundaries."""

import json
import pytest

from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import (
    PROPOSAL_PROTOCOL_ID, ProposalContext, ProposalRuntime, ProtocolError,
)


def _runtime() -> ProposalRuntime:
    return ProposalRuntime(EquationRuntime(
        2, refit_policy="pcpi-expanded-fixed-inner-v1"), 2, None, 4)


def _response(expressions):
    return {"protocol_id": PROPOSAL_PROTOCOL_ID, "round_id": 1,
            "island": "novelty", "candidates": [
                {"candidate_id": f"c{i}", "parent_hash": "p",
                 "action": "PROPOSE_NEW_SKELETON", "equation": expression,
                 "rationale": "controlled grammar check"}
                for i, expression in enumerate(expressions)]}


EXPRESSIONS = (
    "exp(-0.7*x0)", "log(1+Abs(x1))",
    "sqrt(1+x0**2)", "x0/(1+Abs(x1))",
)


@pytest.mark.parametrize("quota", (1, 2, 4))
def test_skeleton_breadth_capacity_has_no_silent_clamping(quota):
    runtime = _runtime()
    context = ProposalContext(1, "novelty", "p", 2, 4,
                              new_skeleton_quota=quota)
    rows, rejected, _ = runtime._validate_batch(
        _response(EXPRESSIONS[:quota]), context, "p", "r")
    assert len(rows) == quota and not rejected
    assert all(row.action == "PROPOSE_NEW_SKELETON" for row in rows)
    if quota > 1:
        shorter, rejected, _ = runtime._validate_batch(
            _response(EXPRESSIONS[:quota - 1]), context, "p", "r")
        assert len(shorter) == quota - 1 and not rejected
    if quota < 4:
        with pytest.raises(ProtocolError, match="new_skeleton_quota_exceeded"):
            runtime._validate_batch(_response(EXPRESSIONS[:quota + 1]),
                                    context, "p", "r")


def test_breadth_rejects_insufficient_batch_capacity_before_provider_call():
    runtime = _runtime()
    runtime.candidates_per_island = 2
    with pytest.raises(ProtocolError, match="invalid_new_skeleton_quota"):
        runtime.propose(task_name="fixture", task_desc="fixture", round_id=1,
            island="novelty", parent_hash="p",
            island_context={"new_skeleton_quota": 4},
            library_rows=(), ephemeral_refinements=())
    with pytest.raises(ValueError, match="quota"):
        DiscoveryConfig.from_mapping({"refit_policy":
            "pcpi-expanded-fixed-inner-v1", "candidates_per_island": 2,
            "new_skeleton_quota": 4})


def test_historical_parser_rejects_new_skeleton_action():
    runtime = ProposalRuntime(EquationRuntime(2), 2, None, 4)
    context = ProposalContext(1, "novelty", "p", 2, 4)
    with pytest.raises(ProtocolError, match="new_skeleton_not_registered"):
        runtime._candidate(_response(EXPRESSIONS[:1])["candidates"][0],
                           0, context, "p", "r")


def test_repeated_topology_is_not_counted_twice():
    runtime = _runtime()
    context = ProposalContext(1, "novelty", "p", 2, 4,
                              new_skeleton_quota=2)
    repeated = _response((EXPRESSIONS[0], EXPRESSIONS[0]))
    accepted, rejected, _ = runtime._validate_batch(repeated, context, "p", "r")
    assert len(accepted) == 1 and len(rejected) == 1
    assert rejected[0]["error"] == "duplicate_new_skeleton_support"


def test_actual_proposal_request_carries_four_skeletons(monkeypatch):
    runtime = _runtime()
    observed = {}

    def answer(messages, prompt_hash):
        observed["payload"] = json.loads(messages[1]["content"])
        return json.dumps(_response(EXPRESSIONS)), {"actual_provider": "fixture"}

    monkeypatch.setattr(runtime, "_request", answer)
    batch = runtime.propose(task_name="fixture", task_desc="fixture",
        round_id=1, island="novelty", parent_hash="p",
        island_context={"new_skeleton_quota": 4},
        library_rows=(), ephemeral_refinements=())
    assert batch.protocol_valid and len(batch.candidates) == 4
    assert observed["payload"]["contract"]["new_skeleton_quota"] == 4
    assert batch.telemetry["valid_new_skeletons"] == 4


def test_zero_quota_payload_does_not_request_forbidden_action():
    runtime = _runtime()
    context = ProposalContext(1, "novelty", "p", 2, 2,
                              new_skeleton_quota=0)
    payload = runtime._proposal_payload("fixture", "fixture", context,
                                        {}, (), ())
    assert "PROPOSE_NEW_SKELETON" not in payload["contract"]["allowed_actions"]
    assert "not registered" in payload["contract"]["instruction"]


def test_registered_iterative_config_has_real_topology_budget():
    from pathlib import Path
    import yaml
    from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig
    from hypothesis_mvp.discovery.proposal_runtime import ProviderRoute, ProviderSettings

    raw = yaml.safe_load((Path(__file__).parents[1] / "configs" /
        "aistats_three_arm_formula_expanded_candidate.yaml").read_text())
    config = DiscoveryAgentConfig(**raw["agent_config"])
    assert config.iterative_posterior_refinement
    assert config.new_skeleton_quota >= 1
    assert config.llm_evaluation_reserve >= (config.new_skeleton_quota
        * len(config.discovery_islands) * config.discovery_rounds)
    DiscoveryAgent(config, ProviderSettings(routes=(ProviderRoute(
        "https://example.invalid", "fixture", "fixture"),)))
