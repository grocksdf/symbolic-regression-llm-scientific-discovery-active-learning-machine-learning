"""Response-free correctness fixtures for executable localized Gap repair."""
import numpy as np

from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.localized_gap_refinement import (
    certify_localized_refinement, compile_executable_gap_brief,
    materialize_localized_refinement,
)
from hypothesis_mvp.discovery.proposal_runtime import (
    PROPOSAL_PROTOCOL_ID, ProposalContext, ProposalRuntime,
)
from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.feedback_generation import (
    SUPPORT_PRESERVING_POLICY, validate_policy,
)
from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig


def _brief():
    return {
        "schema": "pcpi-frozen-bank-exploration-brief-v1",
        "region_axis": 0, "propose_allowed": True,
        "regions": [
            {"region": 0, "lower_bound": None, "upper_bound": 0.},
            {"region": 1, "lower_bound": 0., "upper_bound": None}],
        "independent_adequacy": [
            {"region": 0, "count": 10, "pit_above_median": 8,
             "directional_pit_imbalance": True,
             "directional_pit_lower_bound": .6,
             "score_degradation": False, "undercoverage": False},
            {"region": 1, "count": 10, "pit_above_median": 5,
             "directional_pit_imbalance": False,
             "score_degradation": False, "undercoverage": False}],
    }


def test_gap_brief_compiles_to_explicit_region_direction_and_constraints():
    value = compile_executable_gap_brief(_brief())
    assert value["propose_allowed"]
    assert len(value["target_regions"]) == 1
    target = value["target_regions"][0]
    assert target["region"] == 0
    assert target["direction"] == "increase-parent-prediction"
    assert target["upper_bound"] == 0.
    assert len(target["requirements"]) == 3


def test_multiple_gap_regions_select_exactly_one_deterministic_target():
    brief = _brief()
    brief["independent_adequacy"][1].update(
        directional_pit_imbalance=True,
        directional_pit_lower_bound=.8,
        pit_above_median=1)
    value = compile_executable_gap_brief(brief)
    assert value["eligible_region_count"] == 2
    assert len(value["target_regions"]) == 1
    assert value["target_regions"][0]["region"] == 1
    assert value["target_regions"][0]["direction"] == (
        "decrease-parent-prediction")


def test_code_owned_localized_formula_and_covariate_only_certificate():
    runtime = EquationRuntime(
        1, refit_policy="pcpi-expanded-fixed-inner-v1")
    expression, audit = materialize_localized_refinement(
        runtime, "x0", "(1-tanh(8*x0))/2", "0.5*x0")
    X = np.linspace(-1., 1., 200)[:, None]
    labels = (X[:, 0] >= 0).astype(int)
    certificate = certify_localized_refinement(
        runtime, "x0", expression, audit["gate_expression"],
        X, labels, (0,))
    assert certificate["passed"]
    assert certificate["non_target_change_rms"] <= (
        certificate["target_change_rms"])
    assert certificate["response_accessed"] is False


def test_global_change_fails_locality_certificate():
    runtime = EquationRuntime(
        1, refit_policy="pcpi-expanded-fixed-inner-v1")
    expression, audit = materialize_localized_refinement(
        runtime, "x0", "(1+tanh(8*x0))/2", "x0")
    X = np.linspace(-1., 1., 200)[:, None]
    labels = (X[:, 0] >= 0).astype(int)
    certificate = certify_localized_refinement(
        runtime, "x0", expression, audit["gate_expression"],
        X, labels, (0,))
    assert not certificate["passed"]


def test_proposal_protocol_materializes_gate_and_correction():
    runtime = ProposalRuntime(EquationRuntime(
        1, refit_policy="pcpi-expanded-fixed-inner-v1"), 1, None, 1)
    context = ProposalContext(
        1, "nmse", "parent", 1, 1,
        incumbent_expression="x0", gap_directed=True,
        new_skeleton_quota=0)
    payload = runtime._proposal_payload(
        "fixture", "fixture", context, {
            "posterior_gap_brief": {
                "propose_allowed": True, "regions": [{"region": 0}]},
            "current_expression": "x0",
            "posterior_gap_existing_supports": (),
            "localized_refinement_contract": True,
            "new_skeleton_quota": 0,
        }, (), ())
    assert payload["contract"]["localized_refinement_contract"]
    assert "local_gate" in payload["contract"]["required_candidate_fields"]
    parsed = {
        "protocol_id": PROPOSAL_PROTOCOL_ID,
        "round_id": 1, "island": "nmse",
        "candidates": [{
            "candidate_id": "localized-1",
            "parent_hash": "parent", "action": "ADD",
            "local_gate": "(1-tanh(8*x0))/2",
            "local_correction": "0.5*x0",
            "rationale": "localized synthetic correctness fixture"}]}
    rows, rejected, normalized = runtime._validate_batch(
        parsed, context, "prompt", "response")
    assert len(rows) == 1 and not rejected
    assert normalized[0]["localized_refinement"] is True
    assert rows[0].local_gate
    assert rows[0].local_correction


def test_localized_transport_binds_envelope_and_identity_in_code(
        monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(
        1, refit_policy="pcpi-expanded-fixed-inner-v1"), 1, None, 1)
    context = {
        "posterior_gap_brief": {
            "propose_allowed": True, "regions": [{"region": 0}]},
        "current_expression": "x0",
        "posterior_gap_existing_supports": (),
        "localized_refinement_contract": True,
        "new_skeleton_quota": 0,
    }
    raw = {
        "wrong_protocol": "ignored",
        "round_id": 999, "island": "wrong",
        "candidates": [{
            "candidate_id": "provider-owned-id-is-ignored",
            "parent_hash": "wrong", "action": "DELETE",
            "local_gate": "(1-tanh(8*x0))/2",
            "local_correction": "0.5*x0",
            "rationale": "provider supplies scientific localized fields only"}]}

    monkeypatch.setattr(
        runtime, "_request",
        lambda messages, prompt_hash: (
            __import__("json").dumps(raw), {"fixture": True}))
    batch = runtime.propose(
        task_name="fixture", task_desc="fixture", round_id=1,
        island="nmse", parent_hash="parent",
        island_context=context, library_rows=(),
        ephemeral_refinements=(), max_candidates=1,
        allow_protocol_repair=False)
    assert batch.protocol_valid
    candidate = batch.candidates[0]
    assert candidate.candidate_id == "localized-1-nmse-0"
    assert candidate.parent_hash == "parent"
    assert candidate.action == "ADD"


def test_localized_policy_is_explicit_and_requires_support_preservation():
    inner = DiscoveryConfig.from_mapping({
        "localized_gap_refinement": True})
    assert inner.localized_gap_refinement
    valid = DiscoveryAgentConfig(
        feedback_generation_policy=SUPPORT_PRESERVING_POLICY,
        localized_gap_refinement=True,
        iterative_posterior_refinement=True,
        posterior_gap_directed=True,
        expanded_formula_synthesis=True,
        typed_inner_augmentation=True,
        candidates_per_island=2, new_skeleton_quota=1)
    validate_policy(valid)
    invalid = DiscoveryAgentConfig(
        localized_gap_refinement=True)
    import pytest
    with pytest.raises(ValueError, match="support-preserving"):
        validate_policy(invalid)
