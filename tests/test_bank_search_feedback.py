from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import t

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.bank_search_feedback import (
    mixture_location_feedback, feedback_from_posterior, rank_materials,
    attach_feedback, install_feedback, validate_bank_prompt_feedback,
    BANK_PROPOSAL_FEEDBACK_SCHEMA,
)
from hypothesis_mvp.discovery.pcpi_adapter import freeze_expanded_formula_model, EXPANDED_FORMULA_POLICY
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def test_location_score_matches_finite_difference_of_entire_mixture():
    mu = np.array([[0., 2.], [-1., 1.], [2., -2.]])
    scale = np.array([[.5, 2.], [1., .7], [2., 1.]])
    df, weights, y = np.array([3., 8.]), np.log([.4, .6]), np.array([1., .3, 3.])
    logp, score, residual = mixture_location_feedback(mu, scale, df, weights, y)
    def shifted(delta):
        return logsumexp(t.logpdf(y[:, None], df, loc=mu + delta, scale=scale) + weights, axis=1)
    np.testing.assert_allclose(logp, shifted(0.))
    np.testing.assert_allclose(score, (shifted(1e-5) - shifted(-1e-5)) / 2e-5, rtol=1e-8)
    np.testing.assert_allclose(residual, y - mu[:, 1])
    assert not np.allclose(score, residual)


def test_gaussian_limit_and_zero_weight_component():
    logp, score, _ = mixture_location_feedback([[0., 1e6]], [[2., 1.]], [1e9, 3.], [0., -np.inf], [1.])
    assert score[0] == pytest.approx(.25, rel=1e-8)
    assert np.isfinite(logp[0])


def test_component_permutation_and_extreme_finite_values():
    arguments = (np.array([[0., 2.]]), np.array([[1., 2.]]), np.array([3., 7.]), np.log([.3, .7]))
    first = mixture_location_feedback(*arguments, [1e100])
    reordered = mixture_location_feedback(arguments[0][:, ::-1], arguments[1][:, ::-1],
                                         arguments[2][::-1], arguments[3][::-1], [1e100])
    np.testing.assert_allclose(first[0], reordered[0])
    np.testing.assert_allclose(first[1], reordered[1])
    assert np.all(np.isfinite(first[1]))


@pytest.mark.parametrize("weights,scale", [([0., 0.], [[1., 1.]]),
                                         (np.log([.5, .5]), [[0., 1.]]),
                                         (np.log([.5, .5]), [[np.nan, 1.]])])
def test_invalid_law_fails_closed(weights, scale):
    with pytest.raises(ValueError):
        mixture_location_feedback([[0., 1.]], scale, [3., 4.], weights, [0.])


def fitted(*, include_bank_inventory=False):
    x = np.linspace(-1., 1., 12)[:, None]
    data = RoleDataset(DataRole.DEVELOPMENT, x, x[:, 0] ** 2 + .2 * x[:, 0])
    model = freeze_expanded_formula_model(
        [{"expression": "x0", "source": "engine:pysr"},
         {"expression": "x0**2", "source": "engine:pysr"}],
        n_features=1, prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    engine = model.engine(model.stable_hash)
    posterior = engine.fit_batch(data.X, data.y)
    feedback = feedback_from_posterior(
        engine, posterior, data, search_row_ids=tuple(f"fit:{i}" for i in range(12)),
        forbidden_row_ids=("report:0",), bank_identity=model.stable_hash,
        include_bank_inventory=include_bank_inventory)
    return engine, posterior, data, feedback


def test_actual_posterior_density_and_alignment():
    engine, posterior, data, feedback = fitted()
    np.testing.assert_allclose(feedback.log_density, engine.predictive_logpdf(posterior, data.X, data.y))
    score = np.asarray(feedback.location_score)
    result = rank_materials(feedback, {"aligned": score, "constant": np.ones(12)}, mode="bank")
    assert result[0]["expression"] == "aligned"
    assert result[0]["priority"] > 0
    assert result[1]["constant"]
    assert attach_feedback({"allowed_edits": ["ADD"]}, feedback, mode="bank")["allowed_edits"] == ["ADD"]
    assert feedback.prompt_payload("blind")["regions"] == []


def test_sealed_role_and_overlap_rejected():
    engine, posterior, data, _ = fitted()
    for role, forbidden in ((DataRole.UNTOUCHED_HELDOUT, ("r",)),
                            (DataRole.DEVELOPMENT, ("fit:0",))):
        with pytest.raises(ValueError):
            feedback_from_posterior(engine, posterior, RoleDataset(role, data.X, data.y),
                                    search_row_ids=tuple(f"fit:{i}" for i in range(12)),
                                    forbidden_row_ids=forbidden, bank_identity="fixed")


def test_feedback_enters_real_runtime_context_without_modifying_rights():
    from hypothesis_mvp.discovery.scientific_runtime import ScientificDiscoveryRuntime
    _, _, _, feedback = fitted()
    runtime = SimpleNamespace(config=SimpleNamespace(feedback_generation_policy="legacy", new_skeleton_quota=1),
                              orchestration_context={}, evaluation=SimpleNamespace(
                                  validity_attempts=[], failure_signature=lambda *args: ()))
    current = SimpleNamespace(compact=lambda: {}, dag=SimpleNamespace(expression="x0"))
    exploration = SimpleNamespace(as_prompt_dict=lambda: {})
    before = ScientificDiscoveryRuntime._proposal_context(runtime, current, exploration, "balanced")
    install_feedback(runtime, feedback, mode="bank")
    after = ScientificDiscoveryRuntime._proposal_context(runtime, current, exploration, "balanced")
    assert before["allowed_edits"] == after["allowed_edits"]
    assert before["new_skeleton_quota"] == after["new_skeleton_quota"]
    assert after["bank_predictive_search_feedback"]["regions"]
    with pytest.raises(ValueError):
        install_feedback(runtime, feedback, mode="residual")
    with pytest.raises(ValueError):
        attach_feedback({"admission_gain": 3.}, feedback, mode="bank")


@pytest.mark.parametrize("field", ["structure_library_read", "structure_library_write",
                                  "task_local_memory_read", "task_local_memory_write"])
def test_memory_cannot_contaminate_information_condition(field):
    _, _, _, feedback = fitted()
    runtime = SimpleNamespace(config=SimpleNamespace(**{field: True}), orchestration_context={})
    with pytest.raises(ValueError, match="dedicated"):
        install_feedback(runtime, feedback, mode="bank")


def test_rich_inventory_is_common_and_blind_has_no_empirical_diagnostics():
    _, posterior, _, feedback = fitted(include_bank_inventory=True)
    packets = [feedback.prompt_payload(mode) for mode in ("bank", "residual", "blind")]
    assert all(packet["schema"] == BANK_PROPOSAL_FEEDBACK_SCHEMA for packet in packets)
    assert packets[0]["bank_inventory"] == packets[1]["bank_inventory"] == packets[2]["bank_inventory"]
    assert len(packets[0]["bank_inventory"]) == len(posterior.members)
    assert all(set(row) == {"structure_id", "expression", "basis_terms"}
               for row in packets[0]["bank_inventory"])
    assert all("formula_ast_v1:" not in row["expression"]
               for row in packets[0]["bank_inventory"])
    assert packets[2]["signal_identity"] is None and packets[2]["regions"] == []
    assert "mean_location_score" not in packets[1]["regions"][0]
    assert "mean_residual" not in packets[0]["regions"][0]
    for packet in packets:
        assert validate_bank_prompt_feedback(packet)


def test_legacy_packet_identity_is_preserved_without_opt_in():
    from hypothesis_mvp.discovery.bank_search_feedback import _hash
    _, _, _, feedback = fitted()
    old_fields = {key: getattr(feedback, key) for key in (
        "bank_identity", "search_row_ids", "search_data_identity", "log_density",
        "location_score", "residual", "regions_json")}
    assert feedback.identity == _hash(old_fields)
    assert feedback.prompt_payload()["schema"] == "bank-search-feedback-v1"
    assert "bank_inventory" not in feedback.prompt_payload()


def test_actual_runtime_rejects_mismatched_fitting_rows_before_exploration():
    from hypothesis_mvp.discovery.scientific_runtime import ScientificDiscoveryRuntime
    _, _, data, feedback = fitted(include_bank_inventory=True)
    runtime = SimpleNamespace(config=SimpleNamespace(), orchestration_context={},
                              _explore=lambda *args: pytest.fail("explored wrong task"))
    install_feedback(runtime, feedback, mode="bank")
    with pytest.raises(ValueError, match="fitting rows"):
        ScientificDiscoveryRuntime._request_batches(
            runtime, {}, 0, (data.X, data.y + 1.), [])


def test_live_memory_flag_change_is_rejected_before_exploration():
    from hypothesis_mvp.discovery.scientific_runtime import ScientificDiscoveryRuntime
    _, _, data, feedback = fitted(include_bank_inventory=True)
    runtime = SimpleNamespace(config=SimpleNamespace(), orchestration_context={})
    install_feedback(runtime, feedback, mode="bank")
    runtime.config.task_local_memory_read = True
    with pytest.raises(ValueError, match="isolated KnowledgeRuntime"):
        ScientificDiscoveryRuntime._request_batches(runtime, {}, 0, (data.X, data.y), [])


@pytest.mark.parametrize("corruption", ["schema", "blind_regions", "inventory_identity", "extra_metric"])
def test_malformed_feedback_rejected_before_transport(monkeypatch, corruption):
    from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
    from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime
    _, _, _, feedback = fitted(include_bank_inventory=True)
    packet = feedback.prompt_payload("bank")
    if corruption == "schema":
        packet["schema"] = "guessed-contract"
    elif corruption == "blind_regions":
        packet.update(mode="blind", signal_identity=None)
    elif corruption == "inventory_identity":
        packet["bank_inventory"][0]["expression"] = "x0**4"
    else:
        packet["regions"][0]["mean_residual"] = 2.
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    monkeypatch.setattr(runtime, "_request", lambda *args: pytest.fail("transport called"))
    with pytest.raises(ValueError):
        runtime.propose(task_name="fixture", task_desc="synthetic correctness only",
                        round_id=0, island="balanced", parent_hash="p",
                        island_context={"bank_predictive_search_feedback": packet},
                        library_rows=[], ephemeral_refinements=[])


@pytest.mark.parametrize("quota", [0, 1])
def test_real_proposal_parser_receives_bank_context_and_identical_search_rights(monkeypatch, quota):
    from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
    from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime, ProviderSettings, ProviderRoute
    from hypothesis_mvp.pcpi.reference.expanded_formula_basis import compile_fixed_formula_support
    _, _, data, feedback = fitted(include_bank_inventory=True)
    payloads = []
    def fake_post(url, **kwargs):
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        payloads.append(payload)
        content = {"protocol_id": "hypothesis-proposal-v1", "round_id": 0,
                   "island": "balanced", "candidates": [{
                       "candidate_id": "fixture", "parent_hash": "p",
                       "action": "PROPOSE_NEW_SKELETON" if quota else "REPLACE",
                       "equation": "sin(x0)", "rationale": "correctness fixture, no efficacy claim"}]}
        return SimpleNamespace(status_code=200, text="", json=lambda: {
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12,
                      "extra_metadata": "must-not-be-recorded"}})
    monkeypatch.setattr("hypothesis_mvp.discovery.proposal_runtime.requests.post", fake_post)
    for mode in ("bank", "residual", "blind"):
        runtime = ProposalRuntime(EquationRuntime(1, refit_policy="pcpi-expanded-fixed-inner-v1"), 1,
                                  ProviderSettings(routes=(ProviderRoute("https://fixture.invalid", "fixture", "secret"),),
                                                   attempts=1, min_request_interval_s=0), 1)
        packet = attach_feedback({"new_skeleton_quota": quota}, feedback, mode=mode)
        batch = runtime.propose(task_name="fixture", task_desc="synthetic correctness only",
                                round_id=0, island="balanced", parent_hash="p", island_context=packet,
                                library_rows=[], ephemeral_refinements=[], allow_protocol_repair=False)
        assert batch.protocol_valid and len(batch.candidates) == 1
        assert compile_fixed_formula_support(batch.candidates[0].equation, 1)
        assert np.all(np.isfinite(runtime.equation_runtime.predict(batch.candidates[0].equation, data.X)))
        assert batch.telemetry["bank_feedback_mode"] == mode
        outcome = batch.telemetry["provider_requests"][0]["provider_outcomes"][0]
        assert outcome["provider_prompt_tokens"] == 7 and outcome["provider_usage_reported"]
        assert outcome["provider_completion_tokens"] == 5 and outcome["provider_total_tokens"] == 12
        assert "secret" not in json.dumps(batch.telemetry)
        assert "must-not-be-recorded" not in json.dumps(batch.telemetry)
        assert runtime.attempt_count == runtime.call_count == 1
    contracts = [payload["contract"] for payload in payloads]
    assert contracts[0] == contracts[1] == contracts[2]
    assert "Refine only the frozen parent" not in contracts[0]["instruction"]
    assert "whole frozen Bank" in contracts[0]["instruction"]


@pytest.mark.parametrize("usage,status", [(None, "missing"), ({}, "invalid"),
    ({"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2}, "invalid"),
    ({"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 1}, "invalid")])
def test_missing_or_invalid_usage_is_unknown_without_retry(monkeypatch, usage, status):
    from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
    from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime, ProviderSettings, ProviderRoute
    monkeypatch.setattr("hypothesis_mvp.discovery.proposal_runtime.requests.post", lambda *args, **kwargs:
                        SimpleNamespace(status_code=200, text="", json=lambda: {
                            "choices": [{"message": {"content": "{}"}}], "usage": usage}))
    route = ProviderRoute("https://fixture.invalid", "fixture", "secret")
    runtime = ProposalRuntime(EquationRuntime(1), 1, ProviderSettings(routes=(route,), min_request_interval_s=0), 1)
    _, telemetry = runtime._request([], "fixture")
    outcome = telemetry["provider_outcomes"][0]
    assert runtime.attempt_count == 1 and not outcome["provider_usage_reported"]
    assert outcome["provider_usage_status"] == status
    assert all(outcome[f"provider_{key}_tokens"] is None for key in ("prompt", "completion", "total"))
