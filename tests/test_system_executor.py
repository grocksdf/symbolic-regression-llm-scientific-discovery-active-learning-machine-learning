"""End-to-end wiring fixtures only; no real data, provider or engine execution."""
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

from hypothesis_mvp.discovery import system_executor as executor
from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig
from hypothesis_mvp.discovery.proposal_runtime import ProviderSettings, ProviderRoute
from hypothesis_mvp.data.system_protocol import OpenSystemData, ROLE_NAMES
from hypothesis_mvp.data.roles import SelectionData, RoleDataset, DataRole
from hypothesis_mvp.data.oracle import PoolOracle
from hypothesis_mvp.hypotheses import EvidenceRegistry, EvidenceEventType


def _settings():
    return ProviderSettings(routes=(ProviderRoute("https://fixture.invalid", "fixture-model", "fixture-key"),))


def _config():
    return {"schema": "scientific-system-development-registration-v1",
        "data": [{"dataset": "uci_gas_turbine_co", "source": str(Path(__file__).resolve().parent / "unopened-fixture"), "split_seed": 7,
                  "counts": {r: 2 for r in ROLE_NAMES}}], "seeds": [11],
        "agent": asdict(DiscoveryAgentConfig(cycles=1, engine_workers=1, engine_retries=0,
                    engine_budget=4, engine_repeats=2, discovery_budget=10)),
        "single_engine": "polynomial_lasso", "prior": {},
        "scoring": {"minimum_samples": 8, "maximum_samples": 16, "error_safety_factor": 2.},
        "measurement_budget": 2, "exploration_seconds": 30, "policy_seconds": 30,
        "hypothesis_bank_gate": {"schema": "scientific-hypothesis-bank-gate-v1",
                                 "exact_eig_epsabs": 1e-10,
                                 "require_all_variants": True},
        "marginal_influence_gate": {
            "schema": "scientific-source-marginal-decision-influence-gate-v2",
            "exact_eig_epsabs": 1e-10,
            "required_contributions": ["llm", "engine:mcts"],
            "decision_rule": "full-target-certified-regret-v1",
            "require_all_contributions": True},
        "data_loading_seconds": 30, "provider_attempt_ceiling": 2,
        "provider_public_identity": executor.public_provider_identity(_settings()),
        "coefficient_policy": "discard-fitted-coefficients-refit-closed-basis",
        "user_execution_authorized": True}


def _data():
    def role(kind, x): return RoleDataset(kind, np.array(x, dtype=float)[:, None], np.array(x, dtype=float))
    return OpenSystemData(SelectionData(role(DataRole.DEVELOPMENT, [0, 1]),
        role(DataRole.VALIDATION, [2, 3]), None, ()),
        role(DataRole.DEVELOPMENT, [4, 5]), role(DataRole.VALIDATION, [8, 9]),
        PoolOracle(np.array([[6.], [7.]]), np.array([6., 7.])),
        {"family": "gas_turbine", "heldout_opened": False, "fixture_only": True,
         "scientific_context": {"task_name": "fixture_law",
                                "task_description": "fixture description",
                                "feature_names": ["fixture_feature"],
                                "feature_units": ["unit"],
                                "target_name": "fixture_target", "target_unit": "unit"}})


def _inline(function, args=(), kwargs=None, **limits):
    return function(*args, **(kwargs or {})), {"fixture_only": True}


def _patch(monkeypatch, supported=True):
    monkeypatch.setattr(executor, "verify_system_freeze", lambda *args: {})
    monkeypatch.setattr(executor, "registered_provider_settings", lambda *args: _settings())
    monkeypatch.setattr(executor, "load_registered_system_data", lambda registration: _data())
    monkeypatch.setattr(executor, "run_bounded", _inline)
    monkeypatch.setattr("hypothesis_mvp.discovery.system_run.run_bounded", _inline)
    monkeypatch.setattr(executor, "audit_frozen_hypothesis_bank", lambda *args, **kwargs: {
        "schema": "scientific-hypothesis-bank-viability-v1", "passed": True,
        "candidate_response_accessed": False, "heldout_opened": False})
    monkeypatch.setattr(executor, "freeze_initial_eig_interval_profile",
                        lambda variant, *args, **kwargs: variant)
    monkeypatch.setattr(executor, "leave_one_source_out_candidates",
                        lambda candidates, contribution: candidates[:-1])
    monkeypatch.setattr(executor, "audit_marginal_influence",
        lambda profiles, **kwargs: {
            "schema": "scientific-marginal-decision-influence-family-gate-v1",
            "passed": True, "candidate_response_accessed": False,
            "heldout_opened": False, "efficacy_demonstrated": False})
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked",
        lambda components, *args, **kwargs: SimpleNamespace(ranking_certified=True,
            estimate=SimpleNamespace(scores=np.arange(components.locations.shape[1], dtype=float),
                                     error_bounds=np.zeros(components.locations.shape[1]))))
    def exploration(root, selection, **kwargs):
        rows = []
        for variant in ("full", "no_llm", "single_engine"):
            registry = EvidenceRegistry(root / variant / "evidence_registry.jsonl")
            if not registry.events():
                registry.append(hypothesis_id="fixture-hypothesis",
                    event_type=EvidenceEventType.EVIDENCE_ATTACHED, payload={"fixture_only": True})
            candidates = [
                {"expression": "x0", "source": "engine:a", "origin": "deterministic"},
                {"expression": "x0**2" if supported else "tan(x0)",
                 "source": "engine:b" if variant == "full" else "engine:a",
                 "origin": "deterministic"},
            ]
            if variant != "no_llm":
                candidates.append({"expression": "x0**3", "source": "llm_proposal",
                                   "origin": "llm"})
            rows.append({"variant": variant, "candidates": candidates})
        return {"rows": rows, "fixture_only": True}
    monkeypatch.setattr(executor, "run_exploration_ablations", exploration)


def test_entire_pipeline_and_completed_recovery(tmp_path, monkeypatch):
    _patch(monkeypatch)
    config = _config()
    root, source = tmp_path / "output", tmp_path / "source"
    result = executor.execute_registered_system(source, root, config, {}, execution_role="user")
    assert result["protocol_complete"] and not result["superiority_demonstrated"]
    for comparison in result["results"][0]["comparisons"].values():
        assert {m["target"] for m in comparison["manifests"].values()} == {comparison["target"]}
        for manifest in comparison["manifests"].values():
            assert manifest["completed_queries"] == 2
            assert len(manifest["development_curve"]["rmse"]) == 3
    registry = EvidenceRegistry(root / "uci_gas_turbine_co" / "11" / "exploration" / "full" / "evidence_registry.jsonl")
    count = len(registry.events())
    def forbidden(*args): raise AssertionError("complete recovery must not reveal again")
    monkeypatch.setattr(PoolOracle, "acquire_indices", forbidden)
    assert executor.execute_registered_system(source, root, config, {}, execution_role="user") == result
    assert len(registry.events()) == count


def test_unsupported_hypothesis_terminally_blocks_without_filtering(tmp_path, monkeypatch):
    _patch(monkeypatch, supported=False)
    root = tmp_path / "output"
    with pytest.raises(ValueError):
        executor.execute_registered_system(tmp_path / "source", root, _config(), {}, execution_role="user")


def test_degenerate_bank_stops_before_any_pool_response(tmp_path, monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(executor, "audit_frozen_hypothesis_bank", lambda *args, **kwargs: {
        "schema": "scientific-hypothesis-bank-viability-v1", "passed": False,
        "candidate_response_accessed": False, "heldout_opened": False})
    def forbidden(*args): raise AssertionError("degenerate bank must not reveal a response")
    monkeypatch.setattr(PoolOracle, "acquire_indices", forbidden)
    root = tmp_path / "output"
    with pytest.raises(executor.HypothesisBankNotViable):
        executor.execute_registered_system(
            tmp_path / "source", root, _config(), {}, execution_role="user")
    gate = root / "uci_gas_turbine_co" / "11" / "HYPOTHESIS_BANK_VIABILITY.json"
    assert gate.is_file()
    assert not (root / "uci_gas_turbine_co" / "11" / "measured").exists()
    assert (root / "TERMINAL_FAILURE.json").exists()
    assert not (root / "SYSTEM_MANIFEST.json").exists()
    with pytest.raises(ValueError, match="terminally failed"):
        executor.execute_registered_system(tmp_path / "source", root, _config(), {}, execution_role="user")


def test_marginal_influence_failure_stops_before_any_pool_response(tmp_path, monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(executor, "audit_marginal_influence",
        lambda profiles, **kwargs: {
            "schema": "scientific-marginal-decision-influence-family-gate-v1",
            "passed": False, "candidate_response_accessed": False,
            "heldout_opened": False, "efficacy_demonstrated": False})
    def forbidden(*args): raise AssertionError("influence failure must not reveal a response")
    monkeypatch.setattr(PoolOracle, "acquire_indices", forbidden)
    root = tmp_path / "output"
    with pytest.raises(executor.MarginalDecisionInfluenceNotCertified):
        executor.execute_registered_system(
            tmp_path / "source", root, _config(), {}, execution_role="user")
    report = root / "uci_gas_turbine_co" / "11" / "MARGINAL_DECISION_INFLUENCE.json"
    assert report.is_file()
    assert not (root / "uci_gas_turbine_co" / "11" / "measured").exists()


def test_authorization_and_dirty_source_block_before_data(tmp_path, monkeypatch):
    def forbidden(*args): raise AssertionError("must not load data")
    monkeypatch.setattr(executor, "load_registered_system_data", forbidden)
    config = _config(); config["user_execution_authorized"] = False
    with pytest.raises(PermissionError):
        executor.execute_registered_system(tmp_path / "source", tmp_path / "output", config, {}, execution_role="user")
    config["user_execution_authorized"] = True
    def dirty(*args): raise RuntimeError("dirty source")
    monkeypatch.setattr(executor, "verify_system_freeze", dirty)
    with pytest.raises(RuntimeError, match="dirty"):
        executor.execute_registered_system(tmp_path / "source", tmp_path / "output", config, {}, execution_role="user")
    assert not (tmp_path / "output").exists()


def test_provider_identity_excludes_credentials():
    from dataclasses import replace
    changed = replace(_settings(), routes=(replace(_settings().routes[0], api_key="different-fixture-key"),))
    assert executor.public_provider_identity(changed) == executor.public_provider_identity(_settings())


def test_draft_identity_is_never_executable():
    config = _config(); config["user_execution_authorized"] = False
    config["provider_public_identity"] = None
    assert executor.validate_system_registration(config) is config
    config["user_execution_authorized"] = True
    with pytest.raises(ValueError): executor.validate_system_registration(config)


def test_unrelated_generic_credentials_are_not_project_authority(monkeypatch):
    for name in ("HYPOTHESIS_LLM_API_BASE", "HYPOTHESIS_LLM_MODEL", "HYPOTHESIS_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-fixture-key")
    with pytest.raises(ValueError, match="missing project LLM"):
        executor.registered_provider_settings()


def test_existing_provider_file_is_selected_without_network(tmp_path, monkeypatch):
    import json
    path = tmp_path / "config" / "bigmodel_glm_5_2.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"api_base_url": "https://fixture.invalid", "api_path": "/chat/completions",
        "api_method": "POST", "model": "fixture-model", "api_key": "fixture-only",
        "thinking_type": "enabled", "reasoning_effort": "max"}), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-fixture-key")
    def forbidden(*args, **kwargs): raise AssertionError("must not use env or network")
    monkeypatch.setattr(executor.ProviderSettings, "from_environment", forbidden)
    monkeypatch.setattr("hypothesis_mvp.discovery.proposal_runtime.requests.post", forbidden)
    settings = executor.registered_provider_settings(tmp_path)
    assert settings.routes[0].model == "fixture-model"
    assert settings.thinking_type == "enabled" and settings.reasoning_effort == "max"
    config = _config(); config["provider_public_identity"] = executor.public_provider_identity(settings)
    executor.verify_registered_provider(tmp_path, config)
    config["provider_public_identity"] = "0" * 64
    with pytest.raises(ValueError): executor.verify_registered_provider(tmp_path, config)


def test_invalid_saved_provider_does_not_fall_back(tmp_path, monkeypatch):
    path = tmp_path / "config" / "bigmodel_glm_5_2.json"
    path.parent.mkdir(); path.write_text("{}", encoding="utf-8")
    def forbidden(*args, **kwargs): raise AssertionError("no silent env fallback")
    monkeypatch.setattr(executor.ProviderSettings, "from_environment", forbidden)
    with pytest.raises(ValueError): executor.registered_provider_settings(tmp_path)


def test_ignored_local_provider_overrides_tracked_default(tmp_path):
    import json
    root = tmp_path / "config"; root.mkdir()
    common = {"api_base_url": "https://fixture.invalid", "api_path": "/chat/completions",
        "api_method": "POST", "api_key": "fixture-only"}
    (root / "bigmodel_glm_5_2.json").write_text(json.dumps({**common, "model": "old"}), encoding="utf-8")
    (root / "bigmodel.local.json").write_text(json.dumps({**common, "model": "local"}), encoding="utf-8")
    assert executor.registered_provider_settings(tmp_path).routes[0].model == "local"


@pytest.mark.parametrize("key,value", [("measurement_budget", 3), ("policy_seconds", float("nan")),
    ("seeds", [11, 11]), ("coefficient_policy", "silent-refit")])
def test_registration_rejects_invalid_contract(key, value):
    config = _config(); config[key] = value
    with pytest.raises(ValueError): executor.validate_system_registration(config)
