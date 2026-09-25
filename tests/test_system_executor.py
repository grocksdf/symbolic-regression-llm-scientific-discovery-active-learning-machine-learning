"""End-to-end wiring fixtures only; no real data, provider or engine execution."""
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path
from hashlib import sha256
import json
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
        "hypothesis_bank_gate": {"schema": "scientific-hypothesis-bank-gate-v4",
                                 "exact_eig_epsabs": 1e-10,
                                 "maximum_candidates": 4,
                                 "selection_rule": "two-fold-safe-half-core-source-stacking-operational-entropy-v3",
                                 "source_safety_folds": 2,
                                 "source_stacking_baseline": "core",
                                 "source_stacking_dyadic_depth": 8,
                                 "source_stacking_max_optional_mass": 0.5,
                                 "require_all_variants": True},
        "marginal_influence_gate": {
            "schema": "scientific-source-admission-influence-gate-v5",
            "exact_eig_epsabs": 1e-10,
            "required_contributions": ["llm", "engine:mcts"],
            "required_active_contributions": ["llm"],
            "rejectable_contributions": ["engine:mcts"],
            "source_admission_rule": "independent-sourcewise-fold-safe-half-core-log-score-stacking-v2",
            "decision_rule": "full-target-certified-regret-v1",
            "quality_rule": "positive-paired-cumulative-log-predictive-ratio-v1",
            "arbitration_fraction": 0.5,
            "require_all_contributions": True},
        "data_loading_seconds": 30, "provider_attempt_ceiling": 2,
        "provider_public_identity": executor.public_provider_identity(_settings()),
        "coefficient_policy": "discard-fitted-coefficients-refit-closed-basis",
        "user_execution_authorized": True}


def test_registration_accepts_four_typed_engine_skills():
    config = _config()
    config["agent"]["engines"] = [
        "polynomial_lasso", "mcts", "sparse_library",
        "additive_mechanisms"]
    config["agent"]["engine_repeats"] = 1
    config["agent"]["engine_budget"] = 4
    config["marginal_influence_gate"]["required_contributions"] = [
        "llm", "engine:mcts", "engine:sparse_library",
        "engine:additive_mechanisms"]
    config["marginal_influence_gate"]["rejectable_contributions"] = [
        "engine:mcts", "engine:sparse_library",
        "engine:additive_mechanisms"]
    assert executor.validate_system_registration(config) is config


def test_registration_accepts_policy_level_scientist_ablation():
    config = _config()
    gate = config["marginal_influence_gate"]
    gate["schema"] = "scientific-policy-and-source-influence-gate-v6"
    gate["required_contributions"] = ["scientist_policy", "engine:mcts"]
    gate["required_active_contributions"] = ["scientist_policy"]
    gate["scientist_policy_ablation"] = "full-vs-no_llm-matched-budget-v1"
    assert executor.validate_system_registration(config) is config


def test_registration_accepts_task_local_memory_only_for_scientist_policy():
    config = _config()
    config["agent"]["scientist_orchestration"] = True
    config["agent"]["task_local_memory"] = True
    gate = config["marginal_influence_gate"]
    gate["schema"] = "scientific-policy-and-source-influence-gate-v6"
    gate["required_contributions"] = ["scientist_policy", "engine:mcts"]
    gate["required_active_contributions"] = ["scientist_policy"]
    gate["scientist_policy_ablation"] = "full-vs-no_llm-matched-budget-v1"
    assert executor.validate_system_registration(config) is config
    config["agent"]["scientist_orchestration"] = False
    with pytest.raises(ValueError, match="task-local memory"):
        executor.validate_system_registration(config)


def test_bayesian_skill_policy_certificate_is_hash_and_replay_bound(tmp_path):
    reliability = {
        name: {"posterior_mean": .5, "lower_credible_bound": .2}
        for name in ("polynomial_lasso", "mcts", "sparse_library",
                     "additive_mechanisms")}
    certificate = {
        "schema": "scientific-bayesian-skill-policy-certificate-v1",
        "passed": True, "candidate_response_accessed": False,
        "heldout_opened": False, "replay": {"identity": "replay"},
        "skill_reliability": reliability}
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(certificate), encoding="utf-8")
    config = _config()
    gate = config["marginal_influence_gate"]
    gate["schema"] = "scientific-policy-and-source-influence-gate-v6"
    gate["required_contributions"] = ["scientist_policy", "engine:mcts"]
    gate["required_active_contributions"] = ["scientist_policy"]
    gate["scientist_policy_ablation"] = "full-vs-no_llm-matched-budget-v1"
    config["skill_policy"] = {
        "schema": "scientific-bayesian-skill-policy-binding-v1",
        "certificate": str(path.resolve()),
        "certificate_sha256": sha256(path.read_bytes()).hexdigest(),
        "replay_identity": "replay",
        "allocation_method":
            "posterior-lower-mean-llm-preference-diminishing-returns-v1"}
    assert executor.validate_system_registration(config) is config
    loaded, identity = executor._load_skill_policy_certificate(config)
    assert loaded == reliability and identity == "replay"
    config["skill_policy"]["certificate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        executor._load_skill_policy_certificate(config)


def test_task_local_probe_certificate_is_hash_model_and_replay_bound(tmp_path):
    model = {"schema": "scientific-task-local-probe-model-v1",
        "identity": "model", "candidate_response_accessed": False,
        "heldout_opened": False}
    certificate = {
        "schema": "scientific-task-local-probe-skill-certificate-v1",
        "passed": True, "candidate_response_accessed": False,
        "heldout_opened": False, "replay": {"identity": "replay"},
        "model": model}
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(certificate), encoding="utf-8")
    config = _config()
    gate = config["marginal_influence_gate"]
    gate["schema"] = "scientific-policy-and-source-influence-gate-v6"
    gate["required_contributions"] = ["scientist_policy", "engine:mcts"]
    gate["required_active_contributions"] = ["scientist_policy"]
    gate["scientist_policy_ablation"] = "full-vs-no_llm-matched-budget-v1"
    config["probe_skill_policy"] = {
        "schema": "scientific-task-local-probe-policy-binding-v1",
        "certificate": str(path.resolve()),
        "certificate_sha256": sha256(path.read_bytes()).hexdigest(),
        "replay_identity": "replay", "model_identity": "model",
        "allocation_method":
            "one-probe-each-laplace-posterior-llm-preference-v1"}
    assert executor.validate_system_registration(config) is config
    loaded, identity = executor._load_probe_skill_policy_certificate(config)
    assert loaded == model and identity == "replay"
    config["probe_skill_policy"]["model_identity"] = "other"
    with pytest.raises(ValueError, match="not eligible"):
        executor._load_probe_skill_policy_certificate(config)


def test_negative_llm_preference_certificate_disables_job_influence(tmp_path):
    certificate = {
        "schema": "scientific-llm-preference-policy-certificate-v1",
        "passed": False, "candidate_response_accessed": False,
        "heldout_opened": False,
        "replay": {"identity": "preference",
                   "preference_beta_90pct_lower": -.1}}
    path = tmp_path / "preference.json"
    path.write_text(json.dumps(certificate), encoding="utf-8")
    config = _config()
    config["probe_skill_policy"] = {
        "schema": "fixture-probe-binding"}
    config["llm_preference_policy"] = {
        "schema": "scientific-llm-preference-policy-binding-v1",
        "certificate": str(path.resolve()),
        "certificate_sha256": sha256(path.read_bytes()).hexdigest(),
        "replay_identity": "preference",
        "decision": "disable-llm-job-preference-v1"}
    mode, identity = executor._load_llm_preference_policy_decision(config)
    assert mode == "probe-only" and identity == "preference"
    certificate["replay"]["preference_beta_90pct_lower"] = .1
    path.write_text(json.dumps(certificate), encoding="utf-8")
    config["llm_preference_policy"]["certificate_sha256"] = sha256(
        path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not certified"):
        executor._load_llm_preference_policy_decision(config)


def _data():
    def role(kind, x): return RoleDataset(kind, np.array(x, dtype=float)[:, None], np.array(x, dtype=float))
    return OpenSystemData(SelectionData(role(DataRole.DEVELOPMENT, [0, 1]),
        role(DataRole.VALIDATION, [2, 3]), None, ()),
        role(DataRole.DEVELOPMENT, [4, 5]), role(DataRole.VALIDATION, [8, 9, 10, 11]),
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
    def select_fixture(candidates, *args, **kwargs):
        families = sorted({executor.source_family(row) for row in candidates})
        weights = {family: 1.0 / len(families) for family in families}
        return tuple(candidates), {"schema": "fixture-capacity-bank",
            "candidate_response_accessed": False, "source_safety_passed": True,
            "source_prior_weights": weights, "heldout_opened": False}
    monkeypatch.setattr(executor, "select_operational_capacity_bank", select_fixture)
    monkeypatch.setattr(executor, "filter_fold_safe_source_candidates",
        lambda candidates, *args, **kwargs: (tuple(candidates), {
            "schema": "fixture-candidate-admission", "candidate_certificates": [],
            "candidate_response_accessed": False, "heldout_opened": False}))
    monkeypatch.setattr(executor,
        "filter_conditionally_complementary_engine_candidates",
        lambda candidates, *args, **kwargs: (tuple(candidates), {
            "schema": "fixture-engine-complementarity",
            "candidate_certificates": [], "retained_engine_count": 1,
            "candidate_response_accessed": False, "heldout_opened": False}))
    def admission_fixture(candidates, *args, **kwargs):
        families = sorted({executor.source_family(row) for row in candidates})
        core = families.index("core")
        weights = np.zeros(len(families))
        optional = [index for index in range(len(families)) if index != core]
        if optional:
            weights[core] = .5
            weights[optional] = .5 / len(optional)
        else:
            weights[core] = 1.0
        certificate = SimpleNamespace(source_weights=dict(zip(families, weights)),
            stable_hash="admission", to_dict=lambda: {"fixture_only": True})
        sources = {family: {"weight": float(weights[index]), "admitted": True,
            "negative_transfer_certified": False,
            "fold_log_score_gains_vs_core": [1., 1.],
            "fold_numerical_tolerances": [0., 0.]}
            for index, family in enumerate(families)}
        return certificate, sources
    monkeypatch.setattr(executor, "calibrate_source_admission", admission_fixture)
    monkeypatch.setattr(executor, "freeze_discovery_model",
        lambda candidates, **kwargs: SimpleNamespace(stable_hash="model"))
    monkeypatch.setattr(executor, "freeze_discovery_target",
        lambda *args, **kwargs: SimpleNamespace(stable_hash="target"))
    monkeypatch.setattr(executor, "initial_eig_interval_profile",
                        lambda variant, *args, **kwargs: variant)
    monkeypatch.setattr(executor, "predictive_quality_profile",
                        lambda variant, *args, **kwargs: variant)
    monkeypatch.setattr(executor, "leave_one_source_out_candidates",
                        lambda candidates, contribution: candidates[:-1])
    monkeypatch.setattr(executor, "audit_marginal_influence",
        lambda profiles, **kwargs: {
            "schema": "scientific-marginal-decision-influence-family-gate-v1",
            "passed": True, "candidate_response_accessed": False,
            "heldout_opened": False, "efficacy_demonstrated": False})
    monkeypatch.setattr(executor, "_bind_source_admission_to_influence",
                        lambda report, admission: report)
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
                {"expression": "x0", "source": "engine:polynomial_lasso", "origin": "deterministic"},
                {"expression": "x0**2" if supported else "tan(x0)",
                 "source": "engine:mcts" if variant == "full" else "engine:polynomial_lasso",
                 "origin": "deterministic"},
            ]
            if variant != "no_llm":
                candidates.append({"expression": "x0**3", "source": "llm_proposal",
                                   "origin": "llm"})
            rows.append({"variant": variant, "candidates": candidates,
                "hypothesis_provenance": {"candidate_count": len(candidates),
                    "distinct_sources": sorted({row["source"] for row in candidates}),
                    "engine_sources": sorted({row["source"] for row in candidates
                                              if row["source"].startswith("engine:")}),
                    "llm_retained_candidate_count": sum(
                        row["origin"] == "llm" for row in candidates)}})
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


def test_fresh_screen_stops_before_all_pool_responses(tmp_path, monkeypatch):
    _patch(monkeypatch)
    def forbidden(*args):
        raise AssertionError("fresh source screen must not reveal a pool response")
    monkeypatch.setattr(PoolOracle, "acquire_indices", forbidden)
    root = tmp_path / "fresh-screen"
    result = executor.execute_registered_system(
        tmp_path / "source", root, _config(), {}, execution_role="user",
        measurement_authorized=False)
    assert result["protocol_complete"] is True
    assert result["measurement_authorized"] is False
    assert result["candidate_response_accessed"] is False
    assert (root / "SCREEN_MANIFEST.json").is_file()
    assert not list(root.rglob("DECISION-*.json"))
    assert not list(root.rglob("RECEIPT-*.json"))
    assert not list(root.rglob("measured"))


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
    gate = root / "uci_gas_turbine_co" / "11" / "H0_HYPOTHESIS_BANK_VIABILITY.json"
    assert gate.is_file()
    assert not (root / "uci_gas_turbine_co" / "11" / "measured").exists()
    assert (root / "TERMINAL_FAILURE.json").exists()
    assert not (root / "SYSTEM_MANIFEST.json").exists()
    with pytest.raises(ValueError, match="terminally failed"):
        executor.execute_registered_system(tmp_path / "source", root, _config(), {}, execution_role="user")


def test_full_composition_accepts_certified_optional_engine_rejection():
    candidates = [
        {"source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"source": "llm_proposal", "origin": "llm"},
    ]
    admission = {"candidate_certificates": [
        {"family": "engine:mcts", "admitted": False,
         "negative_transfer_certified": True},
        {"family": "engine:mcts", "admitted": False,
         "negative_transfer_certified": True},
    ]}
    decisions = executor._variant_composition("full", candidates, admission)
    assert all(decisions.values())
    admission["candidate_certificates"][0]["negative_transfer_certified"] = False
    decisions = executor._variant_composition("full", candidates, admission)
    assert not decisions[
        "full_optional_engine_retained_or_candidatewise_safe_rejection_certified"]


def test_full_composition_audits_multiple_optional_engine_families():
    candidates = [
        {"source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"source": "llm_proposal", "origin": "llm"},
        {"source": "engine:sparse_library", "origin": "deterministic"},
    ]
    admission = {"candidate_certificates": [{
        "family": "engine:mcts", "admitted": False,
        "negative_transfer_certified": True}, {
        "family": "engine:additive_mechanisms", "admitted": False,
        "redundant_support_certified": True}]}
    decisions = executor._variant_composition(
        "full", candidates, admission,
        optional_engine_families=(
            "engine:mcts", "engine:sparse_library",
            "engine:additive_mechanisms"),
        required_active_contributions=("llm",))
    assert decisions[
        "full_optional_engine_retained_or_candidatewise_safe_rejection_certified"]
    details = decisions["optional_engine_decisions"]
    assert details["engine:sparse_library"]["retained"] is True
    assert details["engine:mcts"]["safely_rejected"] is True
    assert details["engine:additive_mechanisms"]["safely_rejected"] is True


def test_conditional_complementarity_certificate_overrides_earlier_admission():
    candidates = [
        {"source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"source": "llm_evidence_synthesis", "origin": "llm"}]
    admission = {
        "candidate_certificates": [{
            "family": "engine:additive_mechanisms", "admitted": True,
            "negative_transfer_certified": False}],
        "conditional_engine_complementarity": {
            "candidate_certificates": [{
                "family": "engine:additive_mechanisms", "admitted": False,
                "negative_transfer_certified": True}]}}
    decisions = executor._variant_composition(
        "full", candidates, admission,
        optional_engine_families=("engine:additive_mechanisms",))
    detail = decisions["optional_engine_decisions"][
        "engine:additive_mechanisms"]
    assert detail["passed"] is True
    assert detail["safely_rejected"] is True
    assert detail["certificate_stage"] == "conditional-complementarity"


def test_policy_level_composition_does_not_require_llm_formula_retention():
    candidates = [
        {"source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"source": "engine:sparse_library", "origin": "deterministic"}]
    trace = [{"research_plan": {"engine_calls": [{"engine": "a", "jobs": 1}]},
              "scientist_review": {"stop": True},
              "provider_calls": 2,
              "candidate_response_accessed": False,
              "heldout_opened": False}]
    decisions = executor._variant_composition(
        "full", candidates,
        {"candidate_certificates": []},
        optional_engine_families=("engine:sparse_library",),
        required_active_contributions=("scientist_policy",),
        scientist_policy_mode=True,
        scientist_policy_trace=trace, provider_configured=True)
    assert decisions["scientist_policy_execution_audited"] is True
    assert "llm_enabled_variant_retains_llm_hypothesis" not in decisions


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


def test_source_safety_failure_stops_before_validation_or_pool_response(tmp_path, monkeypatch):
    _patch(monkeypatch)
    def failed_selection(candidates, *args, **kwargs):
        families = sorted({executor.source_family(row) for row in candidates})
        return tuple(candidates), {"schema": "fixture-capacity-bank",
            "candidate_response_accessed": False,
            "source_arbitration_validation_response_accessed": False,
            "source_safety_passed": False,
            "source_prior_weights": {family: 1.0 / len(families) for family in families},
            "heldout_opened": False}
    monkeypatch.setattr(executor, "select_operational_capacity_bank", failed_selection)
    def forbidden(*args, **kwargs):
        raise AssertionError("source-safety failure must precede validation and pool responses")
    monkeypatch.setattr(executor, "predictive_quality_profile", forbidden)
    monkeypatch.setattr(PoolOracle, "acquire_indices", forbidden)
    root = tmp_path / "output"
    with pytest.raises(executor.HypothesisBankNotViable):
        executor.execute_registered_system(
            tmp_path / "source", root, _config(), {}, execution_role="user")
    coordinate = root / "uci_gas_turbine_co" / "11"
    assert (coordinate / "H0_HYPOTHESIS_BANK_VIABILITY.json").is_file()
    assert not (coordinate / "MARGINAL_DECISION_INFLUENCE.json").exists()
    assert not (root / "uci_gas_turbine_co" / "11" / "measured").exists()


def test_predictive_safety_is_scoped_to_full_not_negative_controls():
    candidates = [
        {"source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"source": "engine:mcts", "origin": "deterministic"},
        {"source": "llm_proposal", "origin": "llm"},
    ]
    registered = ("origin:llm", "engine:mcts")
    assert executor._selection_safety_roles("full", candidates, registered) == registered
    assert executor._selection_safety_roles("no_llm", candidates[:2], registered) == ()
    assert executor._selection_safety_roles("single_engine", candidates[::2], registered) == ()
    with pytest.raises(ValueError, match="lost"):
        executor._selection_safety_roles("full", candidates[:2], registered)
    with pytest.raises(ValueError, match="unknown"):
        executor._selection_safety_roles("unregistered", candidates, registered)


def test_continuation_screen_branch_precedes_measured_comparisons():
    import inspect
    source = inspect.getsource(executor.execute_registered_system_continuation)
    assert source.index("if measurement_authorized is not True") < source.index(
        "comparisons = {row[\"variant\"]: _run_comparison_variant")
    assert '"candidate_response_accessed": False' in source
    assert '"measurement_authorized": False' in source


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
