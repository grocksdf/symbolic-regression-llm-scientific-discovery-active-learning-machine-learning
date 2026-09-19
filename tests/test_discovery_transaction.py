import json
import numpy as np
import pytest

from hypothesis_mvp.discovery.pcpi_adapter import freeze_discovery_model, freeze_discovery_target
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.discovery_transaction import DiscoveryScoringControls, DiscoveryTransaction
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.data.oracle import PoolOracle


def _session(tmp_path, source="source-frozen-correctness-fixture"):
    model = freeze_discovery_model([
        {"expression": "x0", "source": "engine:a"},
        {"expression": "x0**2", "source": "llm"},
    ], n_features=1, prior=NormalInverseGammaPrior(), exploration_identity="a"*64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis")
    initial = RoleDataset(DataRole.DEVELOPMENT, np.array([[0.], [1.], [2.]]), np.array([.1, 1., 2.]))
    domain = np.array([[3.], [4.]])
    target = freeze_discovery_target(model, initial, domain, measurement_budget=2,
                                    expected_model_identity=model.stable_hash)
    return DiscoveryTransaction(tmp_path, model, target, domain,
                                DiscoveryScoringControls(8, 16, 2.), source_identity=source)


def _broad_analytic(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.analytic_class_eig_bounds",
        lambda components: SimpleNamespace(
            lower_bounds=np.zeros(components.locations.shape[1]),
            upper_bounds=np.ones(components.locations.shape[1]),
            numerical_outward_tolerance=1e-12))


def test_decision_before_response_and_reconstruction(tmp_path):
    session = _session(tmp_path)
    with pytest.raises(FileNotFoundError): session.admit(10, np.array([3.]), 3.)
    decision = session.plan(np.array([10]), np.array([[3.]]))
    assert (tmp_path / "DECISION-001.json").is_file()
    assert not (tmp_path / "RECEIPT-001.json").exists()
    assert decision == session.plan(np.array([10]), np.array([[3.]]))
    session.admit(10, np.array([3.]), 3.)
    recovered = _session(tmp_path)
    assert len(recovered.receipts) == 1
    assert session.prefix_hash == recovered.prefix_hash
    np.testing.assert_allclose([m.probability for m in session.posterior.members],
                               [m.probability for m in recovered.posterior.members])
    with pytest.raises(ValueError): recovered.plan(np.array([10]), np.array([[3.]]))
    recovered.plan(np.array([11]), np.array([[4.]]))
    recovered.admit(11, np.array([4.]), 4.)
    with pytest.raises(ValueError, match="budget"): recovered.plan(np.array([12]), np.array([[3.]]))


def test_information_audit_reports_current_not_frozen_class_entropy(tmp_path):
    from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition
    session = _session(tmp_path)
    first = session.plan(np.array([10]), np.array([[3.]]))
    assert first["information_audit"]["class_entropy_nats"] == pytest.approx(
        session.target.partition.entropy)
    session.admit(10, np.array([3.]), 20.)
    current = predictive_components_for_partition(session.engine, session.posterior,
        session.target.partition, np.array([[4.]])).partition
    second = session.plan(np.array([11]), np.array([[4.]]))
    assert second["information_audit"]["class_entropy_nats"] == pytest.approx(current.entropy)
    assert second["information_audit"]["posterior_effective_class_count"] == pytest.approx(
        np.exp(current.entropy))
    assert second["information_audit"]["strict_prefix_response_count"] == 1
    assert current.entropy != pytest.approx(session.target.partition.entropy)


def test_response_and_candidate_identity_fail_closed(tmp_path):
    session = _session(tmp_path)
    session.plan(np.array([10]), np.array([[3.]]))
    with pytest.raises(ValueError): session.plan(np.array([11]), np.array([[4.]]))
    with pytest.raises(ValueError): session.admit(11, np.array([3.]), 3.)
    with pytest.raises(ValueError): session.admit(10, np.array([4.]), 3.)
    with pytest.raises(ValueError): session.admit(10, np.array([3.]), float("nan"))
    assert not (tmp_path / "RECEIPT-001.json").exists()


def test_tampered_receipt_and_cross_source_are_rejected(tmp_path):
    session = _session(tmp_path)
    session.plan(np.array([10]), np.array([[3.]])); session.admit(10, np.array([3.]), 3.)
    with pytest.raises(ValueError): _session(tmp_path, source="different-source")
    path = tmp_path / "RECEIPT-001.json"
    payload = json.loads(path.read_text()); payload["response"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): _session(tmp_path)


def test_overlapping_exact_intervals_publish_minimax_regret_certificate(monkeypatch, tmp_path):
    from types import SimpleNamespace
    _broad_analytic(monkeypatch)
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked",
                        lambda *args, **kwargs: SimpleNamespace(ranking_certified=False))
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.exact_class_eig_shared_actions",
        lambda components: SimpleNamespace(scores=np.array([.5, .5]),
            quadrature_errors=np.array([1e-4, 1e-4])))
    session = _session(tmp_path)
    decision = session.plan(np.array([10, 11]), np.array([[3.], [4.]]))
    assert decision["candidate_id"] == 10
    assert decision["selection_certificate"] == "exact-interval-minimax-regret"
    assert decision["integration_method"].endswith("interval-minimax-regret")
    assert decision["utility_regret_upper_bound"] == pytest.approx(2e-4)
    assert (tmp_path / "DECISION-001.json").is_file()


def test_uncertified_fast_rule_uses_independent_exact_interval_certificate(monkeypatch, tmp_path):
    from types import SimpleNamespace
    _broad_analytic(monkeypatch)
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked",
        lambda *args, **kwargs: SimpleNamespace(ranking_certified=False))
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.exact_class_eig_shared_actions",
        lambda components: SimpleNamespace(scores=np.array([.2, .7]),
            quadrature_errors=np.array([1e-8, 1e-8])))
    decision = _session(tmp_path).plan(np.array([10, 11]), np.array([[3.], [4.]]))
    assert decision["candidate_id"] == 11
    assert decision["integration_method"] == (
        "shared-action-adaptive-scipy-quad-exact-finite-mixture-interval-ranking")
    assert decision["certified"] is True
    assert decision["selection_certificate"] == "strict-interval-maximizer"
    assert decision["utility_regret_upper_bound"] == 0.0


def test_minimax_regret_uses_lower_bound_not_point_estimate(monkeypatch, tmp_path):
    from types import SimpleNamespace
    _broad_analytic(monkeypatch)
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked",
                        lambda *args, **kwargs: SimpleNamespace(ranking_certified=False))
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.exact_class_eig_shared_actions",
        lambda components: SimpleNamespace(scores=np.array([.55, .50]),
            quadrature_errors=np.array([.20, .01])))
    decision = _session(tmp_path).plan(np.array([10, 11]), np.array([[3.], [4.]]))
    assert decision["candidate_id"] == 11
    assert decision["utility_regret_upper_bound"] == pytest.approx(.26)


def test_invalid_exact_certificate_is_public_and_response_free(monkeypatch, tmp_path):
    from types import SimpleNamespace
    _broad_analytic(monkeypatch)
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked",
                        lambda *args, **kwargs: SimpleNamespace(ranking_certified=False))
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.exact_class_eig_shared_actions",
        lambda components: SimpleNamespace(scores=np.array([np.nan, .5]),
            quadrature_errors=np.array([1e-4, 1e-4])))
    session = _session(tmp_path)
    with pytest.raises(RuntimeError, match="invalid-exact") as caught:
        session.plan(np.array([10, 11]), np.array([[3.], [4.]]))
    assert caught.value.public_diagnostic == "invalid-exact-class-eig-certificate"
    assert not (tmp_path / "DECISION-001.json").exists()


def test_infeasible_structure_quadrature_budget_is_public_and_response_free(
        monkeypatch, tmp_path):
    from hypothesis_mvp.pcpi.acquisition import PredictiveComponents
    session = _session(tmp_path)
    _broad_analytic(monkeypatch)
    structure_count = 11
    components = PredictiveComponents(
        np.full(structure_count, 1.0 / structure_count),
        np.full(structure_count, 7.0),
        np.zeros((structure_count, 2)),
        np.ones((structure_count, 2)),
        session.target.partition,
    )
    monkeypatch.setattr(
        "hypothesis_mvp.pcpi.discovery_transaction.predictive_components_for_partition",
        lambda *args: components)
    with pytest.raises(RuntimeError, match="four-nodes-per-predictive-structure") as caught:
        session.plan(np.array([10, 11]), np.array([[3.], [4.]]))
    assert caught.value.public_diagnostic.endswith("four-nodes-per-predictive-structure")
    assert not (tmp_path / "DECISION-001.json").exists()


def test_analytic_dominance_prunes_only_impossible_maximizers(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.analytic_class_eig_bounds",
        lambda components: SimpleNamespace(
            lower_bounds=np.array([.40, .00]),
            upper_bounds=np.array([.50, .20]),
            numerical_outward_tolerance=1e-12))
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked",
        lambda *args, **kwargs: SimpleNamespace(ranking_certified=False))
    seen = []
    def shared(components):
        seen.append(components.locations.shape[1])
        return SimpleNamespace(scores=np.array([.42]),
                               quadrature_errors=np.array([.01]))
    monkeypatch.setattr(
        "hypothesis_mvp.pcpi.discovery_transaction.exact_class_eig_shared_actions", shared)
    session = _session(tmp_path)
    decision = session.plan(np.array([10, 11]), np.array([[3.], [4.]]))
    assert decision["candidate_id"] == 10
    assert seen == [1]
    assert decision["errors"][1] is None
    assert decision["information_audit"]["analytic_frontier_count"] == 1
    assert decision["information_audit"]["analytically_dominated_count"] == 1
    assert decision["information_audit"]["candidate_response_accessed"] is False


def test_historical_workspace_cannot_be_reused(tmp_path):
    (tmp_path / "IDENTITY.json").write_text(json.dumps({"schema": "pcpi-p3m3"}))
    with pytest.raises(ValueError): _session(tmp_path)


def test_measured_oracle_is_called_only_after_durable_selection(monkeypatch, tmp_path):
    oracle = PoolOracle(np.array([[3.], [4.]]), np.array([3., 4.]))
    original = PoolOracle.acquire_indices
    def checked(self, indices):
        assert (tmp_path / "DECISION-001.json").is_file()
        assert indices.tolist() == [0]
        return original(self, indices)
    monkeypatch.setattr(PoolOracle, "acquire_indices", checked)
    session = _session(tmp_path)
    receipt = session.execute_measured_query(oracle, np.array([0]), np.array([[3.]]))
    assert receipt["candidate_id"] == 0 and receipt["response"] == 3.
    recovered = _session(tmp_path)
    assert len(recovered.receipts) == 1
    with pytest.raises(ValueError): recovered.plan(np.array([0]), np.array([[3.]]))


def test_oracle_mismatch_never_admits_response(tmp_path):
    session = _session(tmp_path)
    oracle = PoolOracle(np.array([[4.]]), np.array([4.]))
    with pytest.raises(ValueError, match="coordinates"):
        session.execute_measured_query(oracle, np.array([0]), np.array([[3.]]))
    assert not (tmp_path / "RECEIPT-001.json").exists()


def test_completed_policy_manifest_and_resume_without_rereveal(monkeypatch, tmp_path):
    session = _session(tmp_path)
    oracle = PoolOracle(np.array([[3.], [4.]]), np.array([3., 4.]))
    ids, actions = np.array([0, 1]), oracle.X_pool
    # The estimator is replaced only to exercise transaction control flow;
    # singleton tests above exercise the real estimator. Not efficacy evidence.
    from types import SimpleNamespace
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked",
        lambda components, *args, **kwargs: SimpleNamespace(ranking_certified=True,
            estimate=SimpleNamespace(scores=np.arange(components.locations.shape[1], dtype=float),
                                     error_bounds=np.zeros(components.locations.shape[1]))))
    manifest = session.run_measured_pool(oracle, ids, actions)
    assert manifest["completed_queries"] == 2 and not manifest["efficacy_demonstrated"]
    def forbidden(*args): raise AssertionError("completed run must never rereveal")
    monkeypatch.setattr(PoolOracle, "acquire_indices", forbidden)
    assert _session(tmp_path).run_measured_pool(oracle, ids, actions) == manifest
    with pytest.raises(ValueError):
        _session(tmp_path).run_measured_pool(oracle, ids[::-1], actions[::-1])
