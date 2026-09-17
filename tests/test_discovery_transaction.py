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


def test_uncertified_ranking_does_not_publish_selection(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked",
                        lambda *args, **kwargs: SimpleNamespace(ranking_certified=False))
    session = _session(tmp_path)
    with pytest.raises(RuntimeError, match="uncertified"):
        session.plan(np.array([10, 11]), np.array([[3.], [4.]]))
    assert not (tmp_path / "DECISION-001.json").exists()


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
