"""Algebraic/control-flow fixtures only; not experimental efficacy."""
import numpy as np
import pytest

from tests.test_discovery_transaction import _session
from hypothesis_mvp.pcpi.discovery_transaction import DiscoveryTransaction
from hypothesis_mvp.data.oracle import PoolOracle
from hypothesis_mvp.discovery.system_run import analyze_system_contract
from hypothesis_mvp.discovery.system_run import run_frozen_system_comparison
from hypothesis_mvp.discovery.system_run import audit_frozen_hypothesis_bank
from hypothesis_mvp.discovery.system_run import audit_frozen_decision_risk_utility


def _random(root, base, seed=17):
    return DiscoveryTransaction(root, base.model, base.target, base.domain,
        base.controls, source_identity="correctness-fixture",
        query_policy="random", random_seed=seed)


def test_random_matched_budget_resume_and_isolation(tmp_path, monkeypatch):
    base = _session(tmp_path / "base")
    oracle = PoolOracle(np.array([[3.], [4.]]), np.array([3., 4.]))
    def forbidden(*args, **kwargs):
        raise AssertionError("random comparator must not invoke EIG")
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked", forbidden)
    random = _random(tmp_path / "random", base)
    decision = random.plan(np.array([0, 1]), oracle.X_pool)
    assert decision["certified"] is False and decision["score"] is None
    assert _random(tmp_path / "random", base).plan(np.array([0, 1]), oracle.X_pool) == decision
    manifest = random.run_measured_pool(oracle, np.array([0, 1]), oracle.X_pool)
    assert manifest["completed_queries"] == base.target.measurement_budget
    assert len({r["candidate_id"] for r in random.receipts}) == 2
    monkeypatch.setattr(PoolOracle, "acquire_indices", forbidden)
    assert _random(tmp_path / "random", base).run_measured_pool(oracle, np.array([0, 1]), oracle.X_pool) == manifest
    with pytest.raises(ValueError): _random(tmp_path / "random", base, seed=18)
    with pytest.raises(ValueError):
        DiscoveryTransaction(tmp_path / "random", base.model, base.target,
            base.domain, base.controls, source_identity="correctness-fixture")


def test_analysis_blocks_unpaired_or_failed_evidence():
    with pytest.raises(ValueError): analyze_system_contract([])
    with pytest.raises(ValueError):
        analyze_system_contract([{"status": "failed", "heldout_opened": False,
                                  "selection_used_heldout": False}])


def test_h0_posterior_contents_are_identity_bound(tmp_path):
    from dataclasses import replace
    base = _session(tmp_path / "base")
    changed = replace(base.target, initial_posterior=replace(
        base.target.initial_posterior, log_evidence=base.posterior.log_evidence + 1))
    assert changed.stable_hash != base.target.stable_hash
    with pytest.raises(ValueError):
        DiscoveryTransaction(tmp_path / "base", base.model, changed, base.domain,
            base.controls, source_identity="source-frozen-correctness-fixture")


def test_coordinator_freezes_shared_target_and_exports(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hypothesis_mvp.data.roles import DataRole, RoleDataset
    from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
    monkeypatch.setattr("hypothesis_mvp.pcpi.discovery_transaction.estimate_class_eig_until_ranked",
        lambda components, *args, **kwargs: SimpleNamespace(ranking_certified=True,
            estimate=SimpleNamespace(scores=np.arange(components.locations.shape[1], dtype=float),
                                     error_bounds=np.zeros(components.locations.shape[1]))))
    oracle = PoolOracle(np.array([[3.], [4.]]), np.array([3., 4.]))
    kwargs = dict(n_features=1, prior=NormalInverseGammaPrior(), exploration_identity="a"*64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        measurement_budget=2, controls=_session(tmp_path / "fixture").controls,
        source_identity="correctness-fixture", random_seed=17,
        source_prior_weights={"core": 1.0})
    args = (tmp_path / "comparison", [{"expression": "x0", "source": "a"},
        {"expression": "x0**2", "source": "b"}],
        RoleDataset(DataRole.DEVELOPMENT, np.array([[0.], [1.], [2.]]), np.array([.1, 1., 2.])),
        oracle, np.array([0, 1]))
    result = run_frozen_system_comparison(*args, **kwargs)
    assert result["protocol_complete"] and not result["formal_experiment_authorized"]
    assert result["source_prior_weights"] == {"core": 1.0}
    assert {m["target"] for m in result["manifests"].values()} == {result["target"]}
    assert all(m["completed_queries"] == 2 for m in result["manifests"].values())
    def forbidden(*args): raise AssertionError("completed comparison must not reveal")
    monkeypatch.setattr(PoolOracle, "acquire_indices", forbidden)
    assert run_frozen_system_comparison(*args, **kwargs) == result


@pytest.mark.parametrize("probabilities,passed", [((.5, .5), True),
                                                    ((1.0 - 1e-12, 1e-12), False)])
def test_response_free_hypothesis_bank_gate_is_resolution_derived(
        monkeypatch, probabilities, passed):
    from types import SimpleNamespace
    from hypothesis_mvp.pcpi.acquisition import ClassPartition
    partition = ClassPartition(("a", "b"), ((0,), (1,)), probabilities, (0, 1))
    model = SimpleNamespace(stable_hash="model", candidate_bindings=(("a",), ("b",)),
                            bank=SimpleNamespace(structures=(1, 2)))
    target = SimpleNamespace(stable_hash="target", partition=partition)
    monkeypatch.setattr("hypothesis_mvp.discovery.system_run._freeze_comparison",
                        lambda *args, **kwargs: (model, target))
    audit = audit_frozen_hypothesis_bank(
        [{}, {}], object(), np.zeros((4, 1)), n_features=1, prior=object(),
        exploration_identity="fixture", coefficient_policy="fixture",
        measurement_budget=2, exact_eig_epsabs=1e-10)
    assert audit["passed"] is passed
    assert audit["candidate_response_accessed"] is False
    assert audit["familywise_utility_resolution_nats"] == 4e-10


def test_decision_risk_utility_gate_is_response_free_and_resolution_derived(monkeypatch):
    from types import SimpleNamespace
    from hypothesis_mvp.pcpi.acquisition import ClassPartition
    partition = ClassPartition(("a", "b"), ((0,), (1,)), (.5, .5), (0, 1))
    model = SimpleNamespace(stable_hash="model", engine=lambda identity: object())
    target = SimpleNamespace(stable_hash="target", partition=partition,
                             initial_posterior=object(), model_identity="model")
    monkeypatch.setattr("hypothesis_mvp.discovery.system_run._freeze_comparison",
                        lambda *args, **kwargs: (model, target))
    monkeypatch.setattr(
        "hypothesis_mvp.discovery.system_run.predictive_components_for_partition",
        lambda *args: object())
    monkeypatch.setattr(
        "hypothesis_mvp.discovery.system_run."
        "exact_class_decision_risk_reduction_shared_actions",
        lambda *args, **kwargs: SimpleNamespace(
            scores=np.array([.1, .3]), quadrature_errors=np.array([1e-12, 1e-12])))
    audit = audit_frozen_decision_risk_utility(
        [{}, {}], object(), np.zeros((2, 1)), n_features=1, prior=object(),
        exploration_identity="fixture", coefficient_policy="fixture",
        measurement_budget=2, exact_epsabs=1e-10)
    assert audit["passed"] is True and audit["selected_action_index"] == 1
    assert audit["candidate_response_accessed"] is False
    assert audit["familywise_resolution"] == 2e-10
