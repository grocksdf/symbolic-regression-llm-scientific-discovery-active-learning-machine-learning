"""Algebraic/control-flow fixtures only; not experimental efficacy."""
import numpy as np
import pytest

from tests.test_discovery_transaction import _session
from hypothesis_mvp.pcpi.discovery_transaction import DiscoveryTransaction
from hypothesis_mvp.data.oracle import PoolOracle
from hypothesis_mvp.discovery.system_run import analyze_system_contract
from hypothesis_mvp.discovery.system_run import run_frozen_system_comparison


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
        source_identity="correctness-fixture", random_seed=17)
    args = (tmp_path / "comparison", [{"expression": "x0", "source": "a"},
        {"expression": "x0**2", "source": "b"}],
        RoleDataset(DataRole.DEVELOPMENT, np.array([[0.], [1.], [2.]]), np.array([.1, 1., 2.])),
        oracle, np.array([0, 1]))
    result = run_frozen_system_comparison(*args, **kwargs)
    assert result["protocol_complete"] and not result["formal_experiment_authorized"]
    assert {m["target"] for m in result["manifests"].values()} == {result["target"]}
    assert all(m["completed_queries"] == 2 for m in result["manifests"].values())
    def forbidden(*args): raise AssertionError("completed comparison must not reveal")
    monkeypatch.setattr(PoolOracle, "acquire_indices", forbidden)
    assert run_frozen_system_comparison(*args, **kwargs) == result
