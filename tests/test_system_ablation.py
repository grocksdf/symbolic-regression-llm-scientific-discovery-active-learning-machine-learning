"""Mocked orchestration correctness only: no engine, provider or real data run."""
from types import SimpleNamespace
import numpy as np
import pytest

from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig
from hypothesis_mvp.discovery.proposal_runtime import ProviderSettings, ProviderRoute
from hypothesis_mvp.discovery.system_ablation import run_exploration_ablations, audit_usage
from hypothesis_mvp.data.roles import SelectionData, RoleDataset, DataRole


def _inputs():
    selection = SelectionData(
        RoleDataset(DataRole.DEVELOPMENT, np.array([[0.], [1.]]), np.array([0., 1.])),
        RoleDataset(DataRole.VALIDATION, np.array([[2.], [3.]]), np.array([2., 3.])), None, ())
    config = DiscoveryAgentConfig(cycles=1, engine_workers=1, engine_retries=0,
        engine_budget=4, engine_repeats=2, discovery_budget=10)
    return selection, config


def _result(config):
    return SimpleNamespace(cycles=[SimpleNamespace(
        engine_report={"run_records": [{"status": "succeeded"}] *
            (len(config.engines) * config.engine_repeats), "failures": []},
        candidate_evaluations=2, provider_attempts=0, provider_calls=0)],
        discovery=SimpleNamespace(report={"best_val_nmse": 1.,
            "final_topk": [{"expression": "x0", "source": "fixture"}]}))


def _inline(function, args=(), kwargs=None, **limits):
    # Test injection exercises orchestration without engines/providers.
    # Actual spawn/deadline enforcement has separate process tests.
    return function(*args, **(kwargs or {})), {"fixture_only": True}


def test_equal_jobs_provider_free_and_completed_recovery(tmp_path, monkeypatch):
    selection, config = _inputs()
    seen = []
    class Agent:
        def __init__(self, config, provider):
            self.config = config; seen.append((config, provider))
        def run(self, **kwargs): return _result(self.config)
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.DiscoveryAgent", Agent)
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.run_bounded", _inline)
    kwargs = dict(dataset="opaque", config=config, provider_settings=ProviderSettings(
        routes=(ProviderRoute("https://fixture.invalid", "fixture-model", "fixture-key"),)),
        single_engine="polynomial_lasso", compute_ceiling=100,
        provider_attempt_ceiling=3, source_identity="correctness-fixture")
    result = run_exploration_ablations(tmp_path, selection, **kwargs)
    assert result["pair_gate"]["passed"] and not result["superiority_demonstrated"]
    assert seen[1][1] is None
    assert seen[2][0].engine_repeats == 4
    assert run_exploration_ablations(tmp_path, selection, **kwargs) == result
    assert len(seen) == 3
    with pytest.raises(ValueError):
        run_exploration_ablations(tmp_path, selection, **{**kwargs, "compute_ceiling": 101})


@pytest.mark.parametrize("field,value", [("candidate_evaluations", 11), ("provider_attempts", 4)])
def test_budget_overrun_blocks_analysis(field, value):
    _, config = _inputs(); result = _result(config)
    setattr(result.cycles[0], field, value)
    with pytest.raises(ValueError): audit_usage(config, result, 1., 100., 3)


def test_compute_and_engine_failures_block_analysis():
    _, config = _inputs(); result = _result(config)
    with pytest.raises(ValueError): audit_usage(config, result, 101., 100., 3)
    result.cycles[0].engine_report["failures"] = ["failure"]
    with pytest.raises(ValueError): audit_usage(config, result, 1., 100., 3)


def test_freeze_is_fail_closed_before_runtime(monkeypatch):
    from hypothesis_mvp.discovery import system_freeze
    def dirty(*args): raise RuntimeError("dirty source")
    def forbidden(): raise AssertionError("runtime should not be reached")
    monkeypatch.setattr(system_freeze, "verify_clean_git_source", dirty)
    monkeypatch.setattr(system_freeze, "runtime_dependency_snapshot", forbidden)
    with pytest.raises(RuntimeError, match="dirty"):
        system_freeze.capture_system_freeze("unused", {})


def test_failed_exploration_cannot_repeat_calls(tmp_path, monkeypatch):
    selection, config = _inputs()
    calls = []
    class Agent:
        def __init__(self, *args): pass
        def run(self, **kwargs):
            calls.append(1)
            raise RuntimeError("fixture infrastructure failure")
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.DiscoveryAgent", Agent)
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.run_bounded", _inline)
    kwargs = dict(dataset="opaque", config=config, provider_settings=ProviderSettings(
        routes=(ProviderRoute("https://fixture.invalid", "fixture-model", "fixture-key"),)),
        single_engine="polynomial_lasso", compute_ceiling=100,
        provider_attempt_ceiling=3, source_identity="correctness-fixture")
    with pytest.raises(RuntimeError): run_exploration_ablations(tmp_path, selection, **kwargs)
    assert (tmp_path / "full" / "FAILURE.json").is_file()
    with pytest.raises(ValueError): run_exploration_ablations(tmp_path, selection, **kwargs)
    assert len(calls) == 1 and not (tmp_path / "ANALYSIS.json").exists()


def test_runtime_config_freeze_mismatch_blocks(monkeypatch):
    from hypothesis_mvp.discovery import system_freeze
    monkeypatch.setattr(system_freeze, "capture_system_freeze", lambda *args: {"config": "actual"})
    with pytest.raises(ValueError): system_freeze.verify_system_freeze("unused", {}, {"config": "different"})


def test_earlier_cycle_provider_failure_blocks_even_if_final_report_clean(tmp_path, monkeypatch):
    from hypothesis_mvp.discovery.system_ablation import _run_variant
    selection, config = _inputs()
    result = _result(config)
    result.cycles[0].provider_errors = 1
    class Agent:
        def __init__(self, *args): pass
        def run(self, **kwargs): return result
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.DiscoveryAgent", Agent)
    with pytest.raises(ValueError, match="provider infrastructure"):
        _run_variant(config, None, selection, tmp_path, 100, 3)
