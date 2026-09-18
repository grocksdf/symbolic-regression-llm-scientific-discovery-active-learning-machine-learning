"""Mocked orchestration correctness only: no engine, provider or real data run."""
from types import SimpleNamespace
import numpy as np
import pytest

from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig
from hypothesis_mvp.discovery.proposal_runtime import ProviderSettings, ProviderRoute
from hypothesis_mvp.discovery.system_ablation import run_exploration_ablations, audit_usage
from hypothesis_mvp.data.roles import SelectionData, RoleDataset, DataRole


CONTEXT = {"task_name": "fixture_law", "task_description": "fixture description",
           "feature_names": ["fixture_feature"], "feature_units": ["unit"],
           "target_name": "fixture_target", "target_unit": "unit"}


def _candidate_audit(score=1., exhausted=False):
    from hypothesis_mvp.discovery.scientific_runtime import ScientificDiscoveryRuntime
    controller = ScientificDiscoveryRuntime.__new__(ScientificDiscoveryRuntime)
    candidate = SimpleNamespace(dag=SimpleNamespace(expression="x0**2"))
    rejections = []
    def build(expression, *args, **kwargs):
        if expression == "invalid":
            rejections.append({"reason": "fixture_invalid_structure"})
            return None
        return candidate
    controller.evaluation = SimpleNamespace(
        budget=SimpleNamespace(exhausted=exhausted), rejections=rejections, build_state=build,
        policy=SimpleNamespace(accept_transition=lambda *args: (True, {"pass": True}),
                               score=lambda *args: score))
    batch = SimpleNamespace(candidates=[SimpleNamespace(candidate_id=str(i), equation=e)
                                       for i, e in enumerate(["x0**2", "invalid"])])
    winner, exploratory, audit = controller._evaluate_batch(batch, object(), (), "balanced", 1)
    return winner, exploratory, audit


def test_mixed_candidate_audit_publishes_complete_strict_json(tmp_path, monkeypatch):
    import json
    winner, exploratory, audit = _candidate_audit()
    assert winner is exploratory[0]
    assert audit[0]["score"] == 1.
    assert audit[1]["score"] is None and not audit[1]["validated"]
    assert audit[1]["score_status"] == "invalid_candidate_no_score"
    assert audit[1]["evaluation_rejections"][0]["reason"] == "fixture_invalid_structure"
    selection, config = _inputs()
    class Agent:
        def __init__(self, config, provider): self.config, self.provider = config, provider
        def run(self, **kwargs):
            result = _result(self.config, self.provider is not None)
            result.discovery.report["llm_rounds"] = [{"candidate_audit": audit}] if self.provider else []
            return result
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.DiscoveryAgent", Agent)
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.run_bounded", _inline)
    run_exploration_ablations(tmp_path, selection, dataset="opaque", config=config,
        provider_settings=ProviderSettings(routes=(ProviderRoute("https://fixture.invalid", "model", "key"),)),
        single_engine="polynomial_lasso", compute_ceiling=100, provider_attempt_ceiling=3,
        source_identity="correctness-fixture", scientific_context=CONTEXT)
    for variant in ("full", "single_engine"):
        row = json.loads((tmp_path / variant / "RESULT.json").read_text(encoding="utf-8"))
        assert row["hypothesis_provenance"]["llm_candidate_lifecycle"][0]["candidate_audit"] == audit
        json.dumps(row, allow_nan=False)
        assert not (tmp_path / variant / "RESULT.json.staging").exists()


@pytest.mark.parametrize("score", [float("inf"), float("-inf"), float("nan")])
def test_validated_nonfinite_score_cannot_be_reported_as_missing(score):
    with pytest.raises(ValueError, match="nonfinite selection score"):
        _candidate_audit(score)


def test_budget_denied_candidates_each_have_explicit_missing_score():
    winner, exploratory, audit = _candidate_audit(exhausted=True)
    assert winner is None and exploratory == [] and len(audit) == 2
    assert all(row["score"] is None and row["score_status"] == "not_evaluated_budget_exhausted"
               for row in audit)


def test_invalid_publication_leaves_neither_result_nor_staging(tmp_path):
    from hypothesis_mvp.pcpi.discovery_transaction import _publish
    path = tmp_path / "RESULT.json"
    with pytest.raises(ValueError):
        _publish(path, {"best_val_nmse": float("inf")})
    assert not path.exists() and not path.with_name("RESULT.json.staging").exists()


def _inputs():
    selection = SelectionData(
        RoleDataset(DataRole.DEVELOPMENT, np.array([[0.], [1.]]), np.array([0., 1.])),
        RoleDataset(DataRole.VALIDATION, np.array([[2.], [3.]]), np.array([2., 3.])), None, ())
    config = DiscoveryAgentConfig(cycles=1, engine_workers=1, engine_retries=0,
        engine_budget=4, engine_repeats=2, discovery_budget=10)
    return selection, config


def _result(config, provider=True):
    return SimpleNamespace(cycles=[SimpleNamespace(
        engine_report={"run_records": [{"status": "succeeded"}] *
            (len(config.engines) * config.engine_repeats), "failures": []},
        candidate_evaluations=2, provider_attempts=(1 if provider else 0), provider_calls=(1 if provider else 0))],
        discovery=SimpleNamespace(report={"best_val_nmse": 1.,
            "final_topk": [{"expression": "x0", "source": "fixture",
                            "origin": "deterministic", "lineage_id": ""}],
            "llm_call_count": (1 if provider else 0), "llm_attempt_count": (1 if provider else 0), "llm_error_count": 0}))


def _inline(function, args=(), kwargs=None, **limits):
    # Test injection exercises orchestration without engines/providers.
    # Actual spawn/deadline enforcement has separate process tests.
    return function(*args, **(kwargs or {})), {"fixture_only": True}


def test_complete_bank_preserves_engine_and_post_refit_rejections(tmp_path, monkeypatch):
    from hypothesis_mvp.discovery.system_ablation import _run_variant
    selection, config = _inputs()
    result = _result(config)
    bank = [
        {"expression": "x0", "source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"expression": "x0**2", "source": "engine:mcts", "origin": "deterministic"},
        {"expression": "tanh(0.01*x0)", "source": "llm_proposal", "origin": "llm"},
    ]
    result.discovery.report["evaluated_hypothesis_bank"] = bank
    result.cycles[0].engine_report["all_results"] = [{"engine": "mcts", "expression": "x0**2"}]
    result.discovery.report["rejected_candidates"] = [{"reason": "evaluation_budget_exhausted"}]
    class Agent:
        def __init__(self, *args): pass
        def run(self, **kwargs): return result
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.DiscoveryAgent", Agent)
    summary = _run_variant(config, object(), selection, tmp_path, 100, 3, CONTEXT)
    assert {row["source"] for row in summary["candidates"]} == {
        "engine:polynomial_lasso", "engine:mcts"}
    audit = summary["hypothesis_provenance"]
    assert audit["pcpi_adapter_rejections"][0]["expression"] == "tanh(0.01*x0)"
    assert audit["llm_retained_candidate_count"] == 0
    assert audit["raw_engine_candidates"][0]["engine"] == "mcts"
    assert audit["evaluation_rejections"][0]["reason"] == "evaluation_budget_exhausted"


def test_equal_jobs_provider_free_and_completed_recovery(tmp_path, monkeypatch):
    selection, config = _inputs()
    seen = []
    class Agent:
        def __init__(self, config, provider):
            self.config = config; self.provider = provider; seen.append((config, provider))
        def run(self, **kwargs): return _result(self.config, self.provider is not None)
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.DiscoveryAgent", Agent)
    monkeypatch.setattr("hypothesis_mvp.discovery.system_ablation.run_bounded", _inline)
    kwargs = dict(dataset="opaque", config=config, provider_settings=ProviderSettings(
        routes=(ProviderRoute("https://fixture.invalid", "fixture-model", "fixture-key"),)),
        single_engine="polynomial_lasso", compute_ceiling=100,
        provider_attempt_ceiling=3, source_identity="correctness-fixture",
        scientific_context=CONTEXT)
    result = run_exploration_ablations(tmp_path, selection, **kwargs)
    assert result["pair_gate"]["passed"] and not result["superiority_demonstrated"]
    assert seen[1][1] is None
    assert seen[2][0].engine_repeats == 4
    assert all(row["hypothesis_provenance"]["all_candidates_source_bound"]
               for row in result["rows"])
    assert all("llm_retained_candidate_count" in row["hypothesis_provenance"]
               for row in result["rows"])
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
        provider_attempt_ceiling=3, source_identity="correctness-fixture",
        scientific_context=CONTEXT)
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
    with pytest.raises(ValueError, match="provider-infrastructure"):
        _run_variant(config, None, selection, tmp_path, 100, 3, CONTEXT)
