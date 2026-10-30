"""Response-free tests for the production shared-E0 experiment adapter."""
from copy import deepcopy

import numpy as np
import pytest
import yaml

from scripts.run_gap_candidate_marginal_value import (
    EXPANDED_REFIT_POLICY, _condition_context, _neutral_brief,
    _admission_function, _filter_engine_result_all_domains,
    _placebo_executable_brief,
    _rotate_scalar_leaves, _runtime_config, _score,
)
from hypothesis_mvp.discovery.localized_gap_refinement import (
    compile_executable_gap_brief,
)
from hypothesis_mvp.data.roles import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from hypothesis_mvp.discovery.factory import (
    build_scientific_discovery_runtime,
)
from hypothesis_mvp.discovery.agent import (
    DiscoveryAgent, DiscoveryAgentConfig,
)
from hypothesis_mvp.discovery.proposal_runtime import (
    ProviderRoute, ProviderSettings,
)


def _base():
    return {
        "current_expression": "x0",
        "new_skeleton_quota": 0,
        "allowed_edits": ["ADD", "DELETE", "REPLACE"],
    }


def _gap():
    return {
        "schema": "fixture-gap",
        "propose_allowed": False,
        "regions": [{
            "region": 0, "weight": .25, "risk": 2.,
            "direction": "underprediction"}],
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }


def test_placebo_preserves_schema_and_scalar_multiset_but_changes_binding():
    gap = _gap()
    placebo = _rotate_scalar_leaves(gap, "fixture")
    assert placebo != gap
    assert set(placebo) == set(gap)
    assert set(placebo["regions"][0]) == set(gap["regions"][0])
    before = sorted(map(str, (
        gap["regions"][0]["region"], gap["regions"][0]["weight"],
        gap["regions"][0]["risk"], gap["regions"][0]["direction"])))
    after = sorted(map(str, (
        placebo["regions"][0]["region"],
        placebo["regions"][0]["weight"],
        placebo["regions"][0]["risk"],
        placebo["regions"][0]["direction"])))
    assert before == after


def test_neutral_brief_preserves_shape_without_real_gap_values():
    gap = _gap()
    neutral = _neutral_brief(gap)
    assert set(neutral) == set(gap)
    assert set(neutral["regions"][0]) == set(gap["regions"][0])
    assert neutral["schema"] == gap["schema"]
    assert neutral["regions"][0]["risk"] == 0
    assert neutral["regions"][0]["direction"] == "neutral"


def test_conditions_change_context_only_not_registered_support():
    gap = _gap()
    neutral = _neutral_brief(gap)
    placebo = _rotate_scalar_leaves(gap, "fixture")
    contexts = {
        condition: _condition_context(
            deepcopy(_base()), condition, gap, neutral, placebo)
        for condition in ("B0", "G_hard0", "G_soft0", "G_placebo0")
    }
    assert all(row["new_skeleton_quota"] == 0
               for row in contexts.values())
    assert len({tuple(row["allowed_edits"])
                for row in contexts.values()}) == 1
    assert "posterior_gap_brief" not in contexts["B0"]
    assert contexts["G_hard0"]["diagnostic_instruction_mode"] == (
        "historical-hard")
    assert contexts["G_soft0"]["diagnostic_instruction_mode"] == (
        contexts["G_placebo0"]["diagnostic_instruction_mode"]
    ) == "non-mandatory-evidence"
    assert contexts["G_soft0"]["posterior_gap_brief"] != (
        contexts["G_placebo0"]["posterior_gap_brief"])


def test_placebo_can_use_same_hard_instruction_as_real_gap():
    gap = _gap()
    context = _condition_context(
        deepcopy(_base()), "G_placebo0", gap,
        _neutral_brief(gap), _rotate_scalar_leaves(gap, "fixture"),
        {"G_placebo0": "historical-hard"})
    assert context["diagnostic_instruction_mode"] == "historical-hard"


def test_localized_real_and_placebo_share_language_but_not_direction():
    gap = _gap()
    gap["schema"] = "pcpi-frozen-bank-exploration-brief-v1"
    gap["propose_allowed"] = True
    gap["regions"] = [{
        "region": 0, "lower_bound": None, "upper_bound": 0.}]
    gap["independent_adequacy"] = [{
        "region": 0, "count": 10, "pit_above_median": 8,
        "directional_pit_imbalance": True,
        "directional_pit_lower_bound": .6,
        "score_degradation": False, "undercoverage": False}]
    gap["region_axis"] = 0
    executable = compile_executable_gap_brief(gap)
    placebo = _placebo_executable_brief(executable)
    assert executable["schema"] == placebo["schema"]
    assert executable["target_regions"][0]["region"] == (
        placebo["target_regions"][0]["region"])
    assert executable["target_regions"][0]["direction"] != (
        placebo["target_regions"][0]["direction"])
    context = _condition_context(
        deepcopy(_base()), "G_placebo0", gap,
        _neutral_brief(gap), _rotate_scalar_leaves(gap, "fixture"),
        {"G_placebo0": "historical-hard"}, placebo)
    assert context["localized_refinement_contract"] is True


def test_executable_gap_without_targets_is_an_explicit_no_call_abstention():
    gap = {
        "schema": "pcpi-frozen-bank-exploration-brief-v1",
        "region_axis": 0, "propose_allowed": False,
        "regions": [{
            "region": 0, "lower_bound": None, "upper_bound": 0.}],
        "independent_adequacy": [{
            "region": 0, "count": 10, "pit_above_median": 5,
            "directional_pit_imbalance": False,
            "score_degradation": False, "undercoverage": False}],
    }
    executable = compile_executable_gap_brief(gap)
    assert executable["target_regions"] == []
    assert executable["propose_allowed"] is False


def test_typed_engine_agent_requires_registered_provider_but_does_not_call_it(
        monkeypatch):
    config = DiscoveryAgentConfig(
        engines=("polynomial_lasso",),
        engine_budget=1,
        discovery_budget=20,
        synthesis_evaluation_reserve=1,
        llm_evaluation_reserve=1,
        scientist_orchestration=True,
        typed_evidence_synthesis=True,
        typed_inner_augmentation=True,
        expanded_formula_synthesis=True,
        refit_policy="pcpi-expanded-fixed-inner-v1",
        posterior_gap_directed=True,
        iterative_posterior_refinement=True,
        discovery_islands=("nmse",),
        discovery_rounds=1,
        candidates_per_island=2,
        new_skeleton_quota=1,
    )
    calls = []
    monkeypatch.setattr(
        "hypothesis_mvp.discovery.proposal_runtime.requests.post",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    agent = DiscoveryAgent(
        config, provider_settings=ProviderSettings(routes=(ProviderRoute(
            "https://fixture.invalid", "fixture", "fixture"),)))
    assert agent.provider_settings is not None
    assert calls == []


def test_runtime_config_uses_refit_policy_not_coefficient_policy(tmp_path):
    config = _runtime_config(seed=7, protected_count=3)
    assert config.refit_policy == EXPANDED_REFIT_POLICY
    runtime = build_scientific_discovery_runtime(
        n_features=2, config=config,
        library_path=tmp_path / "knowledge.jsonl",
        ledger_path=tmp_path / "ledger.jsonl")
    assert runtime.equation.refit_policy == EXPANDED_REFIT_POLICY


def test_reporting_score_uses_explicit_expanded_formula_model():
    X = np.linspace(-1., 1., 40)[:, None]
    y = 1. + 2. * X[:, 0]
    actions = X[30:]
    selection = SelectionData(
        RoleDataset(DataRole.DEVELOPMENT, X[:20], y[:20]),
        RoleDataset(DataRole.VALIDATION, X[20:30], y[20:30]),
        AcquisitionCovariates(
            DataRole.ACQUISITION_POOL, actions,
            covariate_fingerprint(actions)), ())
    score, audit = _score(
        [{"expression": "1", "source": "engine:constant"},
         {"expression": "x0", "source": "engine:linear"}],
        selection, X[30:], y[30:], "a" * 64)
    assert np.isfinite(score)
    assert audit["posterior_member_count"] == 2
    update = RoleDataset(
        DataRole.VALIDATION, X[20:30], y[20:30])
    updated_score, updated_audit = _score(
        [{"expression": "1", "source": "engine:constant"},
         {"expression": "x0", "source": "engine:linear"}],
        selection, X[30:], y[30:], "b" * 64,
        posterior_update=update)
    assert np.isfinite(updated_score)
    assert updated_audit["target_identity"] != audit["target_identity"]


def test_candidate_budget_is_separate_from_shared_parent_construction(
        tmp_path):
    config = _runtime_config(seed=9, protected_count=2)
    runtime = build_scientific_discovery_runtime(
        n_features=1, config=config,
        library_path=tmp_path / "knowledge.jsonl",
        ledger_path=tmp_path / "ledger.jsonl")
    X = np.linspace(-1., 1., 32)[:, None]
    y = 1. + X[:, 0]
    parent, _ = runtime.evaluation.seed_survivor(
        [{"expression": "1", "source": "engine:constant"},
         {"expression": "x0", "source": "engine:linear"}],
        X[:20], y[:20], X[20:], y[20:])
    assert runtime.evaluation.budget.used > 0
    runtime.evaluation.reset()
    assert runtime.evaluation.budget.used == 0
    candidate = runtime.evaluation.build_state(
        "x0 + sin(x0)", X[:20], y[:20], X[20:], y[20:],
        source="fixture", origin="deterministic",
        island="nmse", round_id=1, parent=parent)
    assert candidate is not None
    assert runtime.evaluation.validity_attempts[-1]["evaluated"] is True


def test_admission_method_is_explicit_and_historical_default_is_stable():
    assert _admission_function({}).__name__ == (
        "admit_regional_candidates")
    assert _admission_function({
        "admission_method": "global-protected-fixed-expert-v1"
    }).__name__ == "admit_global_protected_candidates"
    with pytest.raises(ValueError, match="unregistered"):
        _admission_function({"admission_method": "result-tuned"})


def test_admission_response_opens_only_after_all_arm_generation_is_frozen():
    import inspect
    from scripts.run_gap_candidate_marginal_value import (
        _generation_roles, _run_block,
    )
    generation_source = inspect.getsource(_generation_roles)
    block_source = inspect.getsource(_run_block)
    assert 'indices["gap_admission"], response=False' in generation_source
    assert "ARM_GENERATION_FROZEN.json" in block_source
    assert block_source.index("ARM_GENERATION_FROZEN.json") < (
        block_source.index("gap_admission = _gap_admission_role"))


def test_all_domain_definedness_filters_before_gap_or_provider():
    from types import SimpleNamespace
    valid = SimpleNamespace(
        engine="fixture", expression="x0 + 1", score=1.,
        complexity=2.)
    invalid = SimpleNamespace(
        engine="fixture", expression="1/x0", score=.5,
        complexity=2.)
    result = SimpleNamespace(
        all_results=(invalid, valid), best=invalid)
    # dataclasses.replace requires the production dataclass; use one real
    # scheduler result-shaped dataclass container.
    from hypothesis_mvp.symbolic.scheduler import MultiEngineResult
    result = MultiEngineResult(
        invalid, (invalid, valid), (), (), 2, 2, ())
    filtered, audit = _filter_engine_result_all_domains(
        result, 1, (("safe", np.array([[1.], [2.]])),
                    ("zero", np.array([[0.], [1.]]))))
    assert filtered.all_results == (valid,)
    assert filtered.best is valid
    assert audit["rejected_candidate_count"] == 1
    assert audit["response_accessed"] is False


def test_localized_global_protected_config_is_constructible():
    from pathlib import Path
    from hypothesis_mvp.discovery.feedback_generation import validate_policy
    raw = yaml.safe_load((Path(__file__).parents[1] / "configs" /
        "aistats_gap_localized_global_protected.yaml").read_text(
            encoding="utf-8"))
    values = dict(raw["agent_config"])
    values["engines"] = tuple(values["engines"])
    values["discovery_islands"] = tuple(values["discovery_islands"])
    config = DiscoveryAgentConfig(**values)
    validate_policy(config)
    assert config.localized_gap_refinement

