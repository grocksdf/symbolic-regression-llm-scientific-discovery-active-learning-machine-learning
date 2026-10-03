"""Response-free source-composition check; no efficacy or benchmark access."""

from types import SimpleNamespace
from hashlib import sha256
import json

import numpy as np
import pytest

from hypothesis_mvp.data.roles import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig
from hypothesis_mvp.discovery.proposal_runtime import ProviderRoute, ProviderSettings
from hypothesis_mvp.discovery.scientist_policy import (
    ScientistReview, deterministic_plan, ScientistState,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.discovery.scientific_runtime import ScientificDiscoveryRuntime
from hypothesis_mvp.discovery.candidate_region_expansion import FrozenAxisRegions
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY
from hypothesis_mvp.discovery.regional_candidate_admission import (
    admit_regional_candidates,
)
from hypothesis_mvp.discovery.system_ablation import run_exploration_ablations


def _selection_and_roles():
    fit_x = np.linspace(-2., 2., 18)[:, None]
    fit = RoleDataset(DataRole.DEVELOPMENT, fit_x, fit_x[:, 0] ** 2)
    validation = RoleDataset(DataRole.VALIDATION,
        np.array([[-2.5], [2.5]]), np.array([6.25, 6.25]))
    domain = np.linspace(-4., 7., 56)[:, None]
    selection = SelectionData(fit, validation,
        AcquisitionCovariates(DataRole.ACQUISITION_POOL, domain,
                              covariate_fingerprint(domain)), ())
    def role(left, right):
        x = np.linspace(left, right, 24)[:, None]
        return RoleDataset(DataRole.VALIDATION, x, x[:, 0] ** 2)
    return selection, ((role(5., 7.), role(3., 4.)),
                       (role(8., 9.), role(10., 11.)))


def test_admission_changes_actual_frozen_posterior_used_by_second_gap(
        tmp_path, monkeypatch):
    selection, roles = _selection_and_roles()
    config = DiscoveryAgentConfig(
        engines=("polynomial_lasso", "mcts"), engine_budget=2,
        engine_repeats=1, engine_workers=1, cycles=2, discovery_budget=55,
        scientist_orchestration=True, typed_evidence_synthesis=True,
        typed_inner_augmentation=True, posterior_gap_directed=True,
        iterative_posterior_refinement=True,
        expanded_formula_synthesis=True,
        refit_policy="pcpi-expanded-fixed-inner-v1",
        synthesis_evaluation_reserve=3, llm_evaluation_reserve=4)
    agent = DiscoveryAgent(config, ProviderSettings(routes=(ProviderRoute(
        "https://example.invalid", "fixture", "fixture"),)))
    engine_calls = []
    rows = tuple(SimpleNamespace(engine=e, expression=x, mse_val=1.,
        complexity=2, score=1., lineage_id=e, diagnostics={})
        for e, x in (("polynomial_lasso", "x0"), ("mcts", "x0**3")))
    monkeypatch.setattr(agent, "_run_engines", lambda *args:
        (engine_calls.append(1) or SimpleNamespace(all_results=rows)))
    monkeypatch.setattr(agent, "_resolve_plan", lambda *args:
        (deterministic_plan(config.engines, config.engine_budget), {}, None, 0))
    review = ScientistReview(("fixture",), (), (), ("fixture",), False, "fixture")
    monkeypatch.setattr(agent, "_resolve_review", lambda *args, **kwargs:
                        (review, {}, None, 0))
    monkeypatch.setattr(ScientistState, "advance", lambda self, **kwargs: self)
    monkeypatch.setattr(agent, "_compile_cycle_synthesis", lambda *args:
                        ([], {"fixture": True}))
    monkeypatch.setattr(agent, "_cycle_record", lambda *args:
                        SimpleNamespace(posterior_gap_brief={}))
    import hypothesis_mvp.discovery.agent as agent_module
    monkeypatch.setattr(agent_module, "system_evaluation", lambda *args: {})
    candidate = {"expression": "x0**2", "source": "llm_proposal_1_novelty",
                 "origin": "llm", "lineage_id": "fixture-L1"}
    def fake_discover(*args, **kwargs):
        brief = kwargs["orchestration_context"]["posterior_gap_brief"]
        digest = sha256(json.dumps(brief, sort_keys=True,
                                  allow_nan=False).encode()).hexdigest()
        return SimpleNamespace(expression="x0", report={
            "evaluated_hypothesis_bank": [candidate],
            "final_topk": [candidate],
            "llm_rounds": [{"islands": [{"memory": {
                "posterior_gap_identity": digest}}]}]
            if brief["propose_allowed"] else []})
    monkeypatch.setattr(agent, "_discover", fake_discover)
    result = agent.run(selection=selection, task_name="fixture",
        task_description="fixture", output_dir=tmp_path / "run",
        knowledge_dir=tmp_path / "knowledge", variable_metadata={},
        gap_prior=NormalInverseGammaPrior(), gap_measurement_budget=4,
        gap_cycle_roles=roles)
    trace = result.system_evaluation["iterative_feedback_trace"]
    assert len(engine_calls) == 1
    assert trace[0]["admitted_supports"]
    assert trace[0]["bank_after"] == trace[1]["gap_bank"]
    assert trace[0]["posterior_after"] == trace[1]["gap_posterior"]
    assert trace[0]["posterior_before"] != trace[1]["gap_posterior"]
    assert trace[1]["proposal_batch_count"] > 0
    assert result.system_evaluation["iterative_feedback_gate"]["passed"]
    assert result.system_evaluation["iterative_posterior_refinement_verified"]


def test_reused_cycle_response_role_fails_before_output(tmp_path):
    selection, roles = _selection_and_roles()
    agent = DiscoveryAgent(DiscoveryAgentConfig(
        cycles=2, posterior_gap_directed=True,
        iterative_posterior_refinement=True,
        scientist_orchestration=True, typed_evidence_synthesis=True,
        typed_inner_augmentation=True,
        synthesis_evaluation_reserve=3, llm_evaluation_reserve=4,
        discovery_budget=55), ProviderSettings(routes=(ProviderRoute(
            "https://example.invalid", "fixture", "fixture"),)))
    with pytest.raises(ValueError, match="row-disjoint"):
        agent.run(selection=selection, task_name="fixture",
            task_description="fixture", output_dir=tmp_path / "run",
            knowledge_dir=tmp_path / "knowledge", variable_metadata={},
            gap_prior=NormalInverseGammaPrior(), gap_measurement_budget=4,
            gap_cycle_roles=(roles[0], (roles[0][1], roles[1][1])))
    assert not (tmp_path / "run").exists()


def test_inner_proposal_context_binds_actual_gap_identity():
    runtime = object.__new__(ScientificDiscoveryRuntime)
    brief = {"target_identity": "posterior-1", "regions": [{"region": 0}],
             "propose_allowed": True}
    runtime.orchestration_context = {"posterior_gap_brief": brief}
    runtime.config = SimpleNamespace(new_skeleton_quota=2)
    runtime.evaluation = SimpleNamespace(
        failure_signature=lambda current, exploration: ())
    current = SimpleNamespace(dag=SimpleNamespace(expression="x0"),
                              compact=lambda: {"expression": "x0"})
    exploration = SimpleNamespace(as_prompt_dict=lambda: {})
    context = runtime._proposal_context(current, exploration, "novelty")
    assert context["posterior_gap_brief"] == brief
    assert context["posterior_gap_identity"] == sha256(json.dumps(
        brief, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert context["new_skeleton_quota"] == 2


def test_nonclosed_formula_uses_expanded_independent_admission():
    fit_x = np.linspace(-2., 2., 18)[:, None]
    fit = RoleDataset(DataRole.DEVELOPMENT, fit_x, np.exp(fit_x[:, 0]))
    gap_x = np.linspace(4., 5., 24)[:, None]
    admission_x = np.linspace(2.5, 3.5, 24)[:, None]
    gap = RoleDataset(DataRole.VALIDATION, gap_x, np.exp(gap_x[:, 0]))
    admission = RoleDataset(DataRole.VALIDATION, admission_x,
                            np.exp(admission_x[:, 0]))
    core = [{"expression": "x0", "source": "engine:a"},
            {"expression": "x0**3", "source": "engine:b"}]
    optional = [{"expression": "exp(x0)", "source": "llm_proposal",
                 "origin": "llm", "lineage_id": "fixture-exp"}]
    retained, report = admit_regional_candidates(core, optional, fit,
        gap, admission, np.linspace(-4., 6., 56)[:, None],
        FrozenAxisRegions(1, 0, (1.,)), (1,), NormalInverseGammaPrior(), 4,
        alpha=.025, coefficient_policy=EXPANDED_FORMULA_POLICY)
    assert retained == optional
    assert report["coefficient_policy"] == EXPANDED_FORMULA_POLICY


def test_old_paired_runner_rejects_adaptive_mode_before_output(tmp_path):
    selection, _ = _selection_and_roles()
    with pytest.raises(ValueError, match="no paired ablation role forwarding"):
        run_exploration_ablations(tmp_path / "old", selection,
            dataset="fixture", config=DiscoveryAgentConfig(
                iterative_posterior_refinement=True),
            provider_settings=None, single_engine="mcts",
            compute_ceiling=1., provider_attempt_ceiling=0,
            source_identity="fixture", scientific_context={})
    assert not (tmp_path / "old").exists()
