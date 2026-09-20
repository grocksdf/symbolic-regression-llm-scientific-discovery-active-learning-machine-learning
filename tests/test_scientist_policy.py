"""Scientist-policy correctness fixtures; no provider or real data access."""
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime
from hypothesis_mvp.discovery.scientist_policy import (
    ENGINE_REVIEW_PROTOCOL, RESEARCH_PLAN_PROTOCOL,
    ScientistState, deterministic_plan, plan_from_json, review_from_json,
)
from hypothesis_mvp.discovery.inference_router import route_inference
from hypothesis_mvp.discovery.skill_policy import (
    SkillTaskEvidence, fit_skill_reliability, leave_one_task_out_skill_policy,
)
from hypothesis_mvp.symbolic.scheduler import EngineScheduler


def test_typed_research_plan_binds_exact_budget_and_registered_engines():
    raw = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["linear baseline", "nonlinear residual structure"],
        "engine_calls": [
            {"engine": "polynomial_lasso", "jobs": 1,
             "objective": "estimate sparse baseline",
             "expected_evidence": "validated polynomial support"},
            {"engine": "mcts", "jobs": 2,
             "objective": "search nonlinear alternatives",
             "expected_evidence": "diverse typed expressions"},
        ],
        "comparison_questions": ["Does nonlinear structure improve prediction?"],
        "synthesis_goal": "build a falsifiable diverse bank",
        "stop_conditions": ["budget exhausted"]}
    plan = plan_from_json(raw, ("polynomial_lasso", "mcts"), 3)
    assert plan.stable_hash and sum(call.jobs for call in plan.engine_calls) == 3
    with pytest.raises(ValueError):
        plan_from_json(raw, ("polynomial_lasso", "mcts"), 4)


def test_provider_free_plan_is_deterministic_and_covers_budget():
    first = deterministic_plan(("polynomial_lasso", "mcts"), 5)
    second = deterministic_plan(("polynomial_lasso", "mcts"), 5)
    assert first == second
    assert [call.jobs for call in first.engine_calls] == [3, 2]


def test_scientist_review_schema_is_strict():
    review = review_from_json({
        "protocol_id": ENGINE_REVIEW_PROTOCOL,
        "supported_mechanisms": ["sparse linear dependence"],
        "contradicted_mechanisms": [], "cross_engine_conflicts": [],
        "synthesis_instructions": ["retain both falsifiable supports"],
        "stop": False, "stop_reason": "synthesis remains useful"})
    assert review.stable_hash and not review.stop
    with pytest.raises(ValueError):
        review_from_json({"protocol_id": ENGINE_REVIEW_PROTOCOL})


def test_proposal_runtime_uses_single_transport_for_plan_and_review(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(2), 2, None, 2)
    responses = iter([
        {"protocol_id": RESEARCH_PLAN_PROTOCOL,
         "mechanisms": ["linear", "nonlinear"],
         "engine_calls": [
             {"engine": "polynomial_lasso", "jobs": 1, "objective": "baseline",
              "expected_evidence": "support"},
             {"engine": "mcts", "jobs": 1, "objective": "alternatives",
              "expected_evidence": "frontier"}],
         "comparison_questions": ["which generalizes"],
         "synthesis_goal": "diverse bank", "stop_conditions": ["budget"]},
        {"protocol_id": ENGINE_REVIEW_PROTOCOL,
         "supported_mechanisms": ["linear"], "contradicted_mechanisms": [],
         "cross_engine_conflicts": ["different supports"],
         "synthesis_instructions": ["retain testable alternatives"],
         "stop": False, "stop_reason": "requires synthesis"},
    ])
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (next(responses), {"fixture": True}))
    plan, _ = runtime.plan_research(task_context={"description": "fixture"},
        available_engines=("polynomial_lasso", "mcts"), total_jobs=2)
    review, _ = runtime.review_engine_evidence(plan=plan, engine_evidence=[])
    assert len(plan.engine_calls) == 2
    assert review.cross_engine_conflicts == ("different supports",)


def test_engine_scheduler_executes_nonuniform_scientist_allocation(monkeypatch):
    scheduler = EngineScheduler()
    seen = []
    def execute(job, *args):
        seen.append((job.engine, job.repeat))
        result = SimpleNamespace(engine=job.engine, expression=f"x0+{job.repeat}",
            mse_val=float(job.repeat + 1), complexity=1., score=float(job.repeat + 1),
            diagnostics={}, repeats=(), lineage_id=f"{job.engine}-{job.repeat}")
        record = SimpleNamespace(engine=job.engine, repeat=job.repeat, attempt=0,
            seed=job.seed, status="succeeded", elapsed_seconds=0.,
            lineage_id=result.lineage_id, expression=result.expression,
            error_type="", error_message="")
        return (result,), record
    monkeypatch.setattr("hypothesis_mvp.symbolic.scheduler._execute", execute)
    result = scheduler.run_allocated(
        allocations={"polynomial_lasso": 1, "mcts": 2},
        config=SymbolicConfig(), X_train=np.ones((4, 1)), y_train=np.ones(4),
        X_val=np.ones((4, 1)), y_val=np.ones(4), evaluation_budget=3,
        parallel=False)
    assert seen == [("polynomial_lasso", 0), ("mcts", 0), ("mcts", 1)]
    assert result.evaluations_used == 3


def test_scientist_state_carries_prior_evidence_into_replanning():
    plan = deterministic_plan(("polynomial_lasso", "mcts"), 2)
    review = review_from_json({
        "protocol_id": ENGINE_REVIEW_PROTOCOL,
        "supported_mechanisms": ["nonlinear"], "contradicted_mechanisms": [],
        "cross_engine_conflicts": ["different support"],
        "synthesis_instructions": ["retain both"],
        "stop": False, "stop_reason": "another round is registered"})
    state = ScientistState().advance(
        plan=plan, review=review,
        engine_evidence=[
            {"engine": "polynomial_lasso", "selection_score": 2.,
             "expression": "x0"},
            {"engine": "mcts", "selection_score": 1., "expression": "sin(x0)"}],
        surviving_hypotheses=("x0", "sin(x0)"))
    assert state.round_index == 1 and state.cumulative_engine_jobs == 2
    assert state.prior_rounds[0]["engine_summary"]["mcts"][
        "best_expression"] == "sin(x0)"
    assert state.to_dict()["candidate_response_accessed"] is False


def test_inference_router_uses_exact_for_finite_and_blocks_unauthorized_open_smc():
    model = SimpleNamespace(stable_hash="finite-model")
    exact = route_inference(model)
    assert exact.mode == "exact_finite" and not exact.certified_smc_authorized
    with pytest.raises(PermissionError):
        route_inference(model, open_support=True)
    open_plan = route_inference(
        model, requested_mode="certified_open_smc", open_support=True,
        certified_smc_authorized=True)
    assert open_plan.mode == "certified_open_smc"


def test_skill_reliability_counts_tasks_not_folds_as_replicates():
    rows = [
        SkillTaskEvidence("a", "family-a", "mcts", (1., 2.), (.01, .01)),
        SkillTaskEvidence("b", "family-b", "mcts", (-1., 2.), (.01, .01)),
    ]
    result = fit_skill_reliability(rows)[0]
    assert result.successes == 1 and result.failures == 1
    assert result.posterior_alpha == result.posterior_beta == 1.5


def test_leave_one_task_out_skill_gate_requires_cross_family_coverage():
    same_family = tuple(SkillTaskEvidence(
        str(index), "gas", "mcts", (1., 1.), (.01, .01))
        for index in range(4))
    failed = leave_one_task_out_skill_policy(same_family)
    assert failed["passed"] is False
    assert failed["status"] == "insufficient-cross-family-coverage"
    diverse = (
        SkillTaskEvidence("a1", "a", "mcts", (1., 1.), (.01, .01)),
        SkillTaskEvidence("a2", "a", "mcts", (1., 1.), (.01, .01)),
        SkillTaskEvidence("b1", "b", "mcts", (1., 1.), (.01, .01)),
        SkillTaskEvidence("b2", "b", "mcts", (1., 1.), (.01, .01)),
    )
    passed = leave_one_task_out_skill_policy(diverse)
    assert passed["passed"] is True
    assert passed["candidate_response_accessed"] is False
