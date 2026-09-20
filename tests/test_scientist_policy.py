"""Scientist-policy correctness fixtures; no provider or real data access."""
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.agent import (
    DiscoveryAgentConfig, _bounded_seed_bank,
)
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
    scalar = review_from_json({
        "protocol_id": ENGINE_REVIEW_PROTOCOL,
        "supported_mechanisms": [], "contradicted_mechanisms": [],
        "cross_engine_conflicts": [],
        "synthesis_instructions": "retain the simpler falsifiable law",
        "stop": False, "stop_reason": "continue"})
    assert scalar.synthesis_instructions == (
        "retain the simpler falsifiable law",)


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


def test_cross_round_seed_bank_respects_fixed_deterministic_budget():
    engine = SimpleNamespace(all_results=tuple(
        SimpleNamespace(expression=f"x0+{index}", engine=(
            "polynomial_lasso" if index == 0 else "mcts"),
            lineage_id=str(index)) for index in range(4)))
    selection = SimpleNamespace(development=SimpleNamespace(
        X=np.arange(64., dtype=float).reshape(32, 2),
        y=np.arange(32., dtype=float)))
    config = DiscoveryAgentConfig(
        engines=("polynomial_lasso", "mcts"), engine_budget=2,
        discovery_budget=24, llm_evaluation_reserve=8,
        discovery_islands=("balanced",))
    rows, audit = _bounded_seed_bank(
        engine, tuple(f"x0+x1+{index}" for index in range(10)),
        selection, config)
    assert len(rows) == audit["limit"] == 15
    assert {row["source"] for row in rows} >= {
        "engine:polynomial_lasso", "engine:mcts",
        "previous_cycle_survivor", "deterministic_linear_anchor",
        "deterministic_constant_anchor"}

def test_single_engine_plan_projects_only_fixed_dispatch_fields(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    raw = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["sparse predictive structure"],
        "engine_calls": [{"engine": "mcts", "jobs": 1,
            "objective": "estimate sparse polynomial support",
            "expected_evidence": "validated polynomial support"}],
        "comparison_questions": ["which support generalizes"],
        "synthesis_goal": "retain falsifiable structure",
        "stop_conditions": ["registered budget exhausted"]}
    calls = []
    def complete_json(**kwargs):
        calls.append(kwargs)
        return raw, {"fixture": True}
    monkeypatch.setattr(runtime, "complete_json", complete_json)
    plan, telemetry = runtime.plan_research(
        task_context={"description": "fixture"},
        available_engines=("polynomial_lasso",), total_jobs=2)
    assert len(calls) == 1
    assert len(plan.engine_calls) == 1
    assert plan.engine_calls[0].engine == "polynomial_lasso"
    assert plan.engine_calls[0].jobs == 2
    assert plan.engine_calls[0].objective == "estimate sparse polynomial support"
    assert plan.engine_calls[0].expected_evidence == "validated polynomial support"
    assert plan.mechanisms == ("sparse predictive structure",)
    projection = telemetry["singleton_dispatch_projection"]
    assert projection["applied"] is True
    assert projection["original_engine"] == "mcts"
    assert projection["original_jobs"] == 1


def test_single_engine_projection_keeps_capability_validation_and_one_repair(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    responses = iter([
        {"protocol_id": RESEARCH_PLAN_PROTOCOL,
         "mechanisms": ["fixture"],
         "engine_calls": [{"engine": "mcts", "jobs": 1,
             "objective": "test logarithmic alternatives",
             "expected_evidence": "logarithmic support"}],
         "comparison_questions": ["which support generalizes"],
         "synthesis_goal": "retain falsifiable structure",
         "stop_conditions": ["budget"]},
        {"protocol_id": RESEARCH_PLAN_PROTOCOL,
         "mechanisms": ["fixture"],
         "engine_calls": [{"engine": "mcts", "jobs": 1,
             "objective": "estimate sparse polynomial support",
             "expected_evidence": "validated polynomial support"}],
         "comparison_questions": ["which support generalizes"],
         "synthesis_goal": "retain falsifiable structure",
         "stop_conditions": ["budget"]},
    ])
    calls = []
    def complete_json(**kwargs):
        calls.append(kwargs)
        return next(responses), {"fixture": len(calls)}
    monkeypatch.setattr(runtime, "complete_json", complete_json)
    plan, telemetry = runtime.plan_research(
        task_context={"description": "fixture"},
        available_engines=("polynomial_lasso",), total_jobs=2)
    assert len(calls) == 2
    assert telemetry["protocol_repair_attempted"] is True
    assert plan.engine_calls[0].engine == "polynomial_lasso"
    assert plan.engine_calls[0].jobs == 2
    assert "logarithmic" not in plan.engine_calls[0].objective


def test_single_engine_projection_does_not_mask_invalid_non_dispatch_fields(monkeypatch):
    from hypothesis_mvp.discovery.proposal_runtime import ScientistPlanProtocolError
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    invalid = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["fixture"],
        "engine_calls": [{"engine": "mcts", "jobs": 1,
            "objective": "estimate sparse polynomial support",
            "expected_evidence": "validated polynomial support"}],
        "comparison_questions": ["which support generalizes"],
        "synthesis_goal": "retain falsifiable structure",
        "stop_conditions": []}
    responses = iter([invalid, invalid])
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (next(responses), {"fixture": True}))
    with pytest.raises(
            ScientistPlanProtocolError,
            match="scientist-plan-invalid-after-one-provider-repair"
    ) as caught:
        runtime.plan_research(task_context={"description": "fixture"},
            available_engines=("polynomial_lasso",), total_jobs=2)
    assert caught.value.public_diagnostic == (
        "scientist-plan-invalid-after-one-provider-repair")


def test_multi_engine_plan_remains_fail_closed_after_invalid_repair(monkeypatch):
    from hypothesis_mvp.discovery.proposal_runtime import ScientistPlanProtocolError
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    invalid = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["fixture"],
        "engine_calls": [
            {"engine": "polynomial_lasso", "jobs": 1,
             "objective": "logarithmic baseline",
             "expected_evidence": "logarithmic support"},
            {"engine": "mcts", "jobs": 1,
             "objective": "typed symbolic search",
             "expected_evidence": "structural diversity frontier"}],
        "comparison_questions": ["which support generalizes"],
        "synthesis_goal": "retain falsifiable structure",
        "stop_conditions": ["budget"]}
    responses = iter([invalid, invalid])
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (next(responses), {"fixture": True}))
    with pytest.raises(
            ScientistPlanProtocolError,
            match="scientist-plan-invalid-after-one-provider-repair"
    ):
        runtime.plan_research(task_context={"description": "fixture"},
            available_engines=("polynomial_lasso", "mcts"), total_jobs=2)
