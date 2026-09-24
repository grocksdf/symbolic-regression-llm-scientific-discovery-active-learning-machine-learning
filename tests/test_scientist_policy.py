"""Scientist-policy correctness fixtures; no provider or real data access."""
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.agent import (
    DiscoveryAgentConfig, _bounded_seed_bank, _survivors,
)
from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.evaluation_runtime import EvaluationRuntime
from hypothesis_mvp.discovery.initializer import normalize_candidates
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime
from hypothesis_mvp.discovery.proposal_runtime import ProposalContext
from hypothesis_mvp.discovery.scientist_policy import (
    ENGINE_REVIEW_PROTOCOL, RESEARCH_PLAN_PROTOCOL,
    ScientistState, deterministic_plan, plan_from_json, review_from_json,
)
from hypothesis_mvp.discovery.inference_router import route_inference
from hypothesis_mvp.discovery.skill_policy import (
    SkillTaskEvidence, allocate_bayesian_skill_jobs, fit_skill_reliability,
    leave_one_task_out_contextual_skill_policy,
    leave_one_task_out_skill_policy, leave_one_task_out_skill_policy_v2,
)
from hypothesis_mvp.discovery.skill_meta_policy import (
    SkillMetaEvidence, leave_one_task_out_meta_skill_policy,
)
from hypothesis_mvp.discovery.skill_probe_policy import (
    SkillProbeEvidence, allocate_task_local_probe_jobs,
    fit_probe_skill_model, leave_one_task_out_probe_skill_policy,
    predict_probe_skill_model,
)
from hypothesis_mvp.discovery.llm_preference_policy import (
    LLMPreferenceEvidence, leave_one_task_out_llm_preference_policy,
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
    with pytest.raises(ValueError, match="items must be strings"):
        review_from_json({
            "protocol_id": ENGINE_REVIEW_PROTOCOL,
            "supported_mechanisms": [{"name": "not-a-string"}],
            "contradicted_mechanisms": [], "cross_engine_conflicts": [],
            "synthesis_instructions": ["retain"], "stop": False,
            "stop_reason": "continue"})


def test_review_compiler_preserves_structured_evidence_and_typed_stop(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    raw = {"protocol_id": "provider-review-protocol",
        "supported_mechanisms": [
            {"mechanism": "sparse dependence", "evidence": "lower score"}],
        "contradicted_mechanisms": [],
        "cross_engine_conflicts": [{"engines": ["a", "b"], "reason": "rank"}],
        "synthesis_instructions": [
            {"instruction": "retain both supports", "scope": "development"}],
        "stop": "false", "stop_reason": "another comparison remains"}
    calls = []
    def complete_json(**kwargs):
        calls.append(kwargs)
        return raw, {"fixture": True}
    monkeypatch.setattr(runtime, "complete_json", complete_json)
    review, telemetry = runtime.review_engine_evidence(
        plan=deterministic_plan(("polynomial_lasso",), 1),
        engine_evidence=[])
    assert len(calls) == 1 and review.stop is False
    assert review.supported_mechanisms == (
        '{"evidence":"lower score","mechanism":"sparse dependence"}',)
    assert review.cross_engine_conflicts == (
        '{"engines":["a","b"],"reason":"rank"}',)
    projection = telemetry["review_contract_projection"]
    assert projection["protocol_identity_projection"]["bound"] == (
        ENGINE_REVIEW_PROTOCOL)
    assert projection["stop_boolean_projection"]["bound"] is False
    assert projection["structured_statement_projection"][
        "canonical_json_statement_counts"] == {
            "supported_mechanisms": 1,
            "cross_engine_conflicts": 1,
            "synthesis_instructions": 1}


def test_review_compiler_fails_closed_after_invalid_stop_repair(monkeypatch):
    from hypothesis_mvp.discovery.proposal_runtime import (
        ScientistReviewProtocolError,
    )
    invalid = {"protocol_id": ENGINE_REVIEW_PROTOCOL,
        "supported_mechanisms": [], "contradicted_mechanisms": [],
        "cross_engine_conflicts": [],
        "synthesis_instructions": ["retain evidence"],
        "stop": "maybe", "stop_reason": "ambiguous"}
    responses = iter([invalid, invalid])
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (next(responses), {"fixture": True}))
    with pytest.raises(
            ScientistReviewProtocolError,
            match=("scientist-review-invalid-after-one-provider-repair:"
                   "invalid-stop-decision")):
        runtime.review_engine_evidence(
            plan=deterministic_plan(("polynomial_lasso",), 1),
            engine_evidence=[])


def test_proposal_parent_hash_is_code_bound_not_llm_controlled():
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    context = ProposalContext(1, "balanced", "bound-parent", 1, 1)
    candidate, audit = runtime._candidate({
        "candidate_id": "candidate", "parent_hash": "wrong-parent",
        "action": "ADD", "equation": "x0", "rationale": "testable law"},
        0, context, "prompt", "response")
    assert candidate.parent_hash == "bound-parent"
    assert audit["parent_hash_projected"] is True
    assert audit["supplied_parent_hash"] == "wrong-parent"


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


def test_engine_scheduler_compiles_requested_operations_into_job_matrix(
        monkeypatch):
    scheduler = EngineScheduler()
    seen = []

    def execute(job, *args):
        seen.append(job.controls)
        result = SimpleNamespace(
            engine=job.engine, expression=f"x0+{job.repeat}",
            mse_val=float(job.repeat + 1), complexity=1.,
            score=float(job.repeat + 1),
            diagnostics={"skill_controls": list(job.controls)},
            repeats=(), lineage_id=f"{job.engine}-{job.repeat}")
        record = SimpleNamespace(
            engine=job.engine, repeat=job.repeat, attempt=0,
            seed=job.seed, status="succeeded", elapsed_seconds=0.,
            lineage_id=result.lineage_id, expression=result.expression,
            error_type="", error_message="", controls=job.controls)
        return (result,), record

    monkeypatch.setattr(
        "hypothesis_mvp.symbolic.scheduler._execute", execute)
    scheduler.run_allocated(
        allocations={"polynomial_lasso": 6},
        engine_controls={"polynomial_lasso": (
            "linear", "quadratic", "cubic", "quartic", "interactions")},
        config=SymbolicConfig(), X_train=np.ones((4, 1)), y_train=np.ones(4),
        X_val=np.ones((4, 1)), y_val=np.ones(4), evaluation_budget=6,
        parallel=False)
    assert seen[:5] == [
        ("linear",), ("quadratic",), ("cubic",), ("quartic",),
        ("interactions",)]
    assert seen[5] == (
        "linear", "quadratic", "cubic", "quartic", "interactions")


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


def test_skill_replay_v2_requires_per_skill_cross_family_prediction():
    rows = tuple(
        SkillTaskEvidence(
            f"{skill}-{family}-{index}", family, skill,
            ((1., 1.) if outcome else (-1., -1.)), (.01, .01))
        for skill, outcomes in {
            "mcts": (True, False, True, False),
            "sparse_library": (True, True, False, True)}.items()
        for index, (family, outcome) in enumerate(zip(
            ("a", "a", "b", "b"), outcomes)))
    report = leave_one_task_out_skill_policy_v2(
        rows, minimum_training_tasks=2, minimum_resolved_predictions=4)
    assert report["coverage_passed"]
    assert report["resolved_covered_prediction_count"] == 8
    assert report["brier_score"] is not None


def test_bayesian_skill_allocator_combines_reliability_and_llm_preference():
    reliability = {
        "polynomial_lasso": {
            "posterior_mean": .6, "lower_credible_bound": .4},
        "mcts": {"posterior_mean": .2, "lower_credible_bound": .05},
        "sparse_library": {
            "posterior_mean": .7, "lower_credible_bound": .45},
        "additive_mechanisms": {
            "posterior_mean": .5, "lower_credible_bound": .3}}
    result = allocate_bayesian_skill_jobs(
        tuple(reliability), 6, reliability,
        llm_requested_jobs={"polynomial_lasso": 2, "mcts": 1,
                            "sparse_library": 2,
                            "additive_mechanisms": 1})
    assert sum(result["allocated_jobs"].values()) == 6
    assert result["allocated_jobs"]["sparse_library"] >= 2
    assert result["allocated_jobs"]["mcts"] == 1
    assert result["candidate_response_accessed"] is False


def test_contextual_skill_replay_improves_family_specific_prediction():
    rows = tuple(
        SkillTaskEvidence(
            f"{family}-{index}", family, "fixture_skill",
            ((1., 1.) if family == "a" else (-1., -1.)), (.01, .01))
        for family in ("a", "b") for index in range(3))
    report = leave_one_task_out_contextual_skill_policy(rows)
    assert report["passed"]
    assert report["coverage_passed"]
    assert report["contextual_brier_score"] < report["global_brier_score"]
    assert report["contextual_brier_score"] < .25


def test_task_meta_skill_policy_learns_response_free_semantic_signal():
    rows = []
    for index in range(10):
        success = index % 2 == 0
        family = "a" if index < 5 else "b"
        evidence = SkillTaskEvidence(
            f"task-{index}", family, "fixture_skill",
            ((1., 1.) if success else (-1., -1.)), (.01, .01))
        feature_count = 2 if success else 9
        context = {"dataset": "dataset",
            "dataset_family": family, "feature_count": feature_count,
            "feature_names": [f"x{value}" for value in range(feature_count)],
            "feature_units": ["u"] * feature_count,
            "target_name": "target",
            "target_unit": "u",
            "counts": {"exploration_development": 32,
                       "exploration_validation": 32,
                       "inference_initial": 16,
                       "development_evaluation": 64,
                       "acquisition_pool": 8}}
        rows.append(SkillMetaEvidence(evidence, context))
    report = leave_one_task_out_meta_skill_policy(rows)
    assert report["passed"], report
    assert report["meta_brier_score"] < report["global_brier_score"]
    assert report["rank_concordance"] > .5


def test_task_local_probe_policy_predicts_independent_admission():
    rows = []
    for index in range(10):
        success = index % 2 == 0
        family = "a" if index < 5 else "b"
        evidence = SkillTaskEvidence(
            f"task-{index}", family, "fixture_skill",
            ((1., 1.) if success else (-1., -1.)), (.01, .01))
        gain = .4 if success else -.4
        rows.append(SkillProbeEvidence(evidence, {
            "relative_score_gain": gain,
            "relative_mse_gain": gain,
            "log_complexity": 1.0,
            "support_novelty_ratio": .5 if success else .1,
            "log_candidate_count": 1.0}))
    report = leave_one_task_out_probe_skill_policy(rows)
    assert report["passed"], report
    assert report["probe_brier_score"] < report["global_brier_score"]
    assert report["rank_concordance"] > .5
    model = fit_probe_skill_model(rows)
    probabilities = predict_probe_skill_model(
        model, "a", {"fixture_skill": rows[0].probe})
    assert probabilities["fixture_skill"] > .5
    allocation = allocate_task_local_probe_jobs(
        ("fixture_skill", "baseline"), 3,
        {**probabilities, "baseline": .5},
        llm_requested_jobs={"fixture_skill": 2, "baseline": 1})
    assert allocation["allocated_jobs"]["fixture_skill"] == 2


def test_llm_preference_requires_positive_calibrated_increment():
    rows = []
    for task in range(10):
        for skill, preferred in (("good", True), ("bad", False)):
            rows.append(LLMPreferenceEvidence(
                f"task-{task}", skill, .5,
                (1.0 if preferred else -1.0),
                ("success" if preferred else "failure")))
    report = leave_one_task_out_llm_preference_policy(rows)
    assert report["passed"], report
    assert report["preference_beta_90pct_lower"] > 0.0
    assert report["calibrated_brier_score"] < report["probe_brier_score"]


def test_plan_compiler_binds_replay_posterior_to_llm_job_preferences(
        monkeypatch):
    reliability = {
        "polynomial_lasso": {
            "posterior_mean": .5, "lower_credible_bound": 0.0},
        "mcts": {
            "posterior_mean": .278, "lower_credible_bound": .106},
        "sparse_library": {
            "posterior_mean": .5, "lower_credible_bound": .225},
        "additive_mechanisms": {
            "posterior_mean": .875, "lower_credible_bound": .661}}
    runtime = ProposalRuntime(
        EquationRuntime(1), 1, None, 1,
        skill_reliability=reliability, skill_policy_identity="replay")
    raw = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["fixture"],
        "engine_calls": [
            {"engine": "polynomial_lasso", "jobs": 2,
             "objective": "baseline", "expected_evidence": "support"},
            {"engine": "mcts", "jobs": 1,
             "objective": "tree search", "expected_evidence": "frontier"},
            {"engine": "sparse_library", "jobs": 2,
             "objective": "sparse search", "expected_evidence": "path"},
            {"engine": "additive_mechanisms", "jobs": 1,
             "objective": "additive search", "expected_evidence": "terms"}],
        "comparison_questions": ["compare"], "synthesis_goal": "synthesize",
        "stop_conditions": ["budget"]}
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (raw, {"fixture": True}))
    plan, telemetry = runtime.plan_research(
        task_context={"description": "fixture"},
        available_engines=tuple(reliability), total_jobs=6)
    jobs = {call.engine: call.jobs for call in plan.engine_calls}
    assert jobs == {"polynomial_lasso": 1, "mcts": 1,
                    "sparse_library": 2, "additive_mechanisms": 2}
    projection = telemetry["plan_contract_projection"][
        "forced_coverage_dispatch_projection"]
    assert projection["skill_policy_identity"] == "replay"
    assert projection["bayesian_skill_allocation"]["allocated_jobs"] == jobs


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
        "legacy_previous_cycle_survivor", "deterministic_linear_anchor",
        "deterministic_constant_anchor"}


def test_cross_round_survivor_preserves_source_origin_and_lineage():
    survivor = _survivors({"final_topk": [{
        "expression": "x0**2", "source": "llm_round_1",
        "origin": "llm", "lineage_id": "lineage"}]}, "x0")[0]
    assert survivor == {"expression": "x0**2", "source": "llm_round_1",
                        "origin": "llm", "lineage_id": "lineage"}
    normalized, rejected = normalize_candidates([survivor], n_features=1)
    assert rejected == 0 and normalized[0]["origin"] == "llm"
    assert normalized[0]["engine_provenance"][0]["origin"] == "llm"
    runtime = EvaluationRuntime(
        EquationRuntime(1, refit_policy="pcpi-closed-basis-amplitudes"),
        DiscoveryConfig.from_mapping({
            "evaluation_budget": 8, "llm_evaluation_reserve": 2,
            "refit_policy": "pcpi-closed-basis-amplitudes"}))
    X = np.linspace(-1., 1., 8)[:, None]
    _, states = runtime.seed_survivor(
        [survivor, {"expression": "x0", "source": "core"}],
        X, X[:, 0] ** 2, X, X[:, 0] ** 2)
    llm = next(row for row in states if row.source == "llm_round_1")
    assert llm.origin == "llm" and llm.source == "llm_round_1"
    assert llm.lineage_id == "lineage"

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
    assert "registered polynomial_lasso skill" in plan.engine_calls[0].objective
    assert "Validated expressions" in plan.engine_calls[0].expected_evidence
    assert plan.mechanisms == ("sparse predictive structure",)
    projection = telemetry["singleton_dispatch_projection"]
    assert projection["applied"] is True
    assert projection["original_engine"] == "mcts"
    assert projection["original_jobs"] == 1
    assert projection["scientist_objective"] == "estimate sparse polynomial support"


def test_single_engine_projection_replaces_unsupported_dispatch_without_retry(monkeypatch):
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
    assert len(calls) == 1
    assert "protocol_repair_attempted" not in telemetry
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
        "scientist-plan-invalid-after-one-provider-repair:"
        "empty-scientist-text")


def test_plan_protocol_identity_is_code_owned_without_provider_retry(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    raw = {"protocol_id": "provider-invented-protocol",
        "mechanisms": ["compare sparse and nonlinear structure"],
        "engine_calls": [
            {"engine": "polynomial_lasso", "jobs": 1,
             "objective": "estimate sparse polynomial support",
             "expected_evidence": "validated sparse support"},
            {"engine": "mcts", "jobs": 1,
             "objective": "search a nonlinear structural frontier",
             "expected_evidence": "validated structural diversity"}],
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
        available_engines=("polynomial_lasso", "mcts"), total_jobs=2)
    assert len(calls) == 1
    assert plan.protocol_id == RESEARCH_PLAN_PROTOCOL
    projection = telemetry["plan_contract_projection"]
    assert projection["protocol_identity_projection"]["supplied"] == (
        "provider-invented-protocol")
    assert projection["protocol_identity_projection"]["bound"] == (
        RESEARCH_PLAN_PROTOCOL)


def test_plan_compiler_preserves_structured_scientific_statements(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    raw = {"protocol_id": "provider-plan-protocol",
        "mechanisms": [{"name": "sparse baseline", "role": "falsifier"}],
        "engine_calls": [
            {"engine": "mcts", "jobs": 2,
             "objective": {"goal": "search nonlinear support"},
             "expected_evidence": {"artifact": "validated frontier"}}],
        "comparison_questions": {"question": "which support generalizes"},
        "synthesis_goal": {"goal": "retain falsifiable diversity"},
        "stop_conditions": [{"condition": "registered budget exhausted"}]}
    calls = []
    def complete_json(**kwargs):
        calls.append(kwargs)
        return raw, {"fixture": True}
    monkeypatch.setattr(runtime, "complete_json", complete_json)
    plan, telemetry = runtime.plan_research(
        task_context={"description": "fixture"},
        available_engines=("polynomial_lasso", "mcts"), total_jobs=2)
    assert len(calls) == 1
    assert plan.mechanisms == (
        '{"name":"sparse baseline","role":"falsifier"}',)
    assert plan.comparison_questions == (
        '{"question":"which support generalizes"}',)
    assert plan.synthesis_goal == '{"goal":"retain falsifiable diversity"}'
    assert plan.stop_conditions == (
        '{"condition":"registered budget exhausted"}',)
    assert [(call.engine, call.jobs) for call in plan.engine_calls] == [
        ("polynomial_lasso", 1), ("mcts", 1)]
    assert plan.engine_calls[1].objective == (
        '{"goal":"search nonlinear support"}')
    projection = telemetry["plan_contract_projection"]
    assert projection["structured_plan_statement_projection"][
        "canonical_json_statement_counts"] == {
            "mechanisms": 1, "comparison_questions": 1,
            "stop_conditions": 1, "synthesis_goal": 1}
    assert projection["forced_coverage_dispatch_projection"][
        "structured_engine_text_fields"] == [
            {"engine": "mcts", "field": "objective"},
            {"engine": "mcts", "field": "expected_evidence"}]


def test_forced_full_coverage_jobs_are_code_owned(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    raw = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["compare sparse and nonlinear structure"],
        "engine_calls": [
            {"engine": "mcts", "jobs": 2,
             "objective": "search a nonlinear structural frontier",
             "expected_evidence": "validated structural diversity"},
            {"engine": "polynomial_lasso", "jobs": 0,
             "objective": "estimate sparse polynomial support",
             "expected_evidence": "validated sparse support"}],
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
        available_engines=("polynomial_lasso", "mcts"), total_jobs=2)
    assert len(calls) == 1
    assert [(call.engine, call.jobs) for call in plan.engine_calls] == [
        ("polynomial_lasso", 1), ("mcts", 1)]
    projection = telemetry["plan_contract_projection"]
    assert projection["forced_coverage_dispatch_projection"][
        "bound_jobs"] == {"polynomial_lasso": 1, "mcts": 1}


def test_scientist_extra_budget_allocation_is_executable(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    raw = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["compare sparse and nonlinear structure"],
        "engine_calls": [
            {"engine": "polynomial_lasso", "jobs": 3,
             "objective": "fit sparse support",
             "expected_evidence": "validated polynomial support"},
            {"engine": "mcts", "jobs": 1,
             "objective": "search nonlinear support",
             "expected_evidence": "validated nonlinear frontier"}],
        "comparison_questions": ["which support generalizes"],
        "synthesis_goal": "retain falsifiable structure",
        "stop_conditions": ["registered budget exhausted"]}
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (raw, {"fixture": True}))
    plan, telemetry = runtime.plan_research(
        task_context={"description": "fixture"},
        available_engines=("polynomial_lasso", "mcts"), total_jobs=4)
    assert [(call.engine, call.jobs) for call in plan.engine_calls] == [
        ("polynomial_lasso", 3), ("mcts", 1)]
    projection = telemetry["plan_contract_projection"][
        "forced_coverage_dispatch_projection"]
    assert projection["bound_jobs"] == {
        "polynomial_lasso": 3, "mcts": 1}
    assert len(projection["skill_operation_projections"]) == 2


def test_scientist_invalid_extra_budget_is_projected_from_preferences(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    raw = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["fixture"],
        "engine_calls": [
            {"engine": "polynomial_lasso", "jobs": 9,
             "objective": "baseline", "expected_evidence": "support"},
            {"engine": "mcts", "jobs": 0,
             "objective": "search", "expected_evidence": "frontier"}],
        "comparison_questions": ["compare"], "synthesis_goal": "synthesize",
        "stop_conditions": ["budget"]}
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (raw, {"fixture": True}))
    plan, telemetry = runtime.plan_research(
        task_context={"description": "fixture"},
        available_engines=("polynomial_lasso", "mcts"), total_jobs=4)
    assert [(call.engine, call.jobs) for call in plan.engine_calls] == [
        ("polynomial_lasso", 2), ("mcts", 2)]
    assert telemetry["plan_contract_projection"][
        "forced_coverage_dispatch_projection"]["bound_jobs"] == {
            "polynomial_lasso": 2, "mcts": 2}


def test_forced_full_coverage_compiles_missing_registered_skill(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    raw = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["compare sparse and nonlinear structure"],
        "engine_calls": [
            {"engine": "mcts", "jobs": 2,
             "objective": "search a nonlinear structural frontier",
             "expected_evidence": "validated structural diversity"}],
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
        available_engines=("polynomial_lasso", "mcts"), total_jobs=2)
    assert len(calls) == 1
    assert [(call.engine, call.jobs) for call in plan.engine_calls] == [
        ("polynomial_lasso", 1), ("mcts", 1)]
    projection = telemetry["plan_contract_projection"][
        "forced_coverage_dispatch_projection"]
    assert projection["synthesized_required_calls"] == [
        "polynomial_lasso"]
    assert plan.engine_calls[0].requested_operations == (
        "linear", "quadratic", "cubic", "quartic", "interactions")


def test_forced_full_coverage_rejects_entirely_unusable_dispatch(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    invalid = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["fixture"],
        "engine_calls": [{"engine": "invented_engine", "jobs": 2,
            "objective": "unknown search", "expected_evidence": "unknown"}],
        "comparison_questions": ["which support generalizes"],
        "synthesis_goal": "retain falsifiable structure",
        "stop_conditions": ["budget"]}
    responses = iter([invalid, invalid])
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (next(responses), {"fixture": True}))
    from hypothesis_mvp.discovery.proposal_runtime import ScientistPlanProtocolError
    with pytest.raises(
            ScientistPlanProtocolError,
            match=("scientist-plan-invalid-after-one-provider-repair:"
                   "incomplete-or-inconsistent-plan")):
        runtime.plan_research(task_context={"description": "fixture"},
            available_engines=("polynomial_lasso", "mcts"), total_jobs=2)


def test_multi_engine_plan_projects_operations_to_registered_capabilities(
        monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    invalid = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["fixture"],
        "engine_calls": [
            {"engine": "polynomial_lasso", "jobs": 1,
             "objective": "baseline",
             "expected_evidence": "support",
             "requested_operations": ["logarithmic"]},
            {"engine": "mcts", "jobs": 1,
             "objective": "typed symbolic search",
             "expected_evidence": "structural diversity frontier",
             "requested_operations": []}],
        "comparison_questions": ["which support generalizes"],
        "synthesis_goal": "retain falsifiable structure",
        "stop_conditions": ["budget"]}
    calls = []
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (calls.append(kwargs) or invalid, {"fixture": True}))
    plan, telemetry = runtime.plan_research(
        task_context={"description": "fixture"},
        available_engines=("polynomial_lasso", "mcts"), total_jobs=2)
    assert len(calls) == 1
    assert plan.engine_calls[0].requested_operations == (
        "linear", "quadratic", "cubic", "quartic", "interactions")
    assert plan.engine_calls[1].requested_operations == (
        "monomials", "trigonometric", "saturating")
    projections = telemetry["plan_contract_projection"][
        "forced_coverage_dispatch_projection"][
            "skill_operation_projections"]
    assert {row["engine"] for row in projections} == {
        "polynomial_lasso", "mcts"}


def test_explicit_skill_control_protocol_is_required_when_registered(monkeypatch):
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    valid = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["fixture"],
        "engine_calls": [
            {"engine": "polynomial_lasso", "jobs": 1,
             "objective": "linear baseline", "expected_evidence": "support",
             "requested_operations": ["linear"]},
            {"engine": "mcts", "jobs": 1,
             "objective": "nonlinear search", "expected_evidence": "frontier",
             "requested_operations": ["saturating"]}],
        "comparison_questions": ["compare"], "synthesis_goal": "synthesize",
        "stop_conditions": ["budget"]}
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (valid, {"fixture": True}))
    plan, _ = runtime.plan_research(
        task_context={"description": "fixture",
                      "require_explicit_skill_controls": True},
        available_engines=("polynomial_lasso", "mcts"), total_jobs=2)
    assert plan.engine_calls[0].requested_operations == ("linear",)
    assert plan.engine_calls[1].requested_operations == ("saturating",)


def test_missing_explicit_skill_controls_use_registered_capability_sets(
        monkeypatch):
    invalid = {"protocol_id": RESEARCH_PLAN_PROTOCOL,
        "mechanisms": ["fixture"],
        "engine_calls": [
            {"engine": "polynomial_lasso", "jobs": 1,
             "objective": "baseline", "expected_evidence": "support"},
            {"engine": "mcts", "jobs": 1,
             "objective": "search", "expected_evidence": "frontier"}],
        "comparison_questions": ["compare"], "synthesis_goal": "synthesize",
        "stop_conditions": ["budget"]}
    runtime = ProposalRuntime(EquationRuntime(1), 1, None, 1)
    monkeypatch.setattr(runtime, "complete_json",
        lambda **kwargs: (invalid, {"fixture": True}))
    plan, telemetry = runtime.plan_research(
        task_context={"description": "fixture",
                      "require_explicit_skill_controls": True},
        available_engines=("polynomial_lasso", "mcts"), total_jobs=2)
    assert all(call.requested_operations for call in plan.engine_calls)
    assert len(telemetry["plan_contract_projection"][
        "forced_coverage_dispatch_projection"][
            "skill_operation_projections"]) == 2
