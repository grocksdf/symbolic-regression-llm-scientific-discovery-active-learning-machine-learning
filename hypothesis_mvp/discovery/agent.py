"""Single production agent for multi-engine and LLM discovery orchestration.

Measured-pool acquisition is owned exclusively by :mod:`hypothesis_mvp.pcpi`
and its P3B runner.  This agent deliberately cannot select or reveal pool
labels, so it cannot become a second acquisition implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.data import SelectionData
from hypothesis_mvp.symbolic import EngineScheduler, merge_multi_engine_results

from .api import DiscoveryRunResult, discover_from_selection
from .contracts import DiscoveryConfig
from .proposal_runtime import (
    ProtocolError, ProviderInfrastructureError, ProviderSettings, ProposalRuntime,
    ScientistPlanProtocolError, ScientistReviewProtocolError,
)
from .equation_runtime import EquationRuntime
from .evidence_synthesis import compile_evidence_synthesis
from .initializer import generic_deterministic_candidates
from .scientist_policy import (
    ResearchPlan, ScientistReview, ScientistState, allocated_plan,
    deterministic_plan,
)
from .skill_policy import (
    allocate_bayesian_skill_jobs, conservative_allocation_decision,
)
from .skill_probe_policy import (
    allocate_task_local_probe_jobs, predict_probe_skill_model,
    probe_features_from_engine_results,
)
from .system_evidence import (
    attach_scientist_policy_evidence, attach_system_evidence, system_evaluation,
)


@dataclass(frozen=True)
class DiscoveryAgentConfig:
    engines: tuple[str, ...] = ("polynomial_lasso", "mcts")
    engine_repeats: int = 2
    engine_budget: int = 8
    engine_workers: int = 2
    engine_timeout_s: float = 300.0
    engine_retries: int = 1
    cycles: int = 3
    discovery_budget: int = 600
    random_seed: int = 42
    search_iterations: int = 120
    mcts_frontier_size: int = 4
    mcts_score_folds: int = 2
    acquisition_enabled: bool = False
    use_knowledge: bool = False
    task_local_memory: bool = False
    discovery_rounds: int = 3
    candidates_per_island: int = 4
    task_local_memory_topk: int = 8
    llm_evaluation_reserve: int = 0
    refit_policy: str = "global-constants"
    discovery_islands: tuple[str, ...] = ("low_complexity", "nmse", "tail", "novelty")
    scientist_orchestration: bool = False
    require_explicit_skill_controls: bool = False
    typed_evidence_synthesis: bool = False
    skill_reliability: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict)
    skill_policy_identity: str = ""
    probe_skill_model: Mapping[str, Any] = field(default_factory=dict)
    probe_skill_policy_identity: str = ""
    probe_allocation_mode: str = "llm-preference-enabled"
    llm_preference_policy_identity: str = ""
    dataset_family: str = ""
    protected_counterfactual_backbone: bool = False
    provider_failure_mode: str = "abort"
    conservative_allocation_policy: Mapping[str, Any] = field(
        default_factory=dict)
    allocation_calibration_role: str = "production"


@dataclass(frozen=True)
class DiscoveryCycle:
    cycle: int
    expression: str
    hypothesis_id: str
    development_rows: int
    engine_report: Mapping[str, Any]
    acquisition: Mapping[str, Any]
    provider_calls: int
    candidate_evaluations: int = 0
    provider_attempts: int = 0
    provider_errors: int = 0
    research_plan: Mapping[str, Any] = field(default_factory=dict)
    scientist_review: Mapping[str, Any] = field(default_factory=dict)
    scientist_state_before: Mapping[str, Any] = field(default_factory=dict)
    scientist_state_after: Mapping[str, Any] = field(default_factory=dict)
    probe_allocation: Mapping[str, Any] = field(default_factory=dict)
    counterfactual_backbone: Mapping[str, Any] = field(default_factory=dict)
    provider_failure_abstention: Mapping[str, Any] = field(
        default_factory=dict)
    conservative_allocation_decision: Mapping[str, Any] = field(
        default_factory=dict)
    evidence_synthesis: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscoveryAgentResult:
    discovery: DiscoveryRunResult
    cycles: tuple[DiscoveryCycle, ...]
    final_selection: SelectionData
    pool_rows_remaining: int
    provider_configured: bool
    system_evaluation: Mapping[str, Any] = field(default_factory=dict)


def _survivors(report: Mapping[str, Any], fallback: str) -> tuple[Mapping[str, str], ...]:
    rows = [{
        "expression": str(row.get("expression") or "").strip(),
        "source": str(row.get("source") or "prior_cycle_candidate"),
        "origin": str(row.get("origin") or "unknown"),
        "lineage_id": str(row.get("lineage_id") or ""),
    }
        for row in report.get("final_topk") or ()
        if isinstance(row, Mapping)
    ]
    unique = {row["expression"].replace(" ", ""): row
              for row in rows if row["expression"]}
    return tuple(unique.values()) or ({
        "expression": fallback, "source": "prior_cycle_final",
        "origin": str(report.get("selected_source") or "unknown"),
        "lineage_id": str(report.get("final_lineage_id") or "")},)


def _engine_payload(result: Any) -> dict[str, Any]:
    return {
        "best": asdict(result.best),
        "all_results": [asdict(row) for row in result.all_results],
        "run_records": [asdict(row) for row in result.run_records],
        "failures": list(result.failures),
        "evaluation_budget": result.evaluation_budget,
        "evaluations_used": result.evaluations_used,
    }


def _engine_evidence(result: Any) -> list[dict[str, Any]]:
    if not hasattr(result, "all_results"):
        return []
    return [{"engine": row.engine, "expression": row.expression,
        "validation_mse": row.mse_val, "complexity": row.complexity,
        "selection_score": row.score, "lineage_id": row.lineage_id,
        "diagnostics": {
            key: value for key, value in dict(row.diagnostics).items()
            if key in {"candidate_rank", "candidate_count", "search_score_method",
                       "search_score_folds", "candidate_set_method",
                       "skill_controls", "job_control_variants"}
        }} for row in result.all_results]


def _bounded_seed_bank(
        engine_result, previous, selection, config, synthesized=()):
    engine_rows = [{"expression": row.expression,
        "source": f"engine:{row.engine}", "lineage_id": row.lineage_id}
        for row in engine_result.all_results]
    previous_rows = [
        (dict(row) if isinstance(row, Mapping) else {
            "expression": str(row), "source": "legacy_previous_cycle_survivor",
            "origin": "unknown", "lineage_id": ""})
        for row in previous]
    generic = generic_deterministic_candidates(
        selection.development.X, selection.development.y)
    limit = (config.discovery_budget - config.llm_evaluation_reserve
             - len(config.discovery_islands))
    if limit < len({row["source"] for row in engine_rows}) + 2:
        raise ValueError("discovery budget cannot preserve required seed roles")
    priority, seen_engines = [], set()
    for row in engine_rows:
        engine = row["source"]
        if engine not in seen_engines:
            priority.append(row); seen_engines.add(engine)
    if previous_rows:
        priority.append(previous_rows[0])
    for marker in ("deterministic_linear_anchor", "deterministic_constant_anchor"):
        match = next((row for row in generic if row["source"] == marker), None)
        if match is not None:
            priority.append(match)
    synthesis_rows = [dict(row) for row in synthesized]
    # Evidence-conditioned synthesis is an additional proposal family.  It
    # may not evict a registered engine frontier or the deterministic anchors
    # from the fixed seed budget; this is a task-independent portfolio rule.
    ordered = [
        *priority, *engine_rows, *synthesis_rows, *previous_rows, *generic]
    selected, seen = [], set()
    for row in ordered:
        key = str(row["expression"]).replace(" ", "")
        if key in seen:
            continue
        seen.add(key); selected.append(row)
        if len(selected) == limit:
            break
    return selected, {"schema": "scientific-bounded-cross-round-seed-bank-v1",
        "limit": limit, "input_engine_candidates": len(engine_rows),
        "input_previous_survivors": len(previous_rows),
        "input_evidence_synthesis_candidates": len(synthesis_rows),
        "synthesis_is_non_destructive": True,
        "input_generic_candidates": len(generic), "selected_count": len(selected),
        "candidate_response_accessed": False, "heldout_opened": False}


class DiscoveryAgent:
    def __init__(
        self, config: DiscoveryAgentConfig,
        provider_settings: ProviderSettings | None = None,
    ) -> None:
        self.config = config
        self.provider_settings = provider_settings
        if config.allocation_calibration_role not in {
                "production", "paired-challenger"}:
            raise ValueError("invalid allocation calibration role")
        if (config.allocation_calibration_role == "paired-challenger"
                and (not config.scientist_orchestration
                     or not config.typed_evidence_synthesis)):
            raise ValueError(
                "paired challenger requires typed Scientist orchestration")
        if config.acquisition_enabled:
            raise ValueError(
                "DiscoveryAgent acquisition was removed; run the canonical P3B "
                "PCPI acquisition protocol instead"
            )
        self.scheduler = EngineScheduler()

    def _run_probe_stages(
            self, selection, cycle, symbolic, allocations, controls):
        first = self.scheduler.run_allocated(
            allocations={name: 1 for name in self.config.engines},
            config=symbolic, X_train=selection.development.X,
            y_train=selection.development.y,
            X_val=selection.validation.X, y_val=selection.validation.y,
            base_seed=self.config.random_seed + cycle,
            max_retries=self.config.engine_retries,
            evaluation_budget=len(self.config.engines),
            parallel=self.config.engine_workers > 1,
            max_workers=self.config.engine_workers,
            timeout_s=self.config.engine_timeout_s,
            engine_controls=controls)
        probes = probe_features_from_engine_results(
            first.all_results, selection.development.X.shape[1])
        probabilities = predict_probe_skill_model(
            self.config.probe_skill_model,
            self.config.dataset_family, probes)
        allocation = allocate_task_local_probe_jobs(
            self.config.engines, self.config.engine_budget, probabilities,
            llm_requested_jobs=(
                None if self.config.probe_allocation_mode == "probe-only"
                else allocations))
        self._last_probe_allocation = {
            **allocation,
            "probe_skill_policy_identity":
                self.config.probe_skill_policy_identity,
            "probe_allocation_mode": self.config.probe_allocation_mode,
            "llm_preference_policy_identity":
                self.config.llm_preference_policy_identity,
            "probe_features": probes}
        extras = {name: count - 1 for name, count
                  in allocation["allocated_jobs"].items() if count > 1}
        if not extras:
            return first
        second = self.scheduler.run_allocated(
            allocations=extras, config=symbolic,
            X_train=selection.development.X,
            y_train=selection.development.y,
            X_val=selection.validation.X, y_val=selection.validation.y,
            base_seed=self.config.random_seed + cycle + 1000003,
            max_retries=self.config.engine_retries,
            evaluation_budget=sum(extras.values()),
            parallel=self.config.engine_workers > 1,
            max_workers=self.config.engine_workers,
            timeout_s=self.config.engine_timeout_s,
            engine_controls=controls)
        return merge_multi_engine_results(
            (first, second), evaluation_budget=self.config.engine_budget)

    def _run_protected_backbone(
            self, selection, cycle, symbolic, allocations, controls):
        """Run one immutable default-control job per engine before adaptation.

        The backbone consumes part of the same registered engine budget.  LLM
        planning can allocate and control only the remaining jobs, so it cannot
        erase the matched provider-free counterfactual evidence.
        """
        baseline_jobs = len(self.config.engines)
        if (self.config.engine_budget < baseline_jobs
                or set(allocations) != set(self.config.engines)
                or any(type(allocations[name]) is not int
                       or allocations[name] < 1
                       for name in self.config.engines)
                or sum(allocations.values()) != self.config.engine_budget):
            raise ValueError(
                "protected backbone requires full engine coverage within budget")
        first = self.scheduler.run_allocated(
            allocations={name: 1 for name in self.config.engines},
            config=symbolic, X_train=selection.development.X,
            y_train=selection.development.y,
            X_val=selection.validation.X, y_val=selection.validation.y,
            base_seed=self.config.random_seed + cycle,
            max_retries=self.config.engine_retries,
            evaluation_budget=baseline_jobs,
            parallel=self.config.engine_workers > 1,
            max_workers=self.config.engine_workers,
            timeout_s=self.config.engine_timeout_s,
            engine_controls={})
        extras = {
            name: allocations[name] - 1 for name in self.config.engines
            if allocations[name] > 1}
        self._last_counterfactual_backbone = {
            "schema":
                "scientific-protected-counterfactual-engine-backbone-v1",
            "engines": list(self.config.engines),
            "backbone_jobs": baseline_jobs,
            "adaptive_jobs": sum(extras.values()),
            "total_jobs": self.config.engine_budget,
            "backbone_controls": "registered-engine-defaults",
            "backbone_candidates": [{
                "engine": str(row.engine),
                "expression": str(row.expression),
                "lineage_id": str(row.lineage_id),
            } for row in first.all_results],
            "adaptive_allocations": extras,
            "candidate_response_accessed": False,
            "heldout_opened": False,
        }
        if not extras:
            return first
        second = self.scheduler.run_allocated(
            allocations=extras, config=symbolic,
            X_train=selection.development.X,
            y_train=selection.development.y,
            X_val=selection.validation.X, y_val=selection.validation.y,
            base_seed=self.config.random_seed + cycle + 1000003,
            max_retries=self.config.engine_retries,
            evaluation_budget=sum(extras.values()),
            parallel=self.config.engine_workers > 1,
            max_workers=self.config.engine_workers,
            timeout_s=self.config.engine_timeout_s,
            engine_controls={
                name: controls.get(name, ()) for name in extras})
        return merge_multi_engine_results(
            (first, second), evaluation_budget=self.config.engine_budget)

    def _run_engines(self, selection: SelectionData, cycle: int) -> Any:
        symbolic = SymbolicConfig(
            niterations=self.config.search_iterations,
            mcts_max_iterations=self.config.search_iterations,
            mcts_random_seed=self.config.random_seed + cycle,
            mcts_frontier_size=self.config.mcts_frontier_size,
            mcts_score_folds=self.config.mcts_score_folds,
            expression_contract=("pcpi-closed-basis-v1" if
                self.config.refit_policy == "pcpi-closed-basis-amplitudes" else "unrestricted"),
        )
        resolved = getattr(self, "_active_research_plan", None)
        allocations = ({call.engine: call.jobs
                        for call in resolved.engine_calls}
                       if resolved is not None else {
                           name: self.config.engine_repeats
                           for name in self.config.engines})
        controls = ({call.engine: call.requested_operations
                     for call in resolved.engine_calls}
                    if resolved is not None else {})
        self._last_counterfactual_backbone = {}
        if (resolved is not None
                and self.config.protected_counterfactual_backbone):
            self._last_probe_allocation = {}
            return self._run_protected_backbone(
                selection, cycle, symbolic, allocations, controls)
        if (self.config.probe_skill_model
                and len(self.config.engines) > 1
                and self.config.engine_budget > len(self.config.engines)):
            return self._run_probe_stages(
                selection, cycle, symbolic, allocations, controls)
        self._last_probe_allocation = {}
        if resolved is None:
            return self.scheduler.run(
                engines=self.config.engines, config=symbolic,
                X_train=selection.development.X, y_train=selection.development.y,
                X_val=selection.validation.X, y_val=selection.validation.y,
                repeats=self.config.engine_repeats,
                base_seed=self.config.random_seed + cycle,
                max_retries=self.config.engine_retries,
                evaluation_budget=self.config.engine_budget,
                parallel=self.config.engine_workers > 1,
                max_workers=self.config.engine_workers,
                timeout_s=self.config.engine_timeout_s,
                engine_controls=controls)
        return self.scheduler.run_allocated(
            allocations=allocations, config=symbolic,
            X_train=selection.development.X, y_train=selection.development.y,
            X_val=selection.validation.X, y_val=selection.validation.y,
            base_seed=self.config.random_seed + cycle,
            max_retries=self.config.engine_retries,
            evaluation_budget=self.config.engine_budget,
            parallel=self.config.engine_workers > 1,
            max_workers=self.config.engine_workers,
            timeout_s=self.config.engine_timeout_s,
            engine_controls=controls,
        )

    def _discover(
        self, selection: SelectionData, engine_result: Any,
        previous: Sequence[str], task_name: str, task_description: str,
        output_dir: Path, knowledge_dir: Path, variable_metadata: Mapping[str, Any],
        cycle: int = 0, orchestration_context: Mapping[str, Any] | None = None,
        synthesized_candidates: Sequence[Mapping[str, Any]] = (),
    ) -> DiscoveryRunResult:
        seeds, seed_audit = _bounded_seed_bank(
            engine_result, previous, selection, self.config,
            synthesized_candidates)
        context = dict(orchestration_context or {})
        context["seed_bank"] = seed_audit
        discovery = discover_from_selection(
            selection=selection,
            task_name=task_name, task_description=task_description,
            base_candidates=seeds, knowledge_dir=knowledge_dir,
            hypothesis_dir=output_dir / "hypotheses",
            evidence_registry_path=output_dir / "evidence_registry.jsonl",
            config=DiscoveryConfig.from_mapping({
                "evaluation_budget": self.config.discovery_budget,
                "llm_evaluation_reserve": self.config.llm_evaluation_reserve,
                "refit_policy": self.config.refit_policy,
                "islands": self.config.discovery_islands,
                "random_seed": self.config.random_seed,
                "use_library": self.config.use_knowledge,
                "task_local_memory_read": self.config.task_local_memory,
                "task_local_memory_write": self.config.task_local_memory,
                "structure_library_topk": self.config.task_local_memory_topk,
                "max_rounds": self.config.discovery_rounds,
                "candidates_per_island": self.config.candidates_per_island,
            }),
            provider_settings=(
                None if self.config.typed_evidence_synthesis
                else self.provider_settings),
            variable_metadata=dict(variable_metadata),
            orchestration_context=context,
            refinement_enabled=True, include_generic_candidates=False,
        )
        attach_system_evidence(discovery, _engine_payload(engine_result), cycle)
        return discovery

    def _provider_abstention(self, phase, error):
        if self.config.provider_failure_mode != (
                "audited-deterministic-abstention"):
            raise error
        return {
            "phase": phase, "error_type": type(error).__name__,
            "message": str(error),
            "mode": self.config.provider_failure_mode}

    def _resolve_plan(self, planner, task_context):
        if planner.enabled and self.config.scientist_orchestration:
            try:
                plan, telemetry = planner.plan_research(
                    task_context=task_context,
                    available_engines=self.config.engines,
                    total_jobs=self.config.engine_budget)
            except (ProtocolError, ProviderInfrastructureError,
                    ScientistPlanProtocolError) as error:
                fallback = self._provider_abstention("plan", error)
                plan = deterministic_plan(
                    self.config.engines, self.config.engine_budget)
                telemetry = {"provider_failure_abstention": fallback}
                return plan, telemetry, fallback, 1
            baseline = deterministic_plan(
                self.config.engines, self.config.engine_budget)
            baseline_jobs = {
                call.engine: call.jobs for call in baseline.engine_calls}
            challenger_jobs = {
                call.engine: call.jobs for call in plan.engine_calls}
            decision = conservative_allocation_decision(
                baseline_jobs, challenger_jobs,
                self.config.conservative_allocation_policy,
                self.config.dataset_family)
            if self.config.allocation_calibration_role == "paired-challenger":
                decision = {
                    **decision,
                    "selected_allocation": challenger_jobs,
                    "challenger_certified": False,
                    "fallback_to_baseline": False,
                    "calibration_override": True,
                    "execution_role": "paired-calibration-only",
                }
            elif not decision["challenger_certified"]:
                plan = allocated_plan(baseline_jobs)
                decision = {
                    **decision,
                    "selected_allocation": baseline_jobs,
                    "controls_fallback_to_baseline": True,
                }
            else:
                decision = {
                    **decision,
                    "controls_fallback_to_baseline": False,
                }
            telemetry = {
                **dict(telemetry),
                "conservative_allocation_decision": decision}
            return plan, telemetry, None, 1
        if self.config.skill_reliability:
            allocation = allocate_bayesian_skill_jobs(
                self.config.engines, self.config.engine_budget,
                self.config.skill_reliability)
            plan = allocated_plan(allocation["allocated_jobs"])
            telemetry = {"bayesian_skill_allocation": allocation,
                "skill_policy_identity": self.config.skill_policy_identity}
            return plan, telemetry, None, 0
        return deterministic_plan(
            self.config.engines, self.config.engine_budget), {}, None, 0

    @staticmethod
    def _abstention_review(fallback):
        return ScientistReview(
            ("provider abstained; deterministic evidence retained",),
            (), (), ("retain all validated engine candidates",),
            False, "audited provider abstention"), {
                "provider_failure_abstention": fallback}

    def _resolve_review(self, planner, plan, evidence, fallback):
        if not (planner.enabled and self.config.scientist_orchestration):
            return ScientistReview(
                ("deterministic engine evidence available",), (), (),
                ("compare all registered engine candidates",), False,
                "provider-free deterministic orchestration"), {}, fallback, 0
        if fallback is not None:
            review, telemetry = self._abstention_review(fallback)
            return review, telemetry, fallback, 0
        try:
            review, telemetry = planner.review_engine_evidence(
                plan=plan, engine_evidence=evidence,
                require_typed_synthesis=self.config.typed_evidence_synthesis)
            return review, telemetry, None, 1
        except (ProtocolError, ProviderInfrastructureError,
                ScientistReviewProtocolError) as error:
            fallback = self._provider_abstention("review", error)
            review, telemetry = self._abstention_review(fallback)
            return review, telemetry, fallback, 1

    def _orchestrate_cycle(self, selection, cycle, planner, task_context):
        counters = (planner.call_count, planner.attempt_count, len(planner.errors))
        plan, plan_telemetry, fallback, plan_calls = self._resolve_plan(
            planner, task_context)
        self._active_research_plan = plan
        engines = self._run_engines(selection, cycle)
        evidence = _engine_evidence(engines)
        review, review_telemetry, fallback, review_calls = (
            self._resolve_review(planner, plan, evidence, fallback))
        orchestration = {"schema": "scientific-llm-engine-orchestration-v1",
            "scientist_state_before": task_context["scientist_state"],
            "research_plan": plan.to_dict(), "research_plan_identity": plan.stable_hash,
            "engine_evidence": evidence, "scientist_review": review.to_dict(),
            "scientist_review_identity": review.stable_hash,
            "provider_telemetry": [dict(plan_telemetry), dict(review_telemetry)],
            "protected_counterfactual_backbone": dict(getattr(
                self, "_last_counterfactual_backbone", {})),
            "provider_failure_abstention": fallback,
            "candidate_response_accessed": False, "heldout_opened": False}
        usage = (plan_calls + review_calls,
                 planner.attempt_count - counters[1],
                 len(planner.errors) - counters[2])
        return plan, review, engines, evidence, orchestration, usage

    def _compile_cycle_synthesis(self, review, evidence, n_features):
        audit = {
            "schema": "scientific-evidence-conditioned-synthesis-v1",
            "directive_count": 0, "compiled_candidate_count": 0,
            "records": [], "candidate_response_accessed": False,
            "heldout_opened": False}
        if self.config.allocation_calibration_role == "paired-challenger":
            return [], {
                **audit,
                "synthesis_unavailable": True,
                "unavailable_reason":
                    "paired-allocation-calibration-isolates-engine-scheduling",
                "candidate_admission": "disabled-by-calibration-contract",
                "engine_candidates_preserved": True}
        if not self.config.typed_evidence_synthesis:
            return [], audit
        candidates, audit = compile_evidence_synthesis(
            review.synthesis_directives, evidence, n_features)
        if not candidates:
            reason = (
                "single-engine-control-has-fewer-than-two-distinct-lineages"
                if len(self.config.engines) == 1 else
                "typed-directives-produced-no-adaptable-novel-candidate")
            return [], {
                **audit,
                "synthesis_unavailable": True,
                "unavailable_reason": reason,
                "candidate_admission": "abstain-no-synthetic-candidate-added",
                "engine_candidates_preserved": True}
        return candidates, audit

    def _cycle_record(
            self, cycle, final, selection, engines, acquisition, usage,
            plan, review, state_before, state_after, orchestration):
        telemetry = orchestration.get("provider_telemetry") or ()
        allocation = next((
            row.get("conservative_allocation_decision")
            for row in telemetry
            if isinstance(row, Mapping)
            and row.get("conservative_allocation_decision")), {})
        return DiscoveryCycle(
            cycle, final.expression, final.hypothesis.hypothesis_id,
            len(selection.development.X), _engine_payload(engines),
            acquisition, int(final.report.get("llm_call_count", 0)) + usage[0],
            int(final.report["evaluation_budget_used"]),
            int(final.report["llm_attempt_count"]) + usage[1],
            int(final.report.get("llm_error_count", 0)) + usage[2],
            plan.to_dict(), review.to_dict(), state_before, state_after,
            dict(getattr(self, "_last_probe_allocation", {})),
            dict(getattr(self, "_last_counterfactual_backbone", {})),
            dict(orchestration.get("provider_failure_abstention") or {}),
            dict(allocation or {}),
            dict(orchestration.get("evidence_synthesis") or {}),
        )

    def run(
        self, *, selection: SelectionData,
        task_name: str, task_description: str, output_dir: str | Path,
        knowledge_dir: str | Path, variable_metadata: Mapping[str, Any],
    ) -> DiscoveryAgentResult:
        output, knowledge = Path(output_dir), Path(knowledge_dir); output.mkdir(parents=True, exist_ok=True)
        previous: tuple[Mapping[str, str], ...] = ()
        history: list[DiscoveryCycle] = []
        final: DiscoveryRunResult | None = None
        planner = ProposalRuntime(
            EquationRuntime(
                selection.development.X.shape[1],
                refit_policy=self.config.refit_policy),
            selection.development.X.shape[1], self.provider_settings, 1,
            skill_reliability=self.config.skill_reliability,
            skill_policy_identity=self.config.skill_policy_identity)
        base_task_context = {"name": task_name, "description": task_description,
            "require_explicit_skill_controls":
                self.config.require_explicit_skill_controls,
            "variables": {key: value for key, value in variable_metadata.items()
                          if key in {"feature_names", "feature_units",
                                     "target_name", "target_unit"}}}
        scientist_state = ScientistState()
        for cycle in range(max(1, self.config.cycles)):
            state_before = scientist_state.to_dict()
            context = {**base_task_context, "scientist_state": state_before}
            plan, review, engines, evidence, orchestration, usage = (
                self._orchestrate_cycle(selection, cycle, planner, context))
            synthesized, synthesis_audit = self._compile_cycle_synthesis(
                review, evidence, selection.development.X.shape[1])
            orchestration = {
                **orchestration, "evidence_synthesis": synthesis_audit}
            final = self._discover(
                selection, engines, previous, task_name, task_description,
                output, knowledge, variable_metadata, cycle=cycle,
                orchestration_context=orchestration,
                synthesized_candidates=synthesized,
            )
            previous = _survivors(final.report, final.expression)
            scientist_state = scientist_state.advance(
                plan=plan, review=review, engine_evidence=evidence,
                surviving_hypotheses=previous)
            if hasattr(final, "evidence_registry_path"):
                attach_scientist_policy_evidence(
                    final, cycle, state_before, scientist_state.to_dict(),
                    plan.to_dict(), review.to_dict())
            if review.stop and cycle + 1 < self.config.cycles:
                acquisition = {"reason": "scientist_stop_condition",
                    "stop_reason": review.stop_reason,
                    "cycle_continues_without_labels": True,
                    "registered_cycles_continue": True}
            elif cycle + 1 >= self.config.cycles:
                acquisition = {"reason": "final_cycle"}
            else:
                acquisition = {
                    "reason": "canonical_p3b_acquisition_required",
                    "cycle_continues_without_labels": True}
            history.append(self._cycle_record(
                cycle, final, selection, engines, acquisition, usage,
                plan, review, state_before, scientist_state.to_dict(),
                orchestration))
        if final is None:
            raise RuntimeError("discovery agent executed no cycle")
        remaining = (len(selection.acquisition_pool.X)
                     if selection.acquisition_pool is not None else 0)
        return DiscoveryAgentResult(
            final, tuple(history), selection, remaining, self.provider_settings is not None,
            system_evaluation(history),
        )


__all__ = [
    "DiscoveryAgent", "DiscoveryAgentConfig", "DiscoveryAgentResult", "DiscoveryCycle",
]
