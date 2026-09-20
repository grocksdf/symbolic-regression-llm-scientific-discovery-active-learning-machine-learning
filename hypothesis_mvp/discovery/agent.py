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
from hypothesis_mvp.symbolic import EngineScheduler

from .api import DiscoveryRunResult, discover_from_selection
from .contracts import DiscoveryConfig
from .proposal_runtime import ProviderSettings
from .proposal_runtime import ProposalRuntime
from .equation_runtime import EquationRuntime
from .scientist_policy import (
    ResearchPlan, ScientistReview, ScientistState, deterministic_plan,
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
    llm_evaluation_reserve: int = 0
    refit_policy: str = "global-constants"
    discovery_islands: tuple[str, ...] = ("low_complexity", "nmse", "tail", "novelty")
    scientist_orchestration: bool = False


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


@dataclass(frozen=True)
class DiscoveryAgentResult:
    discovery: DiscoveryRunResult
    cycles: tuple[DiscoveryCycle, ...]
    final_selection: SelectionData
    pool_rows_remaining: int
    provider_configured: bool
    system_evaluation: Mapping[str, Any] = field(default_factory=dict)


def _survivors(report: Mapping[str, Any], fallback: str) -> tuple[str, ...]:
    expressions = [
        str(row.get("expression") or "").strip()
        for row in report.get("final_topk") or ()
        if isinstance(row, Mapping)
    ]
    values = tuple(dict.fromkeys(value for value in expressions if value))
    return values or (fallback,)


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
                       "search_score_folds", "candidate_set_method"}
        }} for row in result.all_results]


class DiscoveryAgent:
    def __init__(
        self, config: DiscoveryAgentConfig,
        provider_settings: ProviderSettings | None = None,
    ) -> None:
        self.config = config
        self.provider_settings = provider_settings
        if config.acquisition_enabled:
            raise ValueError(
                "DiscoveryAgent acquisition was removed; run the canonical P3B "
                "PCPI acquisition protocol instead"
            )
        self.scheduler = EngineScheduler()

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
                timeout_s=self.config.engine_timeout_s)
        allocations = {call.engine: call.jobs for call in resolved.engine_calls}
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
        )

    def _discover(
        self, selection: SelectionData, engine_result: Any,
        previous: Sequence[str], task_name: str, task_description: str,
        output_dir: Path, knowledge_dir: Path, variable_metadata: Mapping[str, Any],
        cycle: int = 0, orchestration_context: Mapping[str, Any] | None = None,
    ) -> DiscoveryRunResult:
        seeds = [{
            "expression": row.expression, "source": f"engine:{row.engine}",
            "lineage_id": row.lineage_id,
        } for row in engine_result.all_results]
        seeds.extend({
            "expression": expression, "source": "previous_cycle_survivor"
        } for expression in previous)
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
            }),
            provider_settings=self.provider_settings,
            variable_metadata=dict(variable_metadata),
            orchestration_context=dict(orchestration_context or {}),
            refinement_enabled=True, include_generic_candidates=True,
        )
        attach_system_evidence(discovery, _engine_payload(engine_result), cycle)
        return discovery

    def _orchestrate_cycle(self, selection, cycle, planner, task_context):
        counters = (planner.call_count, planner.attempt_count, len(planner.errors))
        if planner.enabled and self.config.scientist_orchestration:
            plan, plan_telemetry = planner.plan_research(
                task_context=task_context, available_engines=self.config.engines,
                total_jobs=self.config.engine_budget)
        else:
            plan, plan_telemetry = deterministic_plan(
                self.config.engines, self.config.engine_budget), {}
        self._active_research_plan = plan
        engines = self._run_engines(selection, cycle)
        evidence = _engine_evidence(engines)
        if planner.enabled and self.config.scientist_orchestration:
            review, review_telemetry = planner.review_engine_evidence(
                plan=plan, engine_evidence=evidence)
        else:
            review = ScientistReview(
                ("deterministic engine evidence available",), (), (),
                ("compare all registered engine candidates",), False,
                "provider-free deterministic orchestration")
            review_telemetry = {}
        orchestration = {"schema": "scientific-llm-engine-orchestration-v1",
            "scientist_state_before": task_context["scientist_state"],
            "research_plan": plan.to_dict(), "research_plan_identity": plan.stable_hash,
            "engine_evidence": evidence, "scientist_review": review.to_dict(),
            "scientist_review_identity": review.stable_hash,
            "provider_telemetry": [dict(plan_telemetry), dict(review_telemetry)],
            "candidate_response_accessed": False, "heldout_opened": False}
        usage = (planner.call_count - counters[0],
                 planner.attempt_count - counters[1],
                 len(planner.errors) - counters[2])
        return plan, review, engines, evidence, orchestration, usage

    def run(
        self, *, selection: SelectionData,
        task_name: str, task_description: str, output_dir: str | Path,
        knowledge_dir: str | Path, variable_metadata: Mapping[str, Any],
    ) -> DiscoveryAgentResult:
        output, knowledge = Path(output_dir), Path(knowledge_dir); output.mkdir(parents=True, exist_ok=True)
        previous: tuple[str, ...] = ()
        history: list[DiscoveryCycle] = []
        final: DiscoveryRunResult | None = None
        planner = ProposalRuntime(
            EquationRuntime(
                selection.development.X.shape[1],
                refit_policy=self.config.refit_policy),
            selection.development.X.shape[1], self.provider_settings, 1)
        base_task_context = {"name": task_name, "description": task_description,
            "variables": {key: value for key, value in variable_metadata.items()
                          if key in {"feature_names", "feature_units",
                                     "target_name", "target_unit"}}}
        scientist_state = ScientistState()
        for cycle in range(max(1, self.config.cycles)):
            state_before = scientist_state.to_dict()
            context = {**base_task_context, "scientist_state": state_before}
            plan, review, engines, evidence, orchestration, usage = (
                self._orchestrate_cycle(selection, cycle, planner, context))
            final = self._discover(
                selection, engines, previous, task_name, task_description,
                output, knowledge, variable_metadata, cycle=cycle,
                orchestration_context=orchestration,
            )
            previous = _survivors(final.report, final.expression)
            scientist_state = scientist_state.advance(
                plan=plan, review=review, engine_evidence=evidence,
                surviving_hypotheses=previous)
            if hasattr(final, "evidence_registry_path"):
                attach_scientist_policy_evidence(
                    final, cycle, state_before, scientist_state.to_dict(),
                    plan.to_dict(), review.to_dict())
            acquisition = (
                {"reason": "final_cycle"}
                if cycle + 1 >= self.config.cycles
                else {
                    "reason": "canonical_p3b_acquisition_required",
                    "cycle_continues_without_labels": True,
                }
            )
            history.append(DiscoveryCycle(
                cycle, final.expression, final.hypothesis.hypothesis_id,
                len(selection.development.X), _engine_payload(engines),
                acquisition, int(final.report.get("llm_call_count", 0)) + usage[0],
                int(final.report["evaluation_budget_used"]),
                int(final.report["llm_attempt_count"]) + usage[1],
                int(final.report.get("llm_error_count", 0)) + usage[2],
                plan.to_dict(), review.to_dict(),
                state_before, scientist_state.to_dict(),
            ))
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
