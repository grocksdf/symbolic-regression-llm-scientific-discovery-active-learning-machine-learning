"""Single production agent for multi-engine and LLM discovery orchestration.

Measured-pool acquisition is owned exclusively by :mod:`hypothesis_mvp.pcpi`
and its P3B runner.  This agent deliberately cannot select or reveal pool
labels, so it cannot become a second acquisition implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.data import SelectionData
from hypothesis_mvp.data.roles import RoleDataset
from hypothesis_mvp.symbolic import EngineScheduler, merge_multi_engine_results

from .api import DiscoveryRunResult, discover_from_selection
from .contracts import DiscoveryConfig
from .proposal_runtime import (
    ProtocolError, ProviderInfrastructureError, ProviderSettings, ProposalRuntime,
    ScientistPlanProtocolError, ScientistReviewProtocolError,
)
from .equation_runtime import EquationRuntime
from .evidence_synthesis import compile_evidence_synthesis
from .candidate_region_expansion import FrozenAxisRegions
from .pcpi_adapter import (freeze_discovery_model, freeze_discovery_target,
    structural_terms, EXPANDED_FORMULA_POLICY, model_factory_for_policy,
    support_parser_for_policy)
from .posterior_gap_diagnosis import diagnose_frozen_bank
from .posterior_gap_evidence import screen_independent_adequacy
from .regional_candidate_admission import admit_regional_candidates
from .knowledge_runtime import KnowledgeRuntime
from .iterative_refinement_gate import verify_iterative_refinement
from hypothesis_mvp.data.roles import DataRole
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
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
    new_skeleton_quota: int = 0
    task_local_memory_topk: int = 8
    llm_evaluation_reserve: int = 0
    refit_policy: str = "global-constants"
    discovery_islands: tuple[str, ...] = ("low_complexity", "nmse", "tail", "novelty")
    scientist_orchestration: bool = False
    require_explicit_skill_controls: bool = False
    typed_evidence_synthesis: bool = False
    typed_inner_augmentation: bool = False
    expanded_formula_synthesis: bool = False
    posterior_gap_directed: bool = False
    iterative_posterior_refinement: bool = False
    synthesis_evaluation_reserve: int = 0
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
    posterior_gap_audit: Mapping[str, Any] = field(default_factory=dict)
    posterior_gap_brief: Mapping[str, Any] = field(default_factory=dict)


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


def _check_cycle_roles(selection: SelectionData, roles, cycles: int) -> None:
    """Validate every adaptive response role before engine or provider work."""
    if selection.acquisition_pool is None:
        raise ValueError("iterative gap requires frozen action covariates")
    if not isinstance(roles, (tuple, list)) or len(roles) != cycles:
        raise ValueError("one fresh gap/admission pair required per cycle")
    datasets = [selection.development, selection.validation]
    for pair in roles:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(not isinstance(row, RoleDataset)
                       or row.role is not DataRole.VALIDATION for row in pair)):
            raise ValueError("each cycle requires two validation response roles")
        datasets.extend(pair)
    # Also exclude the same covariates with altered response values. The
    # selection and audit roles cannot be recycled under a changed label.
    observed = set()
    for data in datasets:
        keys = {sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()
                for x in data.X}
        if keys & observed:
            raise ValueError("iterative gap/admission roles must be row-disjoint")
        observed.update(keys)


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
             - (config.synthesis_evaluation_reserve
                if config.typed_inner_augmentation else 0)
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
    synthesis_rows = ([] if config.typed_inner_augmentation else
                      [dict(row) for row in synthesized])
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
        "input_evidence_synthesis_candidates": len(synthesized),
        "selected_evidence_synthesis_candidates": sum(
            row.get("source") == "llm_evidence_synthesis" for row in selected),
        "synthesis_deferred_to_separate_phase": config.typed_inner_augmentation,
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
        if config.expanded_formula_synthesis and (
                not config.typed_inner_augmentation
                or config.refit_policy != "pcpi-expanded-fixed-inner-v1"
                or config.provider_failure_mode != "abort"):
            raise ValueError("expanded formula synthesis requires explicit "
                             "typed augmentation, fixed-inner refit and provider abort")
        if config.typed_inner_augmentation:
            if (not config.typed_evidence_synthesis
                    or not config.scientist_orchestration
                    or provider_settings is None
                    or not provider_settings.routes
                    or type(config.synthesis_evaluation_reserve) is not int
                    or config.synthesis_evaluation_reserve < 1
                    or type(config.llm_evaluation_reserve) is not int
                    or config.llm_evaluation_reserve < 1
                    or config.discovery_budget <= (
                        config.synthesis_evaluation_reserve
                        + config.llm_evaluation_reserve
                        + len(config.discovery_islands) + len(config.engines) + 2)):
                raise ValueError(
                    "typed inner augmentation requires a provider, Scientist, "
                    "typed synthesis, and separate positive evaluation budgets")
        elif config.synthesis_evaluation_reserve:
            raise ValueError(
                "synthesis evaluation reserve requires typed inner augmentation")
        if config.posterior_gap_directed and not config.typed_inner_augmentation:
            raise ValueError("posterior gap requires typed inner augmentation")
        if config.iterative_posterior_refinement and (
                not config.posterior_gap_directed or config.cycles < 2
                or config.task_local_memory or config.use_knowledge):
            raise ValueError("iterative posterior refinement requires separate gap roles and no memory")
        if (config.iterative_posterior_refinement
                and config.expanded_formula_synthesis
                and config.new_skeleton_quota < 1):
            raise ValueError("iterative expanded refinement requires a positive new skeleton quota")
        if (type(config.new_skeleton_quota) is not int
                or config.new_skeleton_quota < 0
                or config.new_skeleton_quota > config.candidates_per_island
                or (config.new_skeleton_quota > 0
                    and config.refit_policy != "pcpi-expanded-fixed-inner-v1")):
            raise ValueError("new skeleton quota requires expanded contract and candidate slots")
        if (config.new_skeleton_quota > 0
                and config.llm_evaluation_reserve <
                config.new_skeleton_quota * len(config.discovery_islands)
                * config.discovery_rounds):
            raise ValueError("new skeleton evaluation reserve cannot cover all requested batches")
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
        context["typed_inner_augmentation"] = self.config.typed_inner_augmentation
        gap_allowed = (not self.config.posterior_gap_directed or bool(
            context.get("posterior_gap_brief", {}).get("propose_allowed")))
        synthesis_budget = (self.config.synthesis_evaluation_reserve
                            if self.config.typed_inner_augmentation and gap_allowed else 0)
        llm_budget = self.config.llm_evaluation_reserve if gap_allowed else 0
        evaluation_budget = (self.config.discovery_budget if gap_allowed else
            self.config.discovery_budget - self.config.synthesis_evaluation_reserve
            - self.config.llm_evaluation_reserve)
        discovery = discover_from_selection(
            selection=selection,
            task_name=task_name, task_description=task_description,
            base_candidates=seeds, knowledge_dir=knowledge_dir,
            hypothesis_dir=output_dir / "hypotheses",
            evidence_registry_path=output_dir / "evidence_registry.jsonl",
            config=DiscoveryConfig.from_mapping({
                "evaluation_budget": evaluation_budget,
                "llm_evaluation_reserve": llm_budget,
                "synthesis_evaluation_reserve": synthesis_budget,
                "refit_policy": self.config.refit_policy,
                "islands": self.config.discovery_islands,
                "random_seed": self.config.random_seed,
                "use_library": self.config.use_knowledge,
                "task_local_memory_read": self.config.task_local_memory,
                "task_local_memory_write": self.config.task_local_memory,
                "structure_library_topk": self.config.task_local_memory_topk,
                "max_rounds": self.config.discovery_rounds,
                "candidates_per_island": self.config.candidates_per_island,
                "new_skeleton_quota": self.config.new_skeleton_quota,
            }),
            provider_settings=(self.provider_settings if gap_allowed and (
                not self.config.typed_evidence_synthesis
                or self.config.typed_inner_augmentation) else None),
            supplemental_candidates=(
                synthesized_candidates if synthesis_budget
                else ()),
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
        if self.config.posterior_gap_directed:
            # Keep every engine job and seed paired with the provider-free arm.
            return deterministic_plan(self.config.engines,
                self.config.engine_budget), {"engine_schedule":
                    "frozen-deterministic-quality-first"}, None, 0
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

    def _resolve_review(
            self, planner, plan, evidence, fallback,
            scientific_context=None):
        if not (planner.enabled and self.config.scientist_orchestration):
            return ScientistReview(
                ("deterministic engine evidence available",), (), (),
                ("compare all registered engine candidates",), False,
                "provider-free deterministic orchestration"), {}, fallback, 0
        if fallback is not None:
            review, telemetry = self._abstention_review(fallback)
            return review, telemetry, fallback, 0
        try:
            review_kwargs = {
                "plan": plan, "engine_evidence": evidence,
                "require_typed_synthesis": self.config.typed_evidence_synthesis}
            if self.config.typed_inner_augmentation:
                review_kwargs["allow_interactions"] = True
            if self.config.expanded_formula_synthesis:
                review_kwargs["expanded_formula_synthesis"] = True
            if scientific_context:
                review_kwargs["scientific_context"] = scientific_context
            review, telemetry = planner.review_engine_evidence(**review_kwargs)
            return review, telemetry, None, 1
        except (ProtocolError, ProviderInfrastructureError,
                ScientistReviewProtocolError) as error:
            fallback = self._provider_abstention("review", error)
            review, telemetry = self._abstention_review(fallback)
            return review, telemetry, fallback, 1

    def _orchestrate_cycle(self, selection, cycle, planner, task_context,
                           gap_audit=None, gap_prior=None,
                           gap_measurement_budget=0, admitted_rows=()):
        counters = (planner.call_count, planner.attempt_count, len(planner.errors))
        plan, plan_telemetry, fallback, plan_calls = self._resolve_plan(
            planner, task_context)
        self._active_research_plan = plan
        engines = (self._iterative_frozen_engines
                   if self.config.iterative_posterior_refinement and cycle
                   else self._run_engines(selection, cycle))
        if self.config.iterative_posterior_refinement and cycle == 0:
            self._iterative_frozen_engines = engines
        evidence = _engine_evidence(engines)
        gap = (self._independent_gap_brief(
            selection, engines, gap_audit, gap_prior,
            gap_measurement_budget,
            **({"admitted_rows": admitted_rows}
               if self.config.iterative_posterior_refinement else {}))
            if self.config.posterior_gap_directed else None)
        if self.config.iterative_posterior_refinement:
            self._last_cycle_gap = gap
        if gap is not None and not gap["prompt"]["propose_allowed"]:
            review = ScientistReview(
                ("independent region screen did not identify inadequacy",),
                (), (), ("retain engine candidates",), False,
                "gap-directed proposal abstained")
            review_telemetry, review_calls = {"gap_abstention": True}, 0
        else:
            review_context = (None if gap is None else {
                "posterior_gap_brief": gap["prompt"],
                "posterior_gap_existing_supports": gap["existing_supports"],
                "candidate_response_accessed": False,
                "heldout_opened": False,
            })
            review, review_telemetry, fallback, review_calls = (
                self._resolve_review(
                    planner, plan, evidence, fallback,
                    scientific_context=review_context))
        orchestration = {"schema": "scientific-llm-engine-orchestration-v1",
            "scientist_state_before": task_context["scientist_state"],
            "research_plan": plan.to_dict(), "research_plan_identity": plan.stable_hash,
            "engine_evidence": evidence, "scientist_review": review.to_dict(),
            "scientist_review_identity": review.stable_hash,
            "provider_telemetry": [dict(plan_telemetry), dict(review_telemetry)],
            "protected_counterfactual_backbone": dict(getattr(
                self, "_last_counterfactual_backbone", {})),
            "provider_failure_abstention": fallback,
            **({"posterior_gap_brief": gap["prompt"],
                "posterior_gap_existing_supports": gap["existing_supports"],
                "posterior_gap_audit": gap["audit"]} if gap else {}),
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
            review.synthesis_directives, evidence, n_features,
            allow_interactions=self.config.typed_inner_augmentation,
            coefficient_policy=(EXPANDED_FORMULA_POLICY
                if self.config.expanded_formula_synthesis else
                "discard-fitted-coefficients-refit-closed-basis"))
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
            dict(orchestration.get("posterior_gap_audit") or {}),
            dict(orchestration.get("posterior_gap_brief") or {}),
        )

    def _independent_gap_brief(
        self, selection: SelectionData, engines: Any,
        audit: RoleDataset, prior: NormalInverseGammaPrior,
        measurement_budget: int,
        admitted_rows: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        if (type(measurement_budget) is not int or measurement_budget < 1
                or selection.acquisition_pool is None):
            raise ValueError("posterior gap requires registered budget and pool covariates")
        features = selection.development.X.shape[1]
        coefficient_policy = (EXPANDED_FORMULA_POLICY
            if self.config.expanded_formula_synthesis else
            "discard-fitted-coefficients-refit-closed-basis")
        parser = support_parser_for_policy(coefficient_policy)
        model_factory = model_factory_for_policy(coefficient_policy)
        engine_base = self.config.discovery_budget - (
            self.config.synthesis_evaluation_reserve
            + self.config.llm_evaluation_reserve)
        source_rows = [{"expression": row.expression,
                        "source": f"engine:{row.engine}"}
                       for row in engines.all_results]
        score_reference_rows = generic_deterministic_candidates(
            selection.development.X, selection.development.y)
        source_rows.extend(score_reference_rows)
        reference_rows, reference_supports = [], set()
        for row in score_reference_rows:
            try:
                support = parser(row["expression"], features)
            except (SyntaxError, ValueError):
                continue
            if support not in reference_supports:
                reference_rows.append(row)
                reference_supports.add(support)
        seen: set[tuple[str, ...]] = set()
        candidates = []
        for row in source_rows:
            try:
                supports = parser(row["expression"], features)
            except (SyntaxError, ValueError):
                continue
            if supports in seen:
                continue
            candidates.append(row)
            seen.add(supports)
            if len(candidates) >= engine_base:
                break
        for row in admitted_rows:
            support = parser(str(row["expression"]), features)
            if support in seen:
                raise ValueError("admitted structure collided with frozen engine bank")
            candidates.append(dict(row))
            seen.add(support)
        identity = sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()
        model = model_factory(
            candidates, n_features=features, prior=prior,
            exploration_identity=identity,
            coefficient_policy=coefficient_policy)
        domain = selection.acquisition_pool.X
        target = freeze_discovery_target(
            model, selection.development, domain,
            measurement_budget=measurement_budget,
            expected_model_identity=model.stable_hash)
        reference_model = reference_target = None
        if reference_rows:
            reference_identity = sha256(json.dumps(
                reference_rows, sort_keys=True).encode()).hexdigest()
            reference_model = model_factory(
                reference_rows, n_features=features, prior=prior,
                exploration_identity=reference_identity,
                coefficient_policy=coefficient_policy,
                minimum_supports=1)
            reference_target = freeze_discovery_target(
                reference_model, selection.development, domain,
                measurement_budget=measurement_budget,
                expected_model_identity=reference_model.stable_hash)
        cuts = (float(np.median(domain[:, 0])),)
        regions = FrozenAxisRegions(features, 0, cuts)
        diagnosis = diagnose_frozen_bank(
            model, target, domain, np.full(len(domain), 1. / len(domain)),
            regions)
        evidence = screen_independent_adequacy(
            model, target, diagnosis, regions, audit,
            discovery_development=selection.development,
            discovery_validation=selection.validation,
            alpha=(.05 / self.config.cycles
                   if self.config.iterative_posterior_refinement else .05),
            score_reference_model=reference_model,
            score_reference_target=reference_target)
        return {"prompt": evidence.prompt_brief(diagnosis),
                "existing_supports": [list(row) for row in sorted(seen)],
                "model": model, "target": target, "core_rows": candidates,
                "regions": regions,
                "audit": {"schema": "independent-posterior-gap-screen-v1",
                          "fit_identity": selection.development.fingerprint,
                          "selection_identity": selection.validation.fingerprint,
                          "target_identity": target.stable_hash,
                          "score_reference_identity": (
                              reference_model.stable_hash if reference_model else ""),
                          "audit_identity": audit.fingerprint,
                          "audit_row_count": int(evidence.audit_row_count),
                          "model_identity": model.stable_hash,
                          "gap_bank_inputs": ("frozen_engine_plus_prior_admitted"
                              if self.config.iterative_posterior_refinement else
                              "current_cycle_engine_rows_only"),
                          "prior_cycle_admitted_candidates_used": bool(admitted_rows),
                          **({"iterative_gate_pending": True}
                             if self.config.iterative_posterior_refinement else
                             {"iterative_posterior_refinement_verified": False}),
                          "undefined_row_count": int(
                              evidence.undefined_row_count),
                          "undefined_row_meaning": (
                              "audit rows outside the registered action "
                              "domain where a banked structure is singular; "
                              "excluded from the numeric screen and counted, "
                              "never scored as ordinary observations"),
                          "rows": [asdict(row) for row in evidence.rows],
                          "candidate_response_accessed": False,
                          "heldout_opened": False}}

    def run(
        self, *, selection: SelectionData,
        task_name: str, task_description: str, output_dir: str | Path,
        knowledge_dir: str | Path, variable_metadata: Mapping[str, Any],
        gap_audit: RoleDataset | None = None,
        gap_prior: NormalInverseGammaPrior | None = None,
        gap_measurement_budget: int = 0,
        gap_cycle_roles: Sequence[tuple[RoleDataset, RoleDataset]] | None = None,
    ) -> DiscoveryAgentResult:
        iterative = self.config.iterative_posterior_refinement
        if iterative:
            if gap_audit is not None:
                raise ValueError("iterative mode forbids a reused single gap audit")
            _check_cycle_roles(selection, gap_cycle_roles, self.config.cycles)
            if (not isinstance(gap_prior, NormalInverseGammaPrior)
                    or type(gap_measurement_budget) is not int
                    or gap_measurement_budget < 1):
                raise ValueError("iterative mode requires a registered prior and budget")
        output = Path(output_dir)
        knowledge = Path(knowledge_dir)
        if iterative and output.exists() and any(output.iterdir()):
            raise ValueError("iterative output already exists; provider calls cannot be replayed")
        output.mkdir(parents=True, exist_ok=True)
        previous: tuple[Mapping[str, str], ...] = ()
        admitted_rows: list[dict[str, Any]] = []
        attempted_lineages: set[str] = set()
        feedback_trace: list[dict[str, Any]] = []
        expected_next_bank = expected_next_target = None
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
        if self.config.posterior_gap_directed and not iterative and (
                gap_audit is None or gap_prior is None
                or gap_measurement_budget < 1):
            raise ValueError("posterior gap needs independent rows, prior and budget")
        for cycle in range(max(1, self.config.cycles)):
            cycle_audit = gap_cycle_roles[cycle][0] if iterative else gap_audit
            cycle_admission = gap_cycle_roles[cycle][1] if iterative else None
            state_before = scientist_state.to_dict()
            context = {**base_task_context, "scientist_state": state_before}
            plan, review, engines, evidence, orchestration, usage = (
                self._orchestrate_cycle(
                    selection, cycle, planner, context, cycle_audit,
                    gap_prior, gap_measurement_budget,
                    admitted_rows=tuple(admitted_rows) if iterative else ()))
            gap = self._last_cycle_gap if iterative else None
            if iterative and (gap is None or (
                    expected_next_bank is not None and (
                        gap["model"].stable_hash != expected_next_bank or
                        gap["target"].stable_hash != expected_next_target))):
                raise ValueError("next gap did not use the admitted posterior bank")
            synthesized, synthesis_audit = self._compile_cycle_synthesis(
                review, evidence, selection.development.X.shape[1])
            if (self.config.posterior_gap_directed
                    and not orchestration["posterior_gap_brief"]["propose_allowed"]):
                synthesized = []
                synthesis_audit = {**synthesis_audit,
                    "deferred_no_independent_gap": True}
            orchestration = {
                **orchestration, "evidence_synthesis": synthesis_audit}
            final = self._discover(
                selection, engines, previous, task_name, task_description,
                output, knowledge, variable_metadata, cycle=cycle,
                orchestration_context=orchestration,
                synthesized_candidates=synthesized,
            )
            if iterative:
                parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY
                    if self.config.expanded_formula_synthesis else
                    "discard-fitted-coefficients-refit-closed-basis")
                policy = (EXPANDED_FORMULA_POLICY
                    if self.config.expanded_formula_synthesis else
                    "discard-fitted-coefficients-refit-closed-basis")
                optional = []
                for row in final.report.get("evaluated_hypothesis_bank", ()):
                    if row.get("origin") != "llm":
                        continue
                    lineage = str(row.get("lineage_id") or "")
                    if not lineage:
                        raise ValueError("iterative candidate lacks an auditable lineage")
                    if lineage not in attempted_lineages:
                        optional.append(dict(row))
                        attempted_lineages.add(lineage)
                if len(optional) > (self.config.llm_evaluation_reserve
                                    + self.config.synthesis_evaluation_reserve):
                    raise ValueError("iterative candidate attempt budget exceeded")
                eligible = tuple(sorted({int(r["region"]) for r in
                    gap["prompt"].get("regions", ())}))
                if optional and eligible and gap["prompt"].get("propose_allowed"):
                    retained, admission_report = admit_regional_candidates(
                        gap["core_rows"], optional, selection.development,
                        cycle_audit, cycle_admission,
                        selection.acquisition_pool.X, gap["regions"], eligible,
                        gap_prior, gap_measurement_budget,
                        alpha=.05 / self.config.cycles,
                        coefficient_policy=policy)
                else:
                    retained, admission_report = [], {
                        "attempts": len(optional), "admitted": 0,
                        "reason": "no-independent-gap-or-no-new-candidate",
                        "admission_identity": cycle_admission.fingerprint}
                retained_supports = [parser(str(row["expression"]),
                    selection.development.X.shape[1]) for row in retained]
                if len(set(retained_supports)) != len(retained_supports):
                    raise ValueError("duplicate admitted support in adaptive cycle")
                stage_ids = final.report.get("knowledge_stage_ids", ())
                if stage_ids:
                    stage_store = KnowledgeRuntime(
                        knowledge / "structure_library.jsonl",
                        knowledge / "runtime_ledger.jsonl")
                    for stage_id in stage_ids:
                        stage_store.finalize_gap_stage(
                            stage_id,
                            admission_identity=cycle_admission.fingerprint,
                            target_identity=gap["target"].stable_hash,
                            candidate_records=admission_report.get("candidates", ()))
                admitted_rows.extend(dict(row) for row in retained)
                after_rows = [*gap["core_rows"], *retained]
                next_identity = sha256(json.dumps(after_rows,
                    sort_keys=True, allow_nan=False).encode()).hexdigest()
                after_model = model_factory_for_policy(policy)(
                    after_rows, n_features=selection.development.X.shape[1],
                    prior=gap_prior, exploration_identity=next_identity,
                    coefficient_policy=policy)
                after_target = freeze_discovery_target(
                    after_model, selection.development, selection.acquisition_pool.X,
                    measurement_budget=gap_measurement_budget,
                    expected_model_identity=after_model.stable_hash)
                expected_next_bank = after_model.stable_hash
                expected_next_target = after_target.stable_hash
                map_id = after_target.initial_posterior.map_structure_id
                map_expressions = sorted(expression for _, expression, identifier
                                         in after_model.candidate_bindings
                                         if identifier == map_id)
                if not map_expressions:
                    raise ValueError("posterior MAP structure lacks a bank expression")
                support_key = lambda row: json.dumps(parser(
                    str(row["expression"]), selection.development.X.shape[1]))
                brief_identity = sha256(json.dumps(gap["prompt"],
                    sort_keys=True, allow_nan=False).encode()).hexdigest()
                proposal_audits = [island.get("memory", {})
                    for round_row in final.report.get("llm_rounds", ())
                    for island in round_row.get("islands", ())]
                if gap["prompt"].get("propose_allowed"):
                    if not proposal_audits or any(
                            row.get("posterior_gap_identity") != brief_identity
                            for row in proposal_audits):
                        raise ValueError("proposal batch was not bound to current gap")
                payload_identity = (brief_identity if not proposal_audits else
                    proposal_audits[0]["posterior_gap_identity"])
                feedback_trace.append({
                    "bank_before": gap["model"].stable_hash,
                    "core_rows_before": [dict(r) for r in gap["core_rows"]],
                    "posterior_before": gap["target"].stable_hash,
                    "gap_bank": gap["model"].stable_hash,
                    "gap_posterior": gap["target"].stable_hash,
                    "gap_identity": brief_identity,
                    "proposal_gap_identity": payload_identity,
                    "bank_after": after_model.stable_hash,
                    "posterior_after": after_target.stable_hash,
                    "bank_supports_after": [support_key(r) for r in after_rows],
                    "bank_rows_after": [dict(r) for r in after_rows],
                    "posterior_map_structure_id": map_id,
                    "posterior_map_expression": map_expressions[0],
                    "admitted_supports": [support_key(r) for r in retained],
                    "fit_rows": sorted(selection.development.row_fingerprints),
                    "selection_rows": sorted(selection.validation.row_fingerprints),
                    "gap_rows": sorted(cycle_audit.row_fingerprints),
                    "admission_rows": sorted(cycle_admission.row_fingerprints),
                    "admission_report": admission_report,
                    "attempted_candidate_count": len(optional),
                    "proposal_batch_count": len(proposal_audits),
                    "cumulative_admitted_count": len(admitted_rows),
                    "conditional_bank_posterior_only": True,
                })
            previous = (tuple(dict(row) for row in admitted_rows)
                        if iterative else _survivors(final.report, final.expression))
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
        feedback_gate = (verify_iterative_refinement(feedback_trace)
                         if iterative else None)
        return DiscoveryAgentResult(
            final, tuple(history), selection, remaining, self.provider_settings is not None,
            {**system_evaluation(history),
             "iterative_posterior_refinement_verified": bool(
                 feedback_gate and feedback_gate["passed"]),
             **({"iterative_feedback_trace": feedback_trace,
                 "iterative_feedback_gate": feedback_gate} if iterative else {}),
             "posterior_feedback_boundary": ("bank-conditional feedback only; "
                 "measured executor remains unauthorized" if iterative else
                 "regional admission runs after discovery cycles; "
                 "subsequent gaps cannot include admitted LLM candidates")},
        )


__all__ = [
    "DiscoveryAgent", "DiscoveryAgentConfig", "DiscoveryAgentResult", "DiscoveryCycle",
]
