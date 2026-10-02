"""True multi-engine Scientist searcher for the frozen AISTATS DRR protocol."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Mapping

from hypothesis_mvp.data import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior

from ._mainline import load
from .drr_adapter import (
    candidate_rows, candidate_rows_operational_qd, evaluate_drr_candidates,
    evaluate_drr_prefix_candidates, split_three_arm_training_samples,
    split_training_samples,
)
from .searcher import PCPISearcher, ProviderSettings


_agent = load("agent")
_bank = load("bank_selection")


class DRRBenchmarkSearcher(PCPISearcher):
    """Run the canonical DiscoveryAgent, then issue one response-free DRR certificate."""

    def __init__(self, *args, condition: str, agent_config: Mapping[str, Any],
                 portfolio_method: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition = str(condition)
        values = dict(agent_config)
        values["engines"] = tuple(values["engines"])
        values["discovery_islands"] = tuple(values["discovery_islands"])
        values["random_seed"] = self.random_seed
        self.agent_config = _agent.DiscoveryAgentConfig(**values)
        methods = {
            "v5": _bank.PORTFOLIO_CAPACITY_METHOD,
            "v6": _bank.DECISION_RISK_CAPACITY_METHOD}
        if portfolio_method not in methods:
            raise ValueError("unknown DRR portfolio method")
        self.portfolio_method = methods[portfolio_method]

    def discover(self, task):
        self._task_counter += 1
        three_arm = self.condition.startswith("three_arm_")
        roles = (
            split_three_arm_training_samples(
                task.samples, task_name=str(task.name), seed=self.random_seed)
            if three_arm else
            split_training_samples(
                task.samples, task_name=str(task.name), seed=self.random_seed))
        selection = SelectionData(
            RoleDataset(
                DataRole.DEVELOPMENT,
                roles.X_development, roles.y_development),
            RoleDataset(
                DataRole.VALIDATION,
                roles.X_validation, roles.y_validation),
            (AcquisitionCovariates(
                DataRole.ACQUISITION_POOL, roles.X_actions,
                covariate_fingerprint(roles.X_actions))
             if three_arm else None), ())
        provider = None
        if self.llm_enabled:
            provider = ProviderSettings.from_environment(**{
                "base_url": self.llm_api_url, "model": self.llm_model,
                "api_key": self.llm_api_key, "attempts": 3,
                "connect_timeout_s": 15.0,
                "read_timeout_s": self.llm_timeout_s,
                "retry_backoff_s": 4.0,
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens,
                "thinking_type": self.llm_thinking_type,
                "reasoning_effort": self.llm_reasoning_effort,
                "do_sample": self.llm_do_sample})
        dataset_family = str(
            os.environ.get("PCPI_TASK_FAMILY")
            or self.agent_config.dataset_family)
        agent_config = replace(
            self.agent_config, dataset_family=dataset_family)
        agent = _agent.DiscoveryAgent(
            agent_config, provider_settings=provider)
        symbols, descs, props = self._input_fields(
            task, roles.X_development.shape[1])
        metadata = {
            "feature_names": symbols, "feature_units": props,
            "target_name": str(task.name), "target_unit": "benchmark"}
        knowledge_dir = self.output_dir / "scientist_knowledge"
        if three_arm and knowledge_dir.exists() and any(
                knowledge_dir.iterdir()):
            raise ValueError(
                "three-arm knowledge namespace must start empty")
        result = agent.run(
            selection=selection, task_name=str(task.name),
            task_description=str(task.desc or ""),
            output_dir=self.output_dir / "scientist_agent",
            knowledge_dir=knowledge_dir,
            variable_metadata=metadata,
            gap_audit=(RoleDataset(
                DataRole.VALIDATION, roles.X_gap_audit, roles.y_gap_audit)
                if self.condition == "three_arm_l_gap_v1" else None),
            gap_prior=(NormalInverseGammaPrior()
                if self.condition == "three_arm_l_gap_v1" else None),
            gap_measurement_budget=(2
                if self.condition == "three_arm_l_gap_v1" else 0))
        report = dict(result.discovery.report)
        protected = [
            candidate
            for cycle in result.cycles
            for candidate in cycle.counterfactual_backbone.get(
                "backbone_candidates", ())
        ]
        if three_arm:
            candidates = candidate_rows(
                report, result.discovery.expression,
                roles.X_development.shape[1], protected)
            qd_audit = {
                "schema": "scientific-three-arm-generation-only-v1",
                "evaluated": False,
                "reason": "inference-and-report-responses-remain-sealed",
                "candidate_response_accessed": False,
                "heldout_opened": False,
            }
            readiness = {
                "schema": "scientific-three-arm-generation-readiness-v1",
                "indicator": 0, "ready": False,
                "generation_completed": True,
                "candidate_response_accessed": False,
                "test_or_ood_accessed": False,
                "heldout_opened": False,
            }
            prefix_curve = {
                "schema": "scientific-three-arm-prefix-not-evaluated-v1",
                "candidate_response_accessed": False,
                "heldout_opened": False,
            }
            alternate = None
        else:
            candidates, qd_audit = candidate_rows_operational_qd(
                report, result.discovery.expression, roles,
                task_name=str(task.name), seed=self.random_seed,
                protected_expressions=protected)
            readiness = evaluate_drr_candidates(
                candidates, roles, condition=self.condition,
                task_name=str(task.name), seed=self.random_seed,
                selection_method=self.portfolio_method)
            alternate = None
            prefix_curve = evaluate_drr_prefix_candidates(
                candidates, roles, condition=self.condition,
                task_name=str(task.name), seed=self.random_seed,
                selection_method=self.portfolio_method)
        if not three_arm and self.condition == "full_scientist_v6":
            alternate = evaluate_drr_candidates(
                candidates, roles, condition="entropy_portfolio_v5",
                task_name=str(task.name), seed=self.random_seed,
                selection_method=_bank.PORTFOLIO_CAPACITY_METHOD)
        logical_calls = sum(
            int(cycle.provider_calls) for cycle in result.cycles)
        provider_attempts = sum(
            int(cycle.provider_attempts) for cycle in result.cycles)
        report.update({
            "drr_readiness": (
                readiness if isinstance(readiness, dict)
                else readiness.to_dict()),
            "drr_alternate_readiness": (
                alternate.to_dict() if alternate is not None else None),
            "drr_condition": self.condition,
            "drr_portfolio_method": self.portfolio_method,
            "drr_candidate_rows": candidates,
            "drr_operational_qd": qd_audit,
            "drr_prefix_curve": (
                prefix_curve if isinstance(prefix_curve, dict)
                else prefix_curve.to_dict()),
            "drr_role_row_indices": {
                key: list(value)
                for key, value in roles.role_row_indices.items()},
            "three_arm_role_protocol": (
                "sha256-disjoint-nine-role-v1" if three_arm else None),
            "knowledge_namespace": (
                str(knowledge_dir.resolve()) if three_arm else ""),
            "knowledge_namespace_started_empty": bool(three_arm),
            "knowledge_cross_arm_read_enabled": False,
            "candidate_response_accessed": False,
            "action_response_accessed": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
            "scientist_agent_cycle_count": len(result.cycles),
            "scientist_agent_logical_llm_call_count": logical_calls,
            "scientist_agent_provider_attempt_count": provider_attempts,
            "scientist_agent_counterfactual_backbones": [
                dict(cycle.counterfactual_backbone)
                for cycle in result.cycles],
            "scientist_agent_engine_reports": [
                dict(cycle.engine_report) for cycle in result.cycles],
            "scientist_agent_provider_abstentions": [
                dict(cycle.provider_failure_abstention)
                for cycle in result.cycles
                if cycle.provider_failure_abstention],
            "scientist_agent_conservative_allocation_decisions": [
                dict(cycle.conservative_allocation_decision)
                for cycle in result.cycles
                if cycle.conservative_allocation_decision],
            "scientist_agent_synthesis_audits": [
                dict(cycle.evidence_synthesis)
                for cycle in result.cycles],
            "scientist_agent_dataset_family": dataset_family,
            "scientist_agent_system_evaluation":
                dict(result.system_evaluation),
        })
        self._write_artifact(str(task.name), report)
        return [self._result(
            task, result.discovery.expression, report,
            roles.X_development.shape[1])]

__all__ = ["DRRBenchmarkSearcher"]
