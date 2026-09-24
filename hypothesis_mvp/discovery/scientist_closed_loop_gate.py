"""Response-free correctness gate for the task-local Scientist closed loop."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Mapping

from .agent import DiscoveryAgentConfig
from .contracts import DiscoveryConfig, LineageStep
from .knowledge_runtime import KnowledgeRuntime
from .scientific_runtime import ScientificDiscoveryRuntime
from .system_executor import validate_system_registration
from hypothesis_mvp.symbolic.scheduler import _expand_job_controls


def _memory_fixture() -> Mapping[str, Any]:
    metrics_before = {
        "val_nmse": 1.0, "val_p99": 1.0, "val_strict": 1.0,
        "complexity": 1.0,
    }
    metrics_after = {
        "val_nmse": 0.5, "val_p99": 0.5, "val_strict": 0.5,
        "complexity": 2.0, "stress_nmse": 0.6, "stress_p99": 0.6,
        "stress_strict": 0.6, "ood_proxy_nmse": 0.7,
        "ood_proxy_strict": 0.7, "ood_stability_penalty": 0.1,
    }
    step = LineageStep(
        "llm", 1, "balanced", "lineage", "parent-lineage", "parent-hash",
        "candidate", "x0 + x1", "x0", "x0 + x1", "REPLACE", "REPLACE",
        "generic edit", "0" * 64, "1" * 64, {}, metrics_before, metrics_after,
        {}, {},
    )
    final = SimpleNamespace(
        is_llm=True, lineage=(step,), lineage_id="lineage",
        dag=SimpleNamespace(canonical_hash="canonical"),
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = KnowledgeRuntime(
            root / "structure_library.jsonl", root / "runtime_ledger.jsonl")
        before = runtime.library_digest()
        deterministic = SimpleNamespace(
            is_llm=False, lineage=(), lineage_id="",
            dag=SimpleNamespace(canonical_hash="deterministic"))
        controller = SimpleNamespace(
            config=SimpleNamespace(
                structure_library_write=False,
                task_local_memory_write=True),
            knowledge=runtime,
            evaluation=SimpleNamespace(
                failure_signature=lambda state: ("high_validation_error",)))
        staged = ScientificDiscoveryRuntime._stage(
            controller, deterministic, deterministic, (final,))
        local = runtime.retrieve_task_local(
            ("high_validation_error",), topk=8)
        confirmed = runtime.retrieve(("high_validation_error",), topk=8)
        promotion_failed_closed = False
        try:
            runtime.promote_stage(
                staged["stage_id"],
                evidence_registry_path=root / "missing_registry.jsonl",
                hypothesis_id="unconfirmed")
        except (KeyError, RuntimeError, ValueError):
            promotion_failed_closed = True
        return {
            "staged": staged,
            "local": local,
            "confirmed": confirmed,
            "library_unchanged": runtime.library_digest() == before,
            "promotion_failed_closed": promotion_failed_closed,
        }


def run_scientist_closed_loop_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate registered wiring without opening any dataset or response."""
    validated = validate_system_registration(dict(config))
    agent = DiscoveryAgentConfig(**validated["agent"])
    resolved = DiscoveryConfig.from_mapping({
        "evaluation_budget": agent.discovery_budget,
        "llm_evaluation_reserve": agent.llm_evaluation_reserve,
        "refit_policy": agent.refit_policy,
        "islands": agent.discovery_islands,
        "use_library": agent.use_knowledge,
        "task_local_memory_read": agent.task_local_memory,
        "task_local_memory_write": agent.task_local_memory,
        "structure_library_topk": agent.task_local_memory_topk,
        "max_rounds": agent.discovery_rounds,
        "candidates_per_island": agent.candidates_per_island,
    })
    memory = _memory_fixture()
    local = list(memory["local"])
    checks = {
        "scientist_policy_enabled": agent.scientist_orchestration is True,
        "four_or_more_registered_engine_skills": len(agent.engines) >= 4,
        "multiple_outer_cycles": agent.cycles >= 2,
        "multiple_discovery_islands": len(agent.discovery_islands) >= 3,
        "multiple_inner_refinement_rounds": resolved.max_rounds >= 2,
        "job_level_skill_evidence_matrix": (
            _expand_job_controls((
                "linear", "quadratic", "cubic", "quartic", "interactions"), 6)
            == (("linear",), ("quadratic",), ("cubic",), ("quartic",),
                ("interactions",),
                ("linear", "quadratic", "cubic", "quartic",
                 "interactions"))),
        "bounded_candidates_per_island": 1 <= resolved.candidates_per_island <= 4,
        "task_local_memory_enabled": (
            resolved.task_local_memory_read
            and resolved.task_local_memory_write),
        "cross_task_library_disabled": (
            not resolved.structure_library_read
            and not agent.use_knowledge),
        "staged_memory_retrievable": bool(
            len(local) == 1
            and local[0].get("memory_scope")
                == "task-local-development-staged"),
        "validated_nonfinal_improvements_staged": (
            memory["staged"].get("staged_lineage_count") == 1),
        "staged_memory_not_promoted": (
            memory["library_unchanged"]
            and not memory["confirmed"]),
        "unconfirmed_promotion_fails_closed": memory["promotion_failed_closed"],
        "matched_policy_ablation_registered": (
            validated["marginal_influence_gate"]["schema"]
                == "scientific-policy-and-source-influence-gate-v6"
            and validated["marginal_influence_gate"][
                "scientist_policy_ablation"]
                == "full-vs-no_llm-matched-budget-v1"),
        "explicit_skill_controls_required":
            agent.require_explicit_skill_controls is True,
    }
    return {
        "schema": "scientific-task-local-scientist-closed-loop-gate-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "registered_outer_cycles": agent.cycles,
        "registered_inner_rounds": resolved.max_rounds,
        "registered_islands": list(agent.discovery_islands),
        "registered_engines": list(agent.engines),
        "memory_scope": "same-task-development-staged-only",
        "candidate_response_accessed": False,
        "reporting_validation_response_accessed": False,
        "heldout_opened": False,
        "real_data_accessed": False,
        "measurement_authorized": False,
        "formal_experiment_authorized": False,
    }


__all__ = ["run_scientist_closed_loop_gate"]
