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
    acquisition_enabled: bool = False
    use_knowledge: bool = False


@dataclass(frozen=True)
class DiscoveryCycle:
    cycle: int
    expression: str
    hypothesis_id: str
    development_rows: int
    engine_report: Mapping[str, Any]
    acquisition: Mapping[str, Any]
    provider_calls: int


@dataclass(frozen=True)
class DiscoveryAgentResult:
    discovery: DiscoveryRunResult
    cycles: tuple[DiscoveryCycle, ...]
    final_selection: SelectionData
    pool_rows_remaining: int
    provider_configured: bool


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
        )
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
        )

    def _discover(
        self, selection: SelectionData, engine_result: Any,
        previous: Sequence[str], task_name: str, task_description: str,
        output_dir: Path, knowledge_dir: Path, variable_metadata: Mapping[str, Any],
    ) -> DiscoveryRunResult:
        seeds = [{
            "expression": row.expression, "source": f"engine:{row.engine}",
            "lineage_id": row.lineage_id,
        } for row in engine_result.all_results]
        seeds.extend({
            "expression": expression, "source": "previous_cycle_survivor"
        } for expression in previous)
        return discover_from_selection(
            selection=selection,
            task_name=task_name, task_description=task_description,
            base_candidates=seeds, knowledge_dir=knowledge_dir,
            hypothesis_dir=output_dir / "hypotheses",
            evidence_registry_path=output_dir / "evidence_registry.jsonl",
            config=DiscoveryConfig.from_mapping({
                "evaluation_budget": self.config.discovery_budget,
                "random_seed": self.config.random_seed,
                "use_library": self.config.use_knowledge,
            }),
            provider_settings=self.provider_settings,
            variable_metadata=dict(variable_metadata),
            refinement_enabled=True, include_generic_candidates=True,
        )

    def run(
        self, *, selection: SelectionData,
        task_name: str, task_description: str, output_dir: str | Path,
        knowledge_dir: str | Path, variable_metadata: Mapping[str, Any],
    ) -> DiscoveryAgentResult:
        output, knowledge = Path(output_dir), Path(knowledge_dir)
        output.mkdir(parents=True, exist_ok=True)
        current = selection
        previous: tuple[str, ...] = ()
        history: list[DiscoveryCycle] = []
        final: DiscoveryRunResult | None = None
        for cycle in range(max(1, self.config.cycles)):
            engines = self._run_engines(current, cycle)
            final = self._discover(
                current, engines, previous, task_name, task_description,
                output, knowledge, variable_metadata,
            )
            previous = _survivors(final.report, final.expression)
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
                len(current.development.X), _engine_payload(engines),
                acquisition, int(final.report.get("llm_call_count", 0)),
            ))
        if final is None:
            raise RuntimeError("discovery agent executed no cycle")
        remaining = len(current.acquisition_pool.X) if current.acquisition_pool is not None else 0
        return DiscoveryAgentResult(
            final, tuple(history), current, remaining, self.provider_settings is not None,
        )


__all__ = [
    "DiscoveryAgent", "DiscoveryAgentConfig", "DiscoveryAgentResult", "DiscoveryCycle",
]
