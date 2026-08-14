"""Construction of the single production discovery stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import DiscoveryConfig, RuntimeEvent
from .equation_runtime import EquationRuntime, PrimitiveRegistry
from .evaluation_runtime import EvaluationRuntime
from .exploration_runtime import ExplorationRuntime
from .knowledge_runtime import KnowledgeRuntime
from .proposal_runtime import ProposalRuntime, ProviderSettings
from .scientific_runtime import ScientificDiscoveryRuntime


def build_scientific_discovery_runtime(
    *, n_features: int, config: DiscoveryConfig | Mapping[str, Any] | None,
    library_path: str | Path, ledger_path: str | Path,
    provider_settings: ProviderSettings | None = None,
    variable_metadata: Mapping[str, Any] | None = None,
    primitive_registry: PrimitiveRegistry | None = None,
    event_callback: Callable[[RuntimeEvent], None] | None = None,
) -> ScientificDiscoveryRuntime:
    resolved = config if isinstance(config, DiscoveryConfig) else DiscoveryConfig.from_mapping(config)
    registry = primitive_registry or PrimitiveRegistry()
    equation = EquationRuntime(
        n_features, registry,
        max_numeric_parameters=resolved.max_numeric_parameters,
        max_abs_coefficient=resolved.max_abs_coefficient,
        optimize_exponents=resolved.optimize_exponents,
        variable_metadata=variable_metadata,
    )
    return ScientificDiscoveryRuntime(
        equation=equation,
        exploration=ExplorationRuntime(equation, resolved),
        proposal=ProposalRuntime(
            equation, n_features, provider_settings, resolved.candidates_per_island
        ),
        evaluation=EvaluationRuntime(equation, resolved),
        knowledge=KnowledgeRuntime(
            library_path, ledger_path, resolved.structure_library_max_entries
        ),
        config=resolved, event_callback=event_callback,
    )


__all__ = ["build_scientific_discovery_runtime"]
