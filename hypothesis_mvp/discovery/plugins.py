"""Optional domain extension contracts for scientific discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .contracts import DiscoveryConfig
from .equation_runtime import PrimitiveRegistry

Candidate = Mapping[str, Any] | str


@dataclass(frozen=True)
class DiscoveryContext:
    """Train-only context exposed to optional structure plugins.

    Dataset names, task labels, target names, and filenames are deliberately absent.
    Only caller-declared units/dimensions or other explicit domain constraints may be
    included in ``structure_metadata``.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    structure_metadata: Mapping[str, Any]
    config: DiscoveryConfig


@runtime_checkable
class DiscoveryPlugin(Protocol):
    """Explicit extension point; plugins are never enabled implicitly."""

    name: str

    def candidates(
        self,
        context: DiscoveryContext,
        registry: PrimitiveRegistry,
    ) -> Sequence[Candidate]: ...


def plugin_candidates(
    plugins: Sequence[DiscoveryPlugin],
    context: DiscoveryContext,
    registry: PrimitiveRegistry,
) -> tuple[list[Candidate], list[str]]:
    """Collect candidates from caller-selected plugins in stable order."""
    candidates: list[Candidate] = []
    names: list[str] = []
    for plugin in plugins:
        name = str(getattr(plugin, "name", type(plugin).__name__)).strip()
        if not name:
            raise ValueError("discovery plugin name must not be empty")
        proposed = list(plugin.candidates(context, registry))
        candidates.extend(proposed)
        names.append(name)
    return candidates, names
