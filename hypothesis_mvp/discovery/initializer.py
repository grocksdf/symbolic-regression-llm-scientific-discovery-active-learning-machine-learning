"""Deterministic, dataset-agnostic candidate initialization."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import DiscoveryConfig
from .equation_runtime import EquationRuntime, PrimitiveRegistry


def generic_deterministic_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: DiscoveryConfig | Mapping[str, Any] | None = None,
    registry: PrimitiveRegistry | None = None,
) -> list[dict[str, str]]:
    """Build constant, generic-grammar, and linear train-only anchors."""
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or len(X) != len(y) or X.shape[1] <= 0:
        raise ValueError("X_train and y_train must contain aligned finite samples")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("initializer inputs must be finite")
    resolved = config if isinstance(config, DiscoveryConfig) else DiscoveryConfig.from_mapping(config)
    primitive_registry = registry or PrimitiveRegistry()
    specs = primitive_registry.generic_anchor_basis(X)
    keeps = sorted(set((
        max(2, X.shape[1] + 1),
        max(4, 2 * X.shape[1] + 2),
        min(12, max(4, len(specs))),
        min(24, max(4, len(specs))),
        min(40, max(4, len(specs))),
    )))
    candidates = [{
        "expression": f"({float(np.mean(y)):.16g})",
        "source": "deterministic_constant_anchor",
    }]
    for keep in keeps:
        candidates.append({
            "expression": primitive_registry.fit_sparse_linear_expression(
                specs, y, keep=keep, ridge=resolved.ridge_alpha,
            ),
            "source": f"deterministic_generic_grammar_keep_{keep}",
        })
    linear = [(f"x{i}", X[:, i]) for i in range(X.shape[1])]
    candidates.append({
        "expression": primitive_registry.fit_sparse_linear_expression(
            linear, y, keep=max(1, X.shape[1]), ridge=resolved.ridge_alpha,
        ),
        "source": "deterministic_linear_anchor",
    })
    return candidates


def normalize_candidates(
    candidates: Sequence[Mapping[str, Any] | str],
    *,
    n_features: int,
    X_probe: np.ndarray | None = None,
    registry: PrimitiveRegistry | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Canonicalize, validate, and deduplicate candidates by expression hash."""
    runtime = EquationRuntime(int(n_features), registry or PrimitiveRegistry())
    probe = None if X_probe is None else np.asarray(X_probe, dtype=float)
    accepted: dict[str, dict[str, Any]] = {}
    rejected = 0
    for item in candidates:
        candidate = {"expression": item, "source": "caller_seed"} if isinstance(item, str) else dict(item)
        try:
            dag = runtime.dag(str(candidate.get("expression") or ""))
            if probe is not None:
                runtime.predict(dag.expression, probe[:min(8, len(probe))])
            accepted.setdefault(dag.canonical_hash, {**candidate, "expression": dag.expression})
        except Exception:
            rejected += 1
    return list(accepted.values()), rejected
