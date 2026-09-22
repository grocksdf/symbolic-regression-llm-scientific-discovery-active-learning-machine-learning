"""Typed registry for auditable symbolic-engine skills."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolicEngineSpec:
    name: str
    capabilities: tuple[str, ...]
    inductive_bias: str
    forbidden_requests: tuple[str, ...] = ()
    baseline: bool = False

    @property
    def source_family(self) -> str:
        return "core" if self.baseline else f"engine:{self.name}"


REGISTERED_SYMBOLIC_ENGINES = (
    SymbolicEngineSpec(
        "polynomial_lasso",
        ("sparse polynomial support", "fast deterministic baseline",
         "global additive interactions"),
        "degree-bounded sparse polynomial",
        ("trigonometric", "logarithmic", "exponential", "reciprocal",
         "division"),
        baseline=True),
    SymbolicEngineSpec(
        "mcts",
        ("typed symbolic search", "nonlinear primitive search",
         "structural diversity frontier", "degree-at-most-four monomials",
         "sin(xi), cos(xi), tanh(xi)"),
        "tree-search over registered closed basis",
        ("reciprocal", "logarithmic", "exponential", "division", "ratio")),
    SymbolicEngineSpec(
        "sparse_library",
        ("sequential thresholded sparse library identification",
         "mixed polynomial and bounded nonlinear support",
         "threshold-path closed-basis frontier",
         "sin(xi), cos(xi), tanh(xi)"),
        "sequential thresholded least squares over one frozen mixed library",
        ("reciprocal", "logarithmic", "exponential", "division", "ratio")),
    SymbolicEngineSpec(
        "additive_mechanisms",
        ("additive mechanism discovery", "residual-guided transform selection",
         "per-variable nonlinear mechanism screening",
         "sin(xi), cos(xi), tanh(xi)"),
        "strongly interpretable additive residual-forward-selection",
        ("cross-variable interaction", "reciprocal", "logarithmic",
         "exponential", "division", "ratio")),
)


def engine_spec(name: str) -> SymbolicEngineSpec:
    matches = [spec for spec in REGISTERED_SYMBOLIC_ENGINES
               if spec.name == str(name)]
    if len(matches) != 1:
        raise ValueError(f"unregistered symbolic engine: {name}")
    return matches[0]


def registered_engine_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in REGISTERED_SYMBOLIC_ENGINES)


def baseline_engine_name() -> str:
    values = tuple(spec.name for spec in REGISTERED_SYMBOLIC_ENGINES
                   if spec.baseline)
    if len(values) != 1:
        raise RuntimeError("symbolic registry must have one baseline engine")
    return values[0]


__all__ = [
    "REGISTERED_SYMBOLIC_ENGINES", "SymbolicEngineSpec",
    "baseline_engine_name", "engine_spec", "registered_engine_names",
]
