"""Configuration used by the production symbolic-engine scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SymbolicConfig:
    """Dataset-agnostic search limits shared by every symbolic backend."""

    engine: str = "mcts"
    niterations: int = 120
    population_size: int = 64
    loss: str = "loss(x, y) = (x - y)^2"
    binary_operators: list[str] = field(default_factory=lambda: ["+", "-", "*", "/", "pow"])
    unary_operators: list[str] = field(default_factory=lambda: ["sin", "cos", "exp", "log"])
    seed_expressions: list[str] = field(default_factory=list)
    pysr_model_selection: str = "best"
    maxsize: int = 20
    complexity_of_constants: int = 2
    complexity_of_operators: dict[str, int] = field(default_factory=dict)
    complexity_penalty: float = 0.01
    polynomial_degree: int = 4
    polynomial_alpha: float = 1.0e-3
    mcts_max_iterations: int = 120
    mcts_expansion_factor: int = 5
    mcts_ucb_c: float = 1.4142
    mcts_max_ops: int = 30
    mcts_random_seed: int = 0
    mcts_candidate_sample_k: int = 0
    mcts_max_depth: int = 6
    mcts_max_nodes: int = 40
    mcts_constants: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    math_eps: float = 1.0e-6
    math_max_exp: float = 20.0
    math_max_pow_abs: int = 4


__all__ = ["SymbolicConfig"]
