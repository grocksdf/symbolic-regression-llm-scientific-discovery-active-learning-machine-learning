"""Typed-grammar Monte Carlo tree search for symbolic regression.

The search evolves explicit ASTs.  It never uses dataset metadata, task names,
ground-truth expressions, or an untyped metadata proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np
import sympy as sp

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.symbolic.sympy_utils import sympy_symbols

from .base import SymbolicRegressor
from .typed_grammar import (
    ExprNode,
    Var,
    ast_from_sympy,
    canonicalize_ast,
    expand_ast,
    is_valid_ast,
)


@dataclass
class MCTSNode:
    expression: str
    ast: ExprNode
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)
    score: float = float("inf")
    visits: int = 0
    reward_sum: float = 0.0

    def ucb_score(self, exploration: float) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = max(1, self.parent.visits if self.parent is not None else self.visits)
        exploitation = self.reward_sum / self.visits
        return exploitation + exploration * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )


class MCTSSymbolicAgent(SymbolicRegressor):
    """Budgeted MCTS over a numerically guarded typed grammar."""

    def __init__(
        self,
        cfg: Optional[SymbolicConfig] = None,
        max_iterations: int = 200,
        expansion_factor: int = 5,
        candidate_ops: Optional[list[str]] = None,
        ucb_c: float = 1.4142,
        complexity_penalty: float = 0.0,
        seed_expressions: Optional[Sequence[str]] = None,
        reward_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        config = cfg or SymbolicConfig(engine="mcts")
        self.max_iterations = max(
            1, int(getattr(config, "mcts_max_iterations", max_iterations))
        )
        self.expansion_factor = max(
            1, int(getattr(config, "mcts_expansion_factor", expansion_factor))
        )
        self.ucb_c = float(getattr(config, "mcts_ucb_c", ucb_c))
        self.complexity_penalty = max(
            0.0, float(getattr(config, "complexity_penalty", complexity_penalty))
        )
        self.max_ops = max(1, int(getattr(config, "mcts_max_ops", 30)))
        self.max_depth = max(1, int(getattr(config, "mcts_max_depth", 6)))
        self.max_nodes = max(2, int(getattr(config, "mcts_max_nodes", 40)))
        self.candidate_sample_k = max(
            0, int(getattr(config, "mcts_candidate_sample_k", 0))
        )
        self.constants = tuple(
            float(value) for value in getattr(config, "mcts_constants", [1.0, 2.0, 3.0])
        )
        self.math_eps = float(getattr(config, "math_eps", 1.0e-6))
        self.math_max_exp = float(getattr(config, "math_max_exp", 20.0))
        self.math_max_pow_abs = int(getattr(config, "math_max_pow_abs", 4))
        raw_binary = candidate_ops or getattr(
            config, "binary_operators", ["+", "-", "*", "/"]
        )
        self._binary_ops = tuple(
            "**" if str(operator) == "pow" else str(operator)
            for operator in raw_binary
        )
        self._allowed_unary = tuple(
            str(value) for value in getattr(config, "unary_operators", [])
        )
        self.random_seed = int(getattr(config, "mcts_random_seed", 0))
        self._rng = np.random.default_rng(self.random_seed)
        self.reward_fn = reward_fn or (lambda mse: 1.0 / (1.0 + max(0.0, mse)))
        self.best_expr = ""
        self.best_score = float("inf")
        self._seed_asts = self._load_source_only_seeds(config, seed_expressions)

    def _load_source_only_seeds(
        self,
        config: SymbolicConfig,
        explicit: Optional[Sequence[str]],
    ) -> tuple[ExprNode, ...]:
        seeds = [str(value) for value in (explicit or ())]
        seeds.extend(str(value) for value in getattr(config, "seed_expressions", ()))
        output: dict[str, ExprNode] = {}
        for expression in seeds:
            try:
                node = ast_from_sympy(sp.sympify(expression))
                if node is not None:
                    canonical = canonicalize_ast(node)
                    output[canonical.to_string()] = canonical
            except Exception:
                continue
        return tuple(output.values())

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MCTSSymbolicAgent":
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(target) or not len(features):
            raise ValueError("MCTS requires aligned non-empty two-dimensional data")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
            raise ValueError("MCTS inputs must be finite")
        self._n_features = int(features.shape[1])
        sample_count = min(128, len(features))
        sample_index = (
            self._rng.choice(len(features), size=sample_count, replace=False)
            if len(features) > sample_count
            else np.arange(len(features))
        )
        self._X_sample = features[sample_index]
        root_ast = canonicalize_ast(Var(0))
        root = MCTSNode(root_ast.to_string(), root_ast)
        self.best_expr = ""
        self.best_score = float("inf")
        for _ in range(self.max_iterations):
            leaf = self._select(root)
            node = self._expand(leaf)
            reward = self._evaluate(node, features, target)
            self._backpropagate(node, reward)
        if not self.best_expr:
            self._evaluate(root, features, target)
        return self

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.children:
            unvisited = [child for child in node.children if child.visits == 0]
            if unvisited:
                return unvisited[int(self._rng.integers(0, len(unvisited)))]
            node = max(node.children, key=lambda child: child.ucb_score(self.ucb_c))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        proposals = list(
            expand_ast(
                node.ast,
                n_features=self._n_features,
                allowed_unary=list(self._allowed_unary),
                allowed_binary=list(self._binary_ops),
                constants=list(self.constants),
                rng=self._rng,
                max_depth=self.max_depth,
                max_nodes=self.max_nodes,
                max_new=max(
                    self.expansion_factor,
                    self.candidate_sample_k or self.expansion_factor * 2,
                ),
                X_sample=self._X_sample,
                math_eps=self.math_eps,
                math_max_exp=self.math_max_exp,
                math_max_pow_abs=self.math_max_pow_abs,
            )
        )
        if node.parent is None:
            proposals.extend(self._seed_asts)
        unique: dict[str, ExprNode] = {}
        for proposal in proposals:
            try:
                canonical = canonicalize_ast(proposal)
                if is_valid_ast(
                    canonical,
                    n_features=self._n_features,
                    allowed_unary=list(self._allowed_unary),
                    allowed_binary=list(self._binary_ops),
                    max_depth=self.max_depth,
                    max_nodes=self.max_nodes,
                    X_sample=self._X_sample,
                    math_eps=self.math_eps,
                    math_max_exp=self.math_max_exp,
                    math_max_pow_abs=self.math_max_pow_abs,
                ):
                    unique[canonical.to_string()] = canonical
            except Exception:
                continue
        choices = list(unique.items())
        self._rng.shuffle(choices)
        for expression, ast_node in choices[: self.expansion_factor]:
            if expression == node.expression:
                continue
            node.children.append(MCTSNode(expression, ast_node, parent=node))
        if not node.children:
            return node
        return node.children[int(self._rng.integers(0, len(node.children)))]

    def _evaluate(self, node: MCTSNode, X: np.ndarray, y: np.ndarray) -> float:
        if np.isfinite(node.score):
            return float(self.reward_fn(node.score))
        try:
            expression = node.ast.to_sympy()
            if float(sp.count_ops(expression, visual=False)) > self.max_ops:
                return 0.0
            symbols = sympy_symbols(X.shape[1])
            function = sp.lambdify(symbols, expression, "numpy")
            with np.errstate(all="ignore"):
                prediction = function(*X.T)
            if np.isscalar(prediction):
                prediction = np.full(len(y), float(prediction), dtype=float)
            prediction = np.asarray(prediction, dtype=float).reshape(-1)
            if prediction.shape != y.shape or not np.all(np.isfinite(prediction)):
                return 0.0
            mse = float(np.mean((prediction - y) ** 2))
            complexity = float(sp.count_ops(expression, visual=False))
            score = mse + self.complexity_penalty * complexity
            if not expression.free_symbols:
                baseline = float(np.mean((y - np.mean(y)) ** 2))
                if score >= 0.98 * baseline:
                    score += 1.0e6 + baseline
            node.score = score
            if score < self.best_score:
                self.best_score = score
                self.best_expr = node.expression
            return float(self.reward_fn(score))
        except Exception:
            return 0.0

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float) -> None:
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.reward_sum += float(reward)
            current = current.parent

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.best_expr:
            raise ValueError("MCTS has not been fitted")
        features = np.asarray(X, dtype=float)
        symbols = sympy_symbols(features.shape[1])
        function = sp.lambdify(symbols, sp.sympify(self.best_expr), "numpy")
        with np.errstate(all="ignore"):
            prediction = function(*features.T)
        if np.isscalar(prediction):
            prediction = np.full(features.shape[0], float(prediction), dtype=float)
        output = np.asarray(prediction, dtype=float).reshape(-1)
        if output.shape[0] != features.shape[0] or not np.all(np.isfinite(output)):
            raise ValueError("MCTS best expression produced invalid predictions")
        return output.reshape(-1, 1)

    def best_expression(self) -> str:
        return self.best_expr

    def info(self) -> dict[str, Any]:
        return {
            "engine": "mcts",
            "grammar": "typed",
            "max_iterations": self.max_iterations,
            "expansion_factor": self.expansion_factor,
            "best_expression": self.best_expr,
            "best_score": self.best_score,
            "ucb_c": self.ucb_c,
            "complexity_penalty": self.complexity_penalty,
            "seed_count": len(self._seed_asts),
        }


__all__ = ["MCTSNode", "MCTSSymbolicAgent"]
