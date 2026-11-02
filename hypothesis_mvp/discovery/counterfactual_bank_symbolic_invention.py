"""Counterfactual Bank-Guided Symbolic Invention (CB-SI) primitives.

The module owns three deliberately separate concerns:

1. bounded, provider-free structural edits;
2. matched counterfactual edit-credit computation;
3. a standard Bayesian operator policy consuming either raw or matched credit.

The Bayesian policy is not claimed as novel.  The research hypothesis is the
matched Bank-conditional credit assigned to a structural edit while holding
parent expression, edit site, donor material and evaluation budget fixed.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import sympy as sp

from hypothesis_mvp.hypotheses.expression_limits import AST_NODE_BUDGET
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    compile_fixed_formula_support,
)

from .equation_runtime import EquationRuntime


STRUCTURAL_OPERATORS = (
    "subtree_replacement",
    "subtree_composition",
    "operator_substitution",
    "factor_or_expand",
    "interaction_insertion",
    "correction_term",
    "structural_simplification",
    "llm_proposed_edit",
)

ZERO_PROVIDER_OPERATORS = tuple(
    value for value in STRUCTURAL_OPERATORS if value != "llm_proposed_edit"
)


def _stable_hash(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, default=str,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _ast_paths(root: ast.AST) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = []

    def visit(node: ast.AST, path: tuple[int, ...]) -> None:
        # Function identifiers (e.g. ``sin`` in ``sin(x0)``) are not
        # expression-valued edit sites.  Replacing one with a subtree would
        # manufacture a malformed function call rather than a local edit.
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name)):
            paths.append(path)
        children = list(ast.iter_child_nodes(node))
        for index, child in enumerate(children):
            if (isinstance(node, ast.Call) and child is node.func
                    and isinstance(child, ast.Name)):
                continue
            visit(child, (*path, index))

    visit(root, ())
    return paths


def _node_at_path(root: ast.AST, path: Sequence[int]) -> ast.AST:
    node = root
    for index in path:
        children = list(ast.iter_child_nodes(node))
        if index < 0 or index >= len(children):
            raise ValueError("edit path outside expression tree")
        node = children[index]
    return node


def _replace_at_path(
    root: ast.AST,
    path: Sequence[int],
    replacement: ast.AST,
) -> ast.AST:
    if not path:
        return copy.deepcopy(replacement)
    target_parent_path = tuple(path[:-1])
    child_index = int(path[-1])
    parent = _node_at_path(root, target_parent_path)
    children = list(ast.iter_child_nodes(parent))
    if child_index < 0 or child_index >= len(children):
        raise ValueError("edit child index outside expression tree")
    target = children[child_index]
    for field_name, value in ast.iter_fields(parent):
        if isinstance(value, list):
            for index, child in enumerate(value):
                if child is target:
                    value[index] = copy.deepcopy(replacement)
                    return root
        elif value is target:
            setattr(parent, field_name, copy.deepcopy(replacement))
            return root
    raise ValueError("edit target is not attached to parent")


def _parse_body(expression: str) -> ast.AST:
    return ast.parse(str(expression), mode="eval").body


def _operator_substitute(node: ast.AST, seed: int) -> ast.AST:
    copied = copy.deepcopy(node)
    if isinstance(copied, ast.BinOp):
        allowed = (ast.Add, ast.Sub, ast.Mult, ast.Div)
        current = type(copied.op)
        candidates = [value for value in allowed if value is not current]
        copied.op = candidates[int(seed) % len(candidates)]()
        return copied
    if (isinstance(copied, ast.Call)
            and isinstance(copied.func, ast.Name)
            and copied.func.id in {"sin", "cos", "tanh", "exp", "log"}):
        candidates = [
            value for value in ("sin", "cos", "tanh", "exp", "log")
            if value != copied.func.id]
        copied.func.id = candidates[int(seed) % len(candidates)]
        return copied
    if isinstance(copied, ast.UnaryOp):
        copied.op = ast.UAdd() if isinstance(
            copied.op, ast.USub) else ast.USub()
        return copied
    raise ValueError("selected subtree has no substitutable operator")


@dataclass(frozen=True)
class StructuralEditRequest:
    operator: str
    parent_expression: str
    donor_expression: str | None
    site_path: tuple[int, ...]
    seed: int
    task_identity: str = ""
    mock_llm_expression: str | None = None
    matched_control_of: str | None = None

    @property
    def identity(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True)
class StructuralEditResult:
    request_identity: str
    operator: str
    parent_expression: str
    donor_expression: str | None
    site_path: tuple[int, ...]
    expression: str | None
    canonical_hash: str | None
    structural_hash: str | None
    lineage_id: str
    complexity: float | None
    valid: bool
    novel: bool
    equivalent_to_parent: bool
    failure_reason: str | None
    registered_domain_count: int
    provider_calls: int = 0

    def row(self) -> dict:
        return asdict(self)


class StructuralEditingEngine:
    """Bounded structural editor using the canonical EquationRuntime."""

    def __init__(
        self,
        n_features: int,
        *,
        refit_policy: str = "pcpi-expanded-fixed-inner-v1",
        maximum_ast_nodes: int = AST_NODE_BUDGET,
    ) -> None:
        self.n_features = int(n_features)
        self.runtime = EquationRuntime(
            self.n_features, refit_policy=refit_policy)
        self.maximum_ast_nodes = min(
            int(maximum_ast_nodes), int(AST_NODE_BUDGET))

    def editable_paths(self, expression: str) -> tuple[tuple[int, ...], ...]:
        self.runtime.dag(expression)
        return tuple(_ast_paths(_parse_body(expression)))

    def deterministic_request(
        self,
        operator: str,
        parent_expression: str,
        donor_expression: str | None,
        *,
        seed: int,
        task_identity: str = "",
        mock_llm_expression: str | None = None,
        matched_control_of: str | None = None,
        site_path: Sequence[int] | None = None,
    ) -> StructuralEditRequest:
        if operator not in STRUCTURAL_OPERATORS:
            raise ValueError("unknown structural edit operator")
        paths = self.editable_paths(parent_expression)
        if not paths:
            raise ValueError("parent expression has no editable site")
        if site_path is None:
            digest = sha256(
                f"{task_identity}:{seed}:{operator}:site".encode()).digest()
            site = paths[int.from_bytes(digest[:8], "big") % len(paths)]
        else:
            site = tuple(int(index) for index in site_path)
            if site not in paths:
                raise ValueError("requested edit site is not editable")
        return StructuralEditRequest(
            operator=operator,
            parent_expression=parent_expression,
            donor_expression=donor_expression,
            site_path=site,
            seed=int(seed),
            task_identity=str(task_identity),
            mock_llm_expression=mock_llm_expression,
            matched_control_of=matched_control_of,
        )

    @staticmethod
    def _operator_supported_at_site(operator: str, selected: ast.AST) -> bool:
        if operator == "operator_substitution":
            return (isinstance(selected, (ast.BinOp, ast.UnaryOp))
                    or (isinstance(selected, ast.Call)
                        and isinstance(selected.func, ast.Name)
                        and selected.func.id in
                        {"sin", "cos", "tanh", "exp", "log"}))
        return operator in ZERO_PROVIDER_OPERATORS

    def matched_control_request(
        self,
        primary: StructuralEditRequest,
        *,
        control_operator: str | None = None,
    ) -> StructuralEditRequest:
        if control_operator is None:
            selected = _node_at_path(
                _parse_body(primary.parent_expression), primary.site_path)
            alternatives = [
                value for value in ZERO_PROVIDER_OPERATORS
                if value != primary.operator
                and self._operator_supported_at_site(value, selected)]
            if not alternatives:
                raise ValueError("no matched control operator available")
            digest = sha256(
                f"{primary.identity}:control".encode()).digest()
            control_operator = alternatives[
                int.from_bytes(digest[:8], "big") % len(alternatives)]
        selected = _node_at_path(
            _parse_body(primary.parent_expression), primary.site_path)
        if (control_operator not in ZERO_PROVIDER_OPERATORS
                or control_operator == primary.operator
                or not self._operator_supported_at_site(
                    control_operator, selected)):
            raise ValueError("invalid or inapplicable matched control operator")
        return StructuralEditRequest(
            operator=control_operator,
            parent_expression=primary.parent_expression,
            donor_expression=primary.donor_expression,
            site_path=primary.site_path,
            seed=primary.seed,
            task_identity=primary.task_identity,
            matched_control_of=primary.identity,
        )

    def _raw_edit(self, request: StructuralEditRequest) -> str:
        # Requests may be constructed directly: validate the site here too.
        if tuple(request.site_path) not in self.editable_paths(
                request.parent_expression):
            raise ValueError("requested edit site is not editable")
        if request.operator == "llm_proposed_edit":
            if not request.mock_llm_expression:
                raise ValueError(
                    "LLM edit is dry-run only without a frozen mock response")
            return str(request.mock_llm_expression)

        parent = _parse_body(request.parent_expression)
        selected = _node_at_path(parent, request.site_path)
        donor = (
            _parse_body(request.donor_expression)
            if request.donor_expression else None)

        if request.operator == "subtree_replacement":
            if donor is None:
                raise ValueError("subtree replacement needs donor material")
            edited = _replace_at_path(parent, request.site_path, donor)
        elif request.operator == "subtree_composition":
            if donor is None:
                raise ValueError("subtree composition needs donor material")
            operation = ast.Add() if request.seed % 2 == 0 else ast.Mult()
            replacement = ast.BinOp(
                left=copy.deepcopy(selected), op=operation,
                right=copy.deepcopy(donor))
            edited = _replace_at_path(parent, request.site_path, replacement)
        elif request.operator == "operator_substitution":
            edited = _replace_at_path(
                parent, request.site_path,
                _operator_substitute(selected, request.seed))
        elif request.operator == "factor_or_expand":
            # Apply the rewrite to the selected subtree, not the full parent.
            local_expression = self.runtime.registry.parse(
                ast.unparse(selected), self.n_features, evaluate=False)
            transformed = (
                sp.expand(local_expression)
                if request.seed % 2 == 0 else sp.factor(local_expression))
            replacement = _parse_body(sp.sstr(transformed, order="lex"))
            edited = _replace_at_path(parent, request.site_path, replacement)
        elif request.operator == "interaction_insertion":
            left = request.seed % self.n_features
            right = (left + 1 + (
                request.seed // max(1, self.n_features))
                % max(1, self.n_features - 1)) % self.n_features
            interaction = ast.BinOp(
                left=ast.Name(id=f"x{left}", ctx=ast.Load()),
                op=ast.Mult(),
                right=ast.Name(id=f"x{right}", ctx=ast.Load()))
            replacement = ast.BinOp(
                left=copy.deepcopy(selected), op=ast.Add(),
                right=interaction)
            edited = _replace_at_path(
                parent, request.site_path, replacement)
        elif request.operator == "correction_term":
            if donor is None:
                raise ValueError("correction term needs donor material")
            replacement = ast.BinOp(
                left=copy.deepcopy(selected), op=ast.Add(),
                right=copy.deepcopy(donor))
            edited = _replace_at_path(
                parent, request.site_path, replacement)
        elif request.operator == "structural_simplification":
            local_expression = self.runtime.registry.parse(
                ast.unparse(selected), self.n_features, evaluate=False)
            transformed = sp.cancel(sp.factor_terms(local_expression))
            replacement = _parse_body(sp.sstr(transformed, order="lex"))
            edited = _replace_at_path(
                parent, request.site_path, replacement)
        else:
            raise ValueError("unsupported structural edit operator")
        ast.fix_missing_locations(edited)
        return ast.unparse(edited)

    def apply(
        self,
        request: StructuralEditRequest,
        *,
        registered_domains: Sequence[np.ndarray],
        known_structural_hashes: Iterable[str] = (),
    ) -> StructuralEditResult:
        lineage_id = "cbsi-" + request.identity[:24]
        try:
            parent = self.runtime.dag(request.parent_expression)
            raw = self._raw_edit(request)
            root = ast.parse(raw, mode="eval")
            if len(list(ast.walk(root))) > self.maximum_ast_nodes:
                raise ValueError("edited expression exceeds AST budget")
            dag = self.runtime.dag(raw)
            compile_fixed_formula_support(dag.expression, self.n_features)
            if not registered_domains:
                raise ValueError("registered domain set must not be empty")
            candidate_predictions = []
            parent_predictions = []
            for domain in registered_domains:
                values = np.asarray(domain)
                candidate_predictions.append(
                    self.runtime.predict(dag.expression, values))
                parent_predictions.append(
                    self.runtime.predict(parent.expression, values))
            equivalent = dag.canonical_hash == parent.canonical_hash
            if not equivalent:
                equivalent = all(
                    np.allclose(left, right, rtol=1.0e-10, atol=1.0e-12)
                    for left, right in zip(
                        parent_predictions, candidate_predictions,
                        strict=True))
            known = set(str(value) for value in known_structural_hashes)
            novel = (
                not equivalent
                and dag.structural_hash != parent.structural_hash
                and dag.structural_hash not in known)
            return StructuralEditResult(
                request.identity, request.operator,
                request.parent_expression, request.donor_expression,
                request.site_path, dag.expression, dag.canonical_hash,
                dag.structural_hash, lineage_id, dag.complexity,
                True, novel, equivalent, None, len(registered_domains), 0)
        except Exception as error:
            return StructuralEditResult(
                request.identity, request.operator,
                request.parent_expression, request.donor_expression,
                request.site_path, None, None, None, lineage_id, None,
                False, False, False,
                f"{type(error).__name__}:{error}",
                len(registered_domains), 0)


@dataclass(frozen=True)
class CounterfactualCredit:
    operator: str
    primary_fold_gains: tuple[float, ...]
    parent_fold_gains: tuple[float, ...]
    control_fold_gains: tuple[tuple[float, ...], ...]
    primary_cost: float
    control_cost_mean: float
    raw_bank_gain: float
    parent_adjusted_gain: float
    matched_control_gain: float
    cost_adjustment: float
    credit: float
    standard_error: float
    control_count: int

    def row(self) -> dict:
        return asdict(self)


def counterfactual_edit_credit(
    operator: str,
    primary_fold_gains: Sequence[float],
    parent_fold_gains: Sequence[float],
    control_fold_gains: Sequence[Sequence[float]],
    *,
    primary_cost: float = 0.0,
    control_costs: Sequence[float] = (),
    cost_weight: float = 0.0,
) -> CounterfactualCredit:
    """Compute paired edit credit with parent and matched-control baselines.

    For fold ``k`` the identifying contrast is

        (gain(edit)_k - gain(parent)_k)
        - mean_c(gain(control_c)_k - gain(parent)_k).

    Parent quality therefore cancels algebraically, but is retained in the
    audit to verify that the matched controls really used the same parent.
    """
    primary = np.asarray(primary_fold_gains, dtype=float)
    parent = np.asarray(parent_fold_gains, dtype=float)
    controls = np.asarray(control_fold_gains, dtype=float)
    if (primary.ndim != 1 or parent.shape != primary.shape
            or not len(primary) or controls.ndim != 2
            or controls.shape[1] != len(primary)
            or not len(controls)
            or not np.all(np.isfinite(primary))
            or not np.all(np.isfinite(parent))
            or not np.all(np.isfinite(controls))):
        raise ValueError("counterfactual credit needs finite matched folds")
    control_cost_array = np.asarray(
        list(control_costs) if control_costs else np.zeros(len(controls)),
        dtype=float)
    if (control_cost_array.shape != (len(controls),)
            or not np.all(np.isfinite(control_cost_array))
            or not math.isfinite(float(primary_cost))
            or not math.isfinite(float(cost_weight))
            or cost_weight < 0):
        raise ValueError("counterfactual credit has invalid costs")
    paired = (
        (primary - parent)
        - np.mean(controls - parent[None, :], axis=0))
    cost_adjustment = float(cost_weight) * (
        float(primary_cost) - float(np.mean(control_cost_array)))
    credit = float(np.mean(paired) - cost_adjustment)
    standard_error = (
        float(np.std(paired, ddof=1) / math.sqrt(len(paired)))
        if len(paired) > 1 else 0.0)
    return CounterfactualCredit(
        operator=str(operator),
        primary_fold_gains=tuple(float(v) for v in primary),
        parent_fold_gains=tuple(float(v) for v in parent),
        control_fold_gains=tuple(
            tuple(float(v) for v in row) for row in controls),
        primary_cost=float(primary_cost),
        control_cost_mean=float(np.mean(control_cost_array)),
        raw_bank_gain=float(np.mean(primary)),
        parent_adjusted_gain=float(np.mean(primary - parent)),
        matched_control_gain=float(np.mean(controls)),
        cost_adjustment=cost_adjustment,
        credit=credit,
        standard_error=standard_error,
        control_count=len(controls),
    )


@dataclass
class NormalGammaArm:
    """Scalar Normal-Gamma posterior for an operator reward."""

    prior_mean: float = 0.0
    prior_strength: float = 1.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("operator reward is nonfinite")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def posterior_parameters(self) -> tuple[float, float, float, float]:
        n = float(self.count)
        strength = self.prior_strength + n
        location = (
            self.prior_strength * self.prior_mean + n * self.mean
        ) / strength
        alpha = self.prior_alpha + n / 2.0
        beta = self.prior_beta + 0.5 * self.m2
        if self.count:
            beta += (
                self.prior_strength * n
                * (self.mean - self.prior_mean) ** 2
                / (2.0 * strength))
        return location, strength, alpha, max(beta, 1.0e-12)

    def sample(self, rng: np.random.Generator) -> float:
        location, strength, alpha, beta = self.posterior_parameters()
        precision = rng.gamma(alpha, 1.0 / beta)
        return float(rng.normal(location, 1.0 / math.sqrt(
            max(strength * precision, 1.0e-15))))

    def posterior_mean(self) -> float:
        return float(self.posterior_parameters()[0])

    def audit(self) -> dict:
        location, strength, alpha, beta = self.posterior_parameters()
        return {
            "count": self.count,
            "sample_mean": self.mean,
            "sample_m2": self.m2,
            "posterior_location": location,
            "posterior_strength": strength,
            "posterior_alpha": alpha,
            "posterior_beta": beta,
        }


@dataclass
class BayesianOperatorPolicy:
    """Standard operator selector; novelty is not attributed to this class."""

    operators: tuple[str, ...] = ZERO_PROVIDER_OPERATORS
    seed: int = 0
    arms: dict[str, NormalGammaArm] = field(init=False)
    selections: list[str] = field(default_factory=list)
    rewards: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (not self.operators
                or any(value not in ZERO_PROVIDER_OPERATORS
                       for value in self.operators)
                or len(set(self.operators)) != len(self.operators)):
            raise ValueError("invalid Bayesian operator set")
        self.arms = {value: NormalGammaArm() for value in self.operators}

    def choose(self, step: int, *, mode: str) -> str:
        if mode not in {"fixed", "greedy", "thompson"}:
            raise ValueError("unknown operator policy mode")
        if mode == "fixed":
            choice = self.operators[int(step) % len(self.operators)]
        elif mode == "greedy":
            choice = min(
                self.operators,
                key=lambda value: (
                    -self.arms[value].posterior_mean(), value))
        else:
            rng = np.random.default_rng(
                int(self.seed) + 104729 * (int(step) + 1))
            samples = {
                value: self.arms[value].sample(rng)
                for value in self.operators}
            choice = min(
                self.operators,
                key=lambda value: (-samples[value], value))
        self.selections.append(choice)
        return choice

    def update(
        self,
        operator: str,
        reward: float,
        *,
        reward_kind: str,
        audit: Mapping | None = None,
    ) -> None:
        if operator not in self.arms:
            raise ValueError("operator absent from policy")
        self.arms[operator].update(float(reward))
        self.rewards.append({
            "operator": operator,
            "reward": float(reward),
            "reward_kind": str(reward_kind),
            "audit": dict(audit or {}),
        })

    def audit(self) -> dict:
        return {
            "schema": "cbsi-bayesian-operator-policy-v1",
            "operators": list(self.operators),
            "seed": self.seed,
            "selections": list(self.selections),
            "rewards": list(self.rewards),
            "posteriors": {
                value: self.arms[value].audit()
                for value in self.operators},
            "novelty_claim_authorized": False,
            "note": (
                "The Bayesian selector is a standard component. "
                "CB-SI novelty, if established, is in matched "
                "Bank-conditional structural credit."),
        }

