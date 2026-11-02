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

# Algebraic normalizers change the representation, not the predictive law.
# They remain available for intermediate rewriting, but cannot consume a
# semantic invention slot in the feasibility-aware search arm.
INVENTION_OPERATORS = tuple(
    op for op in ZERO_PROVIDER_OPERATORS
    if op not in {"factor_or_expand", "structural_simplification"}
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
    symbolically_equivalent: bool = False
    empirically_equivalent_on_domains: bool = False
    support_hash: str | None = None

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

    def eligible_paths(
        self,
        operator: str,
        parent_expression: str,
        donor_expression: str | None,
    ) -> tuple[tuple[int, ...], ...]:
        """Response-free, syntax-level eligibility (never inspects labels)."""
        if operator not in STRUCTURAL_OPERATORS:
            raise ValueError("unknown structural edit operator")
        paths = self.editable_paths(parent_expression)
        root = _parse_body(parent_expression)
        return tuple(
            path for path in paths
            if self._operator_supported_at_site(
                operator, _node_at_path(root, path), donor_expression)
        )

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
            # Do not blindly send operator substitutions to bare variables.
            eligible = self.eligible_paths(
                operator, parent_expression, donor_expression)
            if not eligible:
                raise ValueError("no applicable edit site for operator")
            digest = sha256(
                f"{task_identity}:{seed}:{operator}:site".encode()).digest()
            site = eligible[int.from_bytes(digest[:8], "big") % len(eligible)]
        else:
            # Explicit frozen opportunities are permitted to fail and must be
            # recorded by the runner; changing them would break matching.
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

    def _operator_supported_at_site(
        self,
        operator: str,
        selected: ast.AST,
        donor_expression: str | None = None,
    ) -> bool:
        if operator in {
                "subtree_replacement", "subtree_composition",
                "correction_term"} and not donor_expression:
            return False
        if operator == "interaction_insertion" and self.n_features < 2:
            return False
        if operator == "operator_substitution":
            return (isinstance(selected, (ast.BinOp, ast.UnaryOp))
                    or (isinstance(selected, ast.Call)
                        and isinstance(selected.func, ast.Name)
                        and selected.func.id in
                        {"sin", "cos", "tanh", "exp", "log"}))
        if operator in {"factor_or_expand", "structural_simplification"}:
            return isinstance(selected, (ast.BinOp, ast.UnaryOp, ast.Call))
        return operator in ZERO_PROVIDER_OPERATORS or operator == "llm_proposed_edit"

    def applicable_operators(
        self,
        parent_expression: str,
        donor_expression: str | None,
        site_path: Sequence[int],
        *,
        seed: int,
        operators: Sequence[str] = INVENTION_OPERATORS,
        known_support_hashes: Iterable[str] = (),
    ) -> tuple[str, ...]:
        """Preflight bounded syntax/grammar without response access or retries.

        Only the selected operator consumes a primary slot.  All operator
        preflight checks must be logged as planning work by the caller.
        """
        root = _parse_body(parent_expression)
        selected = _node_at_path(root, site_path)
        parent_support = tuple(compile_fixed_formula_support(
            parent_expression, self.n_features))
        eligible = []
        known_support = set(map(str, known_support_hashes))
        for operator in operators:
            if operator not in INVENTION_OPERATORS or not self._operator_supported_at_site(
                    operator, selected, donor_expression):
                continue
            request = StructuralEditRequest(
                operator, parent_expression, donor_expression,
                tuple(site_path), int(seed))
            try:
                raw = self._raw_edit(request)
                if len(list(ast.walk(ast.parse(raw, mode="eval")))) > self.maximum_ast_nodes:
                    continue
                support = tuple(compile_fixed_formula_support(
                    raw, self.n_features))
                if (support != parent_support
                        and _stable_hash(support) not in known_support):
                    eligible.append(operator)
            except (ValueError, SyntaxError, TypeError, OverflowError):
                continue
        return tuple(eligible)

    def matched_control_request(
        self,
        primary: StructuralEditRequest,
        *,
        control_operator: str | None = None,
    ) -> StructuralEditRequest:
        """Deterministic *distinct* control under the same frozen opportunity.

        No response arrays are opened. If there is no admissible control the
        caller must record an invalid pair, not substitute a zero reward.
        """
        selected = _node_at_path(
            _parse_body(primary.parent_expression), primary.site_path)
        if control_operator is not None and (
                control_operator not in ZERO_PROVIDER_OPERATORS
                or control_operator == primary.operator
                or not self._operator_supported_at_site(
                    control_operator, selected, primary.donor_expression)):
            raise ValueError("invalid or inapplicable matched control operator")
        # A grammar-level check avoids controls with missing donor, duplicate
        # proposals or provably identical expanded-formula support.
        try:
            primary_support = tuple(compile_fixed_formula_support(
                self._raw_edit(primary), self.n_features))
        except (ValueError, SyntaxError, TypeError, OverflowError) as error:
            raise ValueError("primary edit cannot form a matched pair") from error
        eligible = []
        options = ((control_operator,) if control_operator is not None
                   else INVENTION_OPERATORS)
        for operator in options:
            if (operator == primary.operator
                    or not self._operator_supported_at_site(
                        operator, selected, primary.donor_expression)):
                continue
            trial = StructuralEditRequest(
                operator, primary.parent_expression,
                primary.donor_expression, primary.site_path, primary.seed,
                primary.task_identity, matched_control_of=primary.identity)
            try:
                raw = self._raw_edit(trial)
                if len(list(ast.walk(ast.parse(raw, mode="eval")))) > self.maximum_ast_nodes:
                    continue
                signature = tuple(compile_fixed_formula_support(
                    raw, self.n_features))
                if signature != primary_support:
                    eligible.append(trial)
            except (ValueError, SyntaxError, TypeError, OverflowError):
                continue
        if not eligible:
            raise ValueError("no distinct executable matched control")
        digest = sha256(f"{primary.identity}:control".encode()).digest()
        return eligible[int.from_bytes(digest[:8], "big") % len(eligible)]

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
            # Composition means a multiplicative interaction, distinct from
            # the additive correction_term operator (v3 protocol semantic).
            operation = ast.Mult()
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
        known_canonical_hashes: Iterable[str] = (),
        known_support_hashes: Iterable[str] = (),
    ) -> StructuralEditResult:
        lineage_id = "cbsi-" + request.identity[:24]
        try:
            # Every registered covariate domain must contain observable rows.
            # Empty arrays previously made allclose vacuously report equality.
            if not registered_domains:
                raise ValueError("registered domain set must not be empty")
            checked_domains = []
            for domain in registered_domains:
                values = np.asarray(domain, dtype=float)
                if (values.ndim != 2 or len(values) == 0
                        or values.shape[1] != self.n_features
                        or not np.all(np.isfinite(values))):
                    raise ValueError("registered domain needs finite nonempty covariates")
                checked_domains.append(values)
            parent = self.runtime.dag(request.parent_expression)
            raw = self._raw_edit(request)
            root = ast.parse(raw, mode="eval")
            if len(list(ast.walk(root))) > self.maximum_ast_nodes:
                raise ValueError("edited expression exceeds AST budget")
            dag = self.runtime.dag(raw)
            support = tuple(compile_fixed_formula_support(
                dag.expression, self.n_features))
            support_hash = _stable_hash(support)
            parent_support = tuple(compile_fixed_formula_support(
                parent.expression, self.n_features))
            candidate_predictions = []
            parent_predictions = []
            for values in checked_domains:
                candidate_predictions.append(
                    self.runtime.predict(dag.expression, values))
                parent_predictions.append(
                    self.runtime.predict(parent.expression, values))
            empirical_equivalence = all(
                np.allclose(left, right, rtol=1.0e-10, atol=1.0e-12)
                for left, right in zip(
                    parent_predictions, candidate_predictions,
                    strict=True))
            symbolic_equivalence = dag.canonical_hash == parent.canonical_hash
            # Small expressions admit a symbolic identity check.  Numerical
            # agreement on finite covariates must NEVER imply this identity.
            if not symbolic_equivalence and len(list(ast.walk(root))) <= 96:
                try:
                    left = self.runtime.registry.parse(
                        parent.expression, self.n_features, evaluate=False)
                    right = self.runtime.registry.parse(
                        dag.expression, self.n_features, evaluate=False)
                    if sp.count_ops(left) + sp.count_ops(right) <= 72:
                        symbolic_equivalence = bool(sp.simplify(left - right) == 0)
                except (ValueError, TypeError, ArithmeticError):
                    pass
            known_structures = set(map(str, known_structural_hashes))
            known_canonicals = set(map(str, known_canonical_hashes))
            known_supports = set(map(str, known_support_hashes))
            novelty = (
                not symbolic_equivalence
                and dag.structural_hash != parent.structural_hash
                and dag.structural_hash not in known_structures
                and dag.canonical_hash not in known_canonicals
                and support != parent_support
                and support_hash not in known_supports)
            return StructuralEditResult(
                request.identity, request.operator,
                request.parent_expression, request.donor_expression,
                request.site_path, dag.expression, dag.canonical_hash,
                dag.structural_hash, lineage_id, dag.complexity,
                True, novelty, symbolic_equivalence, None,
                len(checked_domains), 0,
                symbolically_equivalent=symbolic_equivalence,
                empirically_equivalent_on_domains=empirical_equivalence,
                support_hash=support_hash)
        except Exception as error:
            return StructuralEditResult(
                request.identity, request.operator,
                request.parent_expression, request.donor_expression,
                request.site_path, None, None, None, lineage_id, None,
                False, False, False,
                f"{type(error).__name__}:{error}",
                len(registered_domains), 0)


def validate_matched_edit_pair(
    primary_request: StructuralEditRequest,
    control_request: StructuralEditRequest,
    primary: StructuralEditResult,
    control: StructuralEditResult,
) -> None:
    """Fail closed on counterfactual identity before accepting fold credit.

    Numerical fold arrays cannot prove matching on their own.  The runner
    MUST call this guard before counterfactual_edit_credit.
    """
    if (control_request.matched_control_of != primary_request.identity
            or control_request.operator == primary_request.operator
            or any(getattr(primary_request, field) !=
                   getattr(control_request, field)
                   for field in ("parent_expression", "donor_expression",
                                 "site_path", "seed", "task_identity"))
            or primary.request_identity != primary_request.identity
            or control.request_identity != control_request.identity
            or not primary.valid or not primary.novel
            or not control.valid or not control.novel
            or primary.support_hash is None or control.support_hash is None
            or primary.support_hash == control.support_hash):
        raise ValueError("invalid or non-distinct matched edit evidence")


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
    # Cancel the shared parent algebraically, before floating subtraction.
    # Subtracting a large parent first can erase an identifiable edit contrast.
    paired = primary - np.mean(controls, axis=0)
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
        next_count = self.count + 1
        delta = value - self.mean
        next_mean = self.mean + delta / next_count
        next_m2 = self.m2 + delta * (value - next_mean)
        if not math.isfinite(next_mean) or not math.isfinite(next_m2):
            raise ValueError("operator reward update produced nonfinite moments")
        self.count, self.mean, self.m2 = next_count, next_mean, next_m2

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



@dataclass
class FeasibilityAwareContextualPolicy:
    """Experimental CB-SI operator policy with contextual evidence accounting.

    The score is a *heuristic* feasibility-weighted posterior draw, not a
    calibrated Bayesian posterior over scientific formula truth.  Failures
    inform a Beta feasibility model, while only genuinely scored matched
    pairs update the separate Normal-Gamma credit model.  Site-specific
    rewards back off to operator-wide evidence until two local observations.

    The sampled score is P(novel) * (raw_gain_given_novel + credit_weight
    * P(pair_valid | novel) * credit_given_valid_pair).  A failed primary
    updates only novelty feasibility; it is not a conditional pair trial.
    """

    operators: tuple[str, ...] = INVENTION_OPERATORS
    seed: int = 0
    credit_weight: float = 0.5
    raw_global: dict[str, NormalGammaArm] = field(init=False)
    credit_global: dict[str, NormalGammaArm] = field(init=False)
    raw_local: dict[str, NormalGammaArm] = field(default_factory=dict)
    credit_local: dict[str, NormalGammaArm] = field(default_factory=dict)
    feasible: dict[str, list[int]] = field(default_factory=dict)
    paired: dict[str, list[int]] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (not self.operators or len(set(self.operators)) != len(self.operators)
                or any(op not in INVENTION_OPERATORS for op in self.operators)
                or not math.isfinite(self.credit_weight)
                or self.credit_weight < 0):
            raise ValueError("invalid contextual policy contract")
        self.raw_global = {op: NormalGammaArm() for op in self.operators}
        self.credit_global = {op: NormalGammaArm() for op in self.operators}

    @staticmethod
    def site_context(parent_expression: str, site_path: Sequence[int]) -> str:
        site = _node_at_path(_parse_body(parent_expression), site_path)
        if isinstance(site, ast.BinOp):
            kind = "binary"
        elif isinstance(site, ast.UnaryOp):
            kind = "unary"
        elif isinstance(site, ast.Call):
            kind = "function"
        else:
            kind = "leaf"
        return f"{'root' if not site_path else 'internal'}:{kind}"

    @staticmethod
    def _key(context: str, operator: str) -> str:
        return f"{context}/{operator}"

    def choose(
        self, step: int, *, context: str, eligible_operators: Sequence[str],
    ) -> str:
        allowed = tuple(op for op in self.operators if op in eligible_operators)
        if not allowed:
            raise ValueError("no eligible contextual operator")
        rng = np.random.default_rng(
            int(self.seed) + 104729 * (int(step) + 1))
        scores = {}
        for operator in allowed:
            key = self._key(context, operator)
            counts = self.feasible.get(key, [0, 0])
            success = float(rng.beta(1 + counts[0], 1 + counts[1]))
            pair_counts = self.paired.get(key, [0, 0])
            paired_probability = float(rng.beta(
                1 + pair_counts[0], 1 + pair_counts[1]))
            # Local backoff is explicit to avoid overfitting a one-shot site.
            local_raw = self.raw_local.get(key)
            local_credit = self.credit_local.get(key)
            raw_arm = (local_raw if local_raw is not None
                       and local_raw.count >= 2 else self.raw_global[operator])
            credit_arm = (local_credit if local_credit is not None
                          and local_credit.count >= 2
                          else self.credit_global[operator])
            score = success * (
                raw_arm.sample(rng)
                + self.credit_weight * paired_probability * credit_arm.sample(rng))
            scores[operator] = float(score)
        choice = min(allowed, key=lambda op: (-scores[op], op))
        self.decisions.append({
            "step": int(step), "context": str(context),
            "eligible": list(allowed), "choice": choice,
            "scores": scores,
        })
        return choice

    def observe(
        self, *, operator: str, context: str, novel: bool,
        raw_gain: float | None, pair_valid: bool,
        matched_credit: float | None,
    ) -> None:
        if operator not in self.raw_global:
            raise ValueError("unknown contextual operator")
        if (type(novel) is not bool or type(pair_valid) is not bool
                or (raw_gain is None) != (not novel)
                or (matched_credit is None) != (not pair_valid)
                or (pair_valid and not novel)):
            raise ValueError("inconsistent contextual evidence identity")
        raw_value = float(raw_gain) if raw_gain is not None else None
        credit_value = (
            float(matched_credit) if matched_credit is not None else None)
        if any(value is not None and not math.isfinite(value)
               for value in (raw_value, credit_value)):
            raise ValueError("contextual evidence reward is nonfinite")
        key = self._key(context, operator)
        # Stage every update, including possible arithmetic failures, before
        # publishing any part of this evidence event.
        pending_updates = []
        for value, global_arms, local_arms in (
            (raw_value, self.raw_global, self.raw_local),
            (credit_value, self.credit_global, self.credit_local),
        ):
            if value is None:
                continue
            global_arm = copy.deepcopy(global_arms[operator])
            local_arm = copy.deepcopy(local_arms.get(key, NormalGammaArm()))
            global_arm.update(value)
            local_arm.update(value)
            pending_updates.append((global_arms, operator, global_arm))
            pending_updates.append((local_arms, key, local_arm))
        for arms, arm_key, arm in pending_updates:
            arms[arm_key] = arm
        novel_counts = self.feasible.setdefault(key, [0, 0])
        novel_counts[0 if novel else 1] += 1
        if novel:
            pair_counts = self.paired.setdefault(key, [0, 0])
            pair_counts[0 if pair_valid else 1] += 1
        self.observations.append({
            "operator": operator, "context": context,
            "novel": novel, "pair_valid": pair_valid,
            "raw_gain": raw_value, "matched_credit": credit_value,
        })

    def audit(self) -> dict:
        return {
            "schema": "cbsi-feasibility-contextual-policy-v2",
            "operator_set": list(self.operators),
            "seed": self.seed, "credit_weight": self.credit_weight,
            "decisions": list(self.decisions),
            "observations": list(self.observations),
            "feasibility_counts": dict(self.feasible),
            "pair_identification_counts": dict(self.paired),
            "pair_identification_condition": "novel_primary",
            "raw_posterior": {k: v.audit() for k, v in self.raw_global.items()},
            "credit_posterior": {k: v.audit() for k, v in self.credit_global.items()},
            "local_raw_posterior": {k: v.audit() for k, v in self.raw_local.items()},
            "local_credit_posterior": {k: v.audit() for k, v in self.credit_local.items()},
            "claim_boundary": "adaptive-score heuristic; no posterior over formula truth",
        }
