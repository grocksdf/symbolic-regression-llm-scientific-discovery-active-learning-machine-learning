"""Explicit structural-refit adapter to PCPI's finite conjugate model.

Discovery coefficients are discarded ONLY under a caller-declared structural
refit contract. This is conditional-on-exploration development inference, not
evidence for a response-independent prior over all possible scientific laws.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

import numpy as np
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.acquisition import ClassPartition, class_partition

from hypothesis_mvp.pcpi.reference import (
    NormalInverseGammaPrior, ReferenceBank, ReferenceStructure,
    SequentialReferencePosterior,
    ExactPosterior, aggregate_decision_equivalent_classes,
    budget_resolved_distance_threshold,
)
from .source_stacking import source_family
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    compile_fixed_formula_support,
)


class DiscoveryAdapterError(ValueError):
    """A response-free adapter failure safe to report across isolation."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.public_diagnostic = code


def _scaled_sum(scale: ast.AST, node: ast.AST) -> ast.AST:
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return ast.BinOp(_scaled_sum(scale, node.left), node.op,
                         _scaled_sum(scale, node.right))
    return ast.BinOp(scale, ast.Mult(), node)


class _ExternalScalarDistributor(ast.NodeTransformer):
    """Distribute numeric outer amplitudes only; never symbolic products."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Mult):
            if _scalar_amplitude(node.left) and isinstance(node.right, ast.BinOp) \
                    and isinstance(node.right.op, (ast.Add, ast.Sub)):
                return self.visit(_scaled_sum(node.left, node.right))
            if _scalar_amplitude(node.right) and isinstance(node.left, ast.BinOp) \
                    and isinstance(node.left.op, (ast.Add, ast.Sub)):
                return self.visit(_scaled_sum(node.right, node.left))
        return node


def additive_closed_form(expression: str, n_features: int) -> str:
    """Canonical additive boundary form for the registered closed basis."""
    if type(n_features) is not int or n_features < 1 or len(expression) > 4096:
        raise ValueError("invalid adapter feature/expression controls")
    root = ast.parse(expression, mode="eval")
    if len(list(ast.walk(root))) > 256:
        raise ValueError("adapter AST budget exceeded")
    transformed = ast.fix_missing_locations(_ExternalScalarDistributor().visit(root))
    if len(list(ast.walk(transformed))) > 256:
        raise ValueError("adapter AST budget exceeded after scalar distribution")
    text = ast.unparse(transformed.body)
    if len(text) > 4096:
        raise ValueError("adapter expression budget exceeded after scalar distribution")
    return text


def _monomial(node: ast.AST, n_features: int) -> tuple[int, ...]:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if not math.isfinite(node.value) or node.value == 0:
            raise ValueError("zero/nonfinite terms are not admissible")
        return (0,) * n_features
    if isinstance(node, ast.Name) and node.id in {f"x{i}" for i in range(n_features)}:
        return tuple(int(node.id == f"x{i}") for i in range(n_features))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _monomial(node.operand, n_features)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return tuple(a + b for a, b in zip(_monomial(node.left, n_features), _monomial(node.right, n_features)))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        if isinstance(node.right, ast.Constant) and type(node.right.value) is int and 1 <= node.right.value <= 4:
            return tuple(v * node.right.value for v in _monomial(node.left, n_features))
    raise DiscoveryAdapterError("unsupported-closed-basis-factor")


def _term(power: tuple[int, ...]) -> str:
    active = [(i, p) for i, p in enumerate(power) if p]
    if not active:
        return "intercept"
    if len(active) == 1 and active[0][1] <= 3:
        i, p = active[0]
        return f"x{i}" + {1: "", 2: "_sq", 3: "_cube"}[p]
    if len(active) == 2 and all(p == 1 for _, p in active):
        return f"x{active[0][0]}_x{active[1][0]}"
    if sum(p for _, p in active) <= 4 and all(1 <= p <= 4 for _, p in active):
        return "monomial_" + "_".join(f"x{i}p{p}" for i, p in active)
    raise DiscoveryAdapterError("monomial-outside-registered-degree-four-library")


def _scalar_amplitude(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _scalar_amplitude(node.operand)
    return (isinstance(node, ast.Constant) and type(node.value) in (int, float)
            and math.isfinite(node.value) and node.value != 0)


def _closed_term(node: ast.AST, n_features: int) -> str:
    """Map one additive term to a registered non-evaluating basis token."""
    # A fitted scalar amplitude carries no structural information and is
    # intentionally discarded by the structural-refit contract.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _closed_term(node.operand, n_features)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if _scalar_amplitude(node.left):
            return _closed_term(node.right, n_features)
        if _scalar_amplitude(node.right):
            return _closed_term(node.left, n_features)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"sin", "cos", "tanh"}
            and not node.keywords and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in {f"x{i}" for i in range(n_features)}):
        return f"{node.func.id}_{node.args[0].id}"
    return _term(_monomial(node, n_features))


def structural_terms(expression: str, n_features: int) -> tuple[str, ...]:
    root = ast.parse(additive_closed_form(expression, n_features), mode="eval").body
    nodes = list(ast.walk(root))
    if len(nodes) > 256:
        raise ValueError("adapter AST budget exceeded")
    def summands(node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return summands(node.left) + summands(node.right)
        return [node]
    terms = tuple(sorted(_closed_term(node, n_features) for node in summands(root)))
    if len(set(terms)) != len(terms):
        raise DiscoveryAdapterError("duplicate-or-cancelling-structural-terms")
    return terms


@dataclass(frozen=True)
class FrozenDiscoveryModel:
    bank: ReferenceBank
    n_features: int
    candidate_bindings: tuple[tuple[str, str, str], ...]
    exploration_identity: str
    refit_policy: str = "discard-fitted-coefficients-refit-closed-basis"
    minimum_supports: int = 2

    def __post_init__(self) -> None:
        if (type(self.minimum_supports) is not int
                or self.minimum_supports not in {1, 2}
                or len(self.bank.structures) < self.minimum_supports):
            raise ValueError("frozen discovery model violates its support floor")

    @property
    def stable_hash(self) -> str:
        return sha256(json.dumps({"bank": self.bank.stable_hash, "features": self.n_features,
            "bindings": self.candidate_bindings, "exploration": self.exploration_identity,
            "policy": self.refit_policy, "minimum_supports": self.minimum_supports},
            sort_keys=True).encode()).hexdigest()

    def engine(self, expected_identity: str) -> SequentialReferencePosterior:
        if expected_identity != self.stable_hash:
            raise ValueError("frozen discovery model identity mismatch")
        return SequentialReferencePosterior(self.bank)


@dataclass(frozen=True)
class FrozenDiscoveryTarget:
    model_identity: str
    initial_data_identity: str
    action_domain_identity: str
    measurement_budget: int
    partition: ClassPartition
    initial_posterior: ExactPosterior

    @property
    def stable_hash(self) -> str:
        material = (self.model_identity, self.initial_data_identity,
                    self.action_domain_identity, self.measurement_budget,
                    self.partition.stable_hash, asdict(self.initial_posterior))
        return sha256(json.dumps(material, sort_keys=True, allow_nan=False,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray)
            else value).encode()).hexdigest()


def freeze_discovery_target(
    model: FrozenDiscoveryModel, initial_data: RoleDataset,
    action_domain: np.ndarray, *, measurement_budget: int,
    expected_model_identity: str,
) -> FrozenDiscoveryTarget:
    """Freeze an H0 pushforward using opened development responses only."""
    if initial_data.role is not DataRole.DEVELOPMENT:
        raise ValueError("H0 requires opened development data")
    if initial_data.X.shape[1] != model.n_features:
        raise ValueError("H0 feature dimension crossed model identity")
    domain = np.ascontiguousarray(action_domain, dtype=np.float64)
    if domain.ndim != 2 or not len(domain) or domain.shape[1] != model.n_features or not np.all(np.isfinite(domain)):
        raise ValueError("invalid registered action domain")
    engine = model.engine(expected_model_identity)
    posterior = engine.fit_batch(initial_data.X, initial_data.y)
    classes = aggregate_decision_equivalent_classes(engine, posterior, domain,
        distance_threshold=budget_resolved_distance_threshold(measurement_budget))
    partition = class_partition(posterior, classes)
    digest = sha256(str(domain.shape).encode() + domain.tobytes()).hexdigest()
    return FrozenDiscoveryTarget(model.stable_hash, initial_data.fingerprint, digest,
                                 int(measurement_budget), partition, posterior)


def freeze_discovery_model(
    candidates: Sequence[Mapping[str, str]], *, n_features: int,
    prior: NormalInverseGammaPrior, exploration_identity: str,
    coefficient_policy: str, source_prior_weights: Mapping[str, float] | None = None,
    minimum_supports: int = 2,
) -> FrozenDiscoveryModel:
    if coefficient_policy != "discard-fitted-coefficients-refit-closed-basis":
        raise ValueError("explicit structural-refit authorization required")
    return _freeze_model_with_support_parser(
        candidates, n_features=n_features, prior=prior,
        exploration_identity=exploration_identity,
        coefficient_policy=coefficient_policy,
        source_prior_weights=source_prior_weights,
        minimum_supports=minimum_supports, support_parser=structural_terms)


def freeze_expanded_formula_model(
    candidates: Sequence[Mapping[str, str]], *, n_features: int,
    prior: NormalInverseGammaPrior, exploration_identity: str,
    coefficient_policy: str,
    source_prior_weights: Mapping[str, float] | None = None,
    minimum_supports: int = 2,
) -> FrozenDiscoveryModel:
    """Opt-in finite bank; freezes inner constants, refits outer amplitudes.

    This independent reference entry point does not authorize measured PCPI,
    generation, admission, or any historical frozen protocol.
    """
    if coefficient_policy != "discard-outer-amplitudes-freeze-inner-parameters-v1":
        raise ValueError("explicit expanded-formula refit contract required")
    return _freeze_model_with_support_parser(
        candidates, n_features=n_features, prior=prior,
        exploration_identity=exploration_identity,
        coefficient_policy=coefficient_policy,
        source_prior_weights=source_prior_weights,
        minimum_supports=minimum_supports,
        support_parser=compile_fixed_formula_support)


def _freeze_model_with_support_parser(
    candidates, *, n_features, prior, exploration_identity,
    coefficient_policy, source_prior_weights, minimum_supports,
    support_parser,
):
    if len(exploration_identity) != 64 or any(c not in "0123456789abcdef" for c in exploration_identity):
        raise ValueError("exploration identity must be SHA-256")
    if type(minimum_supports) is not int or minimum_supports not in {1, 2}:
        raise ValueError("discovery model support floor must be one or two")
    supports: dict[tuple[str, ...], tuple[str, str]] = {}
    bindings = []
    for candidate in candidates:
        expression = str(candidate["expression"])
        try:
            terms = support_parser(expression, n_features)
        except (SyntaxError, ValueError) as error:
            digest = sha256(expression.encode("utf-8", errors="replace")).hexdigest()[:16]
            code = getattr(error, "public_diagnostic", type(error).__name__)
            raise DiscoveryAdapterError(
                f"candidate-not-adaptable:{digest}:{code}"
            ) from error
        identifier = "discovery-" + sha256(json.dumps(terms).encode()).hexdigest()[:24]
        family = source_family(candidate)
        existing = supports.setdefault(terms, (identifier, family))
        if existing[1] != family:
            raise DiscoveryAdapterError("duplicate-support-crosses-source-families")
        bindings.append((str(candidate["source"]), expression, identifier))
    if len(supports) < minimum_supports:
        raise ValueError(f"at least {minimum_supports} distinct model supports required")
    families = {family for _, family in supports.values()}
    if source_prior_weights is None:
        weights = {family: sum(value[1] == family for value in supports.values()) / len(supports)
                   for family in families}
    else:
        weights = {str(key): float(value) for key, value in source_prior_weights.items()}
        if (set(weights) != families or any(not math.isfinite(value) or value < 0.0
                for value in weights.values())
                or not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=2e-12)):
            raise ValueError("invalid hierarchical source prior weights")
    active = {family for family, weight in weights.items() if weight > 2e-12}
    counts = {family: sum(value[1] == family for value in supports.values())
              for family in active}
    structures = tuple(ReferenceStructure(identifier, " + ".join(terms), terms,
        weights[family] / counts[family])
        for terms, (identifier, family) in sorted(supports.items()) if family in active)
    bindings = [binding for binding in bindings
                if any(binding[2] == structure.structure_id for structure in structures)]
    if len(structures) < minimum_supports:
        raise ValueError(
            f"source fallback leaves fewer than {minimum_supports} model supports")
    return FrozenDiscoveryModel(ReferenceBank(structures, prior), n_features,
                                tuple(sorted(bindings)), exploration_identity,
                                refit_policy=coefficient_policy,
                                minimum_supports=minimum_supports)
