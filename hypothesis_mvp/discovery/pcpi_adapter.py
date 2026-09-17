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
        if isinstance(node.right, ast.Constant) and type(node.right.value) is int and 1 <= node.right.value <= 3:
            return tuple(v * node.right.value for v in _monomial(node.left, n_features))
    raise ValueError("unsupported expression; only additive closed-library polynomial terms are supported")


def _term(power: tuple[int, ...]) -> str:
    active = [(i, p) for i, p in enumerate(power) if p]
    if not active:
        return "intercept"
    if len(active) == 1 and active[0][1] <= 3:
        i, p = active[0]
        return f"x{i}" + {1: "", 2: "_sq", 3: "_cube"}[p]
    if len(active) == 2 and all(p == 1 for _, p in active):
        return f"x{active[0][0]}_x{active[1][0]}"
    raise ValueError("monomial outside PCPI's closed basis library")


def structural_terms(expression: str, n_features: int) -> tuple[str, ...]:
    if type(n_features) is not int or n_features < 1 or len(expression) > 4096:
        raise ValueError("invalid adapter feature/expression controls")
    root = ast.parse(expression, mode="eval").body
    nodes = list(ast.walk(root))
    if len(nodes) > 256:
        raise ValueError("adapter AST budget exceeded")
    def summands(node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return summands(node.left) + summands(node.right)
        return [node]
    terms = tuple(sorted(_term(_monomial(node, n_features)) for node in summands(root)))
    if len(set(terms)) != len(terms):
        raise ValueError("duplicate/cancelling terms require prior canonical validation")
    return terms


@dataclass(frozen=True)
class FrozenDiscoveryModel:
    bank: ReferenceBank
    n_features: int
    candidate_bindings: tuple[tuple[str, str, str], ...]
    exploration_identity: str
    refit_policy: str = "discard-fitted-coefficients-refit-closed-basis"

    @property
    def stable_hash(self) -> str:
        return sha256(json.dumps({"bank": self.bank.stable_hash, "features": self.n_features,
            "bindings": self.candidate_bindings, "exploration": self.exploration_identity,
            "policy": self.refit_policy}, sort_keys=True).encode()).hexdigest()

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
    coefficient_policy: str,
) -> FrozenDiscoveryModel:
    if coefficient_policy != "discard-fitted-coefficients-refit-closed-basis":
        raise ValueError("explicit structural-refit authorization required")
    if len(exploration_identity) != 64 or any(c not in "0123456789abcdef" for c in exploration_identity):
        raise ValueError("exploration identity must be SHA-256")
    supports: dict[tuple[str, ...], str] = {}
    bindings = []
    for candidate in candidates:
        expression = str(candidate["expression"])
        terms = structural_terms(expression, n_features)
        identifier = "discovery-" + sha256(json.dumps(terms).encode()).hexdigest()[:24]
        supports[terms] = identifier
        bindings.append((str(candidate["source"]), expression, identifier))
    if len(supports) < 2:
        raise ValueError("at least two distinct model supports required")
    structures = tuple(ReferenceStructure(identifier, " + ".join(terms), terms, 1.0 / len(supports))
                       for terms, identifier in sorted(supports.items()))
    return FrozenDiscoveryModel(ReferenceBank(structures, prior), n_features,
                                tuple(sorted(bindings)), exploration_identity)
