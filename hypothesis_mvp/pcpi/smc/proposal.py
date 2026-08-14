"""Explicit reversible structure proposals for corrected collapsed P2B SMC."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Iterable

import numpy as np

from hypothesis_mvp.pcpi.reference import ReferenceBank


MOVE_TYPES = ("birth", "death", "replace")


@dataclass(frozen=True)
class ProposalEdge:
    source_id: str
    target_id: str
    move_type: str
    forward_probability: float
    reverse_probability: float
    log_abs_jacobian: float = 0.0

    def __post_init__(self) -> None:
        if self.source_id == self.target_id:
            raise ValueError("proposal edges must change structure")
        if self.move_type not in MOVE_TYPES:
            raise ValueError(f"unsupported proposal move: {self.move_type}")
        values = (
            self.forward_probability,
            self.reverse_probability,
            self.log_abs_jacobian,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("proposal probabilities and Jacobian must be finite")
        if min(self.forward_probability, self.reverse_probability) <= 0.0:
            raise ValueError("forward and reverse proposal probabilities must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "move_type": self.move_type,
            "forward_probability": self.forward_probability,
            "reverse_probability": self.reverse_probability,
            "log_abs_jacobian": self.log_abs_jacobian,
        }


class StructureProposalCatalog:
    """Finite stochastic kernel with explicit reversible move probabilities.

    Continuous coefficients and noise are analytically integrated, so the
    dimension-changing move acts on the collapsed structure marginal. The
    corresponding deterministic dimension matching is the empty auxiliary
    map and has unit Jacobian. Conditional parameters are reconstructed only
    after SMC reaches the final target.
    """

    def __init__(
        self,
        bank: ReferenceBank,
        moves: Iterable[tuple[str, str, str]],
    ) -> None:
        self.bank = bank
        self.structure_ids = tuple(item.structure_id for item in bank.structures)
        self.dimensions = {
            item.structure_id: len(item.basis_terms) for item in bank.structures
        }
        identifiers = set(self.structure_ids)
        raw = tuple(moves)
        if not raw:
            raise ValueError("proposal catalog requires at least one move")
        if len(raw) != len(set(raw)):
            raise ValueError("proposal catalog contains duplicate moves")
        grouped: dict[str, dict[str, list[str]]] = {
            identifier: {} for identifier in self.structure_ids
        }
        move_lookup: dict[tuple[str, str], str] = {}
        for source, target, move_type in raw:
            if source not in identifiers or target not in identifiers:
                raise ValueError("proposal move references an unknown structure")
            if source == target or move_type not in MOVE_TYPES:
                raise ValueError("proposal move must change structure using a known type")
            key = (source, target)
            if key in move_lookup:
                raise ValueError("only one move type is allowed per directed edge")
            move_lookup[key] = move_type
            grouped[source].setdefault(move_type, []).append(target)
        for source, target in move_lookup:
            if (target, source) not in move_lookup:
                raise ValueError(f"proposal edge lacks reverse support: {source}->{target}")
        probabilities: dict[tuple[str, str], float] = {}
        for source, by_type in grouped.items():
            if not by_type:
                raise ValueError(f"structure has no outgoing proposal support: {source}")
            type_probability = 1.0 / len(by_type)
            for targets in by_type.values():
                edge_probability = type_probability / len(targets)
                for target in sorted(targets):
                    probabilities[(source, target)] = edge_probability
        self._edges = tuple(
            ProposalEdge(
                source,
                target,
                move_lookup[(source, target)],
                probabilities[(source, target)],
                probabilities[(target, source)],
                0.0,
            )
            for source, target in sorted(move_lookup)
        )
        self._outgoing = {
            identifier: tuple(edge for edge in self._edges if edge.source_id == identifier)
            for identifier in self.structure_ids
        }
        self._validate_dimensions()
        if not self.is_irreducible:
            raise ValueError("proposal graph must be irreducible")

    @classmethod
    def complete_replace(cls, bank: ReferenceBank) -> "StructureProposalCatalog":
        """Recover the complete symmetric P2A proposal as the default behavior."""

        identifiers = tuple(item.structure_id for item in bank.structures)
        dimensions = {
            item.structure_id: len(item.basis_terms) for item in bank.structures
        }

        def move_type(source: str, target: str) -> str:
            change = dimensions[target] - dimensions[source]
            if change > 0:
                return "birth"
            if change < 0:
                return "death"
            return "replace"

        return cls(
            bank,
            (
                (source, target, move_type(source, target))
                for source in identifiers
                for target in identifiers
                if source != target
            ),
        )

    @property
    def edges(self) -> tuple[ProposalEdge, ...]:
        return self._edges

    def outgoing(self, structure_id: str) -> tuple[ProposalEdge, ...]:
        try:
            return self._outgoing[structure_id]
        except KeyError as error:
            raise KeyError(structure_id) from error

    def sample(self, structure_id: str, rng: np.random.Generator) -> ProposalEdge:
        edges = self.outgoing(structure_id)
        probabilities = np.asarray(
            [edge.forward_probability for edge in edges], dtype=float
        )
        return edges[int(rng.choice(len(edges), p=probabilities))]

    @property
    def proposal_matrix(self) -> np.ndarray:
        locations = {identifier: index for index, identifier in enumerate(self.structure_ids)}
        matrix = np.zeros((len(self.structure_ids), len(self.structure_ids)), dtype=float)
        for edge in self._edges:
            matrix[locations[edge.source_id], locations[edge.target_id]] += (
                edge.forward_probability
            )
        return matrix

    @property
    def row_normalization_error(self) -> float:
        return float(np.max(np.abs(self.proposal_matrix.sum(axis=1) - 1.0)))

    @property
    def is_irreducible(self) -> bool:
        adjacency = {
            identifier: {edge.target_id for edge in self.outgoing(identifier)}
            for identifier in self.structure_ids
        }
        for start in self.structure_ids:
            reached = {start}
            frontier = [start]
            while frontier:
                current = frontier.pop()
                for target in adjacency[current] - reached:
                    reached.add(target)
                    frontier.append(target)
            if reached != set(self.structure_ids):
                return False
        return True

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": "pcpi-structure-proposal-catalog-v1",
            "bank_hash": self.bank.stable_hash,
            "dimensions": self.dimensions,
            "edges": [edge.to_dict() for edge in self._edges],
        }
        material = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return sha256(material).hexdigest()

    def _validate_dimensions(self) -> None:
        for edge in self._edges:
            change = self.dimensions[edge.target_id] - self.dimensions[edge.source_id]
            if edge.move_type == "birth" and change <= 0:
                raise ValueError("birth moves must increase coefficient dimension")
            if edge.move_type == "death" and change >= 0:
                raise ValueError("death moves must decrease coefficient dimension")
            if edge.move_type == "replace" and change != 0:
                raise ValueError("replace moves must preserve coefficient dimension")


def p2b_structure_proposal_catalog(bank: ReferenceBank) -> StructureProposalCatalog:
    """Frozen diagnostic proposal graph, independent of observed outcomes."""

    required = {
        "constant",
        "linear",
        "linear_alias",
        "quadratic",
        "cubic",
        "sinusoid",
        "reciprocal",
    }
    if set(item.structure_id for item in bank.structures) != required:
        raise ValueError("P2B diagnostic proposal requires the frozen diagnostic bank")
    moves = (
        ("constant", "linear", "birth"),
        ("linear", "constant", "death"),
        ("constant", "linear_alias", "birth"),
        ("linear_alias", "constant", "death"),
        ("constant", "sinusoid", "birth"),
        ("sinusoid", "constant", "death"),
        ("constant", "reciprocal", "birth"),
        ("reciprocal", "constant", "death"),
        ("linear", "quadratic", "birth"),
        ("quadratic", "linear", "death"),
        ("quadratic", "cubic", "birth"),
        ("cubic", "quadratic", "death"),
        ("linear", "linear_alias", "replace"),
        ("linear_alias", "linear", "replace"),
        ("linear", "sinusoid", "replace"),
        ("sinusoid", "linear", "replace"),
        ("linear", "reciprocal", "replace"),
        ("reciprocal", "linear", "replace"),
        ("linear_alias", "sinusoid", "replace"),
        ("sinusoid", "linear_alias", "replace"),
        ("linear_alias", "reciprocal", "replace"),
        ("reciprocal", "linear_alias", "replace"),
        ("sinusoid", "reciprocal", "replace"),
        ("reciprocal", "sinusoid", "replace"),
    )
    return StructureProposalCatalog(bank, moves)


__all__ = [
    "MOVE_TYPES",
    "ProposalEdge",
    "StructureProposalCatalog",
    "p2b_structure_proposal_catalog",
]
