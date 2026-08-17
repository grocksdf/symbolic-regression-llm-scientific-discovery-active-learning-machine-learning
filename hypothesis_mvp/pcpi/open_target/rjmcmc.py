"""Exact transition-matrix checks for collapsed finite-slice RJMCMC moves.

Continuous coefficients, discrepancy coordinates, and noise variance are
analytically integrated before these moves.  Dimension matching therefore
uses an empty auxiliary variable and a unit Jacobian.  This is a correctness
reference for reversible target correction, not a scalable search kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Literal

import numpy as np

from .posterior import OpenTargetExactPosterior


ProposalKind = Literal["complete-uniform", "prior-independence"]


@dataclass(frozen=True)
class CollapsedStateDescriptor:
    state_id: str
    raw_ast_id: str
    node_count: int
    discrepancy_active: bool
    kernel_state_id: str
    integrated_parameter_dimension: int


@dataclass(frozen=True)
class CollapsedRJMCMCProposal:
    proposal_kind: ProposalKind
    descriptors: tuple[CollapsedStateDescriptor, ...]
    matrix: np.ndarray
    log_abs_jacobian: float = 0.0

    def __post_init__(self) -> None:
        matrix = np.ascontiguousarray(self.matrix, dtype=float)
        count = len(self.descriptors)
        if count < 2 or matrix.shape != (count, count):
            raise ValueError("RJMCMC proposal matrix has an invalid shape")
        if np.any(matrix < 0.0) or not np.all(np.isfinite(matrix)):
            raise ValueError("RJMCMC proposal probabilities must be finite and non-negative")
        if np.max(np.abs(matrix.sum(axis=1) - 1.0)) > 1e-13:
            raise ValueError("RJMCMC proposal rows must sum to one")
        support = matrix > 0.0
        if not np.array_equal(support, support.T):
            raise ValueError("RJMCMC proposal must have bidirectional support")
        if self.log_abs_jacobian != 0.0:
            raise ValueError("collapsed finite-slice dimension matching has unit Jacobian")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": "pcpi-p3f2-collapsed-rjmcmc-proposal-v1",
            "proposal_kind": self.proposal_kind,
            "log_abs_jacobian": self.log_abs_jacobian,
            "states": [descriptor.__dict__ for descriptor in self.descriptors],
            "matrix": self.matrix.tolist(),
        }
        material = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return sha256(material).hexdigest()

    def move_type(self, source: int, target: int) -> str:
        left = self.descriptors[source]
        right = self.descriptors[target]
        if left.state_id == right.state_id:
            return "self"
        if left.raw_ast_id == right.raw_ast_id:
            if left.discrepancy_active != right.discrepancy_active:
                return "spike-switch"
            return "kernel-transition"
        change = right.integrated_parameter_dimension - left.integrated_parameter_dimension
        if change > 0:
            return "birth"
        if change < 0:
            return "death"
        return "replace"


@dataclass(frozen=True)
class CollapsedRJMCMCTransition:
    proposal: CollapsedRJMCMCProposal
    target_probabilities: np.ndarray
    matrix: np.ndarray
    maximum_detailed_balance_error: float
    maximum_stationarity_error: float

    def __post_init__(self) -> None:
        target = np.ascontiguousarray(self.target_probabilities, dtype=float)
        matrix = np.ascontiguousarray(self.matrix, dtype=float)
        count = len(self.proposal.descriptors)
        if target.shape != (count,) or matrix.shape != (count, count):
            raise ValueError("RJMCMC transition dimensions are inconsistent")
        if np.any(target <= 0.0) or not math.isclose(
            float(target.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("RJMCMC target probabilities must be positive and normalized")
        if np.any(matrix < -1e-15) or np.max(np.abs(matrix.sum(axis=1) - 1.0)) > 1e-12:
            raise ValueError("RJMCMC transition is not row stochastic")
        target.setflags(write=False)
        matrix.setflags(write=False)
        object.__setattr__(self, "target_probabilities", target)
        object.__setattr__(self, "matrix", matrix)


def _descriptors(posterior: OpenTargetExactPosterior) -> tuple[CollapsedStateDescriptor, ...]:
    expression_lookup = {
        expression.raw_ast_id: expression for expression in posterior.expressions
    }
    return tuple(
        CollapsedStateDescriptor(
            state_id=member.state_id,
            raw_ast_id=member.structure.structure_id,
            node_count=expression_lookup[member.structure.structure_id].node_count,
            discrepancy_active=member.discrepancy_active,
            kernel_state_id=member.kernel_state_id,
            integrated_parameter_dimension=len(member.posterior_mean),
        )
        for member in posterior.generative_posterior.members
    )


def build_collapsed_rjmcmc_proposal(
    posterior: OpenTargetExactPosterior,
    proposal_kind: ProposalKind,
) -> CollapsedRJMCMCProposal:
    descriptors = _descriptors(posterior)
    count = len(descriptors)
    if proposal_kind == "complete-uniform":
        matrix = np.full((count, count), 1.0 / (count - 1), dtype=float)
        np.fill_diagonal(matrix, 0.0)
    elif proposal_kind == "prior-independence":
        prior = np.asarray(
            [member.joint_prior_probability for member in posterior.generative_posterior.members],
            dtype=float,
        )
        prior /= prior.sum()
        matrix = np.tile(prior, (count, 1))
    else:
        raise ValueError(f"unknown P3F.2 proposal kind: {proposal_kind}")
    return CollapsedRJMCMCProposal(proposal_kind, descriptors, matrix)


def metropolis_hastings_transition(
    proposal: CollapsedRJMCMCProposal,
    target_probabilities: np.ndarray,
) -> CollapsedRJMCMCTransition:
    target = np.asarray(target_probabilities, dtype=float).reshape(-1)
    count = len(proposal.descriptors)
    if target.shape != (count,) or np.any(target <= 0.0):
        raise ValueError("positive target probabilities must align with proposal states")
    target = target / target.sum()
    transition = np.zeros_like(proposal.matrix)
    for source in range(count):
        for destination in range(count):
            if source == destination:
                continue
            forward = proposal.matrix[source, destination]
            if forward == 0.0:
                continue
            reverse = proposal.matrix[destination, source]
            log_ratio = (
                math.log(target[destination])
                - math.log(target[source])
                + math.log(reverse)
                - math.log(forward)
                + proposal.log_abs_jacobian
            )
            acceptance = 1.0 if log_ratio >= 0.0 else math.exp(log_ratio)
            transition[source, destination] = forward * acceptance
        transition[source, source] = 1.0 - float(transition[source].sum())
    flow = target[:, None] * transition
    detailed_balance_error = float(np.max(np.abs(flow - flow.T)))
    stationarity_error = float(np.max(np.abs(target @ transition - target)))
    return CollapsedRJMCMCTransition(
        proposal,
        target,
        transition,
        detailed_balance_error,
        stationarity_error,
    )
