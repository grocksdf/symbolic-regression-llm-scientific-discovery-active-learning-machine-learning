"""Deterministic exhaustive-state SMC/RJMCMC integration reference."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .posterior import (
    OpenTargetContract,
    OpenTargetExactPosterior,
    fit_open_target_exact_posterior,
)
from .rjmcmc import (
    ProposalKind,
    build_collapsed_rjmcmc_proposal,
    metropolis_hastings_transition,
)


@dataclass(frozen=True)
class ExhaustiveSMCStep:
    step: int
    observation_index: int
    log_evidence_increment: float
    posterior_probabilities: np.ndarray
    post_move_probabilities: np.ndarray
    maximum_move_invariance_error: float
    maximum_detailed_balance_error: float

    def __post_init__(self) -> None:
        before = np.ascontiguousarray(self.posterior_probabilities, dtype=float)
        after = np.ascontiguousarray(self.post_move_probabilities, dtype=float)
        if before.shape != after.shape or before.ndim != 1:
            raise ValueError("SMC step probability vectors must align")
        if not math.isclose(float(before.sum()), 1.0, abs_tol=1e-12):
            raise ValueError("SMC posterior weights must be normalized")
        before.setflags(write=False)
        after.setflags(write=False)
        object.__setattr__(self, "posterior_probabilities", before)
        object.__setattr__(self, "post_move_probabilities", after)


@dataclass(frozen=True)
class ExhaustiveSequentialSMCResult:
    proposal_kind: ProposalKind
    observation_order: tuple[int, ...]
    prior: OpenTargetExactPosterior
    final_posterior: OpenTargetExactPosterior
    steps: tuple[ExhaustiveSMCStep, ...]
    log_evidence: float
    batch_log_evidence: float
    maximum_batch_sequential_probability_error: float
    evidence_telescoping_error: float


def _validated_sequence(
    registered_actions: np.ndarray,
    targets: np.ndarray,
    observation_order: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    x = np.asarray(registered_actions, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(targets, dtype=float).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(x) < 3:
        raise ValueError("registered actions and targets must be finite and aligned")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("registered actions and targets must be finite")
    order = tuple(range(len(x))) if observation_order is None else tuple(observation_order)
    if sorted(order) != list(range(len(x))):
        raise ValueError("observation order must be a permutation of registered rows")
    return x, y, order


def _advance_exact_step(
    contract: OpenTargetContract,
    x: np.ndarray,
    y: np.ndarray,
    order: tuple[int, ...],
    proposal_kind: ProposalKind,
    step: int,
    previous_log_evidence: float,
) -> tuple[OpenTargetExactPosterior, ExhaustiveSMCStep]:
    indices = order[:step]
    current = fit_open_target_exact_posterior(
        contract,
        x,
        y[np.asarray(indices, dtype=int)],
        sequential=True,
        observation_indices=indices,
    )
    target = np.asarray(
        [member.posterior_probability for member in current.generative_posterior.members],
        dtype=float,
    )
    proposal = build_collapsed_rjmcmc_proposal(current, proposal_kind)
    transition = metropolis_hastings_transition(proposal, target)
    moved = target @ transition.matrix
    result = ExhaustiveSMCStep(
        step=step,
        observation_index=order[step - 1],
        log_evidence_increment=(
            current.generative_posterior.log_evidence - previous_log_evidence
        ),
        posterior_probabilities=target,
        post_move_probabilities=moved,
        maximum_move_invariance_error=float(np.max(np.abs(moved - target))),
        maximum_detailed_balance_error=transition.maximum_detailed_balance_error,
    )
    return current, result


def run_exhaustive_sequential_smc_reference(
    contract: OpenTargetContract,
    registered_actions: np.ndarray,
    targets: np.ndarray,
    proposal_kind: ProposalKind,
    *,
    observation_order: tuple[int, ...] | None = None,
) -> ExhaustiveSequentialSMCResult:
    """Sequentially reweight every registered latent state, then apply RJ moves.

    Keeping one exhaustive state per component removes Monte Carlo error.  The
    routine therefore isolates update algebra, evidence telescoping, and
    reversible-kernel invariance without making an efficiency or efficacy claim.
    """

    x, y, order = _validated_sequence(
        registered_actions, targets, observation_order
    )
    prior = fit_open_target_exact_posterior(
        contract,
        x,
        np.empty(0, dtype=float),
        observation_indices=(),
    )
    previous_log_evidence = prior.generative_posterior.log_evidence
    if abs(previous_log_evidence) > 1e-12:
        raise AssertionError("proper finite-slice prior must have zero log evidence")
    steps: list[ExhaustiveSMCStep] = []
    current = prior
    for step in range(1, len(order) + 1):
        current, diagnostic = _advance_exact_step(
            contract,
            x,
            y,
            order,
            proposal_kind,
            step,
            previous_log_evidence,
        )
        steps.append(diagnostic)
        previous_log_evidence = current.generative_posterior.log_evidence

    batch = fit_open_target_exact_posterior(
        contract,
        x,
        y,
        sequential=False,
        observation_indices=tuple(range(len(x))),
    )
    sequential_probabilities = np.asarray(
        [
            member.posterior_probability
            for member in current.generative_posterior.members
        ]
    )
    batch_probabilities = np.asarray(
        [member.posterior_probability for member in batch.generative_posterior.members]
    )
    log_evidence = math.fsum(step.log_evidence_increment for step in steps)
    return ExhaustiveSequentialSMCResult(
        proposal_kind=proposal_kind,
        observation_order=order,
        prior=prior,
        final_posterior=current,
        steps=tuple(steps),
        log_evidence=log_evidence,
        batch_log_evidence=batch.generative_posterior.log_evidence,
        maximum_batch_sequential_probability_error=float(
            np.max(np.abs(sequential_probabilities - batch_probabilities))
        ),
        evidence_telescoping_error=abs(
            log_evidence - batch.generative_posterior.log_evidence
        ),
    )
