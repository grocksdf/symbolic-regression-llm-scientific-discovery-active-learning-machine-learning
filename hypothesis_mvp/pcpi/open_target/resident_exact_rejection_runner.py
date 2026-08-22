"""CERT.21 guarded exact-rejection selection/confirmation runner.

The pure batch and ledger machinery is executable for deterministic source
checks.  The operational entry point remains guarded before provider, entropy
or output access until an identity-bound user Gate passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable
import uuid

from .grammar import TypedExpression
from .posterior import OpenTargetContract
from .resident_actual_arb_refinement import CertifiedActualArbRefinementPlan
from .resident_certified_function_space import CertifiedResidentFunctionSpacePlan
from .resident_exact_rejection_source import (
    CertifiedExactRejectionSourcePlan,
    CertifiedRationalInterval,
    ExactLazyBernoulliResult,
    ExactRejectionProposal,
    ExternalIdealIndependentBytePremise,
    RandomByteSource,
    SystemEntropyIdealByteSource,
    certify_rejection_acceptance_at_refinement_round,
    draw_exact_rejection_proposal,
    exact_lazy_bernoulli,
    intersect_rejection_acceptance_intervals,
    select_fixed_candidate_from_independent_pilot,
)
from .resident_h0_parameter_balls import CertifiedFullStateH0ParameterBallProvider
from .resident_prebit_refinement import CertifiedPreBitRefinementPlan


P3F4_CERT21_RUNNER_SCHEMA = "pcpi-p3f4-cert21-guarded-exact-rejection-runner-v1"
P3F4_CERT21_LEDGER_SCHEMA = "pcpi-p3f4-cert21-indivisible-evidence-ledger-v1"
P3F4_CERT21_STANDALONE_STATE_MACHINE_AUTHORIZED = True
P3F4_CERT21_ATOMIC_LEDGER_WRITER_AUTHORIZED = True
P3F4_CERT21_OPERATIONAL_EXECUTION_AUTHORIZED = False
P3F4_CERT21_OPERATIONAL_H0_ACCESS_AUTHORIZED = False
P3F4_CERT21_SYSTEM_ENTROPY_ACCESS_AUTHORIZED = False
P3F4_CERT21_REAL_DATA_ACCESS_AUTHORIZED = False
P3F4_CERT21_HELDOUT_ACCESS_AUTHORIZED = False
P3F4_CERT21_ACQUISITION_ACCESS_AUTHORIZED = False
CERT21_FAIL_CLOSED_DRAW_ERRORS = (
    ArithmeticError,
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _hash_payload(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CertifiedExactRejectionRunnerPlan:
    schema: str
    source_plan_hash: str
    actual_plan_hash: str
    refinement_plan_hash: str
    common_target_plan_hash: str
    provider_contract_hash: str
    ideal_byte_premise_hash: str
    selection_coordinate_domain: str
    confirmation_coordinate_domain: str
    ledger_schema: str = P3F4_CERT21_LEDGER_SCHEMA
    incomplete_policy: str = "single-terminal-ledger-no-retry-no-partial-result"
    operational_execution_authorized: bool = False

    def __post_init__(self) -> None:
        identities = (
            self.source_plan_hash,
            self.actual_plan_hash,
            self.refinement_plan_hash,
            self.common_target_plan_hash,
            self.provider_contract_hash,
            self.ideal_byte_premise_hash,
        )
        if (
            self.schema != P3F4_CERT21_RUNNER_SCHEMA
            or not all(identities)
            or self.ledger_schema != P3F4_CERT21_LEDGER_SCHEMA
            or not self.selection_coordinate_domain
            or not self.confirmation_coordinate_domain
            or self.selection_coordinate_domain == self.confirmation_coordinate_domain
            or self.incomplete_policy
            != "single-terminal-ledger-no-retry-no-partial-result"
            or self.operational_execution_authorized
        ):
            raise ValueError("CERT.21 runner plan is invalid or prematurely authorized")

    @property
    def stable_hash(self) -> str:
        return _hash_payload(self.__dict__)


def build_certified_exact_rejection_runner_plan(
    source: CertifiedExactRejectionSourcePlan,
    actual: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    premise: ExternalIdealIndependentBytePremise,
) -> CertifiedExactRejectionRunnerPlan:
    if (
        source.actual_plan_hash != actual.stable_hash
        or source.refinement_plan_hash != refinement.stable_hash
        or source.common_target_plan_hash != common.stable_hash
        or source.provider_contract_hash != provider.parameter_provider_contract_hash
        or source.ideal_byte_premise_hash != premise.stable_hash
    ):
        raise ValueError("CERT.21 crossed source, evaluator, provider or premise identities")
    return CertifiedExactRejectionRunnerPlan(
        schema=P3F4_CERT21_RUNNER_SCHEMA,
        source_plan_hash=source.stable_hash,
        actual_plan_hash=actual.stable_hash,
        refinement_plan_hash=refinement.stable_hash,
        common_target_plan_hash=common.stable_hash,
        provider_contract_hash=provider.parameter_provider_contract_hash,
        ideal_byte_premise_hash=premise.stable_hash,
        selection_coordinate_domain=source.selection_coordinate_domain,
        confirmation_coordinate_domain=source.confirmation_coordinate_domain,
    )


class CoordinateBoundIdealByteSource:
    """Bind every byte request to one declared logical product coordinate."""

    def __init__(self, base: RandomByteSource, coordinate_domain: str) -> None:
        if not coordinate_domain:
            raise ValueError("CERT.21 byte coordinate domain is empty")
        self._base = base
        self.coordinate_domain = str(coordinate_domain)
        self.bytes_requested = 0
        self.request_count = 0

    def bytes(self, length: int) -> bytes:
        result = self._base.bytes(length)
        self.bytes_requested += int(length)
        self.request_count += 1
        return result


@dataclass(frozen=True)
class OperationalRejectionDraw:
    state_id: str
    class_id: str
    proposal_atom_id: str
    proposal_role: str
    accepted: bool
    refinement_rounds: int
    uniform_prefix_bits: int

    def __post_init__(self) -> None:
        if (
            not self.state_id
            or not self.class_id
            or not self.proposal_atom_id
            or self.proposal_role not in {"semantic-core", "analytic-tail"}
            or self.refinement_rounds < 1
            or self.uniform_prefix_bits < 256
            or self.uniform_prefix_bits % 256
        ):
            raise ValueError("CERT.21 operational draw audit is invalid")

    @property
    def stable_hash(self) -> str:
        return _hash_payload(self.__dict__)


@dataclass(frozen=True)
class CompleteExactRejectionBatch:
    coordinate_domain: str
    required_acceptances: int
    proposal_cap: int
    proposal_count: int
    accepted_state_ids: tuple[str, ...]
    accepted_class_ids: tuple[str, ...]
    transcript_hash: str
    total_refinement_rounds: int
    total_uniform_prefix_bits: int
    status: str = "complete"

    def __post_init__(self) -> None:
        if (
            not self.coordinate_domain
            or self.required_acceptances < 1
            or self.proposal_cap < self.required_acceptances
            or not self.required_acceptances == len(self.accepted_state_ids)
            or len(self.accepted_state_ids) != len(self.accepted_class_ids)
            or not 1 <= self.proposal_count <= self.proposal_cap
            or not self.transcript_hash
            or self.total_refinement_rounds < self.proposal_count
            or self.total_uniform_prefix_bits < 256 * self.proposal_count
            or self.status != "complete"
        ):
            raise ValueError("CERT.21 complete rejection batch is invalid")


@dataclass(frozen=True)
class AbstainedExactRejectionBatch:
    coordinate_domain: str
    required_acceptances: int
    proposal_cap: int
    proposal_count: int
    transcript_hash: str
    reason: str
    accepted_state_ids: tuple[str, ...] = ()
    accepted_class_ids: tuple[str, ...] = ()
    status: str = "abstained"

    def __post_init__(self) -> None:
        if (
            not self.coordinate_domain
            or self.required_acceptances < 1
            or self.proposal_count != self.proposal_cap
            or not self.transcript_hash
            or self.reason != "proposal-cap-exhausted"
            or self.accepted_state_ids
            or self.accepted_class_ids
            or self.status != "abstained"
        ):
            raise ValueError("CERT.21 abstention leaked an incomplete batch")


@dataclass(frozen=True)
class FailedExactRejectionBatch:
    coordinate_domain: str
    required_acceptances: int
    proposal_cap: int
    proposal_count: int
    transcript_hash: str
    reason: str = "draw-failure"
    accepted_state_ids: tuple[str, ...] = ()
    accepted_class_ids: tuple[str, ...] = ()
    status: str = "abstained"

    def __post_init__(self) -> None:
        if (
            not self.coordinate_domain
            or self.required_acceptances < 1
            or self.proposal_cap < self.required_acceptances
            or not 0 <= self.proposal_count < self.proposal_cap
            or not self.transcript_hash
            or self.reason != "draw-failure"
            or self.accepted_state_ids
            or self.accepted_class_ids
            or self.status != "abstained"
        ):
            raise ValueError("CERT.21 draw failure leaked an incomplete batch")


ExactRejectionBatchResult = (
    CompleteExactRejectionBatch
    | AbstainedExactRejectionBatch
    | FailedExactRejectionBatch
)


def _advance_transcript(previous: str, draw: OperationalRejectionDraw) -> str:
    return _hash_payload((previous, draw.stable_hash))


def execute_frozen_exact_rejection_batch(
    *,
    coordinate_domain: str,
    required_acceptances: int,
    proposal_cap: int,
    draw_next: Callable[[], OperationalRejectionDraw],
) -> ExactRejectionBatchResult:
    """Execute one fixed-cap batch and erase all partial marks on abstention."""

    required = int(required_acceptances)
    cap = int(proposal_cap)
    if not coordinate_domain or required < 1 or cap < required:
        raise ValueError("CERT.21 batch budget is invalid")
    transcript = _hash_payload((coordinate_domain, required, cap))
    states: list[str] = []
    classes: list[str] = []
    rounds = 0
    bits = 0
    for proposal_count in range(1, cap + 1):
        try:
            draw = draw_next()
        except CERT21_FAIL_CLOSED_DRAW_ERRORS:
            states.clear()
            classes.clear()
            return FailedExactRejectionBatch(
                coordinate_domain,
                required,
                cap,
                proposal_count - 1,
                transcript,
            )
        transcript = _advance_transcript(transcript, draw)
        rounds += draw.refinement_rounds
        bits += draw.uniform_prefix_bits
        if draw.accepted:
            states.append(draw.state_id)
            classes.append(draw.class_id)
        if len(states) == required:
            return CompleteExactRejectionBatch(
                coordinate_domain,
                required,
                cap,
                proposal_count,
                tuple(states),
                tuple(classes),
                transcript,
                rounds,
                bits,
            )
    states.clear()
    classes.clear()
    return AbstainedExactRejectionBatch(
        coordinate_domain,
        required,
        cap,
        cap,
        transcript,
        "proposal-cap-exhausted",
    )


def _proposal_state_id(proposal: ExactRejectionProposal) -> str:
    return f"{proposal.expression.raw_ast_id}:{proposal.component_state_id}"


def _actual_boundary_callback(
    source_plan: CertifiedExactRejectionSourcePlan,
    actual: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    proposal: ExactRejectionProposal,
) -> Callable[[int], CertifiedRationalInterval]:
    accumulated: CertifiedRationalInterval | None = None

    def boundary(round_index: int) -> CertifiedRationalInterval:
        nonlocal accumulated
        current = certify_rejection_acceptance_at_refinement_round(
            source_plan,
            actual,
            refinement,
            common,
            provider,
            proposal,
            round_index=round_index,
        )
        accumulated = (
            current
            if accumulated is None
            else intersect_rejection_acceptance_intervals(accumulated, current)
        )
        return accumulated

    return boundary


def _draw_operational_rejection(
    source_plan: CertifiedExactRejectionSourcePlan,
    actual: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    contract: OpenTargetContract,
    source: RandomByteSource,
) -> OperationalRejectionDraw:
    proposal = draw_exact_rejection_proposal(source_plan, contract, source)
    comparison: ExactLazyBernoulliResult = exact_lazy_bernoulli(
        _actual_boundary_callback(
            source_plan,
            actual,
            refinement,
            common,
            provider,
            proposal,
        ),
        source,
    )
    return OperationalRejectionDraw(
        state_id=_proposal_state_id(proposal),
        class_id=proposal.class_id,
        proposal_atom_id=proposal.atom_id,
        proposal_role=proposal.role,
        accepted=comparison.accepted,
        refinement_rounds=comparison.rounds_used,
        uniform_prefix_bits=comparison.uniform_prefix_bits,
    )


@dataclass(frozen=True)
class ExactRejectionWorkflowLedger:
    schema: str
    runner_plan_hash: str
    source_plan_hash: str
    status: str
    reason: str | None
    selection_transcript_hash: str
    confirmation_transcript_hash: str | None
    candidate_class_id: str | None
    selection_proposal_count: int
    confirmation_proposal_count: int
    confirmation_accepted_count: int
    candidate_member_count: int
    confirmation_state_ids: tuple[str, ...]
    selection_state_ids: tuple[str, ...] = ()
    selection_class_ids: tuple[str, ...] = ()
    no_retry: bool = True
    heldout_access: bool = False
    acquisition_access: bool = False

    def __post_init__(self) -> None:
        terminal = {
            "confirmed",
            "abstained-selection-cap",
            "abstained-confirmation-cap",
            "abstained-selection-failure",
            "abstained-confirmation-failure",
            "abstained-no-boundary",
        }
        if (
            self.schema != P3F4_CERT21_LEDGER_SCHEMA
            or not self.runner_plan_hash
            or not self.source_plan_hash
            or self.status not in terminal
            or not self.selection_transcript_hash
            or not self.no_retry
            or self.heldout_access
            or self.acquisition_access
        ):
            raise ValueError("CERT.21 workflow ledger identity is invalid")
        if self.status != "confirmed" and (
            self.candidate_class_id is not None
            or self.confirmation_state_ids
            or self.selection_state_ids
            or self.selection_class_ids
        ):
            raise ValueError("CERT.21 abstention ledger leaked a partial result")
        if self.status == "confirmed" and (
            not self.candidate_class_id
            or not self.confirmation_transcript_hash
            or not self.confirmation_state_ids
            or not self.selection_state_ids
            or not self.selection_class_ids
        ):
            raise ValueError("CERT.21 confirmed ledger is incomplete")

    @property
    def stable_hash(self) -> str:
        return _hash_payload(self.__dict__)

    def to_dict(self) -> dict[str, object]:
        return {**self.__dict__, "ledger_hash": self.stable_hash}


def _abstention_ledger(
    runner: CertifiedExactRejectionRunnerPlan,
    source: CertifiedExactRejectionSourcePlan,
    selection: ExactRejectionBatchResult,
    *,
    status: str,
    confirmation: ExactRejectionBatchResult | None = None,
) -> ExactRejectionWorkflowLedger:
    return ExactRejectionWorkflowLedger(
        schema=P3F4_CERT21_LEDGER_SCHEMA,
        runner_plan_hash=runner.stable_hash,
        source_plan_hash=source.stable_hash,
        status=status,
        reason=status,
        selection_transcript_hash=selection.transcript_hash,
        confirmation_transcript_hash=(
            None if confirmation is None else confirmation.transcript_hash
        ),
        candidate_class_id=None,
        selection_proposal_count=selection.proposal_count,
        confirmation_proposal_count=(0 if confirmation is None else confirmation.proposal_count),
        confirmation_accepted_count=0,
        candidate_member_count=0,
        confirmation_state_ids=(),
    )


def _confirmation_boundary(
    source: CertifiedExactRejectionSourcePlan,
    accepted_count: int,
    candidate_count: int,
) -> tuple[bool, bool]:
    stages = source.confirmation_plan.accepted_sample_stages
    if accepted_count not in stages:
        return False, False
    confirmed = source.confirmation_plan.certifies(accepted_count, candidate_count)
    exhausted = accepted_count == stages[-1]
    return confirmed, exhausted


def _execute_confirmation(
    source: CertifiedExactRejectionSourcePlan,
    coordinate_source: CoordinateBoundIdealByteSource,
    draw_factory: Callable[[RandomByteSource], OperationalRejectionDraw],
    candidate_class_id: str,
) -> tuple[ExactRejectionBatchResult, int, bool]:
    domain = source.confirmation_coordinate_domain
    maximum = source.confirmation_plan.maximum_accepted_samples
    cap = source.proposal_cap
    transcript = _hash_payload((domain, maximum, cap))
    states: list[str] = []
    classes: list[str] = []
    rounds = 0
    bits = 0
    candidate_count = 0
    for proposal_count in range(1, cap + 1):
        try:
            draw = draw_factory(coordinate_source)
        except CERT21_FAIL_CLOSED_DRAW_ERRORS:
            states.clear()
            classes.clear()
            return (
                FailedExactRejectionBatch(
                    domain,
                    maximum,
                    cap,
                    proposal_count - 1,
                    transcript,
                ),
                0,
                False,
            )
        transcript = _advance_transcript(transcript, draw)
        rounds += draw.refinement_rounds
        bits += draw.uniform_prefix_bits
        if draw.accepted:
            states.append(draw.state_id)
            classes.append(draw.class_id)
            candidate_count += int(draw.class_id == candidate_class_id)
        confirmed, exhausted = _confirmation_boundary(
            source,
            len(classes),
            candidate_count,
        )
        if confirmed or exhausted:
            result = CompleteExactRejectionBatch(
                domain,
                len(states),
                cap,
                proposal_count,
                tuple(states),
                tuple(classes),
                transcript,
                rounds,
                bits,
            )
            return result, candidate_count, confirmed
    states.clear()
    classes.clear()
    return (
        AbstainedExactRejectionBatch(
            domain,
            maximum,
            cap,
            cap,
            transcript,
            "proposal-cap-exhausted",
        ),
        0,
        False,
    )


def execute_exact_rejection_workflow(
    runner: CertifiedExactRejectionRunnerPlan,
    source: CertifiedExactRejectionSourcePlan,
    selection_source: CoordinateBoundIdealByteSource,
    confirmation_source: CoordinateBoundIdealByteSource,
    draw_factory: Callable[[RandomByteSource], OperationalRejectionDraw],
) -> ExactRejectionWorkflowLedger:
    """Execute frozen selection then fresh confirmation as one terminal result."""

    if runner.source_plan_hash != source.stable_hash:
        raise ValueError("CERT.21 workflow crossed runner and source identities")
    if (
        selection_source.coordinate_domain != source.selection_coordinate_domain
        or confirmation_source.coordinate_domain != source.confirmation_coordinate_domain
    ):
        raise ValueError("CERT.21 workflow crossed product-coordinate domains")
    selection = execute_frozen_exact_rejection_batch(
        coordinate_domain=source.selection_coordinate_domain,
        required_acceptances=source.selection_accepted_samples,
        proposal_cap=source.selection_proposal_cap,
        draw_next=lambda: draw_factory(selection_source),
    )
    if isinstance(selection, FailedExactRejectionBatch):
        return _abstention_ledger(
            runner,
            source,
            selection,
            status="abstained-selection-failure",
        )
    if isinstance(selection, AbstainedExactRejectionBatch):
        return _abstention_ledger(
            runner,
            source,
            selection,
            status="abstained-selection-cap",
        )
    candidate = select_fixed_candidate_from_independent_pilot(
        source,
        selection.accepted_class_ids,
        selection_transcript_hash=selection.transcript_hash,
    )
    confirmation, member_count, confirmed = _execute_confirmation(
        source,
        confirmation_source,
        draw_factory,
        candidate.candidate_class_id,
    )
    if isinstance(confirmation, FailedExactRejectionBatch):
        return _abstention_ledger(
            runner,
            source,
            selection,
            status="abstained-confirmation-failure",
            confirmation=confirmation,
        )
    if isinstance(confirmation, AbstainedExactRejectionBatch):
        return _abstention_ledger(
            runner,
            source,
            selection,
            status="abstained-confirmation-cap",
            confirmation=confirmation,
        )
    if not confirmed:
        return _abstention_ledger(
            runner,
            source,
            selection,
            status="abstained-no-boundary",
            confirmation=confirmation,
        )
    return ExactRejectionWorkflowLedger(
        schema=P3F4_CERT21_LEDGER_SCHEMA,
        runner_plan_hash=runner.stable_hash,
        source_plan_hash=source.stable_hash,
        status="confirmed",
        reason=None,
        selection_transcript_hash=selection.transcript_hash,
        confirmation_transcript_hash=confirmation.transcript_hash,
        candidate_class_id=candidate.candidate_class_id,
        selection_proposal_count=selection.proposal_count,
        confirmation_proposal_count=confirmation.proposal_count,
        confirmation_accepted_count=len(confirmation.accepted_class_ids),
        candidate_member_count=member_count,
        confirmation_state_ids=confirmation.accepted_state_ids,
        selection_state_ids=selection.accepted_state_ids,
        selection_class_ids=selection.accepted_class_ids,
    )


def write_indivisible_workflow_ledger(
    ledger: ExactRejectionWorkflowLedger,
    output_path: str | Path,
) -> Path:
    """Publish one complete terminal ledger with fsync plus atomic replace."""

    target = Path(output_path)
    if target.suffix.lower() != ".json":
        raise ValueError("CERT.21 terminal ledger path must end in .json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("CERT.21 terminal ledger already exists; retry is forbidden")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    payload = (_canonical_json(ledger.to_dict()) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("CERT.21 terminal ledger short write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, target)
        temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def load_and_verify_workflow_ledger(
    input_path: str | Path,
) -> ExactRejectionWorkflowLedger:
    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "ledger_hash" not in raw:
        raise ValueError("CERT.21 terminal ledger payload is invalid")
    expected_hash = str(raw.pop("ledger_hash"))
    required = set(ExactRejectionWorkflowLedger.__dataclass_fields__)
    if set(raw) != required:
        raise ValueError("CERT.21 terminal ledger fields changed")
    for name in (
        "confirmation_state_ids",
        "selection_state_ids",
        "selection_class_ids",
    ):
        raw[name] = tuple(raw[name])
    ledger = ExactRejectionWorkflowLedger(**raw)
    if ledger.stable_hash != expected_hash:
        raise ValueError("CERT.21 terminal ledger hash mismatch")
    return ledger


class GuardedOperationalExactRejectionRunner:
    def __init__(self, plan: CertifiedExactRejectionRunnerPlan) -> None:
        self.plan = plan
        self.plan_hash = plan.stable_hash

    def run(
        self,
        source_plan,
        actual,
        refinement,
        common,
        provider,
        contract,
        premise,
        output_path,
    ):
        if not all(
            (
                P3F4_CERT21_OPERATIONAL_EXECUTION_AUTHORIZED,
                P3F4_CERT21_OPERATIONAL_H0_ACCESS_AUTHORIZED,
                P3F4_CERT21_SYSTEM_ENTROPY_ACCESS_AUTHORIZED,
            )
        ):
            raise RuntimeError(
                "CERT.21 operational runner is blocked before H0, entropy or output access"
            )
        observed = build_certified_exact_rejection_runner_plan(
            source_plan,
            actual,
            refinement,
            common,
            provider,
            premise,
        )
        if observed.stable_hash != self.plan_hash:
            raise ValueError("CERT.21 operational runner identity changed after freeze")
        base = SystemEntropyIdealByteSource(premise)
        selection_source = CoordinateBoundIdealByteSource(
            base,
            source_plan.selection_coordinate_domain,
        )
        confirmation_source = CoordinateBoundIdealByteSource(
            base,
            source_plan.confirmation_coordinate_domain,
        )
        draw_factory = lambda source: _draw_operational_rejection(
            source_plan,
            actual,
            refinement,
            common,
            provider,
            contract,
            source,
        )
        ledger = execute_exact_rejection_workflow(
            self.plan,
            source_plan,
            selection_source,
            confirmation_source,
            draw_factory,
        )
        return write_indivisible_workflow_ledger(ledger, output_path)


__all__ = [
    "P3F4_CERT21_ACQUISITION_ACCESS_AUTHORIZED",
    "P3F4_CERT21_ATOMIC_LEDGER_WRITER_AUTHORIZED",
    "P3F4_CERT21_HELDOUT_ACCESS_AUTHORIZED",
    "P3F4_CERT21_LEDGER_SCHEMA",
    "P3F4_CERT21_OPERATIONAL_EXECUTION_AUTHORIZED",
    "P3F4_CERT21_OPERATIONAL_H0_ACCESS_AUTHORIZED",
    "P3F4_CERT21_REAL_DATA_ACCESS_AUTHORIZED",
    "P3F4_CERT21_RUNNER_SCHEMA",
    "P3F4_CERT21_STANDALONE_STATE_MACHINE_AUTHORIZED",
    "P3F4_CERT21_SYSTEM_ENTROPY_ACCESS_AUTHORIZED",
    "AbstainedExactRejectionBatch",
    "CertifiedExactRejectionRunnerPlan",
    "CompleteExactRejectionBatch",
    "CoordinateBoundIdealByteSource",
    "ExactRejectionWorkflowLedger",
    "FailedExactRejectionBatch",
    "GuardedOperationalExactRejectionRunner",
    "OperationalRejectionDraw",
    "build_certified_exact_rejection_runner_plan",
    "execute_exact_rejection_workflow",
    "execute_frozen_exact_rejection_batch",
    "load_and_verify_workflow_ledger",
    "write_indivisible_workflow_ledger",
]
