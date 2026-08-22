"""Response-free CERT.21 guarded runner and indivisible-ledger checks."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import tempfile

import pytest

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT21_ACQUISITION_ACCESS_AUTHORIZED,
    P3F4_CERT21_ATOMIC_LEDGER_WRITER_AUTHORIZED,
    P3F4_CERT21_HELDOUT_ACCESS_AUTHORIZED,
    P3F4_CERT21_OPERATIONAL_EXECUTION_AUTHORIZED,
    P3F4_CERT21_OPERATIONAL_H0_ACCESS_AUTHORIZED,
    P3F4_CERT21_REAL_DATA_ACCESS_AUTHORIZED,
    P3F4_CERT21_STANDALONE_STATE_MACHINE_AUTHORIZED,
    P3F4_CERT21_SYSTEM_ENTROPY_ACCESS_AUTHORIZED,
    AbstainedExactRejectionBatch,
    CompleteExactRejectionBatch,
    CoordinateBoundIdealByteSource,
    ExactRejectionMAPConfirmationPlan,
    ExternalIdealIndependentBytePremise,
    FailedExactRejectionBatch,
    GuardedOperationalExactRejectionRunner,
    OperationalRejectionDraw,
    build_certified_exact_rejection_runner_plan,
    execute_exact_rejection_workflow,
    execute_frozen_exact_rejection_batch,
    load_and_verify_workflow_ledger,
    write_indivisible_workflow_ledger,
)
from tests.test_pcpi_p3f4_actual_arb_refinement import _fixture
from tests.test_pcpi_p3f4_cert20_exact_rejection_source import _source_fixture


ROOT = Path(__file__).resolve().parents[1]


class _Bytes:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def bytes(self, length: int) -> bytes:
        self.calls.append(length)
        return bytes(length)


def _draw(index: int, class_id: str, *, accepted: bool = True) -> OperationalRejectionDraw:
    return OperationalRejectionDraw(
        state_id=f"state-{index}",
        class_id=class_id,
        proposal_atom_id=f"atom-{index}",
        proposal_role="semantic-core",
        accepted=accepted,
        refinement_rounds=1,
        uniform_prefix_bits=256,
    )


def _runner_fixture():
    provider, actual, source, _ = _source_fixture()
    _, _, common, _, _, refinement, _ = _fixture()
    confirmation = ExactRejectionMAPConfirmationPlan(
        rejection_plan_hash=source.rejection_plan.stable_hash,
        operational_estimand_hash="cert21-estimand",
        class_projector_hash="cert21-projector",
        map_regret_budget=Fraction(9, 10),
        failure_probability=Fraction(1, 2),
        accepted_sample_stages=(2, 4),
    )
    small = replace(
        source,
        confirmation_plan=confirmation,
        selection_accepted_samples=2,
    )
    premise = ExternalIdealIndependentBytePremise()
    runner = build_certified_exact_rejection_runner_plan(
        small,
        actual,
        refinement,
        common,
        provider,
        premise,
    )
    return runner, small


def _workflow(selection_classes: tuple[str, ...], confirmation_classes: tuple[str, ...]):
    runner, source = _runner_fixture()
    base = _Bytes()
    selection_source = CoordinateBoundIdealByteSource(
        base,
        source.selection_coordinate_domain,
    )
    confirmation_source = CoordinateBoundIdealByteSource(
        base,
        source.confirmation_coordinate_domain,
    )
    queues = {
        source.selection_coordinate_domain: list(selection_classes),
        source.confirmation_coordinate_domain: list(confirmation_classes),
    }
    counter = 0

    def factory(coordinate_source):
        nonlocal counter
        counter += 1
        queue = queues[coordinate_source.coordinate_domain]
        if not queue:
            raise ArithmeticError("deterministic draw-failure fixture")
        class_id = queue.pop(0)
        return _draw(counter, class_id)

    ledger = execute_exact_rejection_workflow(
        runner,
        source,
        selection_source,
        confirmation_source,
        factory,
    )
    return runner, source, ledger


def test_cert21_authorizes_only_pure_state_machine_and_atomic_writer() -> None:
    assert P3F4_CERT21_STANDALONE_STATE_MACHINE_AUTHORIZED
    assert P3F4_CERT21_ATOMIC_LEDGER_WRITER_AUTHORIZED
    assert not P3F4_CERT21_OPERATIONAL_EXECUTION_AUTHORIZED
    assert not P3F4_CERT21_OPERATIONAL_H0_ACCESS_AUTHORIZED
    assert not P3F4_CERT21_SYSTEM_ENTROPY_ACCESS_AUTHORIZED
    assert not P3F4_CERT21_REAL_DATA_ACCESS_AUTHORIZED
    assert not P3F4_CERT21_HELDOUT_ACCESS_AUTHORIZED
    assert not P3F4_CERT21_ACQUISITION_ACCESS_AUTHORIZED


def test_runner_plan_binds_cert20_source_actual_evaluator_and_premise() -> None:
    runner, source = _runner_fixture()
    assert runner.source_plan_hash == source.stable_hash
    assert runner.actual_plan_hash == source.actual_plan_hash
    assert runner.refinement_plan_hash == source.refinement_plan_hash
    assert runner.selection_coordinate_domain == source.selection_coordinate_domain
    assert runner.confirmation_coordinate_domain == source.confirmation_coordinate_domain


def test_coordinate_sources_are_logically_disjoint_and_account_every_byte() -> None:
    base = _Bytes()
    selection = CoordinateBoundIdealByteSource(base, "selection")
    confirmation = CoordinateBoundIdealByteSource(base, "confirmation")
    assert selection.bytes(3) == bytes(3)
    assert confirmation.bytes(5) == bytes(5)
    assert selection.coordinate_domain != confirmation.coordinate_domain
    assert selection.bytes_requested == 3 and selection.request_count == 1
    assert confirmation.bytes_requested == 5 and confirmation.request_count == 1
    assert base.calls == [3, 5]


def test_fixed_batch_returns_complete_acceptances_and_stable_transcript() -> None:
    draws = iter((_draw(1, "a"), _draw(2, "b", accepted=False), _draw(3, "a")))
    result = execute_frozen_exact_rejection_batch(
        coordinate_domain="selection",
        required_acceptances=2,
        proposal_cap=4,
        draw_next=lambda: next(draws),
    )
    assert isinstance(result, CompleteExactRejectionBatch)
    assert result.proposal_count == 3
    assert result.accepted_state_ids == ("state-1", "state-3")
    assert result.accepted_class_ids == ("a", "a")
    assert len(result.transcript_hash) == 64


def test_cap_abstention_erases_every_partial_acceptance() -> None:
    draws = iter((_draw(1, "a"), _draw(2, "b", accepted=False)))
    result = execute_frozen_exact_rejection_batch(
        coordinate_domain="selection",
        required_acceptances=2,
        proposal_cap=2,
        draw_next=lambda: next(draws),
    )
    assert isinstance(result, AbstainedExactRejectionBatch)
    assert result.accepted_state_ids == () and result.accepted_class_ids == ()
    assert result.reason == "proposal-cap-exhausted"


def test_draw_failure_becomes_terminal_abstention_without_partial_marks() -> None:
    calls = 0

    def draw_next():
        nonlocal calls
        calls += 1
        if calls == 1:
            return _draw(1, "a")
        raise ArithmeticError("deterministic evaluator failure fixture")

    result = execute_frozen_exact_rejection_batch(
        coordinate_domain="selection",
        required_acceptances=2,
        proposal_cap=3,
        draw_next=draw_next,
    )
    assert isinstance(result, FailedExactRejectionBatch)
    assert result.proposal_count == 1 and result.reason == "draw-failure"
    assert result.accepted_state_ids == () and result.accepted_class_ids == ()


def test_independent_selection_and_first_confirmation_stage_publish_one_result() -> None:
    _, source, ledger = _workflow(("a", "a"), ("a", "a"))
    assert ledger.status == "confirmed"
    assert ledger.candidate_class_id == "a"
    assert ledger.confirmation_accepted_count == 2
    assert ledger.candidate_member_count == 2
    assert ledger.selection_transcript_hash != ledger.confirmation_transcript_hash
    assert len(ledger.selection_state_ids) == source.selection_accepted_samples


def test_no_confirmation_boundary_returns_only_indivisible_abstention() -> None:
    _, _, ledger = _workflow(("a", "a"), ("b", "b", "b", "b"))
    assert ledger.status == "abstained-no-boundary"
    assert ledger.candidate_class_id is None
    assert ledger.selection_state_ids == () and ledger.selection_class_ids == ()
    assert ledger.confirmation_state_ids == ()
    assert ledger.confirmation_accepted_count == 0


def test_selection_or_confirmation_draw_failure_never_leaks_candidate() -> None:
    _, _, selection_failure = _workflow(("a",), ())
    assert selection_failure.status == "abstained-selection-failure"
    assert selection_failure.candidate_class_id is None
    assert selection_failure.selection_state_ids == ()
    _, _, confirmation_failure = _workflow(("a", "a"), ("a",))
    assert confirmation_failure.status == "abstained-confirmation-failure"
    assert confirmation_failure.candidate_class_id is None
    assert confirmation_failure.selection_state_ids == ()
    assert confirmation_failure.confirmation_state_ids == ()


def test_atomic_terminal_ledger_round_trip_and_retry_prohibition() -> None:
    _, _, ledger = _workflow(("a", "a"), ("a", "a"))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "cert21_terminal_ledger.json"
        assert write_indivisible_workflow_ledger(ledger, path) == path
        observed = load_and_verify_workflow_ledger(path)
        assert observed == ledger
        assert not tuple(root.glob("*.partial"))
        with pytest.raises(FileExistsError, match="retry is forbidden"):
            write_indivisible_workflow_ledger(ledger, path)


def test_terminal_ledger_hash_detects_tampering() -> None:
    _, _, ledger = _workflow(("a", "a"), ("a", "a"))
    with tempfile.TemporaryDirectory() as directory:
        path = write_indivisible_workflow_ledger(
            ledger,
            Path(directory) / "ledger.json",
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["candidate_member_count"] = 0
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="hash mismatch"):
            load_and_verify_workflow_ledger(path)


def test_operational_guard_precedes_h0_entropy_and_output_access() -> None:
    runner_plan, _ = _runner_fixture()
    runner = GuardedOperationalExactRejectionRunner(runner_plan)

    class _Forbidden:
        def __getattribute__(self, name):
            raise AssertionError(f"forbidden access: {name}")

    forbidden = _Forbidden()
    with pytest.raises(RuntimeError, match="blocked before H0, entropy or output"):
        runner.run(
            forbidden,
            forbidden,
            forbidden,
            forbidden,
            forbidden,
            forbidden,
            forbidden,
            forbidden,
        )


def test_crossed_product_coordinate_domains_fail_before_any_draw() -> None:
    runner, source = _runner_fixture()
    base = _Bytes()
    crossed = CoordinateBoundIdealByteSource(base, "crossed")
    valid = CoordinateBoundIdealByteSource(base, source.confirmation_coordinate_domain)
    with pytest.raises(ValueError, match="coordinate domains"):
        execute_exact_rejection_workflow(
            runner,
            source,
            crossed,
            valid,
            lambda _: (_ for _ in ()).throw(AssertionError("draw accessed")),
        )
    assert base.calls == []


def test_cert21_runner_freeze_matches_source_and_failure_boundaries() -> None:
    freeze = json.loads(
        (ROOT / "configs/p3f_4_cert21_guarded_runner_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    _, source = _runner_fixture()
    assert freeze["coordinate_policy"]["selection"] == source.selection_coordinate_domain
    assert freeze["coordinate_policy"]["confirmation"] == source.confirmation_coordinate_domain
    assert freeze["failure_policy"]["programming_assertion"] == "propagate-do-not-mask"
    assert freeze["ledger"]["existing_target"] == "fail-no-overwrite-no-retry"
    assert freeze["authorization"]["standalone_state_machine"] is True
    assert freeze["authorization"]["operational_execution"] is False
    assert freeze["authorization"]["formal_experiment"] is False
