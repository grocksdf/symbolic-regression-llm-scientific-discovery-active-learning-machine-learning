"""P3J.11 measured-pool ordering tests; no real dataset access."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hypothesis_mvp.pcpi import OperationalClassConditionalDecision
from hypothesis_mvp.pcpi import p3j_measured_pool
from hypothesis_mvp.pcpi.reference import DevelopmentStandardizer
from tests.test_pcpi_p3j2_operational_class_conditional import _scores
from tests.test_pcpi_p3j8_run_identity import _case


class _Oracle:
    def __init__(self, action: np.ndarray, target: float, events: list[str]):
        self.action = np.asarray(action, dtype=float).reshape(1, -1)
        self.target = float(target)
        self.events = events

    def acquire_indices(self, indices: np.ndarray):
        self.events.append("oracle")
        return self.action, np.asarray([self.target]), np.asarray(indices, dtype=int)


def _standardizer() -> DevelopmentStandardizer:
    return DevelopmentStandardizer(
        feature_mean=np.asarray([0.0]),
        feature_scale=np.asarray([1.0]),
        target_mean=0.0,
        target_scale=1.0,
    )


def test_decision_precedes_one_matching_oracle_reveal_and_advance(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, actions, ids, predictive, representative, _, workspace = _case(tmp_path)
    events: list[str] = []
    decision = OperationalClassConditionalDecision(
        prior_state_hash=state.stable_hash,
        selected_candidate_id=3,
        selected_action=actions[1],
        local_index=1,
        scores=_scores(state.target_partition.stable_hash),
    )

    def decide(*args, **kwargs):
        events.append("decision")
        return decision

    def advance(*args, **kwargs):
        events.append("advance")
        assert args[4] == 3
        assert np.array_equal(args[5], actions[1])
        assert args[6] == 0.31
        return state

    monkeypatch.setattr(p3j_measured_pool, "run_p3j_formal_query", decide)
    monkeypatch.setattr(p3j_measured_pool, "admit_p3j_formal_response", advance)
    result = p3j_measured_pool.run_p3j_measured_pool_query(
        workspace, state, actions, ids, predictive, representative,
        _Oracle(actions[1], 0.31, events), _standardizer(),
        eig_min_samples=32, eig_max_samples=512,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
    )
    assert events == ["decision", "oracle", "advance"]
    assert result.revealed_candidate_id == 3
    assert result.revealed_target == 0.31


def test_decision_failure_never_calls_oracle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, actions, ids, predictive, representative, _, workspace = _case(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        p3j_measured_pool,
        "run_p3j_formal_query",
        lambda *a, **k: (_ for _ in ()).throw(FloatingPointError("decision failed")),
    )
    with pytest.raises(FloatingPointError, match="decision failed"):
        p3j_measured_pool.run_p3j_measured_pool_query(
            workspace, state, actions, ids, predictive, representative,
            _Oracle(actions[1], 0.31, events), _standardizer(),
            eig_min_samples=32, eig_max_samples=512,
            eig_error_safety_factor=4.0, eig_growth_factor=2,
        )
    assert events == []


def test_crossed_oracle_id_or_coordinates_never_reaches_advance(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, actions, ids, predictive, representative, _, workspace = _case(tmp_path)
    decision = OperationalClassConditionalDecision(
        prior_state_hash=state.stable_hash, selected_candidate_id=3,
        selected_action=actions[1], local_index=1,
        scores=_scores(state.target_partition.stable_hash),
    )
    monkeypatch.setattr(
        p3j_measured_pool, "run_p3j_formal_query", lambda *a, **k: decision
    )
    advanced = False

    def forbidden(*args, **kwargs):
        nonlocal advanced
        advanced = True
        raise AssertionError("crossed oracle reached advance")

    monkeypatch.setattr(p3j_measured_pool, "admit_p3j_formal_response", forbidden)
    bad_id = _Oracle(actions[1], 0.31, [])
    bad_id.acquire_indices = lambda indices: (
        actions[1:2], np.asarray([0.31]), np.asarray([5])
    )
    with pytest.raises(AssertionError, match="different acquisition index"):
        p3j_measured_pool.run_p3j_measured_pool_query(
            workspace, state, actions, ids, predictive, representative,
            bad_id, _standardizer(), eig_min_samples=32, eig_max_samples=512,
            eig_error_safety_factor=4.0, eig_growth_factor=2,
        )
    with pytest.raises(AssertionError, match="coordinates differ"):
        p3j_measured_pool.run_p3j_measured_pool_query(
            workspace, state, actions, ids, predictive, representative,
            _Oracle(actions[2], 0.31, []), _standardizer(),
            eig_min_samples=32, eig_max_samples=512,
            eig_error_safety_factor=4.0, eig_growth_factor=2,
        )
    assert not advanced


def test_adapter_source_has_single_oracle_call_after_decision() -> None:
    source = inspect.getsource(p3j_measured_pool.run_p3j_measured_pool_query)
    assert source.count("oracle.acquire_indices") == 1
    assert source.index("run_p3j_formal_query") < source.index("oracle.acquire_indices")
    assert source.index("oracle.acquire_indices") < source.index(
        "admit_p3j_formal_response"
    )
    assert not any(token in source for token in (
        "validation", "heldout", "future", "candidate_targets", "rng", "retry"
    ))
