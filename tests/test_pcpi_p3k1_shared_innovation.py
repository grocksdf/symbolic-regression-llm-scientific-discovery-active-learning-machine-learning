"""P3K.1 version and no-data Gate tests."""

from __future__ import annotations

import inspect

import pytest

from hypothesis_mvp.pcpi import (
    P3J_CLASS_CONDITIONAL_JOINT_METHOD,
    P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD,
    P3J_CLASS_POSTERIOR_UPDATE_METHOD,
    P3J_OPERATIONAL_LIFECYCLE,
    P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD,
    P3K_SHARED_INNOVATION_JOINT_METHOD,
    P3K_SHARED_INNOVATION_RESIDUAL_METHOD,
    P3K_OPERATIONAL_LIFECYCLE,
    initialize_class_conditional_residual_state,
)
from scripts import run_pcpi_p3j15_formal_real_acquisition as p3j15
from scripts import run_pcpi_p3k1_shared_innovation_correctness as gate
from tests.test_pcpi_p3j1_class_conditional_semiparametric import _components


def test_p3j_identifiers_remain_immutable_and_p3k_has_new_identity() -> None:
    assert P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD == (
        "frozen-class-conditional-prequential-kt-dyadic-polya-tree-v1"
    )
    assert P3J_CLASS_CONDITIONAL_JOINT_METHOD == (
        "normalized-class-conditional-pit-density-composition-v1"
    )
    assert P3J_CLASS_POSTERIOR_UPDATE_METHOD == (
        "calibrated-class-factor-times-conjugate-structure-update-v1"
    )
    assert P3J_OPERATIONAL_LIFECYCLE == (
        "class-conditional-family-score-select-reveal-advance-exactly-once-v1"
    )
    assert P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD != (
        P3K_SHARED_INNOVATION_RESIDUAL_METHOD
    )
    assert P3J_CLASS_CONDITIONAL_JOINT_METHOD != (
        P3K_SHARED_INNOVATION_JOINT_METHOD
    )
    assert P3J_CLASS_POSTERIOR_UPDATE_METHOD != (
        P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD
    )
    assert P3J_OPERATIONAL_LIFECYCLE != P3K_OPERATIONAL_LIFECYCLE


def test_current_state_is_p3k_and_has_one_within_power_residual_law() -> None:
    state = initialize_class_conditional_residual_state(_components().partition)
    assert state.method == P3K_SHARED_INNOVATION_RESIDUAL_METHOD
    assert not hasattr(state, "residual_states")
    assert state.residual_law is not None


def test_no_data_gate_passes_every_frozen_decision() -> None:
    config = gate._load_config(gate.CONFIG)
    result = gate._evaluate(config)
    assert result["status"] == "passed-correctness-no-real-experiment-authorized"
    assert all(result["decisions"].values())
    assert result["maximum_power_class_update_identity_error"] <= 8e-16
    assert result["real_data_access"] is False
    assert result["heldout_access"] is False
    assert result["simulated_experiment"] is False


def test_retired_p3j15_entry_point_cannot_run_the_p3k_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        p3j15,
        "build_parser",
        lambda *args, **kwargs: type(
            "Parser", (), {"parse_args": lambda self: object()}
        )(),
    )
    with pytest.raises(RuntimeError, match="P3J.15 is frozen"):
        p3j15.main()
    source = inspect.getsource(p3j15.main)
    assert source.index("parse_args") < source.index("P3J.15 is frozen")
    assert "run(" not in source
