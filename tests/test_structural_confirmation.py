"""No-data algebra for independent Scientist confirmation."""

import json

from hypothesis_mvp.discovery.structural_confirmation import (
    evaluate_structural_confirmation, run_confirmation_freeze_gate,
)


def test_degenerate_control_is_a_defined_loss_not_an_exclusion():
    result = evaluate_structural_confirmation(
        full_viable=True, no_llm_viable=False, llm_admitted=True,
        llm_fold_gains=(1., 2.), llm_fold_tolerances=(1e-9, 1e-9),
        full_minus_no_llm_log_score=None, comparison_tolerance=1e-9)
    assert result["passed"] is True
    assert result["no_llm_collapse_counted_as_control_failure"] is True
    assert result["predictive_tie_break_required"] is False


def test_both_ready_requires_positive_proper_score():
    common = dict(
        full_viable=True, no_llm_viable=True, llm_admitted=True,
        llm_fold_gains=(1., 2.), llm_fold_tolerances=(1e-9, 1e-9),
        comparison_tolerance=1e-9)
    assert evaluate_structural_confirmation(
        **common, full_minus_no_llm_log_score=.1)["passed"] is True
    assert evaluate_structural_confirmation(
        **common, full_minus_no_llm_log_score=-.1)["passed"] is False


def test_frozen_registration_has_no_execution_authority():
    path = (__import__("pathlib").Path(__file__).resolve().parents[1] /
            "configs" /
            "scientific_typed_synthesis_independent_confirmation_v1.json")
    result = run_confirmation_freeze_gate(
        json.loads(path.read_text(encoding="utf-8")))
    assert result["passed"] is True
    assert result["eligible_dataset_families"] == []
    assert result["execution_authorized"] is False
    assert result["real_data_accessed"] is False
