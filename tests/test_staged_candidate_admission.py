import numpy as np
import pytest

from hypothesis_mvp.discovery.staged_candidate_admission import (
    staged_global_protected_evidence_decision,
)


def test_direct_pass_preserves_original_gate():
    result = staged_global_protected_evidence_decision(
        np.array([3.0, 3.0]), np.array([0, 1]),
        np.array([-100.0, -100.0]), np.array([0, 1]),
        (0, 1), 5.0)
    assert result["passed"] is True
    assert result["stage"] == "direct"
    assert result["appeal_opened"] is False


def test_independent_appeal_can_rescue_positive_first_stage():
    result = staged_global_protected_evidence_decision(
        np.array([1.0, 1.0]), np.array([0, 1]),
        np.array([3.0, 3.0]), np.array([0, 1]),
        (0, 1), 5.0)
    assert result["passed"] is True
    assert result["stage"] == "appeal"
    assert result["quarantined"] is True
    assert result["appeal_opened"] is True


def test_nonpositive_first_stage_does_not_spend_appeal():
    result = staged_global_protected_evidence_decision(
        np.array([-1.0, .5]), np.array([0, 1]),
        np.array([100.0, 100.0]), np.array([0, 1]),
        (0, 1), 5.0)
    assert result["passed"] is False
    assert result["stage"] == "rejected-before-appeal"
    assert result["appeal_opened"] is False


def test_appeal_remains_region_safe():
    result = staged_global_protected_evidence_decision(
        np.array([2.0, 1.0]), np.array([0, 1]),
        np.array([10.0, -4.0]), np.array([0, 1]),
        (0, 1), 5.0)
    assert result["passed"] is False
    assert result["minimum_region_log_predictive_ratio"] < 0


def test_nonfinite_stage_fails_closed():
    with pytest.raises(ValueError):
        staged_global_protected_evidence_decision(
            np.array([np.nan]), np.array([0]),
            np.array([1.0]), np.array([0]), (0,), 1.0)
