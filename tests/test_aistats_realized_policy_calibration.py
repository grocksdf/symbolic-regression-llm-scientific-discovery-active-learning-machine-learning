"""Bounded symmetric audit helper fixtures."""

from scripts.build_aistats_realized_policy_calibration import _symmetric


def test_symmetric_helper_is_bounded_for_tiny_initial_risk():
    value = _symmetric({"risk_curve": [1e-9, 2e-8, 1e-8]})
    assert -1.0 <= value <= 1.0
