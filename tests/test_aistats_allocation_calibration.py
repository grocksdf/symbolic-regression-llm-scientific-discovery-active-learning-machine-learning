"""Static contracts for paired allocation-only calibration."""

from scripts.run_aistats_allocation_calibration import CONFIGS


def test_calibration_has_only_matched_baseline_and_challenger():
    assert set(CONFIGS) == {
        "allocation_baseline", "allocation_challenger"}
