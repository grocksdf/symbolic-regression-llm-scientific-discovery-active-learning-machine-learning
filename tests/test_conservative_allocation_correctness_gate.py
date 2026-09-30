"""Static surface for the conservative allocation Gate."""

from scripts.run_conservative_allocation_correctness_gate import (
    BASELINE, CHALLENGER,
)


def test_allocation_gate_compares_matched_budget_full_coverage_plans():
    assert set(BASELINE) == set(CHALLENGER)
    assert sum(BASELINE.values()) == sum(CHALLENGER.values()) == 6
