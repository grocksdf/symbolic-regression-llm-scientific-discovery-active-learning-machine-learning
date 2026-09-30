"""Static import for realized DRR correctness Gate."""

from scripts.run_realized_drr_correctness_gate import main


def test_realized_correctness_gate_is_callable():
    assert callable(main)
