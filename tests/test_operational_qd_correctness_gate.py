"""Static import for operational QD correctness Gate."""

from scripts.run_operational_qd_correctness_gate import main


def test_operational_qd_gate_is_callable():
    assert callable(main)
