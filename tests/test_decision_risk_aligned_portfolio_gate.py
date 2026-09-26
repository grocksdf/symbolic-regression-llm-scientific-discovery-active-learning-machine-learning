"""No-data formal Gate fixture for decision-risk-aligned portfolio selection."""

from scripts.run_scientist_decision_risk_aligned_portfolio_gate import main


def test_decision_risk_aligned_portfolio_gate_passes(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert '"passed": true' in output
    assert '"real_data_accessed": false' in output
