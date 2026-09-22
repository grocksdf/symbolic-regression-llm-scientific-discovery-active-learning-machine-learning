"""No-data registration tests for cross-family skill calibration."""
import json
from pathlib import Path

from hypothesis_mvp.discovery.skill_calibration import (
    build_skill_calibration_protocols,
)


ROOT = Path(__file__).resolve().parents[1]


def test_skill_calibration_suite_registers_two_closed_coordinates():
    suite = json.loads((ROOT / "configs" /
        "scientific_system_gas_skill_calibration_suite.json").read_text(
            encoding="utf-8"))
    protocols = build_skill_calibration_protocols(suite)
    assert len(protocols) == 2
    assert {row["data"][0]["dataset"] for row in protocols} == {
        "uci_gas_turbine_co", "uci_gas_turbine_nox"}
    assert all(row["agent"]["engines"] == [
        "polynomial_lasso", "mcts", "sparse_library",
        "additive_mechanisms"] for row in protocols)
    assert all(row["agent"]["engine_budget"] == 4 for row in protocols)
