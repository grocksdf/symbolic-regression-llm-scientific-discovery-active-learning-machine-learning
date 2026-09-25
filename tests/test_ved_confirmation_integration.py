"""No-data VED confirmation integration binding tests."""

import json
from pathlib import Path

from hypothesis_mvp.discovery.ved_confirmation_integration import (
    run_ved_confirmation_integration_gate,
)


def test_ved_integration_binds_passed_gates_without_execution():
    path = (Path(__file__).resolve().parents[1] / "configs" /
            "scientific_ved_confirmation_integration_v1.json")
    result = run_ved_confirmation_integration_gate(
        json.loads(path.read_text(encoding="utf-8")))
    assert result["passed"] is True
    assert result["eligible_dataset_families"] == ["ved_vehicle_energy"]
    assert result["execution_authorized"] is False
    assert result["real_data_accessed"] is False
