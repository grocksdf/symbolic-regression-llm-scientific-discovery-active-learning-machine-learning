"""VED open-loader smoke preflight fixtures; no real data."""

import json
from pathlib import Path

from hypothesis_mvp.discovery import system_executor


def test_system_config_remains_execution_blocked():
    path = (Path(__file__).resolve().parents[1] / "configs" /
            "scientific_system_ved_typed_synthesis_confirmation.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    system_executor.validate_system_registration(config)
    assert config["user_execution_authorized"] is False
    assert config["data"][0]["dataset"] == "ved_fuel_rate"
