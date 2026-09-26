"""Artifact-only VED confirmation eligibility resolution tests."""

import json
from pathlib import Path

from hypothesis_mvp.discovery.ved_confirmation_eligibility import (
    resolve_ved_confirmation_eligibility,
)


def test_observed_zero_coavailability_revokes_ved_eligibility():
    path = (Path(__file__).resolve().parents[1] / "configs" /
            "scientific_ved_confirmation_eligibility_resolution_v1.json")
    result = resolve_ved_confirmation_eligibility(
        json.loads(path.read_text(encoding="utf-8")))
    assert result["passed"] is True
    assert result["ved_vehicle_energy_eligible"] is False
    assert result["eligible_dataset_families"] == []
    assert result["target_change_authorized"] is False
    assert result["heldout_opened"] is False
