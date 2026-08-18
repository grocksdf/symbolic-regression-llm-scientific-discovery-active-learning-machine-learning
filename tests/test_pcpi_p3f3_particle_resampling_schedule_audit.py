"""Static contract tests for the bridge-boundary schedule audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "configs/p3f_3_open_target_particle_resampling_schedule_audit.json"
        ).read_text(encoding="utf-8")
    )


def test_schedule_audit_freezes_target_and_ordering_comparison() -> None:
    config = _config()
    assert config["schema"] == (
        "pcpi-p3f3-open-target-particle-resampling-schedule-audit-v1"
    )
    assert config["particle_counts"] == [512, 2048]
    assert config["proposal_kind"] == "complete-uniform"
    assert config["resampling_kind"] == "systematic"
    assert config["resampling_schedules"] == ["pre-bridge", "post-bridge"]
    assert config["rejuvenation_steps"] == [1]
    assert config["seeds"] == [2026081712, 2026081713, 2026081714, 2026081715]
    assert config["fidelity_envelope"]["formal_gate"] is False


def test_schedule_audit_has_no_downstream_authority() -> None:
    config = _config()
    assert config["real_data_access"] == "forbidden"
    assert config["heldout_state"] == "not-applicable"
    assert config["acquisition_state"] == "blocked"
    assert "diagnostic-only" in config["claim_boundary"]
