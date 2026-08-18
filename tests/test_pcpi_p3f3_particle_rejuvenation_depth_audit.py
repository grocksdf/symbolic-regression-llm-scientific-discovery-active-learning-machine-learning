"""Static contract tests for the P3F.3 rejuvenation-depth audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "configs/p3f_3_open_target_particle_rejuvenation_depth_audit.json"
        ).read_text(encoding="utf-8")
    )


def test_depth_audit_freezes_registered_mixing_comparison() -> None:
    config = _config()
    assert config["schema"] == (
        "pcpi-p3f3-open-target-particle-rejuvenation-depth-audit-v1"
    )
    assert config["particle_counts"] == [512, 2048]
    assert config["proposal_kind"] == "complete-uniform"
    assert config["resampling_kind"] == "systematic"
    assert config["resampling_schedule"] == "pre-bridge"
    assert config["rejuvenation_steps"] == [0, 1, 2, 4]
    assert config["seeds"] == [2026081716, 2026081717, 2026081718, 2026081719]
    assert config["fidelity_envelope"]["formal_gate"] is False


def test_depth_audit_has_no_downstream_authority() -> None:
    config = _config()
    assert config["real_data_access"] == "forbidden"
    assert config["heldout_state"] == "not-applicable"
    assert config["acquisition_state"] == "blocked"
    assert "diagnostic-only" in config["claim_boundary"]
