"""Static contract tests for the P3F.3 mechanism audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/p3f_3_open_target_particle_mechanism_audit.json").read_text(
            encoding="utf-8"
        )
    )


def test_mechanism_audit_design_is_preregistered() -> None:
    config = _config()
    assert config["schema"] == "pcpi-p3f3-open-target-particle-mechanism-audit-v1"
    assert config["particle_counts"] == [512, 2048]
    assert config["proposal_kinds"] == ["prior-independence", "complete-uniform"]
    assert config["rejuvenation_steps"] == [0, 1]
    assert config["seeds"] == [2026081704, 2026081705, 2026081706, 2026081707]
    assert config["fidelity_envelope"]["formal_gate"] is False


def test_mechanism_audit_isolation_boundary_is_frozen() -> None:
    config = _config()
    assert config["real_data_access"] == "forbidden"
    assert config["heldout_state"] == "not-applicable"
    assert config["acquisition_state"] == "blocked"
    assert "not simulated or real-data efficacy" in config["claim_boundary"]
