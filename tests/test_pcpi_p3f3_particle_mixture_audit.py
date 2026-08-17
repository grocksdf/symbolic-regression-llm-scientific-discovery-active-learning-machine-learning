"""Static contract tests for the target-invariant proposal-mixture audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mixture_audit_is_preregistered_and_diagnostic_only() -> None:
    config = json.loads(
        (ROOT / "configs/p3f_3_open_target_particle_mixture_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["schema"] == "pcpi-p3f3-open-target-particle-mixture-audit-v1"
    assert config["proposal_kinds"] == [
        "prior-independence",
        "complete-uniform",
        "prior-uniform-mixture",
    ]
    assert config["proposal_mixture_weight"] == 0.5
    assert config["particle_counts"] == [512, 2048]
    assert config["rejuvenation_steps"] == [1]
    assert config["fidelity_envelope"]["formal_gate"] is False
    assert config["real_data_access"] == "forbidden"
    assert config["acquisition_state"] == "blocked"
