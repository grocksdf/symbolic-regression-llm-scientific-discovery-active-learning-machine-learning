from __future__ import annotations

import json
from pathlib import Path

from scripts.run_pcpi_p3e3_real_predictive_calibration_audit import (
    CONFIG_SCHEMA,
    STAGE,
    _expected_config,
    _load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3e_3_real_predictive_calibration_audit.json"
RUNNER = ROOT / "scripts" / "run_pcpi_p3e3_real_predictive_calibration_audit.py"


def test_p3e3_real_config_is_frozen_and_validation_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config == _expected_config()
    assert config["schema"] == CONFIG_SCHEMA
    assert config["stage"] == STAGE
    assert config["datasets"] == ["uci_ccpp"]
    assert config["seeds"] == list(range(2026080701, 2026080709))
    assert config["heldout_state"] == "closed"
    assert config["validation_role"] == "opened-for-calibration-diagnostic-only"
    assert config["acquisition_comparison"] == "not-run"
    assert config["acquisition_authorization"] == "blocked"
    assert _load_config(CONFIG, ROOT) == config


def test_p3e3_runner_has_no_acquisition_or_heldout_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "prepare_real_pool_oracle" not in source
    assert "score_acquisition_actions" not in source
    assert '"acquisition_comparison_performed": True' not in source
    assert '"heldout_opened": False' in source
    assert "validation_used_for_calibration_diagnostic" in source


def test_p3e3_real_summary_claims_are_fail_closed() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"formal_predictive_calibration_evidence": False' in source
    assert '"formal_real_posterior_adequacy_evidence": False' in source
    assert '"formal_efficacy_evidence": False' in source
    assert '"acquisition_authorized": False' in source


def test_p3e3_correctness_fixture_config_is_frozen() -> None:
    config_path = ROOT / "configs" / "p3e_3_predictive_calibration_diagnostic.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["schema"] == "pcpi-p3e3-predictive-calibration-diagnostic-config-v1"
    assert config["stage"] == "P3E.3"
    assert config["heldout_state"] == "not-applicable"
    assert config["false_alarm_level"] == 0.01
