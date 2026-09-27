"""No-data registration derivation tests for the v6 replay suite."""

import json

from hypothesis_mvp.discovery.bank_selection import (
    DECISION_RISK_CAPACITY_METHOD, PORTFOLIO_CAPACITY_METHOD,
)
from scripts.run_scientist_v6_offline_replay_suite import _derive_config
from tests.test_system_executor import _config


def test_v6_suite_derivation_changes_only_registered_selection_identity(
        tmp_path):
    config = _config()
    config["source_stacking_policy"] = (
        "diversity-preserving-half-core-dyadic-fold-safe-v1")
    config["hypothesis_bank_gate"]["selection_rule"] = (
        PORTFOLIO_CAPACITY_METHOD)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    from hashlib import sha256
    derived = _derive_config(path, sha256(path.read_bytes()).hexdigest())
    assert derived["hypothesis_bank_gate"]["selection_rule"] == (
        DECISION_RISK_CAPACITY_METHOD)
    assert config["hypothesis_bank_gate"]["selection_rule"] == (
        PORTFOLIO_CAPACITY_METHOD)
