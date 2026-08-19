"""Response-free static checks for the P3F.4 certification confirmatory freeze."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import run_pcpi_p3f4_certification_confirmatory as confirmatory


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = (
    ROOT / "configs/p3f_4_semantic_envelope_certification_confirmatory_freeze.json"
)
DEVELOPMENT_PATH = (
    ROOT / "configs/p3f_4_semantic_envelope_certification_development.json"
)


def _contains_key(value: object, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_confirmatory_config_and_dependencies_are_byte_frozen() -> None:
    assert confirmatory.RESPONSES_MATERIALIZED is False
    config, integrity = confirmatory._integrity_preflight(ROOT, FREEZE_PATH)
    assert integrity["config"] == confirmatory.EXPECTED_CONFIG_SHA256
    assert config["stage"] == "P3F.4-CERT.CF.1"
    assert confirmatory.RESPONSES_MATERIALIZED is False


def test_confirmatory_freeze_contains_no_materialized_targets() -> None:
    config = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert not _contains_key(config, "targets")
    assert config["target_generator"]["response_materialization"] == (
        "runner_only_after_integrity_preflight"
    )
    assert all(item["response_free_registration"] for item in config["fixtures"])
    assert confirmatory.RESPONSES_MATERIALIZED is False


def test_confirmatory_thresholds_are_inherited_without_relaxation() -> None:
    confirmatory_config = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    development_config = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    assert confirmatory_config["certification"] == development_config["certification"]
    assert confirmatory_config["target"] == development_config["target"]


def test_confirmatory_fixture_bank_is_new_and_has_eight_unique_runs() -> None:
    confirmatory_config = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    development_config = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    fixtures = confirmatory_config["fixtures"]
    fixture_ids = [item["fixture_id"] for item in fixtures]
    seeds = [int(seed) for item in fixtures for seed in item["seeds"]]
    confirmatory_actions = {tuple(item["actions"]) for item in fixtures}
    development_actions = {
        tuple(item["actions"]) for item in development_config["fixtures"]
    }
    assert len(fixtures) == 4
    assert len(fixture_ids) == len(set(fixture_ids))
    assert len(seeds) == len(set(seeds)) == 8
    assert confirmatory_actions.isdisjoint(development_actions)
    assert confirmatory_config["confirmatory_decision"]["all_runs_must_pass"] is True
    assert confirmatory_config["confirmatory_decision"][
        "threshold_change_after_response"
    ] == "forbidden"
    assert confirmatory.RESPONSES_MATERIALIZED is False


def test_import_and_static_preflight_never_open_response_state() -> None:
    source = Path(confirmatory.__file__).read_text(encoding="utf-8")
    assert "if __name__ == \"__main__\"" in source
    assert "RESPONSES_MATERIALIZED = False" in source
    assert confirmatory.RESPONSES_MATERIALIZED is False

