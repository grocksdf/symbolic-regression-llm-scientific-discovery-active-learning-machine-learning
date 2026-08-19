"""Static protocol tests for the P3F.4-CERT.2 development runner."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.run_pcpi_p3f4_cert2_response_energy_development import (
    ALLOWED_FIXTURE_ROLES,
    EXPECTED_CONTROLS,
    STAGE,
    _load_and_validate_config,
    _materialize_fixture_targets,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3f_4_response_energy_certification_development.json"


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_cert2_development_config_is_static_and_development_only() -> None:
    config = _load_and_validate_config(CONFIG, ROOT)
    assert config["stage"] == STAGE
    assert config["certification"] == EXPECTED_CONTROLS
    assert config["real_data_access"] == "forbidden"
    assert config["heldout_state"] == "not-applicable"
    assert config["resident_smc_execution"] == "forbidden"
    assert config["resident_smc_modification"] is False
    assert config["new_confirmatory_response_materialization"] == "forbidden"
    assert config["provenance"] == {
        "required_python_major_minor": "3.11",
        "required_clean_source": True,
        "future_confirmatory_dependency_lock": (
            "must_be_committed_before_new_response_materialization"
        ),
    }
    assert config["target"]["coefficient_noise_prior"]["coefficient_mean"] == 0.0
    assert {item["role"] for item in config["fixtures"]} == ALLOWED_FIXTURE_ROLES
    assert config["development_decision"]["required_run_count"] == 11
    assert config["development_decision"]["required_seen_postmortem_run_count"] == 8


def test_seen_postmortem_generators_reproduce_frozen_float_semantics() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    fixture = next(
        item
        for item in config["fixtures"]
        if item["fixture_id"] == "cert_confirmatory_irregular_negative_cubic_af"
    )
    runs = _materialize_fixture_targets(fixture)
    assert [seed for seed, _ in runs] == [2026081901, 2026081902]
    energies = [float(np.dot(targets, targets)) for _, targets in runs]
    assert abs(energies[0] - 2.1145039097) < 8e-11
    assert abs(energies[1] - 2.0148885790) < 8e-11


def test_old_confirmatory_freeze_dependencies_remain_byte_identical() -> None:
    expected = {
        "hypothesis_mvp/pcpi/open_target/certification.py": (
            "73b268c5a998be65c4a8ebc245471c2b1c3d1b54faef0fa06faa86ac749d3e8d"
        ),
        "scripts/run_pcpi_p3f4_certification_layer.py": (
            "5dbeee18df3e7d0452034d1febb707d0f8810f93b91f7fa86b25936ea424e6a2"
        ),
        "scripts/run_pcpi_p3f4_certification_confirmatory.py": (
            "ee148fea27fea19445bff42030b6a4bcb7823c48e733fd7e2b987ad07e3606ae"
        ),
        "configs/p3f_4_semantic_envelope_certification_confirmatory_freeze.json": (
            "dc89217920cca81fb91a8d25fa2d1bea1e94086e47f886fbe30a9dbf26d6cfca"
        ),
    }
    assert {path: _sha256(ROOT / path) for path in expected} == expected


def test_cert2_runner_has_no_real_data_or_resident_smc_surface() -> None:
    source = (
        ROOT / "scripts/run_pcpi_p3f4_cert2_response_energy_development.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "ScalableOpenTargetSMC(",
        "MatchedFullPopulationSMC(",
        "run_pcpi_p3b_real",
        "run_pcpi_p3c_real",
        "data_root",
        "heldout_opened = True",
    )
    assert all(token not in source for token in forbidden)
    required = (
        "source_git_commit",
        "source_git_clean",
        "dependency_sha256",
        "interpreter",
        "packages",
        "dependency_snapshot_sha256",
        "evaluation_wall_time_seconds",
        "registered_interpreter_passed",
        "new_confirmatory_responses_materialized",
        "formal_confirmatory_evidence",
        "resident_smc_executed",
        "real_data_accessed",
        "heldout_opened",
    )
    assert all(token in source for token in required)
