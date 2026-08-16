from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.pcpi.reference.orthogonal_discrepancy import (
    orthogonal_discrepancy_fixture,
)
from scripts.run_pcpi_p3e2_real_posterior_adequacy_audit import (
    AUDIT_ROLE,
    CONFIG_SCHEMA,
    EXPERIMENT,
    REAL_AUDIT_DATASETS,
    STAGE,
    _domain_indices_and_order,
    _load_config,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3e_2_real_posterior_adequacy_audit.json"
RUNNER = ROOT / "scripts" / "run_pcpi_p3e2_real_posterior_adequacy_audit.py"


def test_real_audit_config_is_frozen_and_initial_development_only() -> None:
    config = _load_config(CONFIG, ROOT)
    assert config["schema"] == CONFIG_SCHEMA
    assert config["stage"] == STAGE
    assert config["experiment"] == EXPERIMENT
    assert config["heldout_state"] == "closed"
    assert config["acquisition_comparison"] == "not-run"
    assert config["acquisition_authorization"] == "blocked"
    assert config["registered_domain_budget"] == 96
    assert tuple(config["datasets"]) == REAL_AUDIT_DATASETS == ("uci_ccpp",)


def test_domain_selection_and_order_do_not_depend_on_targets() -> None:
    row_ids = np.asarray([f"row:{index}" for index in range(160)], dtype=object)
    first = _domain_indices_and_order(row_ids, 96, 2026080701)
    second = _domain_indices_and_order(row_ids, 96, 2026080701)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_fixture_import_proves_the_runner_keeps_correctness_separate() -> None:
    actions, designs, _probabilities, _basis, _nominal, _misspecified, _order, _engine = (
        orthogonal_discrepancy_fixture()
    )
    assert actions.shape[0] == designs[0].shape[0]
    assert "real_data_accessed" not in RUNNER.read_text(encoding="utf-8").split(
        "def _domain_indices_and_order", 1
    )[0]


def test_real_audit_runner_has_no_acquisition_or_heldout_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "prepare_real_pool_oracle" not in source
    assert "score_acquisition" not in source
    assert "acquisition_comparison_performed" in source
    assert "selection_used_heldout" in source
    parser = build_parser()
    args = parser.parse_args(
        ["--data-root", "data", "--output-dir", "out"]
    )
    assert args.phase == STAGE
    assert args.heldout_state == "closed"


def test_real_audit_config_has_no_policy_or_validation_budget() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forbidden = {
        "policies",
        "validation_budget",
        "candidate_pool_budget",
        "acquisition_observation_budget",
    }
    assert not forbidden & set(config)
    assert AUDIT_ROLE == "initial_development_real_posterior_adequacy_audit"
