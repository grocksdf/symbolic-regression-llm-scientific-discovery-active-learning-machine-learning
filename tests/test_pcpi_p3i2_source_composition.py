"""P3I.2 decision-targeted runner composition without measured-data access."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import pytest

from hypothesis_mvp.pcpi import (
    DECISION_REGRET_DISTANCE_METRIC,
    DECISION_TARGETED_POLICY,
    P3I_COPULA_TRANSPORT_METHOD,
)
from scripts import run_pcpi_p3b_real as shared_runner
from scripts.run_pcpi_p3h9_interval_resolved_real_acquisition import P3H9_PROTOCOL
from scripts.run_pcpi_p3i2_decision_targeted_source_gate import (
    P3I2_PROTOCOL,
    _evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3i_2_decision_targeted_source_composition.json"


def test_p3i2_freezes_decision_target_transport_and_class_only_utility() -> None:
    config = shared_runner._load_config(CONFIG, ROOT, P3I2_PROTOCOL)
    assert P3I2_PROTOCOL.pcpi_policy == DECISION_TARGETED_POLICY
    assert P3I2_PROTOCOL.decision_target_alignment
    assert P3I2_PROTOCOL.semiparametric_lifecycle
    assert P3I2_PROTOCOL.fail_fast
    assert not P3I2_PROTOCOL.operational_execution_authorized
    assert config["operational_class_metric"] == DECISION_REGRET_DISTANCE_METRIC
    assert config["operational_class_quantile_levels"] == []
    assert config["conditional_predictive_information_method"] == (
        "excluded-zero-by-decision-target"
    )
    assert config["p3i_semiparametric_transport"] == P3I_COPULA_TRANSPORT_METHOD
    assert config["representative_empty_safe_set_action"] == (
        "terminal-failure-no-utility-switch"
    )


def test_p3i2_gate_passes_without_authorizing_or_accessing_data() -> None:
    result = _evaluate(CONFIG)
    assert result["status"] == "passed-source-composition-real-execution-blocked"
    assert all(result["decisions"].values())
    assert result["real_data_access"] is False
    assert result["candidate_response_access"] is False
    assert result["heldout_access"] is False
    assert result["operational_execution_authorized"] is False


def test_p3i2_operational_runner_is_blocked_before_data_or_output_access(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        data_root=str(tmp_path / "missing-data"),
        output_dir=str(tmp_path / "must-not-exist"),
        source_artifact=None,
        config=str(CONFIG),
        phase="P3I.2",
        heldout_state="closed",
    )
    with pytest.raises(PermissionError, match="source-composition-only"):
        shared_runner.run(args, P3I2_PROTOCOL)
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("operational_class_metric", "pooled-predictive-sd-quantile-rms"),
        ("operational_class_quantile_levels", [0.1, 0.5, 0.9]),
        (
            "conditional_predictive_information_method",
            "class-conditional-gaussian-moment-epig",
        ),
        ("representative_empty_safe_set_action", "minimum-augmented-mmd"),
        ("failure_policy", "fail_closed_record_all_no_seed_replacement"),
        ("operational_execution_authorized", True),
    ),
)
def test_p3i2_contract_tampering_fails_closed(
    tmp_path: Path, key: str, value: object
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[key] = value
    candidate = tmp_path / "tampered.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        shared_runner._load_config(candidate, tmp_path, P3I2_PROTOCOL)


def test_terminal_failure_record_is_durable_exclusive_and_response_free(
    tmp_path: Path,
) -> None:
    failure = {
        "dataset_id": "preflight",
        "dataset_family": "preflight",
        "seed": "preflight",
        "policy": DECISION_TARGETED_POLICY,
        "failure_status": "CorrectnessError: fixture",
    }
    path = shared_runner._write_terminal_failure(
        tmp_path, P3I2_PROTOCOL, failure
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["terminal"]
    assert record["retry_in_same_output_forbidden"]
    assert record["seed_replacement_forbidden"]
    assert record["heldout_opened"] is False
    assert "candidate" not in json.dumps(record).lower()
    with pytest.raises(FileExistsError):
        shared_runner._write_terminal_failure(
            tmp_path, P3I2_PROTOCOL, failure
        )


def test_p3i_dispatch_precedes_and_excludes_legacy_joint_scorers() -> None:
    source = inspect.getsource(shared_runner._run_policy)
    start = source.index("elif is_pcpi and protocol.decision_target_alignment")
    end = source.index(
        "elif is_pcpi and protocol.discrepancy_profile_method", start
    )
    branch = source[start:end]
    assert "score_decision_targeted_actions" in branch
    assert "score_discrepancy_aware_actions" not in branch
    assert "score_acquisition_actions" not in branch
    assert "semiparametric_state.posterior_models" in branch
    assert "semiparametric_state.family" in branch


def test_p3i_reporting_separates_primary_class_risk_from_prediction() -> None:
    source = inspect.getsource(shared_runner._assessment)
    assert "STRONG_DECISION_TARGETED_REAL_ACQUISITION_EVIDENCE" in source
    assert "secondary-reported-not-in-primary-efficacy-decision" in source
    assert "pcpi_target_only_class_eig_used_rate" in source


def test_p3h9_history_remains_executable_under_its_frozen_legacy_contract() -> None:
    assert not P3H9_PROTOCOL.decision_target_alignment
    assert not P3H9_PROTOCOL.fail_fast
    assert P3H9_PROTOCOL.operational_execution_authorized
