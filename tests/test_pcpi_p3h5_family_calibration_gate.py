"""P3H.5 family-wide calibration-only protocol and leakage boundaries."""

from __future__ import annotations

from fractions import Fraction
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.hypotheses import runtime_dependency_snapshot
from hypothesis_mvp.pcpi.reference import (
    DevelopmentStandardizer,
    SequentialReferencePosterior,
    fit_bank_preconditioner,
    generic_real_bank,
)
from scripts import run_pcpi_p3h5_likelihood_power_family_calibration_gate as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3h_5_likelihood_power_family_calibration_gate.json"


def _fixture(config):
    grid = np.linspace(-1.8, 1.8, 28)
    raw_x = np.column_stack((grid, np.sin(grid)))
    raw_y = 0.4 + 0.55 * grid - 0.12 * np.square(grid) + 0.05 * np.cos(grid)
    standardizer = DevelopmentStandardizer.fit(raw_x[:8], raw_y[:8])
    x = standardizer.transform_X(raw_x)
    y = standardizer.transform_y(raw_y)
    bank = generic_real_bank(2)
    preconditioner = fit_bank_preconditioner(bank, x[:8])
    engines = tuple(
        SequentialReferencePosterior(bank, power, preconditioner)
        for power in config["likelihood_powers"]
    )
    return x[:8], y[:8], x[8:16], y[8:16], x[16:], y[16:], engines


def test_frozen_config_closes_96_coordinate_failure_budget() -> None:
    config = runner._load_config(CONFIG)
    calibration = config["predictive_calibration"]
    assert calibration["registered_coordinate_count"] == 3 * 8 * 4
    assert (
        96 * Fraction(calibration["equal_per_coordinate_false_alarm_level"])
        == Fraction(calibration["familywise_false_alarm_level"])
    )
    assert calibration["rejection_boundary"] == 9600
    assert config["initial_base_warmup_budget"] == 16
    assert config["initial_residual_training_budget"] == 16


def test_correctness_prerequisites_pass_before_any_real_data_access() -> None:
    config = runner._load_config(CONFIG)
    results = runner._correctness_prerequisites(ROOT, config)
    assert tuple(results) == ("P3H.1", "P3H.3", "P3H.4")
    assert all(result["status"] == "passed" for result in results.values())


def test_frozen_runtime_identity_is_still_exact() -> None:
    config = runner._load_config(CONFIG)
    snapshot = runtime_dependency_snapshot()
    assert runner._validate_runtime(config, snapshot) == config["runtime_freeze"][
        "runtime_dependency_hash"
    ]


def test_family_calibration_core_is_deterministic_and_power_complete() -> None:
    config = runner._load_config(CONFIG)
    fixture = _fixture(config)
    first_rows, first_hash = runner._calibration_run(*fixture, config)
    second_rows, second_hash = runner._calibration_run(*fixture, config)
    assert first_rows == second_rows
    assert first_hash == second_hash
    assert tuple(row["likelihood_power"] for row in first_rows) == tuple(
        config["likelihood_powers"]
    )
    assert all(row["validation_count"] == 12 for row in first_rows)
    assert all(row["rejection_threshold"] == 9600 for row in first_rows)
    assert all(not row["candidate_response_access"] for row in first_rows)
    assert all(not row["heldout_access"] for row in first_rows)


def test_runner_self_check_closes_before_data_loading() -> None:
    config = runner._load_config(CONFIG)
    result = runner._deterministic_self_check(config)
    assert result["status"] == "passed"
    assert all(result["checks"].values())


def test_conditioning_and_residual_roles_cannot_be_silently_merged() -> None:
    config = runner._load_config(CONFIG)
    warmup_x, warmup_y, residual_x, residual_y, validation_x, validation_y, engines = (
        _fixture(config)
    )
    split_rows, split_hash = runner._calibration_run(
        warmup_x,
        warmup_y,
        residual_x,
        residual_y,
        validation_x,
        validation_y,
        engines,
        config,
    )
    merged_rows, merged_hash = runner._calibration_run(
        np.vstack((warmup_x, residual_x)),
        np.concatenate((warmup_y, residual_y)),
        validation_x[:4],
        validation_y[:4],
        validation_x[4:],
        validation_y[4:],
        engines,
        config,
    )
    assert split_hash != merged_hash
    assert split_rows != merged_rows


def test_source_never_dispatches_acquisition_or_materializes_candidate_targets() -> None:
    source = inspect.getsource(runner)
    assert "score_acquisition_actions" not in source
    assert "score_discrepancy_aware_actions" not in source
    assert "prepare_real_pool_oracle" not in source
    assert "acquisition_pool.y" not in source
    assert "frame.y" not in source
    assert "candidate_targets" not in source
    assert '"acquisition_executed": False' in source
    assert '"candidate_response_access": False' in source


def test_terminal_publication_is_staged_fsynced_atomic_and_no_overwrite(
    tmp_path: Path,
) -> None:
    source = inspect.getsource(runner._publish)
    assert "p3h5-staging" in source
    assert "_write_json_fsynced" in source
    assert "os.replace(staging, output)" in source
    output = tmp_path / "terminal"
    runner._publish(output, {"status": "test-only"}, [{"coordinate": 1}])
    assert output.is_dir()
    assert not (tmp_path / ".terminal.p3h5-staging").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "family_calibration_runs.csv": runner._file_hash(
            output / "family_calibration_runs.csv"
        ),
        "summary.json": runner._file_hash(output / "summary.json"),
    }
    with pytest.raises(FileExistsError, match="already exists"):
        runner._publish(output, {"status": "changed"}, [{"coordinate": 2}])


def test_config_tampering_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["predictive_calibration"][
        "equal_per_coordinate_false_alarm_level"
    ] = "1/2400"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol was modified"):
        runner._load_config(path)


def test_one_step_update_rejects_wrong_posterior_target() -> None:
    config = runner._load_config(CONFIG)
    warmup_x, warmup_y, _, _, _, _, engines = _fixture(config)
    posterior = engines[0].fit_batch(warmup_x, warmup_y)
    with pytest.raises(ValueError, match="target is inconsistent"):
        engines[1].update_one(posterior, warmup_x[0], float(warmup_y[0]))
