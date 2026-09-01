"""P3J.14 shared outer-runner composition tests; no dataset access."""

from __future__ import annotations

import argparse
import inspect

import pytest

from hypothesis_mvp.pcpi import P3JPolicyArtifacts
from hypothesis_mvp.pcpi import p3j_outer_runner
from scripts import run_pcpi_p3j14_shared_outer as runner
from tests.test_pcpi_p3j8_run_identity import _case


CONFIG = runner.Path(__file__).resolve().parents[1] / "configs" / (
    "p3j_14_shared_outer_runner.json"
)


def test_frozen_config_closes_authorization_and_all_protocol_identities() -> None:
    config = runner._load_config(CONFIG, CONFIG.parents[1])
    assert config["operational_execution_authorized"] is False
    assert config["formal_dataset_runner_authorized"] is False
    assert config["p3j_outer_runner_composition"] == (
        p3j_outer_runner.P3J_OUTER_RUNNER_COMPOSITION
    )
    assert config["initial_base_warmup_budget"] == 16
    assert config["initial_residual_training_budget"] == 16


def test_shared_dispatch_keeps_baselines_on_original_runner(tmp_path) -> None:
    state, *_ = _case(tmp_path)
    events: list[str] = []
    pcpi = runner._compose_policy_call(
        runner.PCPI_POLICY,
        state,
        {"tag": "p3j"},
        {"tag": "legacy"},
        p3j_call=lambda **kwargs: events.append(kwargs["tag"]) or "p3j",
        legacy_call=lambda **kwargs: events.append(kwargs["tag"]) or "legacy",
    )
    baseline = runner._compose_policy_call(
        "qbc",
        None,
        {"tag": "p3j"},
        {"tag": "legacy"},
        p3j_call=lambda **kwargs: events.append(kwargs["tag"]) or "p3j",
        legacy_call=lambda **kwargs: events.append(kwargs["tag"]) or "legacy",
    )
    assert (pcpi, baseline, events) == ("p3j", "legacy", ["p3j", "legacy"])


def test_outer_policy_orders_transaction_reporting_then_summary(monkeypatch) -> None:
    events: list[str] = []
    measured = object()
    artifacts = P3JPolicyArtifacts(curve_rows=({},), query_rows=({},))
    monkeypatch.setattr(
        p3j_outer_runner,
        "run_p3j_measured_pool_acquisition",
        lambda *args, **kwargs: events.append("transaction") or measured,
    )
    monkeypatch.setattr(
        p3j_outer_runner,
        "build_p3j_policy_artifacts",
        lambda *args, **kwargs: events.append("reporting") or artifacts,
    )
    monkeypatch.setattr(
        p3j_outer_runner,
        "summarize_p3j_policy_artifacts",
        lambda *args, **kwargs: events.append("summary") or {"metric": 1.0},
    )
    result = p3j_outer_runner.run_p3j_outer_policy(
        runner.Path("unused"), None, None, None, None, None, None, None, None,
        None, None,
        source_git_tree="1" * 40, config_sha256="2" * 64,
        dataset_id="uci_ccpp", dataset_family="uci_ccpp", seed=2026080701,
        policy=runner.PCPI_POLICY, acquisition_budget=32, eig_min_samples=32,
        eig_max_samples=512, eig_error_safety_factor=4.0, eig_growth_factor=2,
        class_distance_threshold=0.2, structure_count=7,
    )
    assert events == ["transaction", "reporting", "summary"]
    assert result.measured_run is measured
    assert result.summary_metrics == {"metric": 1.0}


def test_blocked_runner_fails_before_data_root_or_output_access() -> None:
    args = argparse.Namespace(
        config=str(CONFIG), data_root="missing-data", output_dir="missing-output",
        phase="P3J.14", heldout_state="closed",
    )
    with pytest.raises(PermissionError, match="real data access is blocked"):
        runner.run(args)
    source = inspect.getsource(runner.run)
    assert source.index("_load_config") < source.index("PermissionError")
    assert "load_registered_real_dataset" not in source
