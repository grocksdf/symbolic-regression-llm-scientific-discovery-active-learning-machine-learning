"""P3H.7 formal protocol freeze and top-level source-composition checks."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from hypothesis_mvp.hypotheses import runtime_dependency_hash, runtime_dependency_snapshot
from hypothesis_mvp.pcpi import P3H_OPERATIONAL_LIFECYCLE, P3H_OPERATIONAL_POWERS
from scripts import run_pcpi_p3b_real as shared_runner
from scripts.run_pcpi_p3h7_semiparametric_real_acquisition import (
    P3H7_PROTOCOL,
    RUNTIME_HASH,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3h_7_semiparametric_real_acquisition.json"


def test_p3h7_config_freezes_complete_family_roles_budgets_and_no_fallback() -> None:
    config = shared_runner._load_config(CONFIG, ROOT, P3H7_PROTOCOL)
    assert config["initial_base_warmup_budget"] == 16
    assert config["initial_residual_training_budget"] == 16
    assert config["initial_observation_budget"] == 32
    assert tuple(config["likelihood_power_candidates"]) == P3H_OPERATIONAL_POWERS
    assert config["p3h_operational_lifecycle"] == P3H_OPERATIONAL_LIFECYCLE
    assert config["pcpi_uncertified_eig_action"] == "terminal-abstention-no-fallback"
    assert config["likelihood_power_calibration_method"] == "none-fixed-complete-family"
    assert config["p3h_validation_state_policy"] == (
        "discard-never-enter-operational-state"
    )


def test_p3h7_uses_exact_p3h5_runtime_identity() -> None:
    assert RUNTIME_HASH == runtime_dependency_hash(runtime_dependency_snapshot())
    config = shared_runner._load_config(CONFIG, ROOT, P3H7_PROTOCOL)
    assert config["runtime_dependency_hash"] == RUNTIME_HASH
    assert P3H7_PROTOCOL.required_runtime_dependency_hash == RUNTIME_HASH


def test_p3h7_cli_allows_only_closed_heldout_and_exact_phase() -> None:
    parser = build_parser(P3H7_PROTOCOL)
    assert parser._option_string_actions["--phase"].choices == ("P3H.7",)
    assert parser._option_string_actions["--heldout-state"].choices == ("closed",)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("initial_base_warmup_budget",), 15),
        (("pcpi_uncertified_eig_action",), "posterior-epistemic-variance"),
        (("likelihood_power_calibration_method",), "prequential-r-log-safebayes"),
        (("p3h_validation_state_policy",), "reuse-calibration-state"),
    ),
)
def test_p3h7_protocol_tampering_fails_closed(tmp_path, path, value) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[path[0]] = value
    candidate = tmp_path / "tampered.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        shared_runner._load_config(candidate, tmp_path, P3H7_PROTOCOL)


def test_top_level_runner_constructs_family_only_for_pcpi_after_16_16_split() -> None:
    source = inspect.getsource(shared_runner.run)
    warmup = source.index("warmup_indices = initial_indices[:warmup_count]")
    standardizer = source.index("DevelopmentStandardizer.fit", warmup)
    preconditioner = source.index("initial_X[:warmup_count]", standardizer)
    fixed_family = source.index('"method": "none-fixed-complete-family"', preconditioner)
    policy_guard = source.index(
        "protocol.semiparametric_lifecycle and policy == protocol.pcpi_policy",
        fixed_family,
    )
    initialization = source.index(
        "initialize_operational_semiparametric_state", policy_guard
    )
    dispatch = source.index("_run_policy(", initialization)
    assert warmup < standardizer < preconditioner < fixed_family
    assert fixed_family < policy_guard < initialization < dispatch
    assert "initial_X[:warmup_count]" in source[initialization:dispatch]
    assert "initial_X[warmup_count:]" in source[initialization:dispatch]


def test_formal_protocol_is_real_only_user_executed_and_claim_bounded() -> None:
    assert P3H7_PROTOCOL.semiparametric_lifecycle
    assert P3H7_PROTOCOL.policies == (
        "random",
        "uncertainty",
        "qbc",
        "pcpi_representative_safe_discrepancy_robust_joint_eig",
    )
    assert "held-out-closed" in P3H7_PROTOCOL.claim_boundary
    assert "No likelihood power is selected" in P3H7_PROTOCOL.claim_boundary
    assert "paper acceptance" in P3H7_PROTOCOL.claim_boundary
