"""P3M.5 formal-freeze tests; no real or simulated experiment is run."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from hypothesis_mvp.hypotheses import (
    runtime_binary_identity,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from hypothesis_mvp.pcpi import P3M_OPERATIONAL_LIFECYCLE
from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3m5_formal_real_acquisition as runner
from scripts.run_pcpi_p3m5_formal_freeze_gate import _evaluate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3m_5_action_conditional_real_acquisition.json"
P3L2 = ROOT / "configs" / "p3l_2_information_risk_real_acquisition.json"
SUPERVISOR = ROOT / "scripts" / "invoke_pcpi_p3m5_supervised.ps1"


def test_p3m5_preserves_matched_real_design_and_changes_only_estimand_contract() -> None:
    current = runner.validate_p3m5_config(CONFIG, ROOT)
    predecessor = json.loads(P3L2.read_text(encoding="utf-8"))
    invariant = {
        "datasets", "policies", "seeds", "initial_observation_budget",
        "initial_base_warmup_budget", "initial_residual_training_budget",
        "acquisition_observation_budget", "candidate_pool_budget",
        "validation_budget", "eig_quadrature_min_evaluations",
        "eig_quadrature_max_evaluations", "eig_quadrature_growth_factor",
        "eig_quadrature_error_safety_factor", "eig_action_chunk_size",
        "likelihood_power_candidates", "representative_discrepancy",
        "representative_safe_set_rule", "assessment_rules",
        "runtime_dependency_hash", "runtime_binary_identity", "split_seed",
        "heldout_state", "failure_policy",
    }
    assert all(current[key] == predecessor[key] for key in invariant)
    assert current["p3m_operational_lifecycle"] == P3M_OPERATIONAL_LIFECYCLE
    assert not any(key.startswith("p3k_") for key in current)


def test_p3m5_shared_runner_and_manifest_bind_the_p3m_law() -> None:
    config = runner.validate_p3m5_config(CONFIG, ROOT)
    source = inspect.getsource(shared.run)
    assert 'action_conditional_residual=(' in source
    assert '== "p3m"' in source
    manifest = shared._manifest_method_contract(config, runner.P3M5_PROTOCOL)
    for key in (
        "p3m_residual_state_method", "p3m_context_transform",
        "p3m_bandwidth_rule", "p3m_bandwidth_schedule",
        "p3m_joint_information_method", "p3m_checkpoint_schema",
        "pcpi_information_risk_method",
    ):
        assert manifest[key] == config[key]


def test_p3m5_runtime_entry_and_ascii_supervisor_are_frozen() -> None:
    config = runner.validate_p3m5_config(CONFIG, ROOT)
    assert config["runtime_dependency_hash"] == runtime_dependency_hash(
        runtime_dependency_snapshot()
    )
    assert config["runtime_binary_identity"] == runtime_binary_identity()
    entry = inspect.getsource(runner.main)
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    common = (ROOT / "scripts" / "invoke_pcpi_p3k3_supervised.ps1").read_text(
        encoding="utf-8"
    )
    assert "return run(" in entry
    assert "simulate" not in entry and "default_rng" not in entry
    assert supervisor.isascii()
    assert "-VerifyCompleteRuntimeIdentity" in supervisor
    assert "[Parameter(Mandatory = $true)][string]$EvaluationsStash" in supervisor
    assert "Move-Item -LiteralPath $evaluationsPath" in common
    assert "Remove-Item" not in supervisor + common
    assert "heldout-open" not in supervisor + common


def test_p3m5_no_data_gate_passes() -> None:
    result = _evaluate()
    assert result["status"] == "passed-user-real-execution-authorized"
    assert all(result["decisions"].values())
    assert result["real_data_access"] is False
    assert result["simulated_experiment"] is False
    assert result["heldout_access"] is False
    assert result["formal_experiment"] is False
