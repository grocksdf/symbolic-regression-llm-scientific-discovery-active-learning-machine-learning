"""P3I.4 shared-H0 target and supervised execution correctness tests."""

from __future__ import annotations

import base64
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hypothesis_mvp.data import PoolOracle
from hypothesis_mvp.hypotheses import file_sha256
from hypothesis_mvp.pcpi import (
    P3H_OPERATIONAL_POWERS,
    initialize_operational_semiparametric_state,
)
from hypothesis_mvp.pcpi.reference import (
    DevelopmentStandardizer,
    SequentialReferencePosterior,
    fit_bank_preconditioner,
    generic_real_bank,
)
from scripts import run_pcpi_p3b_real as shared_runner
from scripts.run_pcpi_p3i3_decision_targeted_real_acquisition import P3I3_PROTOCOL
from scripts.run_pcpi_p3i4_shared_h0_freeze_gate import (
    P3I3_CONFIG,
    P3I4_CONFIG,
    SUPERVISOR,
    _evaluate,
    _without_repair_identity,
)
from scripts.run_pcpi_p3i4_shared_h0_real_acquisition import (
    P3I4_PROTOCOL,
    PYTHON_EXECUTABLE_HASH,
)
from scripts.progress import ProgressReporter


ROOT = Path(__file__).resolve().parents[1]


def _opened_history() -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([
        [-1.0, 0.5], [-0.8, -0.2], [-0.4, 0.7], [-0.1, -0.6],
        [0.2, 0.4], [0.5, -0.7], [0.8, 0.1], [1.1, 0.9],
    ])
    y = 0.4 + 0.8 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 0] ** 2
    return x, y


def _target_case():
    raw_x, raw_y = _opened_history()
    standardizer = DevelopmentStandardizer.fit(raw_x[:4], raw_y[:4])
    x = standardizer.transform_X(raw_x)
    y = standardizer.transform_y(raw_y)
    bank = generic_real_bank(x.shape[1])
    preconditioner = fit_bank_preconditioner(bank, x[:4])
    engine = SequentialReferencePosterior(bank, 1.0, preconditioner)
    config = json.loads(P3I4_CONFIG.read_text(encoding="utf-8"))
    target = shared_runner._freeze_initial_class_target(
        P3I4_PROTOCOL, engine, x, y, x, config
    )
    return engine, x, y, target


def test_p3i4_changes_only_shared_h0_repair_and_stage_identity() -> None:
    failed_formal = json.loads(P3I3_CONFIG.read_text(encoding="utf-8"))
    repaired = json.loads(P3I4_CONFIG.read_text(encoding="utf-8"))
    assert _without_repair_identity(failed_formal) == _without_repair_identity(
        repaired
    )
    assert not P3I3_PROTOCOL.shared_initial_frozen_target
    assert P3I4_PROTOCOL.shared_initial_frozen_target
    assert repaired["p3i_initial_frozen_target_source"] == (
        shared_runner.P3I_SHARED_INITIAL_TARGET_SOURCE
    )


def test_shared_target_canonicalizes_batch_and_accepts_prequential_roundoff() -> None:
    engine, x, y, target = _target_case()
    sequential = engine.fit_sequential(x, y)
    shared_runner._assert_initial_posterior_numerically_equivalent(
        target.posterior, sequential
    )
    assert target.engine_target_hash == engine.target_hash
    assert target.partition.stable_hash == shared_runner.class_partition(
        target.posterior, target.classes
    ).stable_hash


def test_shared_target_equivalence_check_fails_closed_on_material_state_change() -> None:
    engine, x, y, target = _target_case()
    sequential = engine.fit_sequential(x, y)
    first = sequential.members[0]
    changed_precision = first.state.precision.copy()
    changed_precision[0, 0] += 1e-4
    changed = replace(
        sequential,
        members=(
            replace(first, state=replace(first.state, precision=changed_precision)),
            *sequential.members[1:],
        ),
    )
    with pytest.raises(FloatingPointError, match="not numerically equivalent"):
        shared_runner._assert_initial_posterior_numerically_equivalent(
            target.posterior, changed
        )


def test_policy_loop_injects_identical_frozen_target_into_pcpi_and_baseline(
    tmp_path: Path,
) -> None:
    raw_x, raw_y = _opened_history()
    standardizer = DevelopmentStandardizer.fit(raw_x[:4], raw_y[:4])
    initial_x = standardizer.transform_X(raw_x)
    initial_y = standardizer.transform_y(raw_y)
    bank = generic_real_bank(initial_x.shape[1])
    preconditioner = fit_bank_preconditioner(bank, initial_x[:4])
    engines = tuple(
        SequentialReferencePosterior(bank, power, preconditioner)
        for power in P3H_OPERATIONAL_POWERS
    )
    state = initialize_operational_semiparametric_state(
        engines, initial_x[:4], initial_y[:4], initial_x[4:], initial_y[4:]
    )
    pool_x = np.asarray([[0.15, -0.25], [0.45, 0.35], [0.95, -0.55]])
    pool_y = 0.4 + 0.8 * pool_x[:, 0] - 0.3 * pool_x[:, 1] + 0.2 * pool_x[:, 0] ** 2
    fixed_domain_x = standardizer.transform_X(pool_x)
    config = json.loads(P3I4_CONFIG.read_text(encoding="utf-8"))
    config.update({
        "acquisition_observation_budget": 1,
        "eig_quadrature_min_evaluations": 32,
        "eig_quadrature_max_evaluations": 32,
        "qbc_committee_size": 4,
    })
    target = shared_runner._freeze_initial_class_target(
        P3I4_PROTOCOL, engines[-1], initial_x, initial_y, fixed_domain_x, config
    )
    common = {
        "dataset_id": "shared_h0_correctness_fixture",
        "seed": 1,
        "initial_X": initial_x,
        "initial_y": initial_y,
        "validation_X": fixed_domain_x,
        "validation_y": standardizer.transform_y(pool_y),
        "fixed_domain_X": fixed_domain_x,
        "candidate_indices": np.arange(len(pool_x)),
        "pool_X": pool_x,
        "pool_row_ids": np.asarray(["p0", "p1", "p2"]),
        "standardizer": standardizer,
        "subset_commitments": {
            "initial": "a" * 64,
            "validation": "b" * 64,
            "candidate": "c" * 64,
        },
        "config": config,
        "design_preconditioner": preconditioner,
        "likelihood_power": 1.0,
        "protocol": P3I4_PROTOCOL,
        "frozen_initial_target": target,
    }
    baseline, _, _ = shared_runner._run_policy(
        policy="random",
        oracle=PoolOracle(pool_x, pool_y),
        reporter=ProgressReporter(tmp_path / "baseline.jsonl"),
        **common,
    )
    pcpi, _, _ = shared_runner._run_policy(
        policy=P3I4_PROTOCOL.pcpi_policy,
        oracle=PoolOracle(pool_x, pool_y),
        reporter=ProgressReporter(tmp_path / "pcpi.jsonl"),
        semiparametric_state=state,
        **common,
    )
    assert (
        baseline["initial_frozen_class_partition_hash"]
        == pcpi["initial_frozen_class_partition_hash"]
        == target.partition.stable_hash
    )
    assert (
        baseline["initial_operational_class_scale_hash"]
        == pcpi["initial_operational_class_scale_hash"]
        == target.classes.scale_hash
    )
    assert (
        baseline["initial_frozen_target_source"]
        == pcpi["initial_frozen_target_source"]
        == shared_runner.P3I_SHARED_INITIAL_TARGET_SOURCE
    )


def test_p3i4_config_requires_exact_shared_target_source(tmp_path: Path) -> None:
    config = json.loads(P3I4_CONFIG.read_text(encoding="utf-8"))
    config["p3i_initial_frozen_target_source"] = "result-dependent-repair"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="shared initial frozen target"):
        shared_runner._load_config(path, tmp_path, P3I4_PROTOCOL)


def test_p3i4_no_data_gate_passes_every_registered_decision() -> None:
    result = _evaluate()
    assert result["status"] == (
        "passed-user-real-execution-frozen-codex-execution-forbidden"
    )
    assert all(result["decisions"].values())
    assert result["simulated_experiment"] is False
    assert result["real_data_access"] is False
    assert result["candidate_response_access"] is False
    assert result["heldout_access"] is False


def test_p3i4_freezes_the_canonical_venv_launcher_before_data() -> None:
    canonical = ROOT.parent / ".venv_hypothesis_canonical" / "Scripts" / "python.exe"
    assert file_sha256(canonical) == PYTHON_EXECUTABLE_HASH
    run_source = inspect.getsource(shared_runner.run)
    assert run_source.index("python_executable_hash = file_sha256") < (
        run_source.index("if not data_root.is_dir()")
    )


def test_p3i4_supervisor_is_ascii_and_avoids_ps51_redirect_exitcode_bug() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert source.isascii()
    assert "RedirectStandardOutput" not in source
    assert "RedirectStandardError" not in source
    assert "-EncodedCommand" in source
    assert "$exitCode = $process.ExitCode" in source
    assert "$null -eq $exitCode" in source
    assert "$ExpectedPythonHash" in source
    assert "$manifest.python_executable_hash" in source


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell 5.1 contract")
def test_encoded_redirecting_child_preserves_nonzero_exit_code_in_ps51(
    tmp_path: Path,
) -> None:
    powershell = Path(
        r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
    )
    if not powershell.is_file():
        pytest.skip("Windows PowerShell 5.1 is unavailable")
    stdout = tmp_path / "probe.stdout.log"
    stderr = tmp_path / "probe.stderr.log"
    child = (
        f"& '{Path(sys.executable)}' -c 'import sys; sys.exit(2)' "
        f"1> '{stdout}' 2> '{stderr}'; exit $LASTEXITCODE"
    )
    encoded = base64.b64encode(child.encode("utf-16le")).decode("ascii")
    probe = (
        f"$encoded='{encoded}';"
        "$p=Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') "
        "-ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$encoded) "
        "-WindowStyle Hidden -PassThru;"
        "$p.WaitForExit();$p.Refresh();"
        "if($null -eq $p.ExitCode -or [int]$p.ExitCode -ne 2){exit 1};exit 0"
    )
    completed = subprocess.run(
        [str(powershell), "-NoProfile", "-Command", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert stdout.is_file()
    assert stderr.is_file()
