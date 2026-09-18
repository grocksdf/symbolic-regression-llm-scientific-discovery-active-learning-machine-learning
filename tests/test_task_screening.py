"""Correctness fixtures only; no real data or efficacy evaluation."""
from pathlib import Path
import numpy as np
import pytest

from hypothesis_mvp.data.oracle import PoolOracle
from hypothesis_mvp.data.roles import DataRole, RoleDataset, SelectionData
from hypothesis_mvp.data.system_protocol import OpenSystemData, ROLE_NAMES
from hypothesis_mvp.discovery import task_screening as screening


def _config():
    return {"schema": screening.SCHEMA,
            "data": [{"dataset": "uci_ccpp", "source": str(Path.cwd() / "fixture"),
                      "split_seed": 7, "counts": {name: 2 for name in ROLE_NAMES}}],
            "prior": {}, "measurement_budget": 2,
            "coefficient_policy": "discard-fitted-coefficients-refit-closed-basis",
            "exact_eig_epsabs": 1e-10, "user_execution_authorized": True}


def _data():
    X = np.arange(12, dtype=float).reshape(6, 2)
    def role(kind, offset):
        return RoleDataset(kind, X + offset, (X[:, 0] + X[:, 1]) + offset)
    development = role(DataRole.DEVELOPMENT, 0)
    return OpenSystemData(SelectionData(development, role(DataRole.VALIDATION, 1), None, ()),
        role(DataRole.DEVELOPMENT, 2), role(DataRole.VALIDATION, 3),
        PoolOracle(X + 4, np.arange(6, dtype=float)),
        {"heldout_opened": False, "fixture_only": True})


def test_screening_registration_is_response_free_and_provider_free():
    assert screening.validate_task_screening_registration(_config())["exact_eig_epsabs"] == 1e-10
    changed = _config(); changed["exact_eig_epsabs"] = 1e-6
    with pytest.raises(ValueError):
        screening.validate_task_screening_registration(changed)


def test_screening_exports_all_tasks_without_pool_reveal(tmp_path, monkeypatch):
    monkeypatch.setattr(screening, "load_registered_system_data", lambda registration: _data())
    monkeypatch.setattr(screening, "audit_frozen_hypothesis_bank", lambda *args, **kwargs: {
        "schema": "scientific-hypothesis-bank-viability-v1", "passed": True,
        "candidate_response_accessed": False, "heldout_opened": False})
    monkeypatch.setattr(PoolOracle, "acquire_indices",
                        lambda *args: (_ for _ in ()).throw(AssertionError("response reveal")))
    result = screening.run_task_screening(tmp_path / "out", _config())
    assert result["admitted_tasks"] == ["uci_ccpp"]
    assert result["provider_calls"] == result["engine_jobs"] == 0
    assert not result["candidate_response_accessed"] and not result["heldout_opened"]
    assert (tmp_path / "out" / "TASK_SCREENING.json").is_file()


def test_screening_refuses_overwrite_before_data(tmp_path, monkeypatch):
    root = tmp_path / "existing"; root.mkdir()
    monkeypatch.setattr(screening, "load_registered_system_data",
                        lambda registration: (_ for _ in ()).throw(AssertionError("data accessed")))
    with pytest.raises(ValueError, match="new path"):
        screening.run_task_screening(root, _config())
