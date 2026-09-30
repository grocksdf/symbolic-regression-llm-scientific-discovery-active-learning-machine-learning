"""No-response fixtures for the repaired DRR replay screen."""

import h5py
import numpy as np

from scripts.run_aistats_drr_response_free_replay import (
    _strict_roles, _take,
)


def test_take_restores_registered_hash_order(tmp_path):
    path = tmp_path / "fixture.h5"
    values = np.arange(30.0).reshape(10, 3)
    with h5py.File(path, "w") as handle:
        handle["rows"] = values
    with h5py.File(path, "r") as handle:
        selected = _take(handle["rows"], (7, 2, 9), slice(1, None))
    assert np.array_equal(selected, values[[7, 2, 9], 1:])


def test_strict_roles_never_retains_action_targets(tmp_path):
    path = tmp_path / "fixture.h5"
    values = np.column_stack((
        np.arange(80.0),
        np.arange(80.0) + 100,
        np.arange(80.0) + 200,
    ))
    with h5py.File(path, "w") as handle:
        handle["lsr_synth/family/task/train"] = values
    with h5py.File(path, "r") as handle:
        roles = _strict_roles(handle, "family", "task", 3)
    assert roles.X_actions.shape[1] == 2
    assert not hasattr(roles, "y_actions")
    action_indices = roles.role_row_indices["action_covariates"]
    assert np.array_equal(
        roles.X_actions, values[list(action_indices), 1:])
