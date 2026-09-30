"""Static contracts for realized DRR response opening."""

import h5py
import numpy as np

from scripts.run_aistats_realized_drr_pilot import (
    _correlation, _dataset,
)


def test_constant_predicted_scores_have_zero_registered_correlation():
    rows = [{
        "condition": "full_scientist_v6", "policy": "decision_risk",
        "trajectory": {"queries": [
            {"predicted_lower_bound": 0.0,
             "realized_risk_reduction": 1.0},
            {"predicted_lower_bound": 0.0,
             "realized_risk_reduction": 0.0},
            {"predicted_lower_bound": 0.0,
             "realized_risk_reduction": -1.0},
        ]},
    }]
    assert _correlation(rows) == 0.0


def test_lsr_realized_loader_uses_registered_train_member(tmp_path):
    path = tmp_path / "fixture.h5"
    with h5py.File(path, "w") as handle:
        handle["lsr_transform/task/train"] = np.zeros((40, 3))
    with h5py.File(path, "r") as handle:
        assert _dataset(handle, "lsr_transform", "task").shape == (40, 3)
