"""Run the no-data P3K.4 projected representative-guard Gate."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
    p3k_projected_representative_mmd_safe_set,
)
from hypothesis_mvp.pcpi.acquisition import representative_mmd_safe_set


CONFIG = PROJECT_ROOT / "configs" / "p3k_4_projected_guard_correctness.json"


def _evaluate(path: Path = CONFIG) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    target = np.asarray([[-1.0], [0.0], [1.0]])
    infeasible_observed = target.copy()
    infeasible_candidates = np.asarray([[2.0], [3.0], [4.0]])
    old = representative_mmd_safe_set(
        infeasible_observed, infeasible_candidates, target
    )
    projected = p3k_projected_representative_mmd_safe_set(
        infeasible_observed, infeasible_candidates, target
    )
    feasible_observed = np.asarray([[-1.0]])
    feasible_candidates = np.asarray([[0.0], [1.0], [3.0]])
    old_feasible = representative_mmd_safe_set(
        feasible_observed, feasible_candidates, target
    )
    projected_feasible = p3k_projected_representative_mmd_safe_set(
        feasible_observed, feasible_candidates, target
    )
    source = inspect.getsource(p3k_projected_representative_mmd_safe_set)
    decisions = {
        "historical_empty_set_is_reproduced": not old.safe_set_nonempty,
        "projected_set_is_nonempty": projected.safe_set_nonempty,
        "projection_selects_only_minimum_violation": bool(np.all(
            projected.augmented_mmd_squared[projected.safe_mask]
            <= np.min(projected.augmented_mmd_squared) + projected.tolerance
        )),
        "feasible_historical_set_is_unchanged": bool(np.array_equal(
            old_feasible.safe_mask, projected_feasible.safe_mask
        )),
        "method_identity_is_new": (
            projected.method == P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD
        ),
        "only_covariate_surfaces_exist": not any(token in source for token in (
            "response", "validation", "heldout", "random", "rng",
        )),
        "no_experiment_is_authorized": not any(config[key] for key in (
            "operational_execution_authorized", "real_data_access",
            "response_access", "validation_access", "heldout_access",
            "simulated_experiment",
        )),
    }
    if config["method"] != P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD or not all(decisions.values()):
        raise AssertionError("P3K.4 projected representative-guard Gate failed")
    return {
        "schema": "pcpi-p3k4-projected-representative-guard-result-v1",
        "stage": "P3K.4",
        "status": "passed-correctness-no-real-experiment-authorized",
        "decisions": decisions,
        "minimum_attainable_mmd_change": projected.minimum_augmented_mmd_change,
        "projected_safe_set_size": projected.safe_set_size,
        "real_data_access": False,
        "simulated_experiment": False,
        "heldout_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(_evaluate(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
