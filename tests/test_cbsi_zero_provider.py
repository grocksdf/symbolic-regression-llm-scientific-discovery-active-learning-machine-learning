import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_cbsi_zero_provider_experiment.py")
SPEC = importlib.util.spec_from_file_location("cbsi_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_arm_modes_are_distinct_and_complete():
    assert set(MODULE.ARM_MODES) == {
        "blind_random",
        "single_fitness_proxy",
        "residual_proxy",
        "influence_proxy",
        "bank_greedy",
        "standard_thompson",
        "cbsi_full",
    }
    assert MODULE.ARM_MODES["cbsi_full"] == "thompson_matched"


def test_mixture_gain_is_zero_for_identical_component():
    base = np.asarray([-1.0, -2.0, -1.5])
    gain, weights, audit = MODULE.CrossfitBankScorer._mixture_gain(
        base, base, base, base, seed=1)
    assert abs(gain) < 1e-12
    assert np.isclose(np.sum(weights), 1.0)
    assert audit["converged"]


def test_bootstrap_is_deterministic():
    left = MODULE.bootstrap([0.0, 1.0, 2.0], 7, 200)
    right = MODULE.bootstrap([0.0, 1.0, 2.0], 7, 200)
    assert left == right

