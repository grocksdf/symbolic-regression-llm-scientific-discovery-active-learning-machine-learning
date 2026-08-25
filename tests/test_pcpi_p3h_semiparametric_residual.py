"""P3H semiparametric residual-law algebra and leakage boundaries."""

from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    DyadicPolyaTreeResidualModel,
    RegisteredStructurewiseDiscrepancyEngine,
    SequentialSemiparametricResidualEngine,
    StructurewiseDiscrepancyPrior,
    fit_bank_preconditioner,
    generic_real_bank,
    universal_dyadic_depth,
)
from scripts import run_pcpi_p3h1_semiparametric_residual_correctness as runner
from scripts import run_pcpi_p3g1_structurewise_discrepancy_gate as real_runner


def _base_engine():
    grid = np.linspace(-1.5, 1.5, 15)
    actions = np.column_stack((grid, np.cos(grid)))
    bank = generic_real_bank(2)
    preconditioner = fit_bank_preconditioner(bank, actions[:7])
    designs = {
        structure.structure_id: preconditioner.transform(
            actions, structure.basis_terms
        )
        for structure in bank.structures
    }
    return RegisteredStructurewiseDiscrepancyEngine(
        bank,
        actions,
        (DiscrepancyKernelState("registered", 1.0, 0.8),),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        structure_designs=designs,
        maximum_discrepancy_rank=2,
    )


def test_universal_depth_is_count_only_and_sublinear() -> None:
    expected = {0: 0, 1: 0, 3: 0, 4: 1, 15: 1, 16: 2, 63: 2, 64: 3}
    assert {count: universal_dyadic_depth(count) for count in expected} == expected
    for count in range(1, 1000):
        assert 2 ** universal_dyadic_depth(count) <= np.sqrt(count) + 1e-15
    with pytest.raises(ValueError):
        universal_dyadic_depth(-1)
    with pytest.raises(ValueError):
        universal_dyadic_depth(2.5)


def test_empty_tree_is_exact_identity_map() -> None:
    model = DyadicPolyaTreeResidualModel()
    law = model.predictive_law(model.prior_state())
    grid = np.linspace(0.0, 1.0, 101)
    np.testing.assert_array_equal(law.cdf(grid), grid)
    np.testing.assert_array_equal(law.density(grid), np.ones_like(grid))
    np.testing.assert_array_equal(law.inverse_cdf(grid), grid)


def test_kt_split_masses_equal_exact_beta_half_posterior_prediction() -> None:
    model = DyadicPolyaTreeResidualModel()
    state = model.prior_state()
    for value in (0.1, 0.2, 0.3, 0.9):
        state = model.update(state, value)
    law = model.predictive_law(state)
    assert law.depth == 1
    np.testing.assert_allclose(law.leaf_probabilities, [3.5 / 5.0, 1.5 / 5.0])


def test_tree_posterior_is_order_invariant_but_history_hash_is_audit_sensitive() -> None:
    model = DyadicPolyaTreeResidualModel()
    first = model.prior_state()
    second = model.prior_state()
    values = (0.02, 0.17, 0.31, 0.49, 0.52, 0.78, 0.88, 0.97) * 2
    for value in values:
        first = model.update(first, value)
    for value in reversed(values):
        second = model.update(second, value)
    np.testing.assert_allclose(
        model.predictive_law(first).leaf_probabilities,
        model.predictive_law(second).leaf_probabilities,
    )
    assert first.stable_hash != second.stable_hash


def test_predictive_density_normalizes_and_cdf_inverse_are_exact_per_leaf() -> None:
    model = DyadicPolyaTreeResidualModel()
    state = model.prior_state()
    for value in np.linspace(0.01, 0.99, 64) ** 1.7:
        state = model.update(state, float(value))
    law = model.predictive_law(state)
    assert law.depth == 3
    assert np.isclose(np.sum(law.leaf_probabilities), 1.0, atol=2e-15)
    midpoints = (np.arange(2**law.depth) + 0.5) * law.leaf_width
    integral = float(np.sum(law.density(midpoints) * law.leaf_width))
    assert np.isclose(integral, 1.0, atol=2e-15)
    probabilities = np.linspace(0.0, 1.0, 257)
    np.testing.assert_allclose(
        law.cdf(law.inverse_cdf(probabilities)), probabilities, atol=3e-15
    )


def test_semiparametric_chain_rule_matches_base_density_times_residual_density() -> None:
    engine = SequentialSemiparametricResidualEngine(_base_engine())
    targets = 0.4 - 0.3 * np.linspace(-1.5, 1.5, 15)
    state = engine.prior_state()
    for index in range(8):
        state = engine.update(state, index, float(targets[index]))
    law = engine.sequential_predictive_law(state, (9, 11, 13))
    values = targets[[9, 11, 13]]
    raw_pits = law.base_law.cdf(values)
    np.testing.assert_allclose(
        law.logpdf(values),
        law.base_law.logpdf(values) + law.residual_law.log_density(raw_pits),
        atol=2e-15,
    )
    corrected = law.cdf(values)
    assert np.all(corrected >= 0.0) and np.all(corrected <= 1.0)


def test_update_records_the_base_pit_before_assimilating_current_response() -> None:
    base = _base_engine()
    engine = SequentialSemiparametricResidualEngine(base)
    state = engine.prior_state()
    target = 0.37
    expected = float(
        base.sequential_predictive_law(state.base_state, (3,)).cdf(
            np.asarray([target])
        )[0]
    )
    updated = engine.update(state, 3, target)
    assert updated.residual_state.raw_pits == (expected,)
    assert updated.observation_indices == (3,)


def test_future_response_cannot_change_an_already_issued_forecast() -> None:
    engine = SequentialSemiparametricResidualEngine(_base_engine())
    state = engine.prior_state()
    for index, target in enumerate((0.2, -0.1, 0.4, 0.3)):
        state = engine.update(state, index, target)
    issued = engine.sequential_predictive_law(state, (8,))
    before = issued.cdf(np.asarray([0.15]))
    low_future = engine.update(state, 7, -100.0)
    high_future = engine.update(state, 7, 100.0)
    np.testing.assert_array_equal(issued.cdf(np.asarray([0.15])), before)
    assert low_future.residual_state.raw_pits[:-1] == state.residual_state.raw_pits
    assert high_future.residual_state.raw_pits[:-1] == state.residual_state.raw_pits


def test_base_scientific_posterior_update_is_unchanged_by_residual_layer() -> None:
    base = _base_engine()
    engine = SequentialSemiparametricResidualEngine(base)
    wrapped = engine.prior_state()
    ordinary = base.prior_state()
    for index, target in enumerate((0.1, -0.3, 0.2, 0.7, -0.2)):
        wrapped = engine.update(wrapped, index, target)
        ordinary = base.update(ordinary, index, target)
    np.testing.assert_allclose(wrapped.base_state.probabilities, ordinary.probabilities)
    for left, right in zip(
        wrapped.base_state.means, ordinary.means, strict=True
    ):
        np.testing.assert_allclose(left, right, atol=0.0, rtol=0.0)


def test_production_source_has_no_rng_dataset_seed_or_threshold_surface() -> None:
    source = Path(
        "hypothesis_mvp/pcpi/reference/semiparametric_residual.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "np.random",
        "default_rng",
        "dataset_id",
        "seed",
        "heldout",
        "e_value",
        "rejection",
        "2400",
    ):
        assert forbidden not in lowered


def test_p3h1_correctness_gate_passes_without_data_or_experiment_surface() -> None:
    config = runner._load_config(
        Path("configs/p3h_1_semiparametric_residual_correctness.json")
    )
    result = runner._evaluate(config)
    json.dumps(result, allow_nan=False)
    assert result["status"] == "passed", result
    assert all(result["checks"].values())
    assert result["simulated_experiment"] is False
    assert result["real_data_access"] is False
    assert result["validation_response_access"] is False
    assert result["candidate_response_access"] is False
    assert result["heldout_access"] is False
    assert result["formal_efficacy_evidence"] is False


def test_p3h1_config_rejects_depth_cap_or_response_dependent_tuning() -> None:
    path = Path("configs/p3h_1_semiparametric_residual_correctness.json")
    config = runner._load_config(path)
    assert config["residual_law"]["maximum_depth"] is None
    assert config["residual_law"]["response_dependent_bandwidth"] is False
    assert config["residual_law"]["dataset_seed_target_or_result_branching"] is False


def test_p3h2_real_config_preserves_gate_and_blocks_candidate_responses() -> None:
    config = real_runner._load_config(
        Path("configs/p3h_2_semiparametric_residual_calibration_gate.json")
    )
    assert config["schema"] == real_runner.P3H2_CONFIG_SCHEMA
    assert config["predictive_calibration"]["equal_per_run_false_alarm_level"] == "1/2400"
    residual = config["posterior"]["semiparametric_residual_law"]
    assert residual["maximum_depth"] is None
    assert residual["response_dependent_bandwidth"] is False
    assert residual["construction_future_response_access"] is False
    authorization = config["authorization"]
    assert authorization["conditional_acquisition_execution"] is False
    assert authorization["candidate_responses"] == "sealed-through-this-calibration-only-gate"
    assert authorization["heldout"] is False


def test_p3h2_runner_cannot_open_candidate_oracle_after_calibration_pass() -> None:
    source = Path(
        "scripts/run_pcpi_p3g1_structurewise_discrepancy_gate.py"
    ).read_text(encoding="utf-8")
    eligibility = source.index("eligible = len(calibration) == 24")
    authorization = source.index("acquisition_authorized =", eligibility)
    conjunction = source.index(
        "execute_acquisition = eligible and acquisition_authorized", authorization
    )
    oracle = source.index("_open_candidate_oracle(frame, config)", conjunction)
    override = source.index('payload["acquisition_executed"] = False', oracle)
    publication = source.index("_publish(args.output_dir.resolve()", override)
    assert eligibility < authorization < conjunction < oracle < override < publication


def test_p3h1_prerequisite_runs_before_dependency_or_real_dataset_access() -> None:
    config = real_runner._load_config(
        Path("configs/p3h_2_semiparametric_residual_calibration_gate.json")
    )
    result = real_runner._p3h_correctness_prerequisite(Path.cwd(), config)
    assert result["status"] == "passed"
    assert result["real_data_access"] is False
    source = inspect.getsource(real_runner.main)
    prerequisite = source.index("p3h_correctness = _p3h_correctness_prerequisite")
    dependencies = source.index("dependency_snapshot =", prerequisite)
    real_data = source.index("load_registered_real_dataset", dependencies)
    assert prerequisite < dependencies < real_data


def test_p3h2_loader_rejects_a_changed_base_or_residual_rule(tmp_path: Path) -> None:
    source = Path("configs/p3h_2_semiparametric_residual_calibration_gate.json")
    config = json.loads(source.read_text(encoding="utf-8"))
    config["posterior"]["maximum_discrepancy_rank"] = 17
    changed_base = tmp_path / "changed-base.json"
    changed_base.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        real_runner._load_config(changed_base)

    config = json.loads(source.read_text(encoding="utf-8"))
    config["posterior"]["semiparametric_residual_law"]["maximum_depth"] = 4
    changed_residual = tmp_path / "changed-residual.json"
    changed_residual.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        real_runner._load_config(changed_residual)
