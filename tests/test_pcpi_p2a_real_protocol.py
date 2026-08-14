from dataclasses import fields

import numpy as np
import pytest

from hypothesis_mvp.data import (
    RealDatasetFrame,
    prepare_real_selection,
)
from hypothesis_mvp.data.real_registry import AIRFOIL_HASH, CCPP_HASH, GAS_HASHES
from hypothesis_mvp.pcpi.reference import (
    generic_real_bank,
    SequentialReferencePosterior,
    stable_budget_indices,
)
from hypothesis_mvp.pcpi.smc import FixedUniverseSMC, SMCConfig
from hypothesis_mvp.pcpi.reference.basis import design_matrix
from scripts.run_pcpi_p2a_real import build_parser


def _frame(dataset_id: str, target: str, dimensions: int, groups=None) -> RealDatasetFrame:
    count = 50 if groups is None else len(groups)
    X = np.arange(count * dimensions, dtype=float).reshape(count, dimensions) + 1.0
    y = np.linspace(10.0, 20.0, count)
    return RealDatasetFrame(
        dataset_id=dataset_id,
        X=X,
        y=y,
        row_ids=np.asarray([f"row:{index:04d}" for index in range(count)], dtype=object),
        groups=None if groups is None else np.asarray(groups),
        feature_names=tuple(f"feature_{index}" for index in range(dimensions)),
        target_name=target,
        source_paths=(),
        source_hashes=("a" * 64,),
        provenance={"synthetic": False},
    )


def test_real_protocol_exposes_no_heldout_capability_or_metadata() -> None:
    prepared = prepare_real_selection(_frame("uci_ccpp", "PE", 4))
    selection_fields = {field.name for field in fields(type(prepared.selection))}
    encoded = repr(prepared.selection)
    assert not any("heldout" in name.lower() for name in selection_fields)
    assert "untouched-heldout" not in encoded
    assert not hasattr(prepared.selection.acquisition_pool, "y")
    assert set(dict(prepared.selection.role_manifest)) == {
        "development", "validation", "acquisition-pool"
    }


def test_split_assignment_is_target_blind() -> None:
    left = _frame("uci_ccpp", "PE", 4)
    right = RealDatasetFrame(
        **{**left.__dict__, "y": left.y[::-1].copy()}
    )
    left_prepared = prepare_real_selection(left)
    right_prepared = prepare_real_selection(right)
    assert left_prepared.split_manifest["split_hash"] == right_prepared.split_manifest["split_hash"]
    np.testing.assert_array_equal(left_prepared.development_row_ids, right_prepared.development_row_ids)


def test_gas_targets_share_one_grouped_split() -> None:
    groups = np.repeat(np.arange(2011, 2016), 12)
    co = prepare_real_selection(_frame("uci_gas_turbine_co", "CO", 9, groups))
    nox = prepare_real_selection(_frame("uci_gas_turbine_nox", "NOX", 9, groups))
    assert co.split_manifest["dataset_family"] == "uci_gas_turbine"
    assert co.split_manifest["split_hash"] == nox.split_manifest["split_hash"]
    np.testing.assert_array_equal(co.development_row_ids, nox.development_row_ids)


def test_generic_bank_depends_only_on_feature_dimension() -> None:
    first = generic_real_bank(9)
    second = generic_real_bank(9)
    assert first.stable_hash == second.stable_hash
    encoded = repr(first.to_dict()).lower()
    assert "uci_gas_turbine" not in encoded
    assert "nox" not in encoded


def test_multivariate_basis_is_closed_and_numerically_correct() -> None:
    X = np.asarray([[2.0, 3.0], [5.0, 7.0]])
    matrix = design_matrix(X, ("intercept", "x0", "x1_sq", "x0_x1"))
    np.testing.assert_array_equal(
        matrix,
        np.asarray([[1.0, 2.0, 9.0, 6.0], [1.0, 5.0, 49.0, 35.0]]),
    )


def test_multivariate_feature_tokens_do_not_collide_with_legacy_powers() -> None:
    X = np.asarray([[2.0, 3.0, 5.0, 7.0], [11.0, 13.0, 17.0, 19.0]])
    matrix = design_matrix(X, ("x0", "x1", "x2", "x3"))
    np.testing.assert_array_equal(matrix, X)


@pytest.mark.parametrize("dimensions", (4, 9))
def test_registered_feature_dimensions_fit_the_generic_reference_bank(dimensions: int) -> None:
    rng = np.random.default_rng(20260807 + dimensions)
    X = rng.normal(size=(48, dimensions))
    y = rng.normal(size=48)
    bank = generic_real_bank(dimensions)
    posterior = SequentialReferencePosterior(bank).fit_batch(X, y)
    assert posterior.probability_sum == pytest.approx(1.0, abs=1e-12)
    assert all(np.isfinite(member.log_marginal_likelihood) for member in posterior.members)


def test_multivariate_exact_update_and_smc_are_normalized() -> None:
    X = np.column_stack((np.linspace(-1.0, 1.0, 24), np.linspace(1.0, -1.0, 24) ** 2))
    y = np.linspace(-0.7, 0.9, 24)
    bank = generic_real_bank(2)
    reference = SequentialReferencePosterior(bank)
    batch = reference.fit_batch(X, y)
    sequential = reference.fit_sequential(X, y)
    assert batch.probability_sum == pytest.approx(1.0, abs=1e-12)
    assert sequential.probability_sum == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(
        [member.probability for member in batch.members],
        [member.probability for member in sequential.members],
        atol=1e-10,
        rtol=0.0,
    )
    run = FixedUniverseSMC(bank, SMCConfig(96, 0.5, 1), seed=91).run(X, y)
    assert run.population.probability_sum == pytest.approx(1.0, abs=1e-12)
    assert all(step.weight_normalization_error <= 1e-12 for step in run.steps)


def test_budget_order_is_stable_and_target_independent() -> None:
    row_ids = np.asarray([f"id:{index}" for index in range(30)], dtype=object)
    first = stable_budget_indices(row_ids, 10, 77)
    second = stable_budget_indices(row_ids, 10, 77)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 10


def test_real_runner_has_no_hash_bypass_option() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--data-root" in option_strings
    assert "--skip-hash" not in option_strings
    assert "--no-verify-hash" not in option_strings


def test_official_hashes_are_frozen_exactly() -> None:
    assert AIRFOIL_HASH == "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"
    assert CCPP_HASH == "ccd490981db2a2f079963b3d9f0aea30d9d338900a0285428dfc6385396f4651"
    assert GAS_HASHES == {
        "gt_2011.csv": "d87ceef9aa59533cc7d924d10de241b1b06ecd11f9b26bab59191ea0f8a76b9a",
        "gt_2012.csv": "be54b9d0e1a7de40c55d32fa489e75de892b000c066b5a09f09a19124ee29100",
        "gt_2013.csv": "13c437bb440ec2045bd12057e6654c41dd4107a661eac16ba2e878e897a08f9e",
        "gt_2014.csv": "c2a03c92c9c3207aad0c6be7de8d9b5b4bfa4720ad0efb2c1f21b6cec4d3f3fa",
        "gt_2015.csv": "9b08f35fde0d4b138232a605db4093c2b8bf9d6757e6f1fbd9534ad616c13591",
    }


def test_generated_data_experiment_entrypoints_are_absent() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    assert not (root / "scripts" / "run_pcpi_p1_reference.py").exists()
    assert not (root / "scripts" / "run_pcpi_p2a_smc.py").exists()
    assert not (root / "hypothesis_mvp" / "pcpi" / "reference" / "diagnostic.py").exists()
