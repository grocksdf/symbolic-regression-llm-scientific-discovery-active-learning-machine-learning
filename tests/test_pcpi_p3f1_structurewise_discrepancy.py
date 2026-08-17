"""P3F.1 hand-constructed algebraic correctness tests; no efficacy data."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    StructurewiseDiscrepancyPrior,
    design_matrix,
    fit_structurewise_discrepancy_posterior,
    p3f1_contract_hash,
    structurewise_projected_rbf_basis,
)
from tests._pcpi_fixtures import unit_bank


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([-1.4, -1.05, -0.7, -0.3, 0.0, 0.25, 0.65, 1.05, 1.45])
    y = 0.4 - 0.8 * x + 0.22 * np.square(x) + 0.03 * np.cos(3.0 * x)
    return x[:, None], y


def _kernels() -> tuple[DiscrepancyKernelState, ...]:
    return (
        DiscrepancyKernelState("short", 0.4, 0.55),
        DiscrepancyKernelState("long", 0.6, 1.4),
    )


def test_p3f1_projected_basis_is_response_free_and_structure_specific() -> None:
    x, _ = _fixture()
    bank = unit_bank()
    constant = bank.structures[0]
    quadratic = next(item for item in bank.structures if item.structure_id == "quadratic")
    kernel = _kernels()[0]
    first = structurewise_projected_rbf_basis(
        x, design_matrix(x, constant.basis_terms), constant.structure_id, kernel
    )
    repeated = structurewise_projected_rbf_basis(
        x, design_matrix(x, constant.basis_terms), constant.structure_id, kernel
    )
    other = structurewise_projected_rbf_basis(
        x, design_matrix(x, quadratic.basis_terms), quadratic.structure_id, kernel
    )
    assert first.stable_hash == repeated.stable_hash
    assert not np.allclose(first.covariance, other.covariance)


def test_p3f1_whitened_factors_are_orthogonal_and_psd_by_construction() -> None:
    x, _ = _fixture()
    bank = unit_bank()
    for structure in bank.structures:
        design = design_matrix(x, structure.basis_terms)
        for kernel in _kernels():
            basis = structurewise_projected_rbf_basis(
                x, design, structure.structure_id, kernel
            )
            assert np.max(np.abs(design.T @ basis.factor)) < 2e-12
            assert basis.minimum_covariance_eigenvalue > -2e-12
            assert np.allclose(basis.covariance, basis.factor @ basis.factor.T)


def test_p3f1_joint_component_prior_is_proper_without_spike_duplication() -> None:
    x, y = _fixture()
    posterior = fit_structurewise_discrepancy_posterior(
        unit_bank(), x, y, _kernels()
    )
    assert math.isclose(posterior.joint_prior_probability_sum, 1.0, abs_tol=1e-12)
    for structure in unit_bank().structures:
        inactive = [
            item for item in posterior.members
            if item.structure.structure_id == structure.structure_id
            and not item.discrepancy_active
        ]
        assert len(inactive) == 1
        assert inactive[0].kernel_state_id == "none"


def test_p3f1_joint_posterior_and_marginals_are_normalized() -> None:
    x, y = _fixture()
    posterior = fit_structurewise_discrepancy_posterior(
        unit_bank(), x, y, _kernels()
    )
    assert math.isclose(posterior.probability_sum, 1.0, abs_tol=2e-14)
    structure_sum = sum(
        posterior.structure_probability(item.structure_id)
        for item in unit_bank().structures
    )
    assert math.isclose(structure_sum, 1.0, abs_tol=2e-14)
    assert 0.0 < posterior.discrepancy_probability < 1.0


def test_p3f1_batch_and_sequential_sufficient_statistics_agree() -> None:
    x, y = _fixture()
    batch = fit_structurewise_discrepancy_posterior(
        unit_bank(), x, y, _kernels(), sequential=False
    )
    sequential = fit_structurewise_discrepancy_posterior(
        unit_bank(), x, y, _kernels(), sequential=True
    )
    assert abs(batch.log_evidence - sequential.log_evidence) < 2e-12
    assert np.max(
        np.abs(
            np.asarray([item.posterior_probability for item in batch.members])
            - np.asarray([item.posterior_probability for item in sequential.members])
        )
    ) < 2e-12


def test_p3f1_row_permutation_equivariance() -> None:
    x, y = _fixture()
    order = np.asarray([4, 0, 8, 2, 6, 1, 7, 3, 5])
    direct = fit_structurewise_discrepancy_posterior(unit_bank(), x, y, _kernels())
    permuted = fit_structurewise_discrepancy_posterior(
        unit_bank(), x[order], y[order], _kernels()
    )
    direct_probabilities = {item.state_id: item.posterior_probability for item in direct.members}
    permuted_probabilities = {
        item.state_id: item.posterior_probability for item in permuted.members
    }
    assert direct_probabilities.keys() == permuted_probabilities.keys()
    assert max(
        abs(direct_probabilities[key] - permuted_probabilities[key])
        for key in direct_probabilities
    ) < 2e-11
    assert abs(direct.log_evidence - permuted.log_evidence) < 2e-11


def test_p3f1_predictive_mixture_is_a_normalized_distribution() -> None:
    x, y = _fixture()
    posterior = fit_structurewise_discrepancy_posterior(
        unit_bank(), x, y, _kernels()
    )
    integral, error = quad(
        lambda value: posterior.predictive_density(4, value),
        -np.inf,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    assert error < 1e-8
    assert abs(integral - 1.0) < 1e-9
    cdf_values = [posterior.predictive_cdf(4, value) for value in (-1e6, 0.0, 1e6)]
    assert cdf_values[0] < 1e-10
    assert 0.0 < cdf_values[1] < 1.0
    assert cdf_values[2] > 1.0 - 1e-10


def test_p3f1_predictive_cdf_is_monotone() -> None:
    x, y = _fixture()
    posterior = fit_structurewise_discrepancy_posterior(
        unit_bank(), x, y, _kernels()
    )
    values = np.linspace(-2.0, 2.0, 101)
    cdf = np.asarray([posterior.predictive_cdf(2, value) for value in values])
    assert np.all(np.diff(cdf) >= -1e-14)


def test_p3f1_frozen_contract_hash_is_outcome_independent() -> None:
    bank = unit_bank()
    prior = StructurewiseDiscrepancyPrior(0.3, 0.8)
    first = p3f1_contract_hash(bank, _kernels(), prior)
    second = p3f1_contract_hash(bank, _kernels(), prior)
    assert first == second
    assert len(first) == 64


def test_p3f1_kernel_probabilities_must_be_proper() -> None:
    x, y = _fixture()
    invalid = (
        DiscrepancyKernelState("a", 0.3, 0.5),
        DiscrepancyKernelState("b", 0.3, 1.0),
    )
    try:
        fit_structurewise_discrepancy_posterior(unit_bank(), x, y, invalid)
    except ValueError as error:
        assert "sum to one" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("improper kernel prior was accepted")


def test_p3f1_has_no_real_data_or_acquisition_surface() -> None:
    from hypothesis_mvp.pcpi.reference import structurewise_discrepancy

    source = open(structurewise_discrepancy.__file__, encoding="utf-8").read()
    forbidden = (
        "real_registry",
        "prepare_real_pool_oracle",
        "score_acquisition_actions",
        "heldout",
        "dataset_name",
    )
    assert all(token not in source for token in forbidden)
