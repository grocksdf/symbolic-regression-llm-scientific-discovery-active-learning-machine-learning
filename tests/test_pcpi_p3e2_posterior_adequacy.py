from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.pcpi.reference.orthogonal_discrepancy import (
    ADEQUACY_EPROCESS_METHOD,
    NOMINAL_ELIGIBLE_MODE,
    ORTHOGONAL_DISCREPANCY_METHOD,
    REFERENCE_ONLY_MODE,
    ExactOrthogonalDiscrepancyEngine,
    OrthogonalDiscrepancyPrior,
    orthogonal_discrepancy_fixture,
    orthogonal_rbf_discrepancy_basis,
)
from scripts.run_pcpi_p3e2_posterior_adequacy_diagnostic import (
    FIXTURE_ROLE,
    STAGE,
    _evaluate,
    _load_config,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3e_2_posterior_adequacy_diagnostic.json"


def test_discrepancy_basis_is_response_free_and_union_orthogonal() -> None:
    actions, designs, _, basis, *_ = orthogonal_discrepancy_fixture()
    union = np.column_stack(designs)
    assert basis.method == ORTHOGONAL_DISCREPANCY_METHOD
    assert basis.discrepancy_rank > 0
    np.testing.assert_allclose(union.T @ basis.matrix, 0.0, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        basis.covariance,
        basis.covariance.T,
        atol=1e-14,
        rtol=0.0,
    )
    repeated = orthogonal_rbf_discrepancy_basis(actions, designs)
    assert basis.stable_hash == repeated.stable_hash


def test_null_and_discrepancy_posteriors_normalize() -> None:
    *_, nominal, misspecified, order, engine = orthogonal_discrepancy_fixture()
    for targets in (nominal, misspecified):
        posterior = engine.fit(order, targets)
        assert posterior.probability_sum == pytest.approx(1.0, abs=1e-14)
        assert 0.0 < posterior.discrepancy_probability < 1.0
        assert np.isfinite(posterior.log_bayes_factor)


def test_exact_null_favors_no_discrepancy() -> None:
    *_, nominal, _misspecified, order, engine = orthogonal_discrepancy_fixture()
    posterior = engine.fit(order, nominal)
    assert posterior.log_bayes_factor < 0.0
    assert posterior.discrepancy_probability < 0.5
    process = engine.adequacy_eprocess(
        order, nominal, false_alarm_level=0.01
    )
    assert not process.rejected
    assert process.decision_mode == NOMINAL_ELIGIBLE_MODE


def test_structured_residual_crosses_registered_adequacy_threshold() -> None:
    *_, _nominal, misspecified, order, engine = orthogonal_discrepancy_fixture()
    posterior = engine.fit(order, misspecified)
    assert np.exp(posterior.log_bayes_factor) > 100.0
    process = engine.adequacy_eprocess(
        order, misspecified, false_alarm_level=0.01
    )
    assert process.method == ADEQUACY_EPROCESS_METHOD
    assert process.rejected
    assert process.first_rejection_round is not None
    assert process.decision_mode == REFERENCE_ONLY_MODE


def test_prequential_ratios_telescope_to_batch_bayes_factor() -> None:
    *_, _nominal, misspecified, order, engine = orthogonal_discrepancy_fixture()
    process = engine.adequacy_eprocess(
        order, misspecified, false_alarm_level=0.01
    )
    posterior = engine.fit(order, misspecified)
    assert np.sum(process.log_predictive_ratios) == pytest.approx(
        posterior.log_bayes_factor, abs=1e-13
    )
    np.testing.assert_allclose(
        np.cumsum(process.log_predictive_ratios),
        process.log_e_values[1:],
        atol=1e-13,
        rtol=0.0,
    )


def test_orthogonality_preserves_structural_coefficient_means() -> None:
    _actions, designs, _probabilities, basis, _nominal, misspecified, order, engine = (
        orthogonal_discrepancy_fixture()
    )
    posterior = engine.fit(order, misspecified)
    for position, design in enumerate(designs):
        null = posterior.component(position, False)
        active = posterior.component(position, True)
        np.testing.assert_allclose(
            active.coefficient_mean[: design.shape[1]],
            null.coefficient_mean,
            atol=1e-10,
            rtol=0.0,
        )
        assert active.coefficient_mean.shape[0] == (
            design.shape[1] + basis.discrepancy_rank
        )


def test_domain_permutation_preserves_evidence_and_decision() -> None:
    actions, designs, probabilities, _basis, nominal, misspecified, order, engine = (
        orthogonal_discrepancy_fixture()
    )
    permutation = np.asarray([5, 1, 12, 0, 9, 3, 15, 7, 2, 14, 4, 11, 6, 13, 8, 10])
    permuted_designs = tuple(design[permutation] for design in designs)
    permuted_basis = orthogonal_rbf_discrepancy_basis(
        actions[permutation], permuted_designs
    )
    permuted_engine = ExactOrthogonalDiscrepancyEngine(
        permuted_designs,
        probabilities,
        permuted_basis,
        OrthogonalDiscrepancyPrior(),
    )
    first = engine.fit(order, misspecified)
    second = permuted_engine.fit(order, misspecified[permutation])
    assert second.log_bayes_factor == pytest.approx(
        first.log_bayes_factor, abs=1e-10
    )
    null_first = engine.fit(order, nominal)
    null_second = permuted_engine.fit(order, nominal[permutation])
    assert null_second.log_bayes_factor == pytest.approx(
        null_first.log_bayes_factor, abs=1e-10
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "constant_actions",
        "one_design",
        "bad_probability",
        "nonorthogonal",
        "rows",
        "target_shape",
        "empty",
        "level",
    ),
)
def test_posterior_adequacy_contract_fails_closed(mutation: str) -> None:
    actions, designs, probabilities, basis, nominal, _misspecified, order, _engine = (
        orthogonal_discrepancy_fixture()
    )
    if mutation == "constant_actions":
        with pytest.raises(ValueError):
            orthogonal_rbf_discrepancy_basis(np.ones_like(actions), designs)
        return
    if mutation == "one_design":
        with pytest.raises(ValueError):
            orthogonal_rbf_discrepancy_basis(actions, designs[:1])
        return
    if mutation == "bad_probability":
        probabilities = np.asarray([0.4, 0.4])
    elif mutation == "nonorthogonal":
        altered = basis.matrix.copy()
        altered[:, 0] += 1.0
        basis = type(basis)(
            altered,
            basis.covariance,
            basis.bandwidth_squared,
            basis.union_rank,
            basis.discrepancy_rank,
            basis.maximum_orthogonality_error,
        )
    if mutation in {"bad_probability", "nonorthogonal"}:
        with pytest.raises(ValueError):
            ExactOrthogonalDiscrepancyEngine(
                designs, probabilities, basis, OrthogonalDiscrepancyPrior()
            )
        return
    engine = ExactOrthogonalDiscrepancyEngine(
        designs, probabilities, basis, OrthogonalDiscrepancyPrior()
    )
    if mutation == "rows":
        with pytest.raises(ValueError):
            engine.fit(np.asarray([0, 0]), nominal[:2])
    elif mutation == "target_shape":
        with pytest.raises(ValueError):
            engine.fit(order, nominal[:, None])
    elif mutation == "empty":
        with pytest.raises(ValueError):
            engine.fit(np.asarray([], dtype=np.int64), np.asarray([]))
    else:
        with pytest.raises(ValueError):
            engine.adequacy_eprocess(order, nominal, false_alarm_level=1.0)


def test_p3e2_config_is_correctness_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["stage"] == "P3E.2"
    assert config["fixture_role"] == "inference_correctness_diagnostic_fixture"
    assert config["heldout_state"] == "not-applicable"
    assert config["false_alarm_level"] == 0.01
    forbidden = {"datasets", "data_root", "validation_budget", "policies"}
    assert not forbidden & set(config)


def test_p3e2_is_not_imported_by_real_runtime() -> None:
    source = (ROOT / "hypothesis_mvp" / "pcpi" / "real_acquisition.py").read_text()
    runner = (ROOT / "scripts" / "run_pcpi_p3d_real.py").read_text()
    assert "orthogonal_discrepancy" not in source
    assert "P3E.2" not in runner


def test_p3e2_diagnostic_passes_all_frozen_decisions() -> None:
    config = _load_config(CONFIG, ROOT)
    rows, summary = _evaluate(config)
    assert len(rows) == 34
    assert summary["fixture_role"] == FIXTURE_ROLE
    assert summary["gate_decision_count"] == 11
    assert summary["gate_passed"]
    assert all(summary["gate_decisions"].values())
    assert summary["independent_marginal_max_abs_error"] <= 1e-10
    assert not summary["formal_real_posterior_adequacy_evidence"]
    assert not summary["formal_efficacy_evidence"]
    assert not summary["real_data_accessed"]


def test_p3e2_cli_has_no_real_data_surface() -> None:
    parser = build_parser()
    assert {action.dest for action in parser._actions} == {
        "help", "output_dir", "config", "phase", "heldout_state"
    }
    args = parser.parse_args(["--output-dir", "diagnostic"])
    assert args.phase == STAGE
    assert args.heldout_state == "not-applicable"
