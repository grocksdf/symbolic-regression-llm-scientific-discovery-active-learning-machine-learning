"""P3H.6 correctness tests for score-select-reveal-update ordering."""

from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from hypothesis_mvp.data import PoolOracle
from hypothesis_mvp.pcpi import (
    P3H_OPERATIONAL_POWERS,
    P3H_INTERVAL_FRONTIER_RESOLUTION,
    admit_operational_semiparametric_response,
    initialize_operational_semiparametric_state,
    score_operational_semiparametric_candidates,
)
from hypothesis_mvp.pcpi import operational_semiparametric as operational
from hypothesis_mvp.pcpi.real_acquisition import AcquisitionScores
from hypothesis_mvp.pcpi.reference import SequentialReferencePosterior, generic_real_bank
from hypothesis_mvp.pcpi.reference import DevelopmentStandardizer, fit_bank_preconditioner
from scripts.progress import ProgressReporter
from scripts import run_pcpi_p3b_real as real_runner
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol


def _history():
    grid = np.linspace(-1.4, 1.4, 12)
    actions = np.column_stack((grid, np.square(grid)))
    targets = 0.3 + 0.7 * grid - 0.15 * np.square(grid)
    engines = tuple(
        SequentialReferencePosterior(generic_real_bank(2), power)
        for power in P3H_OPERATIONAL_POWERS
    )
    return actions, targets, engines


def _state():
    actions, targets, engines = _history()
    return initialize_operational_semiparametric_state(
        engines, actions[:4], targets[:4], actions[4:8], targets[4:8]
    )


def _scores(*, certified: bool = True) -> AcquisitionScores:
    values = np.asarray([0.2, 0.9, 0.9])
    zeros = np.zeros(3)
    return AcquisitionScores(
        policy="pcpi_representative_safe_discrepancy_robust_joint_eig",
        scores=values,
        integration_error_bounds=zeros,
        class_count=2,
        estimator_samples=16,
        ranking_certified=certified,
        ranking_margin=0.1,
        ranking_error_bound=0.01,
        ranking_certificate_gap=0.09,
        ranking_error_safety_factor=4.0,
        ranking_planned_looks=2,
        ranking_looks_used=1,
        ranking_certificate_method="correctness-fixture",
        estimator_coarse_samples=8,
        estimator_integration_method="p3h-correctness-fixture",
        utility_mode="representative-safe-discrepancy-robust-p3h-semiparametric-maximin-joint-eig-surrogate",
        target_partition_hash="fixture-partition",
        class_eig_scores=values,
        class_eig_error_bounds=zeros,
        conditional_predictive_eig_scores=zeros,
        joint_class_predictive_scores=values,
        representative_guard_applied=True,
        representative_current_mmd_squared=0.1,
        representative_augmented_mmd_squared=np.asarray([0.1, 0.08, 0.08]),
        representative_safe_mask=np.ones(3, dtype=bool),
        representative_safe_set_nonempty=True,
        representative_safe_set_size=3,
        representative_fallback_used=False,
        representative_mmd_tolerance=1e-12,
        representative_kernel_bandwidth_squared=1.0,
        representative_mmd_method="correctness-fixture",
        robust_likelihood_powers=P3H_OPERATIONAL_POWERS,
        robust_model_count=4,
        least_favorable_likelihood_powers=np.ones(3),
        robust_joint_scores_by_model=np.tile(values, (4, 1)),
        robust_lower_bounds=values,
        robust_upper_bounds=values,
    )


def test_initial_state_uses_four_targets_and_excludes_conditioning_from_residuals() -> None:
    state = _state()
    assert state.family.likelihood_powers == P3H_OPERATIONAL_POWERS
    assert state.family.conditioning_count == 4
    assert state.family.observation_count == 4
    assert len(state.posterior_models) == 4
    assert state.nominal_model.likelihood_power == 1.0
    assert all(
        len(item.residual_state.raw_pits) == 4
        for item in state.family.model_states
    )


def test_scoring_binds_exact_family_before_selecting_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    seen = {}

    def scorer(engine, posterior, classes, actions, **kwargs):
        seen.update(engine=engine, posterior=posterior, classes=classes, kwargs=kwargs)
        return _scores()

    monkeypatch.setattr(operational, "score_discrepancy_aware_actions", scorer)
    candidates = np.asarray([[0.0, 0.0], [0.2, 0.04], [0.4, 0.16]])
    decision = score_operational_semiparametric_candidates(
        state,
        object(),
        object(),
        candidates,
        np.asarray([9, 4, 7]),
        candidates,
        state.residual_actions,
        eig_min_samples=8,
        eig_max_samples=16,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
    )
    assert decision.selected_candidate_id == 4
    np.testing.assert_array_equal(decision.selected_action, candidates[1])
    assert seen["engine"] is state.nominal_model.engine
    assert seen["posterior"] is state.nominal_model.posterior
    assert seen["kwargs"]["semiparametric_residual_family"] is state.family
    assert tuple(
        model.posterior for model in seen["kwargs"]["posterior_models"]
    ) == tuple(item.posterior for item in state.family.model_states)


def test_only_matching_selected_reveal_advances_every_target_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    monkeypatch.setattr(
        operational, "score_discrepancy_aware_actions", lambda *args, **kwargs: _scores()
    )
    candidates = np.asarray([[0.0, 0.0], [0.2, 0.04], [0.4, 0.16]])
    decision = score_operational_semiparametric_candidates(
        state, object(), object(), candidates, np.asarray([9, 4, 7]), candidates,
        state.residual_actions, eig_min_samples=8, eig_max_samples=16,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
    )
    advanced = admit_operational_semiparametric_response(
        state, decision, 4, candidates[1], 0.37
    )
    assert advanced.family.observation_count == state.family.observation_count + 1
    assert advanced.stable_hash != state.stable_hash
    assert all(
        len(item.residual_state.raw_pits) == 5
        for item in advanced.family.model_states
    )
    rebuilt = initialize_operational_semiparametric_state(
        tuple(item.engine for item in state.family.model_states),
        state.conditioning_actions,
        state.conditioning_targets,
        advanced.residual_actions,
        advanced.residual_targets,
    )
    assert advanced.stable_hash == rebuilt.stable_hash
    with pytest.raises(ValueError, match="does not match"):
        admit_operational_semiparametric_response(
            advanced, decision, 4, candidates[1], 0.37
        )


@pytest.mark.parametrize(
    ("candidate_id", "action"),
    ((7, np.asarray([0.2, 0.04])), (4, np.asarray([0.2, 0.05]))),
)
def test_wrong_candidate_or_action_cannot_open_a_response(
    monkeypatch: pytest.MonkeyPatch,
    candidate_id: int,
    action: np.ndarray,
) -> None:
    state = _state()
    monkeypatch.setattr(
        operational, "score_discrepancy_aware_actions", lambda *args, **kwargs: _scores()
    )
    candidates = np.asarray([[0.0, 0.0], [0.2, 0.04], [0.4, 0.16]])
    decision = score_operational_semiparametric_candidates(
        state, object(), object(), candidates, np.asarray([9, 4, 7]), candidates,
        state.residual_actions, eig_min_samples=8, eig_max_samples=16,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
    )
    with pytest.raises(ValueError, match="does not match"):
        admit_operational_semiparametric_response(
            state, decision, candidate_id, action, 0.37
        )


def test_uncertified_or_incomplete_family_score_cannot_authorize_reveal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    monkeypatch.setattr(
        operational,
        "score_discrepancy_aware_actions",
        lambda *args, **kwargs: replace(_scores(), ranking_certified=False),
    )
    candidates = np.asarray([[0.0, 0.0], [0.2, 0.04], [0.4, 0.16]])
    with pytest.raises(FloatingPointError, match="not certified"):
        score_operational_semiparametric_candidates(
            state, object(), object(), candidates, np.asarray([9, 4, 7]), candidates,
            state.residual_actions, eig_min_samples=8, eig_max_samples=16,
            eig_error_safety_factor=4.0, eig_growth_factor=2,
        )


def test_interval_frontier_decision_authorizes_only_secondary_admissible_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    frontier = replace(
        _scores(),
        scores=np.asarray([0.90, 0.91, 0.89]),
        primary_ranking_certified=False,
        possible_maximizer_mask=np.ones(3, dtype=bool),
        possible_maximizer_count=3,
        secondary_resolution_used=True,
        secondary_resolution_method=P3H_INTERVAL_FRONTIER_RESOLUTION,
        selection_admissible_mask=np.asarray([True, False, True]),
    )
    seen = {}

    def scorer(*args, **kwargs):
        seen.update(kwargs)
        return frontier

    monkeypatch.setattr(operational, "score_discrepancy_aware_actions", scorer)
    candidates = np.asarray([[0.0, 0.0], [0.2, 0.04], [0.4, 0.16]])
    decision = score_operational_semiparametric_candidates(
        state, object(), object(), candidates, np.asarray([9, 4, 7]), candidates,
        state.residual_actions, eig_min_samples=8, eig_max_samples=16,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
        unresolved_ranking_action=P3H_INTERVAL_FRONTIER_RESOLUTION,
    )
    assert decision.selected_candidate_id == 7
    assert decision.local_index == 2
    assert seen["semiparametric_unresolved_action"] == (
        P3H_INTERVAL_FRONTIER_RESOLUTION
    )


def test_same_length_changed_history_cannot_forge_an_operational_state() -> None:
    state = _state()
    changed = state.residual_targets.copy()
    changed[1] += 0.01
    with pytest.raises(ValueError, match="frozen lifecycle"):
        operational.OperationalSemiparametricState(
            family=state.family,
            conditioning_actions=state.conditioning_actions,
            conditioning_targets=state.conditioning_targets,
            residual_actions=state.residual_actions,
            residual_targets=changed,
        )


def test_lifecycle_source_has_no_batch_refit_eta_selection_or_response_oracle() -> None:
    source = inspect.getsource(operational)
    assert ".fit_batch(" not in source
    assert "calibrate_likelihood_power" not in source
    assert "prepare_real_pool_oracle" not in source
    assert "acquisition_pool.y" not in source
    assert "validation_y" not in source
    assert "posterior_epistemic_variance" not in source


def test_real_policy_loop_composes_score_select_reveal_and_family_advance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    raw_x, raw_y, _ = _history()
    standardizer = DevelopmentStandardizer.fit(raw_x[:4], raw_y[:4])
    initial_x = standardizer.transform_X(raw_x[:8])
    initial_y = standardizer.transform_y(raw_y[:8])
    bank = generic_real_bank(2)
    preconditioner = fit_bank_preconditioner(bank, initial_x[:4])
    engines = tuple(
        SequentialReferencePosterior(bank, power, preconditioner)
        for power in P3H_OPERATIONAL_POWERS
    )
    state = initialize_operational_semiparametric_state(
        engines, initial_x[:4], initial_y[:4], initial_x[4:], initial_y[4:]
    )
    pool_x = np.asarray([[0.1, 0.01], [0.3, 0.09], [0.5, 0.25]])
    pool_y = np.asarray([0.36, 0.49, 0.61])
    validation_x = raw_x[8:]
    validation_y = raw_y[8:]
    seen_families = []

    def scorer(*args, **kwargs):
        seen_families.append(kwargs["semiparametric_residual_family"].stable_hash)
        return replace(
            _scores(), target_partition_hash=kwargs["target_partition"].stable_hash
        )

    monkeypatch.setattr(real_runner, "score_discrepancy_aware_actions", scorer)
    protocol = RealAcquisitionProtocol(
        stage="P3H.6",
        schema="correctness-only",
        experiment="correctness-only",
        hypothesis_id="correctness-only",
        pcpi_policy="pcpi_representative_safe_discrepancy_robust_joint_eig",
        policies=("pcpi_representative_safe_discrepancy_robust_joint_eig",),
        claim_boundary="correctness-only",
        parent_lineage=("P3H.5",),
        discrepancy_profile_method="correctness-fixture",
        semiparametric_lifecycle=True,
    )
    summary, curves, queries = real_runner._run_policy(
        dataset_id="inference_correctness_diagnostic_fixture",
        seed=1,
        policy=protocol.pcpi_policy,
        initial_X=initial_x,
        initial_y=initial_y,
        validation_X=standardizer.transform_X(validation_x),
        validation_y=standardizer.transform_y(validation_y),
        fixed_domain_X=standardizer.transform_X(pool_x),
        candidate_indices=np.arange(len(pool_x)),
        pool_X=pool_x,
        pool_row_ids=np.asarray(["p0", "p1", "p2"]),
        oracle=PoolOracle(pool_x, pool_y),
        standardizer=standardizer,
        subset_commitments={
            "initial": "a" * 64,
            "validation": "b" * 64,
            "candidate": "c" * 64,
        },
        config={
            "acquisition_observation_budget": 1,
            "operational_class_aggregate_separation": 1.0,
            "operational_class_quantile_levels": [0.1, 0.5, 0.9],
            "eig_quadrature_min_evaluations": 8,
            "eig_quadrature_max_evaluations": 16,
            "eig_quadrature_growth_factor": 2,
            "eig_quadrature_error_safety_factor": 4.0,
            "qbc_committee_size": 1,
            "likelihood_power_candidates": list(P3H_OPERATIONAL_POWERS),
            "pcpi_discrepancy_profile": "correctness-fixture",
            "pcpi_robust_utility": "p3h-semiparametric-correctness-fixture",
        },
        reporter=ProgressReporter(tmp_path / "progress.jsonl"),
        design_preconditioner=preconditioner,
        likelihood_power=1.0,
        protocol=protocol,
        semiparametric_state=state,
    )
    assert len(curves) == 2
    assert len(queries) == 1
    assert len(seen_families) == 1
    assert queries[0]["p3h_operational_lifecycle_applied"]
    assert queries[0]["p3h_family_hash_before_query"] == state.stable_hash
    assert queries[0]["p3h_family_hash_after_query"] != state.stable_hash
    assert summary["pcpi_decision_rule_valid_rate"] == 1.0
