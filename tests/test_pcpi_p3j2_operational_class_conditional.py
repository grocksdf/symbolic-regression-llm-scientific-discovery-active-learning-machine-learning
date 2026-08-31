"""P3J.2 source-composition tests; no real or simulated efficacy data."""

from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3H_OPERATIONAL_POWERS,
    ClassPartition,
    OperationalClassConditionalDecision,
    admit_operational_class_conditional_response,
    initialize_operational_class_conditional_state,
    score_operational_class_conditional_candidates,
)
from hypothesis_mvp.pcpi import operational_class_conditional as operational
from hypothesis_mvp.pcpi.real_acquisition import AcquisitionScores
from hypothesis_mvp.pcpi.reference import SequentialReferencePosterior
from tests._pcpi_fixtures import unit_bank


def _fixture():
    actions = np.linspace(-1.5, 1.5, 12)[:, None]
    targets = 0.4 - 0.8 * actions[:, 0] + 0.3 * np.square(actions[:, 0])
    engines = tuple(
        SequentialReferencePosterior(unit_bank(), power)
        for power in P3H_OPERATIONAL_POWERS
    )
    nominal = engines[-1].fit_batch(actions[:8], targets[:8])
    groups = ((0, 1, 2), tuple(range(3, len(nominal.members))))
    partition = ClassPartition(
        class_ids=("low-order", "higher-order"),
        member_indices=groups,
        class_probabilities=tuple(
            sum(nominal.members[index].probability for index in group)
            for group in groups
        ),
        structure_to_class=(0, 0, 0, 1, 1, 1, 1),
    )
    state = initialize_operational_class_conditional_state(
        engines,
        actions[:4],
        targets[:4],
        actions[4:8],
        targets[4:8],
        partition,
    )
    return actions, targets, state


def _scores(partition_hash: str, *, certified: bool = True) -> AcquisitionScores:
    values = np.asarray([0.2, 0.9, 0.8])
    zeros = np.zeros(3)
    return AcquisitionScores(
        policy="pcpi_representative_safe_robust_class_eig",
        scores=values,
        integration_error_bounds=zeros,
        class_count=2,
        estimator_samples=64,
        ranking_certified=certified,
        ranking_margin=0.1,
        ranking_error_bound=0.01,
        ranking_certificate_gap=0.09,
        ranking_error_safety_factor=4.0,
        ranking_planned_looks=2,
        ranking_looks_used=1,
        ranking_certificate_method="p3j-correctness-fixture",
        estimator_coarse_samples=32,
        estimator_integration_method="p3j-correctness-fixture",
        utility_mode=(
            "representative-safe-robust-class-conditional-semiparametric-"
            "operational-class-eig"
        ),
        target_partition_hash=partition_hash,
        class_eig_scores=values,
        class_eig_error_bounds=zeros,
        conditional_predictive_eig_scores=zeros,
        joint_class_predictive_scores=values,
        representative_guard_applied=True,
        representative_current_mmd_squared=0.1,
        representative_augmented_mmd_squared=np.asarray([0.09, 0.08, 0.09]),
        representative_safe_mask=np.ones(3, dtype=bool),
        representative_safe_set_nonempty=True,
        representative_safe_set_size=3,
        representative_fallback_used=False,
        representative_mmd_tolerance=1e-12,
        representative_kernel_bandwidth_squared=1.0,
        representative_mmd_method="p3j-correctness-fixture",
        robust_likelihood_powers=P3H_OPERATIONAL_POWERS,
        robust_model_count=4,
        least_favorable_likelihood_powers=np.ones(3),
        robust_joint_scores_by_model=np.tile(values, (4, 1)),
        robust_lower_bounds=values,
        robust_upper_bounds=values,
        semiparametric_transport_method=(
            "normalized-class-conditional-pit-density-composition-v1"
        ),
        semiparametric_information_invariance_applied=False,
        decision_target="frozen-operational-class-log-risk",
    )


def test_initial_state_binds_complete_family_and_discards_raw_history() -> None:
    _, _, state = _fixture()
    assert tuple(
        item.engine.likelihood_power for item in state.model_states
    ) == P3H_OPERATIONAL_POWERS
    assert state.conditioning_count == 4
    assert state.residual_observation_count == 4
    assert state.calibrated_update_count == 0
    assert state.nominal_state.engine.likelihood_power == 1.0
    assert not hasattr(state, "conditioning_targets")
    assert not hasattr(state, "residual_targets")


def test_score_select_then_matching_reveal_advances_every_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, state = _fixture()
    candidates = np.asarray([[-0.2], [0.2], [0.7]])
    seen = {}

    def scorer(engine, posterior, actions, **kwargs):
        seen.update(engine=engine, posterior=posterior, actions=actions, kwargs=kwargs)
        return _scores(state.target_partition.stable_hash)

    monkeypatch.setattr(
        operational, "score_class_conditional_decision_actions", scorer
    )
    decision = score_operational_class_conditional_candidates(
        state,
        candidates,
        np.asarray([8, 3, 5]),
        candidates,
        candidates[:1],
        eig_min_samples=8,
        eig_max_samples=16,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
    )
    assert decision.selected_candidate_id == 3
    assert decision.local_index == 1
    assert seen["posterior"] is state.nominal_state.posterior
    assert seen["kwargs"]["calibrated_posterior_states"] == state.model_states
    advanced = admit_operational_class_conditional_response(
        state, decision, 3, candidates[1], 0.31
    )
    assert advanced.residual_observation_count == 5
    assert advanced.calibrated_update_count == 1
    assert advanced.stable_hash != state.stable_hash
    assert all(
        item.calibrated_update_count == 1 for item in advanced.model_states
    )
    with pytest.raises(ValueError, match="does not match"):
        admit_operational_class_conditional_response(
            advanced, decision, 3, candidates[1], 0.31
        )


def test_wrong_reveal_and_uncertified_score_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, state = _fixture()
    candidates = np.asarray([[-0.2], [0.2], [0.7]])
    score = _scores(state.target_partition.stable_hash)
    decision = OperationalClassConditionalDecision(
        prior_state_hash=state.stable_hash,
        selected_candidate_id=3,
        selected_action=candidates[1],
        local_index=1,
        scores=score,
    )
    with pytest.raises(ValueError, match="does not match"):
        admit_operational_class_conditional_response(
            state, decision, 5, candidates[1], 0.31
        )
    monkeypatch.setattr(
        operational,
        "score_class_conditional_decision_actions",
        lambda *args, **kwargs: replace(score, ranking_certified=False),
    )
    with pytest.raises(FloatingPointError, match="not certified"):
        score_operational_class_conditional_candidates(
            state,
            candidates,
            np.asarray([8, 3, 5]),
            candidates,
            candidates[:1],
            eig_min_samples=8,
            eig_max_samples=16,
            eig_error_safety_factor=4.0,
            eig_growth_factor=2,
        )


def test_lifecycle_has_no_candidate_response_validation_or_heldout_surface() -> None:
    source = inspect.getsource(operational)
    admission_source = inspect.getsource(
        operational.admit_operational_class_conditional_response
    )
    assert not any(
        token in source
        for token in (
            "candidate_targets",
            "validation",
            "heldout",
            "PoolOracle",
            "calibrate_likelihood_power",
            "rng",
        )
    )
    assert source.index("score_class_conditional_decision_actions") < source.index(
        "selected = select_acquisition_candidate"
    )
    assert admission_source.index("decision.prior_state_hash") < admission_source.index(
        "advance_calibrated_class_posterior"
    )
