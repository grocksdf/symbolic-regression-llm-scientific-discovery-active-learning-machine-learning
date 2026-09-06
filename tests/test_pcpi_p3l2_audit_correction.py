"""P3L.2 completed-run reporting correction tests; no real data are opened."""

from __future__ import annotations

from scripts.audit_pcpi_p3l2_completed_run import p3l_query_audit
from hypothesis_mvp.pcpi import p3j_reporting


def _row() -> dict[str, object]:
    return {
        "p3k_identity_hash": "a",
        "p3k_prior_state_hash": "b",
        "p3k_next_state_hash": "c",
        "response_receipt_admitted_before_reporting": True,
        "heldout_opened": False,
        "selection_used_validation": False,
        "acquisition_target_partition_hash": "partition",
        "initial_frozen_class_partition_hash": "partition",
        "semiparametric_transport_method": (
            "normalized-shared-innovation-class-conditional-pit-density-v1"
        ),
        "semiparametric_information_invariance_applied": False,
        "selected_conditional_predictive_eig": 0.0,
        "selected_class_eig": 0.9,
        "utility_mode": (
            "representative-safe-robust-class-conditional-semiparametric-"
            "lower-tail-class-entropy-reduction-cvar"
        ),
        "representative_guard_applied": True,
        "representative_safe_set_nonempty": True,
        "representative_fallback_used": False,
        "representative_selected_in_safe_set": True,
        "representative_selected_within_projected_budget": True,
        "eig_selected_from_admissible_set": True,
        "eig_ranking_certified": True,
        "information_risk_method": (
            "lower-tail-cvar-of-frozen-class-entropy-reduction-v1"
        ),
        "information_risk_tail_probability": 0.25,
        "robust_likelihood_powers": [0.125, 0.25, 0.5, 1.0],
        "robust_model_count": 4,
        "score": -0.4,
        "selected_joint_class_predictive_score": -0.4,
        "selected_lower_tail_cvar": -0.4,
        "selected_lower_tail_cvar_by_model": [-0.2, -0.4, -0.4, -0.1],
        "selected_robust_lower_tail_cvar_lower_bound": -0.5,
        "selected_robust_lower_tail_cvar_upper_bound": -0.3,
        "selected_negative_gain_probability_by_model": [0.2, 0.3, 0.1, 0.0],
        "selected_least_favorable_information_risk_power": 0.25,
    }


def test_completed_query_audit_accepts_cvar_not_mean_eig_semantics() -> None:
    row = _row()
    checks = p3l_query_audit(row)
    assert all(checks.values())
    assert p3j_reporting._p3j_decision_valid(
        row,
        "partition",
        "normalized-shared-innovation-class-conditional-pit-density-v1",
    )
    assert p3j_reporting._information_risk_usage((row, row)) == (0.0, 1.0)


def test_completed_query_audit_rejects_wrong_score_and_tie_break() -> None:
    wrong_score = _row() | {"score": 0.1}
    assert not p3l_query_audit(wrong_score)["maximin_cvar_score_valid"]
    wrong_tie = _row() | {
        "selected_least_favorable_information_risk_power": 0.5
    }
    assert not p3l_query_audit(wrong_tie)[
        "negative_mass_and_least_favorable_model_valid"
    ]
