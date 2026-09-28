"""Familywise realized-policy calibration fixtures."""

from hypothesis_mvp.discovery.realized_policy import (
    RealizedPolicyTaskEvidence, fit_familywise_realized_policy,
    realized_policy_decision,
)


def test_realized_policy_authorizes_only_safe_family():
    rows = tuple([
        *(
            RealizedPolicyTaskEvidence(
                f"safe-{index}", "safe", 1.0, .01)
            for index in range(2)),
        *(
            RealizedPolicyTaskEvidence(
                f"{family}-{index}", family,
                (1.0 if index == 0 else -1.0), .01)
            for family in ("mixed-a", "mixed-b", "mixed-c")
            for index in range(2)),
    ])
    policy = fit_familywise_realized_policy(
        rows, policy_name="fixture")
    assert policy["global_safety_passed"] is False
    assert policy["authorized_families"] == []
    decision = realized_policy_decision(
        policy, "safe", active_mode="targeted",
        fallback_mode="random")
    assert decision["selected_mode"] == "random"


def test_missing_realized_policy_fails_closed():
    decision = realized_policy_decision(
        None, "family", active_mode="synthesis",
        fallback_mode="no_llm")
    assert decision["selected_mode"] == "no_llm"
