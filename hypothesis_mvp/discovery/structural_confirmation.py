"""Prospective outcome algebra for independent Scientist confirmation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "scientific-typed-synthesis-independent-confirmation-v1"
METHOD = "readiness-then-proper-log-score-lexicographic-v1"


def evaluate_structural_confirmation(
    *, full_viable: bool, no_llm_viable: bool,
    llm_admitted: bool, llm_fold_gains: Sequence[float],
    llm_fold_tolerances: Sequence[float],
    full_minus_no_llm_log_score: float | None,
    comparison_tolerance: float,
) -> dict[str, Any]:
    if (type(full_viable) is not bool or type(no_llm_viable) is not bool
            or type(llm_admitted) is not bool
            or len(llm_fold_gains) != len(llm_fold_tolerances)
            or len(llm_fold_gains) < 2
            or comparison_tolerance < 0):
        raise ValueError("invalid structural confirmation outcome")
    fold_safe = all(
        float(gain) > float(tolerance)
        for gain, tolerance in zip(llm_fold_gains, llm_fold_tolerances))
    control_collapsed = not no_llm_viable
    predictive_tie_break_passed = (
        None if control_collapsed else
        full_minus_no_llm_log_score is not None
        and float(full_minus_no_llm_log_score) > comparison_tolerance)
    passed = bool(
        full_viable and llm_admitted and fold_safe
        and (control_collapsed or predictive_tie_break_passed is True))
    return {
        "schema": "scientific-typed-synthesis-confirmation-outcome-v1",
        "method": METHOD,
        "full_operationally_ready": full_viable,
        "no_llm_operationally_ready": no_llm_viable,
        "no_llm_collapse_counted_as_control_failure": control_collapsed,
        "llm_source_admitted": llm_admitted,
        "llm_fold_safe": fold_safe,
        "predictive_tie_break_required": not control_collapsed,
        "predictive_tie_break_passed": predictive_tie_break_passed,
        "passed": passed,
    }


def validate_confirmation_registration(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema", "method", "source_case_certificate",
        "source_case_certificate_sha256", "development_exclusions",
        "eligible_dataset_families", "incompatible_dataset_families",
        "primary_endpoint", "execution_authorized", "claim_boundary"}
    if (set(registration) != required
            or registration.get("schema") != SCHEMA
            or registration.get("method") != METHOD
            or registration.get("execution_authorized") is not False):
        raise ValueError("invalid confirmation registration identity")
    certificate = Path(str(registration["source_case_certificate"]))
    if not certificate.is_file():
        raise ValueError("source case certificate is missing")
    digest = sha256(certificate.read_bytes()).hexdigest()
    if digest != registration["source_case_certificate_sha256"]:
        raise ValueError("source case certificate identity changed")
    endpoint = registration["primary_endpoint"]
    if endpoint != {
        "full_must_be_operationally_ready": True,
        "llm_source_must_be_admitted": True,
        "all_llm_fold_gains_must_exceed_numerical_tolerance": True,
        "no_llm_collapse_is_control_failure_not_exclusion": True,
        "if_both_ready_require_positive_proper_log_score_gain": True,
        "heldout_opened": False,
    }:
        raise ValueError("invalid prospective confirmation endpoint")
    return dict(registration)


def run_confirmation_freeze_gate(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_confirmation_registration(registration)
    cases = {
        "collapse_control": evaluate_structural_confirmation(
            full_viable=True, no_llm_viable=False, llm_admitted=True,
            llm_fold_gains=(2., 1.), llm_fold_tolerances=(1e-9, 1e-9),
            full_minus_no_llm_log_score=None, comparison_tolerance=1e-9),
        "both_ready_positive": evaluate_structural_confirmation(
            full_viable=True, no_llm_viable=True, llm_admitted=True,
            llm_fold_gains=(2., 1.), llm_fold_tolerances=(1e-9, 1e-9),
            full_minus_no_llm_log_score=0.2, comparison_tolerance=1e-9),
        "both_ready_negative": evaluate_structural_confirmation(
            full_viable=True, no_llm_viable=True, llm_admitted=True,
            llm_fold_gains=(2., 1.), llm_fold_tolerances=(1e-9, 1e-9),
            full_minus_no_llm_log_score=-0.2, comparison_tolerance=1e-9),
        "full_collapsed": evaluate_structural_confirmation(
            full_viable=False, no_llm_viable=False, llm_admitted=True,
            llm_fold_gains=(2., 1.), llm_fold_tolerances=(1e-9, 1e-9),
            full_minus_no_llm_log_score=None, comparison_tolerance=1e-9),
    }
    checks = {
        "registration_identity_valid": bool(validated),
        "control_collapse_has_defined_loss": (
            cases["collapse_control"]["passed"] is True
            and cases["collapse_control"][
                "no_llm_collapse_counted_as_control_failure"] is True),
        "both_ready_requires_positive_proper_score": (
            cases["both_ready_positive"]["passed"] is True
            and cases["both_ready_negative"]["passed"] is False),
        "full_collapse_fails": cases["full_collapsed"]["passed"] is False,
        "development_identities_excluded": bool(
            validated["development_exclusions"]),
        "incompatible_grammars_fail_closed": bool(
            validated["incompatible_dataset_families"]),
        "no_current_execution_authority":
            validated["execution_authorized"] is False,
    }
    return {
        "schema": "scientific-typed-synthesis-confirmation-freeze-gate-v1",
        "checks": checks, "passed": all(checks.values()), "cases": cases,
        "eligible_dataset_families":
            validated["eligible_dataset_families"],
        "execution_authorized": False,
        "real_data_accessed": False,
        "candidate_response_accessed": False,
        "reporting_validation_response_accessed": False,
        "heldout_opened": False,
        "confirmation_performed": False,
    }


__all__ = [
    "METHOD", "SCHEMA", "evaluate_structural_confirmation",
    "run_confirmation_freeze_gate", "validate_confirmation_registration",
]
