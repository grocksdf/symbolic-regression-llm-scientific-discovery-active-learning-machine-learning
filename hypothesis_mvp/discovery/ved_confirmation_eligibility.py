"""Artifact-only resolution of VED confirmation eligibility."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "scientific-ved-confirmation-eligibility-resolution-v1"


def _read_verified(path: str, expected: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("VED eligibility artifact identity changed")
    return json.loads(source.read_text(encoding="utf-8"))


def resolve_ved_confirmation_eligibility(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema", "source_gate", "source_gate_sha256",
        "integration_gate", "integration_gate_sha256",
        "availability_audit", "availability_audit_sha256",
        "execution_authorized", "claim_boundary"}
    if (set(registration) != required
            or registration.get("schema") != SCHEMA
            or registration.get("execution_authorized") is not False):
        raise ValueError("invalid VED eligibility resolution registration")
    source = _read_verified(
        registration["source_gate"], registration["source_gate_sha256"])
    integration = _read_verified(
        registration["integration_gate"],
        registration["integration_gate_sha256"])
    availability = _read_verified(
        registration["availability_audit"],
        registration["availability_audit_sha256"])
    roles = availability["roles"]
    feature_names = [
        row["name"] for row in availability["feature_hierarchy"]]
    all_zero = all(
        report["complete_row_count_by_feature_set"][name] == 0
        for report in roles.values() for name in feature_names)
    checks = {
        "source_identity_gate_passed": source.get("passed") is True,
        "integration_identity_gate_passed":
            integration.get("passed") is True,
        "availability_audit_completed":
            availability.get("schema")
                == "scientific-ved-fuel-feature-availability-audit-v1",
        "fuel_rate_feature_hierarchy_infeasible": (
            availability.get("passed") is False
            and availability.get("first_feasible_feature_set") is None
            and all_zero),
        "availability_used_no_values": (
            availability.get("observed_values_retained") is False
            and availability.get(
                "observed_value_statistics_computed") is False),
        "confirmation_member_closed": (
            availability.get("confirmation_member_opened") is False
            and source.get("reserved_confirmation_member_opened") is False),
    }
    passed = all(checks.values())
    return {
        "schema": "scientific-ved-confirmation-eligibility-certificate-v1",
        "checks": checks, "passed": passed,
        "status": (
            "ved-fuel-rate-confirmation-ineligible-"
            "no-coobserved-feature-target-rows"
            if passed else "ved-eligibility-resolution-failed"),
        "eligible_dataset_families": [],
        "ved_vehicle_energy_eligible": False,
        "target_change_authorized": False,
        "feature_search_authorized": False,
        "week_replacement_authorized": False,
        "execution_authorized": False,
        "real_observation_values_retained": False,
        "heldout_opened": False,
        "confirmation_performed": False,
        "claim_boundary": (
            "VED Fuel Rate is ineligible for the frozen confirmation because "
            "the preregistered open weeks contain no jointly complete row for "
            "any preregistered feature set. This is data-availability evidence, "
            "not method efficacy evidence."),
    }


__all__ = ["SCHEMA", "resolve_ved_confirmation_eligibility"]
