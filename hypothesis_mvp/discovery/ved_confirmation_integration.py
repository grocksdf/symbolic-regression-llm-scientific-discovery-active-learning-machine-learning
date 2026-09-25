"""Bind VED source, loader, endpoint and system identities without data."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .system_executor import validate_system_registration


SCHEMA = "scientific-ved-independent-confirmation-integration-v1"


def _verified_json(path: str, expected: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("VED integration artifact identity changed")
    return json.loads(source.read_text(encoding="utf-8"))


def run_ved_confirmation_integration_gate(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema", "source_gate", "source_gate_sha256",
        "loader_gate", "loader_gate_sha256",
        "endpoint_gate", "endpoint_gate_sha256",
        "system_config", "system_config_sha256",
        "eligible_dataset_family", "execution_authorized",
        "claim_boundary"}
    if (set(registration) != required
            or registration.get("schema") != SCHEMA
            or registration.get("execution_authorized") is not False
            or registration.get("eligible_dataset_family")
                != "ved_vehicle_energy"):
        raise ValueError("invalid VED integration registration")
    source = _verified_json(
        registration["source_gate"], registration["source_gate_sha256"])
    loader = _verified_json(
        registration["loader_gate"], registration["loader_gate_sha256"])
    endpoint = _verified_json(
        registration["endpoint_gate"], registration["endpoint_gate_sha256"])
    config = _verified_json(
        registration["system_config"], registration["system_config_sha256"])
    validate_system_registration(config)
    data = config["data"]
    checks = {
        "source_gate_passed": source.get("passed") is True,
        "source_gate_opened_no_observations": (
            source.get("real_observation_accessed") is False
            and source.get("csv_content_opened") is False),
        "loader_gate_passed": loader.get("passed") is True,
        "loader_gate_used_no_real_data": (
            loader.get("real_observation_accessed") is False
            and loader.get("extractor_started") is False),
        "endpoint_gate_passed": endpoint.get("passed") is True,
        "endpoint_remains_unexecuted": (
            endpoint.get("confirmation_performed") is False),
        "system_config_is_ved_only": (
            len(data) == 1 and data[0]["dataset"] == "ved_fuel_rate"),
        "system_config_execution_blocked":
            config["user_execution_authorized"] is False,
        "reserved_confirmation_member_closed": (
            source.get("reserved_confirmation_member_opened") is False
            and loader.get("reserved_confirmation_member_opened") is False),
    }
    return {
        "schema": "scientific-ved-independent-confirmation-integration-gate-v1",
        "checks": checks, "passed": all(checks.values()),
        "eligible_dataset_families": (
            ["ved_vehicle_energy"] if all(checks.values()) else []),
        "execution_authorized": False,
        "real_data_accessed": False,
        "candidate_response_accessed": False,
        "reporting_validation_response_accessed": False,
        "heldout_opened": False,
        "confirmation_performed": False,
        "next_gate": (
            "real-open-week-loader-smoke-user-only-no-confirmation-member"),
    }


__all__ = ["SCHEMA", "run_ved_confirmation_integration_gate"]
