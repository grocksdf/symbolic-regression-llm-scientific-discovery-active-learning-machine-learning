"""No-observation source and grammar Gate for VED confirmation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


SCHEMA = "scientific-ved-confirmation-source-registration-v1"


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_names(extractor: Path, archive: Path) -> tuple[str, ...]:
    result = subprocess.run(
        [str(extractor), "l", "-slt", str(archive)],
        check=True, capture_output=True, text=True, encoding="utf-8",
        errors="strict")
    values = []
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"Path = (.+\.csv)", line.strip())
        if match:
            values.append(match.group(1))
    if not values or len(set(values)) != len(values):
        raise ValueError("VED archive member listing is empty or duplicated")
    return tuple(values)


def validate_ved_source_registration(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema", "source_name", "source_citation", "archives",
        "extractor", "open_members", "reserved_confirmation_member_sha256",
        "features", "target", "grammar_contract",
        "execution_authorized", "claim_boundary"}
    if (set(registration) != required
            or registration.get("schema") != SCHEMA
            or registration.get("execution_authorized") is not False
            or registration.get("grammar_contract")
                != "pcpi-closed-basis-v1"):
        raise ValueError("invalid VED source registration")
    if (set(registration["open_members"])
            != {"development", "validation", "acquisition_pool"}
            or len(set(registration["open_members"].values())) != 3):
        raise ValueError("VED open-member roles are invalid")
    features = registration["features"]
    if (not isinstance(features, list) or len(features) < 3
            or len(set(features)) != len(features)
            or registration["target"] in features):
        raise ValueError("VED feature/target registration is invalid")
    commitment = str(registration["reserved_confirmation_member_sha256"])
    if (len(commitment) != 64
            or any(value not in "0123456789abcdef" for value in commitment)):
        raise ValueError("invalid VED confirmation-member commitment")
    return dict(registration)


def run_ved_source_gate(registration: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_ved_source_registration(registration)
    extractor = Path(validated["extractor"]["path"])
    extractor_identity = _sha(extractor)
    archive_members, archive_identities = {}, {}
    for name, row in validated["archives"].items():
        path = Path(row["path"])
        archive_identities[name] = _sha(path)
        archive_members[name] = _member_names(extractor, path)
    available = {member for rows in archive_members.values() for member in rows}
    open_members = set(validated["open_members"].values())
    reserved_hash = validated["reserved_confirmation_member_sha256"]
    reserved_matches = [
        member for member in available
        if sha256(member.encode("utf-8")).hexdigest() == reserved_hash]
    checks = {
        "extractor_identity_matches": (
            extractor_identity == validated["extractor"]["sha256"]),
        "archive_identities_match": all(
            archive_identities[name] == row["sha256"]
            for name, row in validated["archives"].items()),
        "open_members_exist": open_members <= available,
        "reserved_confirmation_member_exists":
            len(reserved_matches) == 1,
        "reserved_confirmation_member_is_not_open":
            bool(reserved_matches and reserved_matches[0] not in open_members),
        "continuous_raw_features_registered":
            len(validated["features"]) == 5,
        "continuous_target_registered":
            validated["target"] == "Fuel Rate[L/hr]",
        "frozen_closed_basis_compatible":
            validated["grammar_contract"] == "pcpi-closed-basis-v1",
        "execution_remains_blocked":
            validated["execution_authorized"] is False,
    }
    return {
        "schema": "scientific-ved-confirmation-source-gate-v1",
        "checks": checks, "passed": all(checks.values()),
        "extractor_sha256": extractor_identity,
        "archive_sha256": archive_identities,
        "archive_member_counts": {
            name: len(rows) for name, rows in archive_members.items()},
        "open_members": dict(validated["open_members"]),
        "reserved_confirmation_member_commitment": reserved_hash,
        "reserved_confirmation_member_opened": False,
        "csv_content_opened": False,
        "real_observation_accessed": False,
        "heldout_opened": False,
        "execution_authorized": False,
    }


__all__ = [
    "SCHEMA", "run_ved_source_gate", "validate_ved_source_registration",
]
