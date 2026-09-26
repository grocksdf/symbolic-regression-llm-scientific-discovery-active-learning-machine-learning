"""User-only missingness audit for a frozen VED Fuel Rate feature hierarchy."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.ved_confirmation_source import _member_names
from hypothesis_mvp.discovery.ved_streaming_loader import (
    audit_ved_feature_set_completeness, stream_ved_archive_member,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


SCHEMA = "scientific-ved-fuel-feature-availability-registration-v1"


def _verified_json(path, expected):
    source = Path(path)
    if sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("VED feature-availability identity changed")
    return json.loads(source.read_text(encoding="utf-8"))


def _preflight(registration):
    if (registration.get("schema") != SCHEMA
            or registration.get("execution_authorized") is not True
            or registration.get("target") != "Fuel Rate[L/hr]"
            or registration.get("confirmation_member_authorized") is not False):
        raise ValueError("invalid VED feature-availability registration")
    source = verify_clean_git_source(ROOT)
    if source != registration["source"]:
        raise ValueError("VED feature-availability source identity changed")
    source_config = _verified_json(
        registration["source_registration"],
        registration["source_registration_sha256"])
    hierarchy = registration["feature_hierarchy"]
    if (not isinstance(hierarchy, list) or not hierarchy
            or any(set(row) != {"name", "features"} for row in hierarchy)
            or len({row["name"] for row in hierarchy}) != len(hierarchy)):
        raise ValueError("invalid VED feature hierarchy")
    previous = None
    for row in hierarchy:
        current = tuple(row["features"])
        if not current or (previous is not None
                           and not set(current) < set(previous)):
            raise ValueError("VED feature hierarchy must be strictly nested")
        previous = current
    return source, source_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    registration = json.loads(
        args.registration.read_text(encoding="utf-8"))
    source, config = _preflight(registration)
    if args.preflight_only:
        print(json.dumps({
            "passed": True, "source": source,
            "real_data_accessed": False,
            "confirmation_member_authorized": False}, sort_keys=True))
        return 0
    if args.output_dir is None or args.output_dir.exists():
        raise ValueError("VED feature audit requires a new output path")
    extractor = config["extractor"]["path"]
    archives = {
        name: Path(row["path"]) for name, row in config["archives"].items()}
    member_archive = {}
    for archive in archives.values():
        for member in _member_names(Path(extractor), archive):
            member_archive[member] = archive
    feature_sets = {
        row["name"]: tuple(row["features"])
        for row in registration["feature_hierarchy"]}
    role_reports = {}
    for role, member in config["open_members"].items():
        lines = stream_ved_archive_member(
            extractor=extractor, archive=member_archive[member],
            member=member, open_members=config["open_members"],
            reserved_confirmation_member_sha256=config[
                "reserved_confirmation_member_sha256"])
        role_reports[role] = audit_ved_feature_set_completeness(
            lines, feature_sets=feature_sets, target=registration["target"])
    required = registration["required_complete_rows"]
    selected = None
    for row in registration["feature_hierarchy"]:
        name = row["name"]
        if all(role_reports[role][
                "complete_row_count_by_feature_set"][name] >= required[role]
                for role in required):
            selected = name
            break
    result = {
        "schema": "scientific-ved-fuel-feature-availability-audit-v1",
        "passed": selected is not None,
        "source": source, "target": registration["target"],
        "feature_hierarchy": registration["feature_hierarchy"],
        "required_complete_rows": required,
        "roles": role_reports,
        "first_feasible_feature_set": selected,
        "selection_rule":
            "first-pre-registered-nested-set-meeting-all-role-counts",
        "observed_values_retained": False,
        "observed_value_statistics_computed": False,
        "confirmation_member_opened": False,
        "heldout_opened": False, "engine_calls": 0, "provider_calls": 0,
        "execution_authorized": False}
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "VED_FUEL_FEATURE_AVAILABILITY.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
