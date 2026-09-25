"""User-only VED open-week completeness audit; exports no observed values."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.ved_confirmation_source import (
    _member_names, run_ved_source_gate,
)
from hypothesis_mvp.discovery.ved_streaming_loader import (
    audit_ved_stream_completeness, stream_ved_archive_member,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


SCHEMA = "scientific-ved-open-completeness-audit-registration-v1"


def _verified_json(path, expected):
    source = Path(path)
    if sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("VED completeness audit identity changed")
    return json.loads(source.read_text(encoding="utf-8"))


def _preflight(registration):
    if (registration.get("schema") != SCHEMA
            or registration.get("execution_authorized") is not True
            or registration.get("confirmation_member_authorized") is not False):
        raise ValueError("invalid VED completeness audit registration")
    source = verify_clean_git_source(ROOT)
    if source != registration["source"]:
        raise ValueError("VED completeness audit source identity changed")
    source_config = _verified_json(
        registration["source_registration"],
        registration["source_registration_sha256"])
    integration = _verified_json(
        registration["integration_gate"],
        registration["integration_gate_sha256"])
    if (source_config.get("execution_authorized") is not False
            or integration.get("passed") is not True):
        raise ValueError("VED completeness prerequisites are invalid")
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
        raise ValueError("VED completeness audit requires a new output path")
    extractor = config["extractor"]["path"]
    archives = {
        name: Path(row["path"]) for name, row in config["archives"].items()}
    member_archive = {}
    for archive in archives.values():
        for member in _member_names(Path(extractor), archive):
            member_archive[member] = archive
    reports = {}
    for role, member in config["open_members"].items():
        lines = stream_ved_archive_member(
            extractor=extractor, archive=member_archive[member],
            member=member, open_members=config["open_members"],
            reserved_confirmation_member_sha256=config[
                "reserved_confirmation_member_sha256"])
        reports[role] = audit_ved_stream_completeness(
            lines, features=config["features"], target=config["target"])
    result = {
        "schema": "scientific-ved-open-completeness-audit-v1",
        "passed": True, "source": source, "roles": reports,
        "registered_features": list(config["features"]),
        "registered_target": config["target"],
        "observed_values_retained": False,
        "observed_value_statistics_computed": False,
        "confirmation_member_opened": False,
        "heldout_opened": False,
        "engine_calls": 0, "provider_calls": 0,
        "execution_authorized": False,
        "claim_boundary": (
            "Open-week parse-feasibility audit only; counts missingness, "
            "retains no values and supports no efficacy claim.")}
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "VED_OPEN_COMPLETENESS_AUDIT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
