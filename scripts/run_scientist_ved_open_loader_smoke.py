"""User-only VED open-week loader smoke; never opens confirmation member."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.data.system_protocol import load_registered_system_data
from hypothesis_mvp.discovery.resource_limits import run_bounded
from hypothesis_mvp.discovery.system_executor import validate_system_registration
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


SCHEMA = "scientific-ved-open-loader-smoke-registration-v1"


def _verified_json(path, expected):
    source = Path(path)
    if sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("VED loader smoke artifact identity changed")
    return json.loads(source.read_text(encoding="utf-8"))


def _preflight(registration):
    if (registration.get("schema") != SCHEMA
            or registration.get("execution_authorized") is not True
            or registration.get("confirmation_member_authorized") is not False):
        raise ValueError("invalid VED loader smoke registration")
    source = verify_clean_git_source(ROOT)
    if source != registration["source"]:
        raise ValueError("VED loader smoke source identity changed")
    integration = _verified_json(
        registration["integration_gate"],
        registration["integration_gate_sha256"])
    system = _verified_json(
        registration["system_config"],
        registration["system_config_sha256"])
    source_config = _verified_json(
        registration["source_registration"],
        registration["source_registration_sha256"])
    validate_system_registration(system)
    if (integration.get("passed") is not True
            or system.get("user_execution_authorized") is not False
            or system["data"][0]["dataset"] != "ved_fuel_rate"
            or source_config.get("execution_authorized") is not False):
        raise ValueError("VED loader smoke prerequisites are invalid")
    return source, system


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    registration = json.loads(
        args.registration.read_text(encoding="utf-8"))
    source, system = _preflight(registration)
    if args.preflight_only:
        print(json.dumps({
            "passed": True, "source": source,
            "real_data_accessed": False, "heldout_opened": False,
            "confirmation_member_authorized": False,
            "loader_smoke_authorized": True}, sort_keys=True))
        return 0
    if args.output_dir is None or args.output_dir.exists():
        raise ValueError("VED loader smoke requires a new output path")
    data, enforcement = run_bounded(
        load_registered_system_data, args=(system["data"][0],),
        seconds=system["data_loading_seconds"], provider_attempts=0)
    manifest = data.manifest
    if (manifest.get("dataset") != "ved_fuel_rate"
            or manifest.get("heldout_opened") is not False
            or manifest.get("reserved_confirmation_member_opened") is not False):
        raise ValueError("VED loader smoke crossed confirmation boundary")
    result = {
        "schema": "scientific-ved-open-loader-smoke-result-v1",
        "passed": True, "source": source,
        "data_manifest": manifest,
        "registered_counts": system["data"][0]["counts"],
        "resource_enforcement": enforcement,
        "provider_calls": 0, "engine_calls": 0,
        "candidate_response_accessed": False,
        "reporting_validation_response_accessed": False,
        "reserved_confirmation_member_opened": False,
        "heldout_opened": False,
        "confirmation_performed": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Open-week loader identity smoke only; no discovery, efficacy "
            "or confirmation claim.")}
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "VED_OPEN_LOADER_SMOKE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
