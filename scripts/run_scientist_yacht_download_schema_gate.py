"""User-only official Yacht download and response-isolated schema Gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.yacht_download_gate import inspect_yacht_zip
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


SCHEMA = "scientific-yacht-download-registration-v1"


def _verified_json(path, expected):
    source = Path(path)
    if sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Yacht download prerequisite identity changed")
    return json.loads(source.read_text(encoding="utf-8"))


def _preflight(registration):
    if (registration.get("schema") != SCHEMA
            or registration.get("execution_authorized") is not True
            or registration.get("confirmation_responses_authorized") is not False):
        raise ValueError("invalid Yacht download registration")
    source = verify_clean_git_source(ROOT)
    if source != registration["source"]:
        raise ValueError("Yacht download source identity changed")
    source_config = _verified_json(
        registration["source_registration"],
        registration["source_registration_sha256"])
    candidate_gate = _verified_json(
        registration["source_candidate_gate"],
        registration["source_candidate_gate_sha256"])
    if (candidate_gate.get("passed") is not True
            or source_config.get("execution_authorized") is not False):
        raise ValueError("Yacht download prerequisites are invalid")
    return source, source_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    registration = json.loads(
        args.registration.read_text(encoding="utf-8"))
    source, source_config = _preflight(registration)
    if args.preflight_only:
        print(json.dumps({
            "passed": True, "source": source, "download_performed": False,
            "real_data_accessed": False,
            "confirmation_responses_authorized": False}, sort_keys=True))
        return 0
    if (args.data_dir is None or args.output_dir is None
            or args.data_dir.exists() or args.output_dir.exists()):
        raise ValueError("Yacht download requires new data and output paths")
    request = urllib.request.Request(
        source_config["official_url"],
        headers={"User-Agent": "PCPI-AISTATS-source-gate/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    report, data_bytes = inspect_yacht_zip(payload, source_config)
    args.data_dir.mkdir(parents=True)
    zip_path = args.data_dir / "yacht_hydrodynamics.zip"
    data_path = args.data_dir / "yacht_hydrodynamics.data"
    zip_path.write_bytes(payload)
    data_path.write_bytes(data_bytes)
    report["source"] = source
    report["source_registration_sha256"] = registration[
        "source_registration_sha256"]
    report["download_url"] = source_config["official_url"]
    report["local_zip_path"] = str(zip_path)
    report["local_data_path"] = str(data_path)
    report["confirmation_responses_authorized"] = False
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "YACHT_DOWNLOAD_SCHEMA_GATE.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
