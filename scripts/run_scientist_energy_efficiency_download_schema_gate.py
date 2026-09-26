"""User-only official Energy Efficiency download and response-isolated Gate."""

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

from hypothesis_mvp.discovery.energy_efficiency_download_gate import (
    inspect_energy_efficiency_zip,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


SCHEMA = "scientific-energy-efficiency-download-registration-v1"


def _preflight(registration):
    if (registration.get("schema") != SCHEMA
            or registration.get("execution_authorized") is not True
            or registration.get("confirmation_responses_authorized") is not False):
        raise ValueError("invalid Energy Efficiency download registration")
    source = verify_clean_git_source(ROOT)
    if source != registration["source"]:
        raise ValueError("Energy Efficiency download source identity changed")
    path = Path(registration["source_registration"])
    if sha256(path.read_bytes()).hexdigest() != registration[
            "source_registration_sha256"]:
        raise ValueError("Energy Efficiency source registration changed")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("execution_authorized") is not False:
        raise ValueError("Energy Efficiency source registration is unsafe")
    return source, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    registration = json.loads(
        args.registration.read_text(encoding="utf-8"))
    source, config = _preflight(registration)
    if args.preflight_only:
        print(json.dumps({
            "passed": True, "source": source, "download_performed": False,
            "real_data_accessed": False,
            "confirmation_responses_authorized": False}, sort_keys=True))
        return 0
    if (args.data_dir is None or args.output_dir is None
            or args.data_dir.exists() or args.output_dir.exists()):
        raise ValueError(
            "Energy Efficiency download requires new data and output paths")
    request = urllib.request.Request(
        config["official_url"],
        headers={"User-Agent": "PCPI-AISTATS-source-gate/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    report, workbook = inspect_energy_efficiency_zip(payload, config)
    args.data_dir.mkdir(parents=True)
    zip_path = args.data_dir / "energy_efficiency.zip"
    data_path = args.data_dir / config["data_member"]
    zip_path.write_bytes(payload)
    data_path.write_bytes(workbook)
    report["source"] = source
    report["source_registration_sha256"] = registration[
        "source_registration_sha256"]
    report["download_url"] = config["official_url"]
    report["local_zip_path"] = str(zip_path)
    report["local_data_path"] = str(data_path)
    args.output_dir.mkdir(parents=True)
    _publish(
        args.output_dir / "ENERGY_EFFICIENCY_DOWNLOAD_SCHEMA_GATE.json",
        report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
