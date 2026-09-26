"""Build an immutable response-free Scientist operational-capacity certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.operational_capacity_case_certificate import (
    build_operational_capacity_case_certificate,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("capacity certificate output must be a new path")
    certificate = build_operational_capacity_case_certificate(
        args.source_output, args.gate_output)
    certificate["builder_source"] = verify_clean_git_source(ROOT)
    args.output_dir.mkdir(parents=True)
    _publish(
        args.output_dir / "OPERATIONAL_CAPACITY_CASE_CERTIFICATE.json",
        certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if certificate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
