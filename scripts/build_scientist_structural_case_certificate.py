"""Build an immutable artifact-only Scientist structural case certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.structural_case_certificate import (
    build_structural_case_certificate,
)
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("case certificate output must be a new path")
    certificate = build_structural_case_certificate(
        args.source_output, args.gate_output)
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "STRUCTURAL_CASE_CERTIFICATE.json", certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if certificate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
