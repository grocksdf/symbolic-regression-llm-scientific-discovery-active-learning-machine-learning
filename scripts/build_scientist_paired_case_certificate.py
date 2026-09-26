"""Build the artifact-only paired Scientist case certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.paired_case_certificate import (
    build_paired_case_certificate,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("gas_viability", "gas_admission", "yacht_viability",
                 "yacht_admission", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}",
                            type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("paired case output must be new")
    result = build_paired_case_certificate(
        args.gas_viability, args.gas_admission,
        args.yacht_viability, args.yacht_admission)
    result["source"] = verify_clean_git_source(ROOT)
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "PAIRED_CASE_CERTIFICATE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
