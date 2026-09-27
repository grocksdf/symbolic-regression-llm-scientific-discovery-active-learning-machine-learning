"""Create the metadata-only AISTATS DRR task and metric freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.drr_benchmark_freeze import (
    build_drr_benchmark_freeze,
)
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("DRR freeze output must be a new path")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = build_drr_benchmark_freeze(ROOT, protocol)
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "AISTATS_DRR_PROTOCOL.json", protocol)
    _publish(args.output_dir / "AISTATS_DRR_BENCHMARK_FREEZE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
