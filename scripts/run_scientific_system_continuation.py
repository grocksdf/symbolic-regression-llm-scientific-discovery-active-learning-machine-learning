"""User-only immutable zero-response scientific-system continuation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.system_executor import execute_registered_system_continuation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    execute_registered_system_continuation(
        ROOT, args.output_dir, args.source_output,
        json.loads(args.config.read_text(encoding="utf-8")),
        json.loads(args.freeze.read_text(encoding="utf-8")),
        json.loads(args.continuation.read_text(encoding="utf-8")),
        execution_role="user", measurement_authorized=not args.screen_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
