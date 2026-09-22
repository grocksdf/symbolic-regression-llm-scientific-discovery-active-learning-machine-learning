"""User-only cross-family symbolic-skill calibration suite."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.skill_calibration import (
    execute_skill_calibration_suite,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = execute_skill_calibration_suite(
        ROOT, args.output_dir,
        json.loads(args.config.read_text(encoding="utf-8")),
        json.loads(args.freeze.read_text(encoding="utf-8")),
        execution_role="user")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
