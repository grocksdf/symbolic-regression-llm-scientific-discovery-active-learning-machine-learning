"""User-only response-free task screening; no LLM, engine, or pool response."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.system_freeze import verify_system_freeze
from hypothesis_mvp.discovery.task_screening import (
    run_task_screening, validate_task_screening_registration,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validate_task_screening_registration(config)
    verify_system_freeze(ROOT, config, freeze)
    if args.preflight_only:
        print(json.dumps({"passed": True, "real_data_access": False,
                          "provider_calls": 0, "engine_jobs": 0,
                          "heldout_access": False,
                          "formal_experiment_authorized": False}))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    run_task_screening(args.output_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

