"""User-only registered development runner; preflight never loads data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.system_executor import execute_registered_system, validate_system_registration, verify_registered_provider
from hypothesis_mvp.discovery.system_freeze import verify_system_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validate_system_registration(config)
    if args.preflight_only:
        verify_system_freeze(ROOT, config, freeze)
        verify_registered_provider(ROOT, config)
        print(json.dumps({"passed": True, "real_data_access": False,
            "heldout_access": False, "provider_public_identity_verified": True,
            "user_pilot_authorized": config["user_execution_authorized"],
            "provider_authentication_tested": False, "formal_experiment_authorized": False}))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for user execution")
    execute_registered_system(ROOT, args.output_dir, config, freeze, execution_role="user",
                              measurement_authorized=not args.screen_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
