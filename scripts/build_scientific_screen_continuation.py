"""Build an immutable continuation certificate for a passed fresh source screen."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.system_executor import (
    _passed_screen_artifact_names, _sha256_file, _verify_passed_source_screen,
)
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.source_output.resolve()
    certificate = {
        "schema": "scientific-passed-source-screen-continuation-v2",
        "source_output": str(source),
        "artifacts": {name: _sha256_file(source / Path(name))
                      for name in sorted(_passed_screen_artifact_names(config))},
        "claim_boundary": (
            "continue immutable passed sourcewise screen without rerunning exploration"),
    }
    _verify_passed_source_screen(source, config, certificate)
    _publish(args.output.resolve(), certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
