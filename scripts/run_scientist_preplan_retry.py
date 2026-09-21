"""User-only retry of a frozen system after certified pre-plan provider failure."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.system_executor import (
    execute_registered_system, validate_system_registration,
)
from hypothesis_mvp.discovery.system_freeze import verify_system_freeze


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    retry = json.loads(args.retry.read_text(encoding="utf-8"))
    validate_system_registration(config)
    verify_system_freeze(ROOT, config, freeze)
    source, output = args.source_output.resolve(), args.output_dir.resolve()
    if (retry.get("schema") != "scientific-preplan-provider-retry-v1"
            or Path(retry.get("source_output", "")).resolve() != source
            or retry.get("scientific_artifact_count") != 0
            or output.exists() or output == source):
        raise ValueError("invalid pre-plan retry request")
    for name, expected in retry["artifacts"].items():
        if _sha(source / name) != expected:
            raise ValueError("pre-plan failure artifact changed")
    execute_registered_system(
        ROOT, output, config, freeze, execution_role="user",
        measurement_authorized=False)


if __name__ == "__main__":
    main()
