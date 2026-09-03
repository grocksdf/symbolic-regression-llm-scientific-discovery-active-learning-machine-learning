"""Guard the immutable historical P3J.1 correctness-stage identity.

The executable implementation moved to P3K after the completed P3J.15 negative
development result. Reproducing P3J.1 requires its historical source tree; the
current tree must not relabel P3K calculations as P3J evidence.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "p3j_1_class_conditional_correctness.json"
CONFIG_SHA256 = "f7b3ae28ba5595e294521b41c5c1c0c3ddfff2c4e93ca54566773fee023b7ea9"


def _validate_archived_config(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or PROJECT_ROOT not in resolved.parents:
        raise ValueError("P3J.1 archived config must be inside the project root")
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3J.1 archived config hash changed")
    config = json.loads(raw.decode("utf-8"))
    if (
        config.get("schema")
        != "pcpi-p3j1-class-conditional-correctness-config-v1"
        or config.get("stage") != "P3J.1"
        or config.get("operational_execution_authorized") is not False
        or config.get("real_data_access") is not False
        or config.get("heldout_access") is not False
        or config.get("simulated_experiment") is not False
    ):
        raise ValueError("P3J.1 archived contract changed")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    _validate_archived_config(args.config)
    raise RuntimeError(
        "P3J.1 is historical and cannot execute the current P3K implementation; "
        "use the frozen P3J source tree for reproduction"
    )


if __name__ == "__main__":
    raise SystemExit(main())
