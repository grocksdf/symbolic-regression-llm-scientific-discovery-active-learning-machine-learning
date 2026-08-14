"""Verify a canonical delivery manifest against local source bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hypothesis_mvp.hypotheses import production_code_hash, verify_local_delivery


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    tree_hash = verify_local_delivery(root)
    manifest = json.loads(
        (root / "DELIVERY_MANIFEST.json").read_text(encoding="utf-8")
    )
    code_hash = production_code_hash(root)
    if manifest.get("source_tree_sha256") != tree_hash:
        raise ValueError("delivery source-tree declaration does not match inventory")
    if manifest.get("production_code_hash") != code_hash:
        raise ValueError("delivery production-code declaration does not match bytes")
    report = {
        "status": "passed",
        "source_tree_sha256": tree_hash,
        "production_code_hash": code_hash,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
