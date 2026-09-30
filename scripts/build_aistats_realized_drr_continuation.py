"""Certify continuation after the immutable LSR train-member loader failure."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


def _sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("realized continuation output must be new")
    rows_path = args.source_output / "REALIZED_ROWS.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    keys = sorted("/".join(map(str, (
        row["family"], row["seed"], row["label"]))) for row in rows)
    if (len(rows) != 16 or any(row["status"] != "success" for row in rows)
            or any(row["family"] not in {
                "bio_pop_growth", "chem_react"} for row in rows)):
        raise ValueError("unexpected realized DRR partial prefix")
    result = {
        "schema": "scientific-aistats-realized-drr-continuation-v1",
        "source_freeze": str(args.freeze.resolve()),
        "source_freeze_sha256": _sha(args.freeze),
        "source_output": str(args.source_output.resolve()),
        "partial_rows_sha256": _sha(rows_path),
        "completed_trajectory_count": len(rows),
        "completed_trajectory_keys": keys,
        "remaining_families": ["lsr_transform", "matsci"],
        "repair_scope": (
            "loader member identity lsr_transform/train plus strict immutable "
            "resume; no model, candidate, seed, budget, policy, response or "
            "decision threshold change"),
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Infrastructure-only continuation certificate; preserves all "
            "completed trajectories and authorizes only missing families."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_REALIZED_DRR_CONTINUATION.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
