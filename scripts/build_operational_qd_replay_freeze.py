"""Freeze response-free BOQD replay over a passed synthesis candidate set."""

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
    parser.add_argument("--synthesis-freeze", type=Path, required=True)
    parser.add_argument("--synthesis-result", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--correctness-gate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("BOQD replay freeze output must be new")
    freeze = json.loads(args.synthesis_freeze.read_text(encoding="utf-8"))
    result = json.loads(args.synthesis_result.read_text(encoding="utf-8"))
    if result.get("passed") is not True:
        raise ValueError("source synthesis pilot did not pass")
    artifacts = {}
    for family, task in freeze["selected_tasks"].items():
        for seed in freeze["seeds"]:
            for condition in freeze["conditions"]:
                path = (
                    args.source_output / "runs" / family / task /
                    f"seed{seed}" / condition / "pcpi_artifacts" /
                    f"{task}.json")
                artifacts[str(path.resolve())] = _sha(path)
    files = [
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "scripts/run_operational_qd_replay.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/operational_qd.py",
        args.synthesis_freeze, args.synthesis_result,
        args.correctness_gate,
    ]
    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    payload = {
        "schema": "scientific-operational-qd-replay-freeze-v1",
        "selected_tasks": freeze["selected_tasks"],
        "seeds": freeze["seeds"],
        "conditions": freeze["conditions"],
        "row_count": 16,
        "prefixes": [8, 16, 32],
        "methods": ["legacy-family-topk", "boqd-v1"],
        "screening_decision": {
            "zero_failures": True,
            "mean_boost_strictly_positive": True,
            "minimum_positive_families": 2,
            "no_negative_transfer_family": True,
            "minimum_llm_novel_niche_families": 2,
            "numerical_tolerance": 1e-12,
        },
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(path.resolve()): _sha(path) for path in files},
        "candidate_artifacts": artifacts,
        "hdf5": {
            "path": str(hdf5.resolve()), "sha256": _sha(hdf5),
            "size_bytes": hdf5.stat().st_size},
        "source_output": str(args.source_output.resolve()),
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Response-free BOQD versus legacy candidate-pool replay only; "
            "not realized efficacy, superiority, or confirmation."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "OPERATIONAL_QD_REPLAY_FREEZE.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
