"""Freeze realized acquisition over a passed synthesis-only candidate bank."""

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
        raise ValueError("realized DRR freeze output must be new")
    synthesis_freeze = json.loads(
        args.synthesis_freeze.read_text(encoding="utf-8"))
    synthesis_result = json.loads(
        args.synthesis_result.read_text(encoding="utf-8"))
    if synthesis_result.get("passed") is not True:
        raise ValueError("synthesis-only pilot did not pass")
    files = [
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "scripts/run_aistats_realized_drr_pilot.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/realized_drr.py",
        args.synthesis_freeze, args.synthesis_result,
        args.correctness_gate,
    ]
    artifacts = {}
    for family, task in synthesis_freeze["selected_tasks"].items():
        for seed in synthesis_freeze["seeds"]:
            for condition in synthesis_freeze["conditions"]:
                path = (
                    args.source_output / "runs" / family / task /
                    f"seed{seed}" / condition / "pcpi_artifacts" /
                    f"{task}.json")
                if not path.is_file():
                    raise FileNotFoundError(path)
                artifacts[str(path.resolve())] = _sha(path)
    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    result = {
        "schema": "scientific-aistats-realized-drr-pilot-freeze-v1",
        "selected_tasks": synthesis_freeze["selected_tasks"],
        "seeds": synthesis_freeze["seeds"],
        "conditions": synthesis_freeze["conditions"],
        "policies": [
            "full_targeted", "full_random",
            "no_llm_targeted", "no_llm_random"],
        "measurement_budget": 2,
        "trajectory_count": 32,
        "primary_metrics": {
            "synthesis_effect":
                "full-targeted-minus-no-llm-targeted-bounded-symmetric-risk-aulc",
            "acquisition_effect":
                "full-targeted-minus-full-random-bounded-symmetric-risk-aulc",
            "score_validity":
                "pearson-correlation-predicted-lower-bound-realized-gain",
            "absolute_sensitivity":
                "unnormalized-bayes-risk-reduction-aulc",
        },
        "screening_decision": {
            "zero_failures": True,
            "synthesis_mean_strictly_positive": True,
            "acquisition_mean_strictly_positive": True,
            "minimum_positive_families_each_effect": 2,
            "predicted_realized_correlation_strictly_positive": True,
            "absolute_synthesis_mean_nonnegative": True,
            "absolute_acquisition_mean_nonnegative": True,
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
        "candidate_response_access_authorized": True,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Matched-budget realized acquisition development pilot over "
            "already-frozen candidates; test/OOD and held-out remain closed."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_REALIZED_DRR_PILOT_FREEZE.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
