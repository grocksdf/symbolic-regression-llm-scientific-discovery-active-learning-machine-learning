"""Freeze paired baseline/challenger allocation calibration coordinates."""

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
    parser.add_argument("--pilot-freeze", type=Path, action="append",
                        required=True)
    parser.add_argument("--correctness-gate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("allocation calibration freeze output must be new")
    tasks = []
    for path in args.pilot_freeze:
        freeze = json.loads(path.read_text(encoding="utf-8"))
        for family, task in freeze["selected_tasks"].items():
            tasks.append((str(family), str(task)))
    tasks = sorted(set(tasks))
    if len(tasks) != 8 or len({family for family, _ in tasks}) != 4:
        raise ValueError(
            "allocation calibration requires all eight prior pilot tasks")
    coordinates = [{
        "family": family, "task": task, "seed": 61 + index}
        for index, (family, task) in enumerate(tasks)]
    files = [
        ROOT / "configs/aistats_allocation_calibration_baseline.yaml",
        ROOT / "configs/aistats_allocation_calibration_challenger.yaml",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_searcher.py",
        ROOT / "scripts/run_aistats_allocation_calibration.py",
        args.correctness_gate, *args.pilot_freeze,
    ]
    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    result = {
        "schema": "scientific-aistats-allocation-calibration-freeze-v1",
        "coordinates": coordinates,
        "conditions": ["allocation_baseline", "allocation_challenger"],
        "child_run_count": 16,
        "prefixes": [8, 16, 32],
        "gain": "challenger-normalized-aulc-minus-baseline-normalized-aulc",
        "numerical_tolerance": 1e-12,
        "calibration_policy": {
            "credible_level": 0.9,
            "minimum_tasks": 8,
            "minimum_families": 3,
            "minimum_family_tasks": 2,
            "global_lower_bound_threshold": 0.5,
            "family_lower_bound_threshold": 0.5,
        },
        "selection_rule": (
            "all tasks from the first two immutable protected-prefix "
            "development pilots; no outcome-based inclusion"),
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(path.resolve()): _sha(path) for path in files},
        "hdf5": {
            "path": str(hdf5.resolve()), "sha256": _sha(hdf5),
            "size_bytes": hdf5.stat().st_size},
        "provider_env": str((ROOT / ".env").resolve()),
        "benchmark_data_root": str(args.data_root.resolve()),
        "benchmark_cache_root": str(
            (ROOT / ".cache/aistats_drr_20260928").resolve()),
        "mainline_root": str((ROOT.parent / "hypothesis_mvp").resolve()),
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Allocation-only development calibration over previously used "
            "tasks; not efficacy, superiority, confirmation, or task "
            "selection evidence."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_ALLOCATION_CALIBRATION_FREEZE.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
