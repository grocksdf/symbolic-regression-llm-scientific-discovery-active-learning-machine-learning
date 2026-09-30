"""Freeze a fresh four-family short-prefix DRR screening pilot."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import h5py


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


FAMILY_PATHS = {
    "bio_pop_growth": "lsr_synth/bio_pop_growth",
    "chem_react": "lsr_synth/chem_react",
    "lsr_transform": "lsr_transform",
    "matsci": "lsr_synth/matsci",
}


def _sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select(names, family):
    return min(
        names,
        key=lambda name: sha256(
            f"aistats-protected-prefix-pilot-v2:{family}:{name}".encode()
        ).digest(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--prior-pilot-freeze", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--adapter-gate", type=Path, required=True)
    parser.add_argument("--resource-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("prefix pilot freeze output must be new")
    old = json.loads(args.benchmark_freeze.read_text(encoding="utf-8"))
    prior = json.loads(
        args.prior_pilot_freeze.read_text(encoding="utf-8"))
    used = {
        family: set(old["selected_tasks"][family])
        for family in FAMILY_PATHS
    }
    used["bio_pop_growth"].add("BPG2")
    for family, task in prior["selected_tasks"].items():
        used[family].add(task)
    hdf5_path = args.data_root / "lsr_bench_data.hdf5"
    selected = {}
    with h5py.File(hdf5_path, "r") as handle:
        for family, path in FAMILY_PATHS.items():
            names = sorted(
                name for name in handle[path].keys()
                if name not in used[family])
            if not names:
                raise ValueError(f"no fresh pilot task remains for {family}")
            selected[family] = _select(names, family)
    files = [
        ROOT / "configs/aistats_drr_full_v6.yaml",
        ROOT / "configs/aistats_drr_no_llm_v6.yaml",
        ROOT / "configs/aistats_drr_single_engine_v6.yaml",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_searcher.py",
        ROOT / "scripts/run_aistats_drr_prefix_pilot.py",
        args.benchmark_freeze, args.prior_pilot_freeze,
        args.adapter_gate, args.resource_gate,
    ]
    result = {
        "schema": "scientific-aistats-drr-prefix-pilot-freeze-v2",
        "selected_tasks": selected,
        "seeds": [43, 44],
        "conditions": [
            "full_scientist_v6", "no_llm_v6", "single_engine_v6"],
        "child_run_count": 24,
        "prefixes": [8, 16, 32],
        "primary_metric":
            "paired-task-mean-full-minus-no-llm-normalized-prefix-aulc",
        "screening_decision": {
            "paired_mean_strictly_positive": True,
            "minimum_positive_task_families": 3,
            "all_four_task_families_included": True,
            "failures_score_zero": True,
        },
        "selection_rule": (
            "minimum SHA-256 of fixed pilot namespace, family and task after "
            "excluding the frozen 30-task set, BPG2 smoke, and every task "
            "used by the first protected-prefix pilot"),
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(path.resolve()): _sha(path) for path in files},
        "hdf5": {
            "path": str(hdf5_path.resolve()),
            "sha256": _sha(hdf5_path),
            "size_bytes": hdf5_path.stat().st_size,
        },
        "provider_env": str((ROOT / ".env").resolve()),
        "benchmark_data_root": str(args.data_root.resolve()),
        "benchmark_cache_root": str(
            (ROOT / ".cache/aistats_drr_20260928").resolve()),
        "mainline_root": str((ROOT.parent / "hypothesis_mvp").resolve()),
        "candidate_response_accessed": False,
        "task_arrays_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": True,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Prospective development screening only; positive direction can "
            "authorize a later measured pilot but is not efficacy or "
            "superiority evidence."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_DRR_PREFIX_PILOT_FREEZE.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
