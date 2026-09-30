"""Freeze fresh-task synthesis-only screening coordinates."""

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
    return min(names, key=lambda name: sha256(
        f"aistats-synthesis-only-pilot-v1:{family}:{name}".encode()
    ).digest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--prior-pilot-freeze", type=Path, action="append",
                        required=True)
    parser.add_argument("--smoke-gate", type=Path, required=True)
    parser.add_argument("--adapter-gate", type=Path, required=True)
    parser.add_argument("--resource-gate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("synthesis-only pilot freeze output must be new")
    base = json.loads(args.benchmark_freeze.read_text(encoding="utf-8"))
    used = {
        family: set(base["selected_tasks"][family])
        for family in FAMILY_PATHS}
    used["bio_pop_growth"].add("BPG2")
    for path in args.prior_pilot_freeze:
        prior = json.loads(path.read_text(encoding="utf-8"))
        for family, task in prior["selected_tasks"].items():
            used[family].add(task)
    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    selected = {}
    with h5py.File(hdf5, "r") as handle:
        for family, group in FAMILY_PATHS.items():
            available = sorted(
                name for name in handle[group].keys()
                if name not in used[family])
            selected[family] = _select(available, family)
    files = [
        ROOT / "configs/aistats_drr_full_v6.yaml",
        ROOT / "configs/aistats_drr_no_llm_v6.yaml",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_searcher.py",
        ROOT / "scripts/run_aistats_synthesis_only_pilot.py",
        args.benchmark_freeze, args.smoke_gate,
        args.adapter_gate, args.resource_gate, *args.prior_pilot_freeze,
    ]
    result = {
        "schema": "scientific-aistats-synthesis-only-pilot-freeze-v1",
        "selected_tasks": selected,
        "seeds": [71, 72],
        "conditions": ["full_scientist_v6", "no_llm_v6"],
        "child_run_count": 16,
        "prefixes": [8, 16, 32],
        "primary_metric":
            "paired-family-mean-full-minus-no-llm-normalized-prefix-aulc",
        "screening_decision": {
            "zero_child_failures": True,
            "identical_engine_execution_within_every_pair": True,
            "paired_mean_strictly_positive": True,
            "minimum_positive_families": 2,
            "minimum_novel_synthesis_families": 2,
            "no_negative_transfer_family": True,
            "numerical_tolerance": 1e-12,
        },
        "selection_rule": (
            "minimum SHA-256 under a fixed synthesis-only namespace after "
            "excluding the frozen 30-task set, BPG2, and both prior pilots"),
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
        "task_arrays_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Fresh-task synthesis-only development screening; not realized "
            "acquisition efficacy, superiority, or confirmation."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_SYNTHESIS_ONLY_PILOT_FREEZE.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
