"""Freeze the MatSci quality-first predictive-increment comparison protocol."""

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


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_selected(value, family: str) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        selected = value.get("selected_tasks")
        if isinstance(selected, dict):
            tasks = selected.get(family)
            if isinstance(tasks, str):
                output.add(tasks)
            elif isinstance(tasks, list):
                output.update(str(task) for task in tasks)
        for child in value.values():
            output.update(_collect_selected(child, family))
    elif isinstance(value, list):
        for child in value:
            output.update(_collect_selected(child, family))
    return output


def _selection_key(task: str) -> bytes:
    return sha256(
        "aistats-matsci-quality-first-increment-v1:"
        f"{task}".encode()
    ).digest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-gate", type=Path, required=True)
    parser.add_argument("--exclude-freeze", type=Path, action="append",
                        required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("fresh", "legacy"), default="fresh",
        help="fresh: four SHA-256-minimum tasks unused by any registered "
             "prior freeze (formal).  legacy: deliberately re-use the four "
             "MatSci tasks of the frozen 20260929 acquisition confirmation "
             "as a development diagnostic against the old records; repeated "
             "seeds are repeat measurements, not new tasks.")
    parser.add_argument(
        "--tasks", action="append", default=None,
        help="explicit task override (recorded verbatim in the freeze)")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("quality-first increment freeze output must be new")

    gate = _read(args.correctness_gate)
    if (
        gate.get("schema")
        != "scientific-aistats-matsci-quality-first-correctness-gate-v1"
        or gate.get("passed") is not True
        or gate.get("benchmark_task_arrays_accessed") is not False
        or gate.get("llm_called") is not False
        or gate.get("heldout_opened") is not False
        or gate.get("benchmark_source") != verify_clean_git_source(ROOT)
        or gate.get("mainline_source") != verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp")
    ):
        raise ValueError("quality-first correctness Gate is invalid")
    if not args.provider_env.is_file():
        raise ValueError("registered provider env is unavailable")

    excluded: set[str] = set()
    exclusion_identities = {}
    for path in args.exclude_freeze:
        payload = _read(path)
        excluded.update(_collect_selected(payload, "matsci"))
        exclusion_identities[str(path.resolve())] = _sha(path)

    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    with h5py.File(hdf5, "r") as handle:
        names = sorted(handle["lsr_synth/matsci"].keys())
    legacy_tasks = ["MatSci6", "MatSci8", "MatSci16", "MatSci25"]
    if args.tasks:
        selected = sorted(set(str(task) for task in args.tasks))
        selection_rule = (
            "tasks named explicitly on the command line and recorded "
            "verbatim in this freeze"
        )
        mode = "explicit"
    elif args.mode == "legacy":
        selected = list(legacy_tasks)
        selection_rule = (
            "the four MatSci tasks of the frozen 20260929 acquisition "
            "confirmation (MatSci6/8/16/25), re-used deliberately as a "
            "development diagnostic against those old records; seeds 81/82 "
            "are repeated measurements and do not constitute new tasks"
        )
        mode = "legacy-development-diagnostic"
    else:
        available = sorted(
            (name for name in names if name not in excluded),
            key=_selection_key,
        )
        if len(available) < 4:
            raise ValueError("fewer than four fresh MatSci tasks remain")
        selected = available[:4]
        selection_rule = (
            "four minimum SHA-256 task identities under the frozen "
            "aistats-matsci-quality-first-increment-v1 namespace after "
            "excluding every task named by the registered prior freezes"
        )
        mode = "fresh-formal"
    missing = [task for task in selected if task not in names]
    if missing:
        raise ValueError(f"selected tasks absent from the dataset: {missing}")
    independent_tasks = mode == "fresh-formal"

    files = [
        ROOT / "eval.py",
        ROOT / "configs/aistats_drr_full_v6.yaml",
        ROOT / "configs/aistats_drr_no_llm_v6.yaml",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "methods/hypothesis_mvp_pcpi/quality_first_augmentation.py",
        ROOT / "scripts/run_aistats_drr_benchmark.py",
        ROOT / "scripts/run_aistats_matsci_quality_first_increment.py",
        ROOT / "scripts/run_aistats_matsci_quality_first_correctness_gate.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/realized_drr.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/pcpi_adapter.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/bank_selection.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/source_stacking.py",
        args.correctness_gate,
        *args.exclude_freeze,
    ]
    result = {
        "schema":
            "scientific-aistats-matsci-quality-first-increment-freeze-v1",
        "protocol_mode": mode,
        "independent_tasks": independent_tasks,
        "selected_tasks": {"matsci": selected},
        "selection_rule": selection_rule,
        "excluded_tasks": sorted(excluded),
        "exclusion_freezes": exclusion_identities,
        "seeds": [81, 82],
        "conditions": ["full_scientist_v6", "no_llm_v6"],
        "discovery_run_count": 16,
        "pair_count": 8,
        "discovery_budget": 48,
        "baseline_evaluations": 48,
        "llm_proposal_limit": 12,
        "measurement_budget": 2,
        "maximum_candidates": 4,
        "initial_rows": 32,
        "report_rows": 64,
        "report_slice_rule": (
            "inference_initial segment in canonical sha256 order; rows "
            "[:32] are the initial data and rows [32:96] are the report "
            "target, disjoint from discovery rows, the initial data, and "
            "the action covariates"
        ),
        "primary_metric": (
            "task-mean paired report log-score difference "
            "(48E+L_LLM minus 48E)"
        ),
        "increment_decision": {
            "zero_pair_failures": True,
            "global_llm_log_score_gain_strictly_positive": True,
            "minimum_strictly_positive_tasks": 2,
            "llm_gain_exceeds_equal_size_control_gain": True,
            "numerical_tolerance": 1e-12,
        },
        "wall_time_seconds_per_discovery": 900,
        "maximum_llm_calls_per_discovery": 4,
        "maximum_provider_attempts_per_discovery": 12,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"
        ),
        "files": {str(path.resolve()): _sha(path) for path in files},
        "hdf5": {
            "path": str(hdf5.resolve()),
            "sha256": _sha(hdf5),
            "size_bytes": hdf5.stat().st_size,
        },
        "benchmark_data_root": str(args.data_root.resolve()),
        "benchmark_cache_root": str(
            (ROOT / ".cache/aistats_matsci_quality_first").resolve()
        ),
        "mainline_root": str((ROOT.parent / "hypothesis_mvp").resolve()),
        "provider_env": str(args.provider_env.resolve()),
        "task_arrays_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "One prospectively frozen fresh-task MatSci development "
            "comparison of paired report-set predictive log scores over "
            "matched-engine banks (48E versus 48E+L_LLM versus "
            "48E+L_control); no acquisition benefit, test/OOD accuracy, "
            "held-out confirmation, or universal superiority claim."
            if independent_tasks else
            "Development diagnostic on the four MatSci tasks already used "
            "by the frozen 20260929 acquisition confirmation, comparing "
            "paired report-set predictive log scores over matched-engine "
            "banks (48E versus 48E+L_LLM versus 48E+L_control); these "
            "tasks are not independent of earlier selection, the seeds are "
            "repeat measurements, and the result cannot replace a "
            "fresh-task confirmation or support any acquisition, "
            "test/OOD, or universal claim."
        ),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / (
        "AISTATS_MATSCI_QUALITY_FIRST_INCREMENT_FREEZE.json"
    )
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
