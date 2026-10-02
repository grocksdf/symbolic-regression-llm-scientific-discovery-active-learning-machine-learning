"""Freeze the Expanded Formula Discovery development experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from scripts.run_aistats_drr_benchmark import _sha


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--correctness-gate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("formula-discovery freeze output must be new")
    registration = _read(args.registration)
    preflight = _read(args.preflight)
    gate = _read(args.correctness_gate)
    if (registration.get("schema") !=
            "scientific-expanded-formula-discovery-registration-v1"
            or registration.get("execution_authorized") is not False
            or preflight.get("schema") !=
                "scientific-expanded-formula-discovery-preflight-v1"
            or preflight.get("execution_authorized") is not False
            or gate.get("schema") !=
                "scientific-expanded-formula-discovery-correctness-gate-v1"
            or gate.get("passed") is not True):
        raise ValueError("formula-discovery prerequisites are invalid")
    metadata_files = {
        family: next(path for path in preflight["metadata_files"]
                     if filename in path)
        for family, filename in {
            "bio_pop_growth": "lsr_synth_bio_pop_growth-",
            "chem_react": "lsr_synth_chem_react-",
            "matsci": "lsr_synth_matsci-",
            "phys_osc": "lsr_synth_phys_osc-",
            "lsr_transform": "lsr_transform-",
        }.items()
    }
    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    files = [
        ROOT / "scripts/run_expanded_formula_discovery.py",
        ROOT / "scripts/expanded_formula_discovery_harness.py",
        ROOT / "scripts/run_aistats_three_arm_child.py",
        ROOT / "configs/aistats_three_arm_l_gap_v1.yaml",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_searcher.py",
        args.registration, args.preflight, args.correctness_gate,
        *map(Path, metadata_files.values()),
    ]
    development_count = sum(
        len(tasks) for tasks in preflight["development_tasks"].values())
    result = {
        "schema": "scientific-expanded-formula-discovery-freeze-v1",
        "method": "diagnose-compose-admit-v1",
        "development_tasks": preflight["development_tasks"],
        "untouched_confirmation_tasks":
            preflight["untouched_confirmation_tasks"],
        "development_task_count": development_count,
        "confirmation_task_count": sum(len(tasks) for tasks in
                                       preflight[
                                           "untouched_confirmation_tasks"
                                       ].values()),
        "seeds": [301, 302],
        "gap_generation_run_count": development_count * 2,
        "admission_arms": preflight["admission_arms"],
        "admission_row_count": development_count * 2 * 3,
        "llm_materialization_limit": 7,
        "llm_user_prompt_utf8_bytes": 65536,
        "wall_time_seconds_per_run": 900,
        "maximum_llm_calls_per_run": 12,
        "maximum_completion_tokens_per_call": 1800,
        "primary_endpoint": "posterior-map-exact-formula-recovery",
        "primary_comparisons": [
            "independent_protected-minus-same_data",
            "independent_protected-minus-accept_all",
        ],
        "secondary_endpoints": [
            "posterior-map-structural-recovery",
            "bank-exact-recall",
            "bank-structural-recall",
            "bank-size",
            "provider-calls-and-token-use",
        ],
        "ground_truth_policy": {
            "development_expression_opened_only_after_all_admission_rows_frozen":
                True,
            "ground_truth_never_available_to_generation_or_admission": True,
            "confirmation_expression_opened": False,
            "confirmation_results_opened": False,
        },
        "metadata_files": metadata_files,
        "hdf5": {
            "path": str(hdf5.resolve()), "sha256": _sha(hdf5),
            "size_bytes": hdf5.stat().st_size},
        "provider_env": str(args.provider_env.resolve()),
        "mainline_root": str((ROOT.parent / "hypothesis_mvp").resolve()),
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(Path(path).resolve()): _sha(path) for path in files},
        "candidate_response_accessed": False,
        "ground_truth_expression_values_opened": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "confirmation_results_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Fresh development Expanded Formula Discovery experiment over "
            "one Gap-generated candidate bank per task/seed and three offline "
            "admission projections. Ground truth opens only after admissions "
            "freeze. Test/OOD and untouched confirmation remain closed."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "EXPANDED_FORMULA_DISCOVERY_FREEZE.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
