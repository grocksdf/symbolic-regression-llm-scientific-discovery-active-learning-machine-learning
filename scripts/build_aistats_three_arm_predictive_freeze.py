"""Freeze the fresh-task E / E+L_blind / E+L_gap predictive Gate."""

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
CONDITIONS = [
    "three_arm_e_v1",
    "three_arm_l_blind_v1",
    "three_arm_l_gap_v1",
]


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


def _selection_key(family: str, task: str) -> bytes:
    return sha256(
        f"aistats-three-arm-predictive-v1:{family}:{task}".encode()
    ).digest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-freeze", type=Path, required=True)
    parser.add_argument("--three-arm-gate", type=Path, required=True)
    parser.add_argument("--exclude-freeze", type=Path, action="append",
                        required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("three-arm freeze output must be new")
    mainline_freeze = _read(args.mainline_freeze)
    gate = _read(args.three_arm_gate)
    current_mainline = verify_clean_git_source(
        ROOT.parent / "hypothesis_mvp")
    if (
        mainline_freeze.get("schema")
        != "scientific-pcpi-60a665f5-correctness-freeze-v1"
        or mainline_freeze.get("passed") is not True
        or mainline_freeze["mainline_source"]["git_commit"]
            != "60a665f5c58f5e7e394e78ccfc57071a6ced8be8"
        or mainline_freeze["mainline_source"]["git_commit"]
            != current_mainline["source_git_commit"]
        or mainline_freeze["mainline_source"]["git_tree"]
            != current_mainline["source_git_tree"]
    ):
        raise ValueError("60a665f5 correctness freeze is invalid")
    if (
        gate.get("schema")
        != "scientific-aistats-three-arm-correctness-gate-v1"
        or gate.get("passed") is not True
        or gate.get("benchmark_task_arrays_accessed") is not False
        or gate.get("llm_called") is not False
    ):
        raise ValueError("three-arm correctness Gate is invalid")
    if not args.provider_env.is_file():
        raise ValueError("registered provider env is unavailable")

    excluded = {family: set() for family in FAMILY_PATHS}
    exclusions = {}
    for path in args.exclude_freeze:
        payload = _read(path)
        for family in FAMILY_PATHS:
            excluded[family].update(_collect_selected(payload, family))
        exclusions[str(path.resolve())] = _sha(path)
    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    selected = {}
    with h5py.File(hdf5, "r") as handle:
        for family, group in FAMILY_PATHS.items():
            available = [
                task for task in handle[group].keys()
                if task not in excluded[family]]
            if not available:
                raise ValueError(
                    f"no fresh three-arm task remains for {family}")
            selected[family] = min(
                available, key=lambda task: _selection_key(family, task))

    files = [
        ROOT / "configs/aistats_three_arm_e_v1.yaml",
        ROOT / "configs/aistats_three_arm_l_blind_v1.yaml",
        ROOT / "configs/aistats_three_arm_l_gap_v1.yaml",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_searcher.py",
        ROOT / "scripts/run_aistats_three_arm_child.py",
        ROOT / "scripts/run_aistats_three_arm_predictive_gate.py",
        args.mainline_freeze, args.three_arm_gate, *args.exclude_freeze,
    ]
    result = {
        "schema": "scientific-aistats-three-arm-predictive-freeze-v1",
        "selected_tasks": selected,
        "selection_rule": (
            "minimum SHA-256 task identity per family under the frozen "
            "aistats-three-arm-predictive-v1 namespace after excluding "
            "every task named by the supplied prior freezes"),
        "excluded_tasks": {
            family: sorted(tasks) for family, tasks in excluded.items()},
        "exclusion_freezes": exclusions,
        "seeds": [101, 102],
        "conditions": CONDITIONS,
        "child_run_count": 24,
        "pair_count": 8,
        "engine_job_budget_per_cycle": 4,
        "cycles": 2,
        "candidate_evaluation_budget_per_cycle": 55,
        "llm_materialization_limit": 7,
        "maximum_llm_calls_per_run": 4,
        "maximum_provider_attempts_per_run": 12,
        "maximum_completion_tokens_per_call": 1800,
        "llm_user_prompt_utf8_bytes": 65536,
        "wall_time_seconds_per_run": 1800,
        "primary_metric":
            "task-mean-E+L_gap-minus-E+L_blind-report-log-score",
        "decision": {
            "zero_pair_failures": True,
            "global_gap_minus_blind_strictly_positive": True,
            "minimum_positive_tasks": 2,
            "global_gap_minus_engine_strictly_positive": True,
            "all_uncertified_decisions_remain_explicit": True,
            "numerical_tolerance": 1e-12,
        },
        "serial_gate": {
            "predictive_gate_first": True,
            "decision_transaction_not_authorized_by_freeze": True,
            "action_response_accessed": False,
        },
        "resource_fairness": {
            "identical_engine_schedule": True,
            "identical_total_candidate_budget": True,
            "blind_gap_equal_llm_call_count_required": True,
            "blind_gap_equal_completion_token_cap_required": True,
            "blind_gap_equal_prompt_byte_cap_required": True,
            "actual_provider_token_usage_recorded": True,
            "wall_time_ceiling_equal_across_arms": True,
            "final_bank_size_and_inference_wall_time_recorded": True,
        },
        "knowledge_isolation": {
            "mode": "empty-distinct-per-arm-task-seed-namespace-v1",
            "cross_arm_read_enabled": False,
            "persistent_library_enabled": False,
            "task_local_memory_enabled": False,
        },
        "mainline_correctness_freeze": {
            "path": str(args.mainline_freeze.resolve()),
            "sha256": _sha(args.mainline_freeze),
        },
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": current_mainline,
        "files": {str(path.resolve()): _sha(path) for path in files},
        "hdf5": {
            "path": str(hdf5.resolve()), "sha256": _sha(hdf5),
            "size_bytes": hdf5.stat().st_size},
        "provider_env": str(args.provider_env.resolve()),
        "benchmark_data_root": str(args.data_root.resolve()),
        "benchmark_cache_root": str(
            (ROOT / ".cache/aistats_three_arm_20261001").resolve()),
        "mainline_root": str((ROOT.parent / "hypothesis_mvp").resolve()),
        "task_arrays_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Fresh-task matched three-arm predictive Gate only: "
            "gap evidence to LLM structure to independent admission to "
            "reporting predictive gain and response-free PCPI decision path. "
            "No action response, realized decision gain, test/OOD, held-out, "
            "or universal superiority claim."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_THREE_ARM_PREDICTIVE_FREEZE.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
