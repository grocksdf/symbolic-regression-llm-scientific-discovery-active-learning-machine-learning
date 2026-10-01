"""Freeze a fresh-task PCPI versus LLM-SR external-baseline comparison."""

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
        f"aistats-external-llmsr-v1:{family}:{task}".encode()
    ).digest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-gate", type=Path, required=True)
    parser.add_argument("--exclude-freeze", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--external-site", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("external LLM-SR freeze output must be new")
    gate = _read(args.resource_gate)
    if (
        gate.get("schema")
        != "scientific-aistats-external-llmsr-resource-gate-v1"
        or gate.get("passed") is not True
        or not gate.get("decisions", {}).get("live_provider_response_received")
        or gate.get("scientific_prompt_sent") is not False
        or gate.get("benchmark_task_arrays_accessed") is not False
    ):
        raise ValueError("external LLM-SR resource Gate is invalid")

    used = {family: set() for family in FAMILY_PATHS}
    exclusion_identities = {}
    for path in args.exclude_freeze:
        payload = _read(path)
        for family in FAMILY_PATHS:
            used[family].update(_collect_selected(payload, family))
        exclusion_identities[str(path.resolve())] = _sha(path)

    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    selected = {}
    with h5py.File(hdf5, "r") as handle:
        for family, group in FAMILY_PATHS.items():
            available = [
                name for name in handle[group].keys()
                if name not in used[family]
            ]
            if not available:
                raise ValueError(f"no fresh external task remains for {family}")
            selected[family] = min(
                available,
                key=lambda task: _selection_key(family, task),
            )

    external_site = args.external_site.resolve()
    package_files = [
        external_site / "openai/__init__.py",
        external_site / "openai/_client.py",
    ]
    metadata = sorted(external_site.glob("openai-*.dist-info/METADATA"))
    if len(metadata) != 1:
        raise ValueError("exactly one external OpenAI metadata file required")
    package_files.append(metadata[0])
    files = [
        ROOT / "eval.py",
        ROOT / "configs/aistats_drr_full_v6.yaml",
        ROOT / "configs/aistats_drr_no_llm_v6.yaml",
        ROOT / "configs/aistats_drr_full_llmchannel_v7.yaml",
        ROOT / "configs/aistats_external_llmsr_glm53.yaml",
        ROOT / "methods/llmsr/sampler.py",
        ROOT / "methods/llmsr/profile.py",
        ROOT / "scripts/run_aistats_external_llmsr_benchmark.py",
        args.resource_gate,
        *args.exclude_freeze,
        *package_files,
    ]
    result = {
        "schema": "scientific-aistats-external-llmsr-freeze-v1",
        "selected_tasks": selected,
        "seeds": [91, 92],
        "conditions": [
            "full_scientist_v6", "no_llm_v6", "full_llmchannel_v7",
            "external_llmsr_glm53"],
        "child_run_count": 32,
        "condition_budgets": {
            # Every internal condition gets the same wall-clock ceiling as the
            # external LLM-SR baseline.  The previous 900s internal ceiling
            # produced spurious wall-time-timeout failures on the heavier
            # task, which silently counted as NMSE=100.
            "full_scientist_v6": {
                "candidate_evaluation_limit": 48,
                "wall_time_seconds": 1800,
            },
            "no_llm_v6": {
                "candidate_evaluation_limit": 48,
                "wall_time_seconds": 1800,
            },
            "full_llmchannel_v7": {
                "candidate_evaluation_limit": 48,
                "wall_time_seconds": 1800,
            },
            "external_llmsr_glm53": {
                "sample_limit": 16,
                "wall_time_seconds": 1800,
            },
        },
        "primary_metrics": {
            "id": "failure-inclusive-restart-nmse",
            "ood": "failure-inclusive-restart-nmse",
            "accuracy": "strict-acc-at-0.1",
            "resource": "search-time-and-registered-sample-or-call-count",
        },
        "analysis_rule": (
            "all frozen rows included; failed runs receive NMSE=100 and "
            "Acc@0.1=0; report task-level seed means and Full minus external "
            "paired effects without task or seed replacement"
        ),
        "selection_rule": (
            "minimum SHA-256 identity per family under the frozen "
            "aistats-external-llmsr-v1 namespace after excluding every task "
            "named by the registered prior freezes"
        ),
        "excluded_tasks": {
            family: sorted(tasks) for family, tasks in used.items()},
        "exclusion_freezes": exclusion_identities,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(path.resolve()): _sha(path) for path in files},
        "hdf5": {
            "path": str(hdf5.resolve()),
            "sha256": _sha(hdf5),
            "size_bytes": hdf5.stat().st_size,
        },
        "provider_env": str(args.provider_env.resolve()),
        "external_site": str(external_site),
        "benchmark_data_root": str(args.data_root.resolve()),
        "benchmark_cache_root": str(
            (ROOT / ".cache/aistats_external_llmsr_20260929").resolve()),
        "mainline_root": str((ROOT.parent / "hypothesis_mvp").resolve()),
        "task_arrays_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "One frozen fresh-task external comparison against LLM-SR; "
            "test/OOD are opened only during this registered run and cannot "
            "change tasks, methods, budgets, or analysis."
        ),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_EXTERNAL_LLMSR_FREEZE.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
