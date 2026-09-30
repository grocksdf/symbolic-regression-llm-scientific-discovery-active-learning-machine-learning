"""Freeze a fresh-task MatSci acquisition confirmation after family authorization."""

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
        f"aistats-matsci-acquisition-confirmation-v1:{task}".encode()
    ).digest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-calibration", type=Path, required=True)
    parser.add_argument("--correctness-gate", type=Path, required=True)
    parser.add_argument("--exclude-freeze", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("MatSci confirmation freeze output must be new")

    policy = _read(args.policy_calibration)
    acquisition = policy.get("acquisition_policy") or {}
    if (
        policy.get("schema")
        != "scientific-aistats-realized-policy-calibration-v1"
        or policy.get("passed") is not True
        or "matsci" not in acquisition.get("authorized_families", ())
        or policy.get("test_or_ood_accessed") is not False
        or policy.get("heldout_opened") is not False
    ):
        raise ValueError("MatSci acquisition was not authorized")
    gate = _read(args.correctness_gate)
    if gate.get("passed") is not True or gate.get("heldout_opened") is not False:
        raise ValueError("realized DRR correctness Gate is invalid")

    excluded: set[str] = set()
    exclusion_identities = {}
    for path in args.exclude_freeze:
        payload = _read(path)
        excluded.update(_collect_selected(payload, "matsci"))
        exclusion_identities[str(path.resolve())] = _sha(path)

    hdf5 = args.data_root / "lsr_bench_data.hdf5"
    with h5py.File(hdf5, "r") as handle:
        names = sorted(handle["lsr_synth/matsci"].keys())
    available = sorted(
        (name for name in names if name not in excluded),
        key=_selection_key,
    )
    if len(available) < 4:
        raise ValueError("fewer than four fresh MatSci tasks remain")
    selected = available[:4]

    files = [
        ROOT / "eval.py",
        ROOT / "configs/aistats_drr_no_llm_v6.yaml",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT / "scripts/run_aistats_matsci_acquisition_confirmation.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/realized_drr.py",
        args.policy_calibration,
        args.correctness_gate,
        *args.exclude_freeze,
    ]
    result = {
        "schema": "scientific-aistats-matsci-acquisition-confirmation-freeze-v1",
        "selected_tasks": {"matsci": selected},
        "seeds": [81, 82],
        "condition": "no_llm_v6",
        "policies": ["decision_risk", "random"],
        "discovery_run_count": 8,
        "trajectory_count": 16,
        "measurement_budget": 2,
        "discovery_budget": 48,
        "wall_time_seconds_per_discovery": 900,
        "primary_metric":
            "task-mean-targeted-minus-random-bounded-symmetric-risk-aulc",
        "confirmation_decision": {
            "zero_failures": True,
            "all_tasks_have_nonnegative_effect": True,
            "minimum_strictly_positive_tasks": 3,
            "global_mean_strictly_positive": True,
            "no_llm_calls": True,
            "numerical_tolerance": 1e-12,
        },
        "selection_rule": (
            "four minimum SHA-256 task identities under the frozen "
            "aistats-matsci-acquisition-confirmation-v1 namespace after "
            "excluding every task named by the registered prior freezes"
        ),
        "excluded_tasks": sorted(excluded),
        "exclusion_freezes": exclusion_identities,
        "authorization_identity": acquisition["identity"],
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
            (ROOT / ".cache/aistats_matsci_confirmation_20260929").resolve()
        ),
        "mainline_root": str((ROOT.parent / "hypothesis_mvp").resolve()),
        "task_arrays_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "One prospectively frozen fresh-task MatSci development "
            "confirmation of decision-risk acquisition versus matched random; "
            "no LLM synthesis, test/OOD, held-out, or universal claim."
        ),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_MATSCI_ACQUISITION_CONFIRMATION_FREEZE.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
