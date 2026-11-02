"""Compare original and staged Protected Admission without provider calls."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.candidate_region_expansion import (
    FrozenAxisRegions,
)
from hypothesis_mvp.discovery.staged_candidate_admission import (
    admit_global_protected_with_quarantine,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from run_gap_candidate_marginal_value import (
    DATASET_GROUPS,
    EXPANDED_FORMULA_POLICY,
    _generation_roles,
    _gap_admission_role,
    _opened,
    _posterior_update_role,
    _reporting_response,
    _score,
    identity,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError("staged diagnostic output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def appeal_role(hdf5, family, task, indices):
    with h5py.File(hdf5, "r") as handle:
        dataset = handle[f"{DATASET_GROUPS[family]}/{task}/train"]
        X, y = _opened(
            dataset, indices["decision_calibration"], response=True)
    return RoleDataset(DataRole.VALIDATION, X, y)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("staged diagnostic output root exists")
    freeze = read(args.freeze)
    if (freeze.get("schema")
            != "staged-admission-development-diagnostic-freeze-v1"
            or freeze.get("execution_authorized") is not True
            or freeze.get("provider_calls_authorized") is not False):
        raise ValueError("invalid staged diagnostic freeze")
    if digest(freeze["hdf5"]) != freeze["hdf5_sha256"]:
        raise ValueError("diagnostic HDF5 identity mismatch")
    for path, expected in freeze["file_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"diagnostic code identity mismatch: {path}")
    for row in freeze["artifact_manifest"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError(
                f"diagnostic source artifact mismatch: {row['path']}")
    args.output_root.mkdir(parents=True)
    write_new(args.output_root / "RUN_FREEZE.json", freeze)
    source = Path(freeze["source_output"])
    hdf5 = Path(freeze["hdf5"])
    rows = []
    for coordinate in freeze["coordinates"]:
        family = coordinate["family"]
        task = coordinate["task"]
        seed = int(coordinate["seed"])
        root = source / "generation" / family / task / f"seed{seed}"
        shared = read(root / "SHARED_BLOCK.json")
        generation = read(root / "B0" / "GENERATION_RESULT.json")
        E0 = [dict(row) for row in shared["shared"]["E0_rows"]]
        optional = [dict(row) for row in generation["optional_rows"]]
        original = [dict(row) for row in generation["admitted_rows"]]
        roles = _generation_roles(hdf5, family, task, seed)
        selection = roles["selection"]
        admission = _gap_admission_role(
            hdf5, family, task, roles["indices"])
        appeal = appeal_role(
            hdf5, family, task, roles["indices"])
        update = _posterior_update_role(
            hdf5, family, task, roles["indices"])
        y_report = _reporting_response(
            hdf5, family, task, roles["indices"])
        regions = FrozenAxisRegions(
            selection.development.X.shape[1], 0,
            (float(np.median(
                selection.acquisition_pool.X[:, 0])),))
        staged, staged_audit = admit_global_protected_with_quarantine(
            E0, optional, selection.development, roles["gap_audit"],
            admission, appeal, selection.acquisition_pool.X,
            regions, (0, 1), NormalInverseGammaPrior(), 2,
            alpha=.025, coefficient_policy=EXPANDED_FORMULA_POLICY)
        baseline, _ = _score(
            E0, selection, roles["reporting_X"], y_report,
            identity({"task": task, "bank": "E0-staged-diagnostic"}),
            posterior_update=update)
        arm_scores = {}
        for name, admitted in (
                ("original_protected", original),
                ("staged_protected", staged)):
            score = baseline
            if admitted:
                score, _ = _score(
                    [*E0, *admitted], selection,
                    roles["reporting_X"], y_report,
                    identity({"task": task, "bank": name}),
                    posterior_update=update)
            arm_scores[name] = score - baseline
        rows.append({
            "task": task, "seed": seed,
            "materialized_candidate_count": len(optional),
            "original_admitted_count": len(original),
            "staged_admitted_count": len(staged),
            "original_gain": arm_scores["original_protected"],
            "staged_gain": arm_scores["staged_protected"],
            "staged_minus_original":
                arm_scores["staged_protected"]
                - arm_scores["original_protected"],
            "staged_admission_audit": staged_audit})
    result = {
        "schema": "staged-admission-development-diagnostic-result-v1",
        "protocol_complete": True,
        "provider_calls": 0,
        "task_count": len(rows),
        "original_admitted_count":
            sum(row["original_admitted_count"] for row in rows),
        "staged_admitted_count":
            sum(row["staged_admitted_count"] for row in rows),
        "staged_direct_count": sum(
            row["staged_admission_audit"]["direct_admitted_count"]
            for row in rows),
        "quarantine_count": sum(
            row["staged_admission_audit"]["quarantine_count"]
            for row in rows),
        "appeal_admitted_count": sum(
            row["staged_admission_audit"]["appeal_admitted_count"]
            for row in rows),
        "original_positive_count": sum(
            row["original_gain"] > 0 for row in rows),
        "original_negative_count": sum(
            row["original_gain"] < 0 for row in rows),
        "staged_positive_count": sum(
            row["staged_gain"] > 0 for row in rows),
        "staged_negative_count": sum(
            row["staged_gain"] < 0 for row in rows),
        "original_task_mean_gain": float(np.mean(
            [row["original_gain"] for row in rows])),
        "staged_task_mean_gain": float(np.mean(
            [row["staged_gain"] for row in rows])),
        "staged_minus_original_task_mean": float(np.mean(
            [row["staged_minus_original"] for row in rows])),
        "rows": rows,
        "post_confirmation_development_only": True,
        "independent_confirmation_claim_authorized": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "claim_boundary": freeze["claim_boundary"],
    }
    write_new(args.output_root / "RESULT.json", result)
    print(json.dumps({
        key: result[key] for key in (
            "original_admitted_count", "staged_admitted_count",
            "staged_direct_count", "quarantine_count",
            "appeal_admitted_count", "original_positive_count",
            "original_negative_count", "staged_positive_count",
            "staged_negative_count", "original_task_mean_gain",
            "staged_task_mean_gain",
            "staged_minus_original_task_mean",
            "protocol_complete")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
