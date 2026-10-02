"""Run fresh realized PCPI trajectories over frozen three-arm candidate banks."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()
from hypothesis_mvp.discovery.realized_drr import run_realized_drr_trajectory
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from scripts.run_aistats_drr_benchmark import _sha
from scripts.run_aistats_three_arm_predictive_gate import (
    CONDITIONS, ARM_NAMES, _admit, _artifact, _baseline_rows, _llm_rows,
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dataset(handle, family, task):
    group = ("lsr_transform" if family == "lsr_transform"
             else f"lsr_synth/{family}")
    return handle[f"{group}/{task}/train"]


def _opened(dataset, indices):
    ordered = sorted(indices)
    rows = np.asarray(dataset[ordered, :])
    lookup = dict(zip(ordered, rows, strict=True))
    return np.asarray([lookup[index] for index in indices])


def _roles(dataset, task, seed):
    frozen = drr_adapter.split_three_arm_training_samples(
        dataset, task_name=task, seed=seed)
    idx = frozen.role_row_indices
    names = ("gap_admission", "decision_selector_update",
             "inference_initial", "action_covariates")
    rows = {name: _opened(dataset, idx[name]) for name in names}
    return {
        "X_initial": rows["inference_initial"][:, 1:],
        "y_initial": rows["inference_initial"][:, 0],
        "X_actions": rows["action_covariates"][:, 1:],
        "y_actions": rows["action_covariates"][:, 0],
        "X_admission": rows["gap_admission"][:, 1:],
        "y_admission": rows["gap_admission"][:, 0],
        "X_selector": rows["decision_selector_update"][:, 1:],
        "y_selector": rows["decision_selector_update"][:, 0],
    }


def _banks(freeze, family, task, seed, roles):
    source = Path(freeze["source_output"])
    reports = {}
    for condition in CONDITIONS:
        reports[condition] = _artifact(
            source / "runs" / family / task / f"seed{seed}" / condition,
            task)
    baseline = _baseline_rows(reports["three_arm_e_v1"])
    n_features = roles["X_initial"].shape[1]
    banks = {"E": baseline}
    for condition in CONDITIONS[1:]:
        optional, _, _ = _llm_rows(
            reports[condition], baseline, n_features,
            freeze["llm_materialization_limit"])
        identity = sha256(
            f"three-arm-admission-v1:{family}:{task}:{seed}:"
            f"{condition}".encode()).hexdigest()
        banks[ARM_NAMES[condition]], _ = _admit(
            baseline, optional, roles, identity=identity,
            n_features=n_features)
    return banks


def _aggregate(rows, tolerance):
    by = {(r["task"], r["seed"], r["arm"]): r for r in rows}
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    effects = {}
    for task in tasks:
        task_seeds = [seed for seed in seeds
                      if (task, seed, "E+L_gap") in by]
        effects[task] = {
            "gap_minus_blind": float(np.mean([
                by[(task, s, "E+L_gap")]["aulc"]
                - by[(task, s, "E+L_blind")]["aulc"]
                for s in task_seeds])),
            "gap_minus_engine": float(np.mean([
                by[(task, s, "E+L_gap")]["aulc"]
                - by[(task, s, "E")]["aulc"]
                for s in task_seeds])),
        }
    gap_blind = float(np.mean(
        [value["gap_minus_blind"] for value in effects.values()]))
    gap_engine = float(np.mean(
        [value["gap_minus_engine"] for value in effects.values()]))
    predicted = np.asarray([
        q["predicted_lower_bound"] for r in rows if r["arm"] == "E+L_gap"
        for q in r["trajectory"].get("queries", ())])
    realized = np.asarray([
        q["realized_risk_reduction"] for r in rows if r["arm"] == "E+L_gap"
        for q in r["trajectory"].get("queries", ())])
    correlation = (0.0 if len(predicted) < 3 or np.std(predicted) == 0
                   or np.std(realized) == 0
                   else float(np.corrcoef(predicted, realized)[0, 1]))
    decisions = {
        "all_24_trajectories_accounted_for": len(rows) == 24,
        "zero_failures": all(r["status"] == "success" for r in rows),
        "gap_minus_blind_strictly_positive": gap_blind > tolerance,
        "gap_minus_engine_strictly_positive": gap_engine > tolerance,
        "minimum_two_positive_gap_minus_blind_tasks": sum(
            v["gap_minus_blind"] > tolerance for v in effects.values()) >= 2,
        "predicted_realized_correlation_strictly_positive": correlation > 0,
    }
    return decisions, effects, gap_blind, gap_engine, correlation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    freeze = _read(args.freeze)
    if (freeze.get("schema") !=
            "scientific-aistats-three-arm-realized-freeze-v1"
            or freeze.get("execution_authorized") is not True):
        raise ValueError("invalid three-arm realized freeze")
    for path, digest in {**freeze["files"],
                         **freeze["candidate_artifacts"]}.items():
        if _sha(path) != digest:
            raise ValueError(f"frozen realized input changed: {path}")
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("benchmark source changed")
    if verify_clean_git_source(
            Path(freeze["mainline_root"])) != freeze["mainline_source"]:
        raise ValueError("mainline source changed")
    if args.output_dir.exists() and not args.resume:
        raise ValueError("three-arm realized output must be new")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    rows_path = args.output_dir / "REALIZED_ROWS.json"
    rows = _read(rows_path) if rows_path.is_file() else []
    completed = {(r["family"], r["task"], r["seed"], r["arm"])
                 for r in rows}
    with h5py.File(freeze["hdf5"]["path"], "r") as handle:
        for family, task in freeze["selected_tasks"].items():
            for seed in freeze["seeds"]:
                roles = _roles(_dataset(handle, family, task), task, seed)
                banks = _banks(freeze, family, task, seed, roles)
                for arm, candidates in banks.items():
                    key = (family, task, seed, arm)
                    if key in completed:
                        continue
                    print(f"realized {arm} {family}/{task} seed={seed}",
                          flush=True)
                    identity = sha256(
                        f"adaptive-realized-v1:{family}:{task}:{seed}:"
                        f"{arm}".encode()).hexdigest()
                    try:
                        trajectory = run_realized_drr_trajectory(
                            candidates, roles["X_initial"],
                            roles["y_initial"], roles["X_actions"],
                            roles["y_actions"], condition=arm,
                            exploration_identity=identity,
                            policy="decision_risk", random_seed=seed)
                        failure = ""
                    except Exception as error:
                        trajectory = {}
                        failure = f"{type(error).__name__}:{error}"
                    rows.append({
                        "family": family, "task": task, "seed": seed,
                        "arm": arm,
                        "aulc": float(trajectory.get(
                            "symmetric_realized_risk_aulc", 0.0)),
                        "status": "failure" if failure else "success",
                        "failure": failure, "trajectory": trajectory})
                    rows_path.write_text(json.dumps(
                        rows, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    values = _aggregate(rows, freeze["numerical_tolerance"])
    decisions, effects, gap_blind, gap_engine, correlation = values
    result = {
        "schema": "scientific-aistats-three-arm-realized-result-v1",
        "passed": all(decisions.values()), "decisions": decisions,
        "task_effects": effects,
        "global_gap_minus_blind_realized_aulc": gap_blind,
        "global_gap_minus_engine_realized_aulc": gap_engine,
        "predicted_realized_correlation": correlation,
        "row_count": len(rows), "rows": rows,
        "adaptive_resource_use_is_diagnostic_not_gate": True,
        "candidate_response_accessed": True,
        "action_response_accessed": True,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "claim_boundary": freeze["claim_boundary"]}
    (args.output_dir / "AISTATS_THREE_ARM_REALIZED_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
