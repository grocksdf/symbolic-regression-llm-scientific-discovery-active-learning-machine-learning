"""Run matched-budget realized acquisition over frozen synthesis candidates."""

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
from hypothesis_mvp.discovery.realized_drr import (
    run_realized_drr_trajectory,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from scripts.run_aistats_drr_benchmark import _sha


def _verify(freeze):
    if freeze.get("schema") != (
            "scientific-aistats-realized-drr-pilot-freeze-v1"):
        raise ValueError("invalid realized DRR freeze")
    for path, expected in {
            **freeze["files"], **freeze["candidate_artifacts"]}.items():
        if _sha(path) != expected:
            raise ValueError(f"realized DRR frozen file changed: {path}")
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("realized DRR benchmark source changed")
    if verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp") != freeze["mainline_source"]:
        raise ValueError("realized DRR mainline source changed")


def _dataset(handle, family, task):
    if family == "lsr_transform":
        return handle[f"lsr_transform/{task}/train"]
    return handle[f"lsr_synth/{family}/{task}/train"]


def _roles_with_responses(handle, family, task, seed):
    dataset = _dataset(handle, family, task)
    order = drr_adapter._ordered_indices(len(dataset), task, seed)
    cuts = (
        int(round(.50 * len(dataset))),
        int(round(.70 * len(dataset))),
        int(round(.90 * len(dataset))),
    )
    initial_indices = tuple(order[cuts[1]:cuts[2]])[:32]
    action_indices = tuple(order[cuts[2]:])
    initial = np.asarray(dataset[list(sorted(initial_indices)), :])
    actions = np.asarray(dataset[list(sorted(action_indices)), :])
    # Restore hash order after h5py's increasing-index requirement.
    initial_lookup = {index: row for index, row in zip(
        sorted(initial_indices), initial, strict=True)}
    action_lookup = {index: row for index, row in zip(
        sorted(action_indices), actions, strict=True)}
    initial = np.asarray([initial_lookup[index] for index in initial_indices])
    actions = np.asarray([action_lookup[index] for index in action_indices])
    return initial[:, 1:], initial[:, 0], actions[:, 1:], actions[:, 0]


def _candidate_rows(freeze, family, task, seed, condition, n_features):
    path = (
        Path(freeze["source_output"]) / "runs" / family / task /
        f"seed{seed}" / condition / "pcpi_artifacts" / f"{task}.json")
    report = json.loads(
        path.read_text(encoding="utf-8"))["scientific_discovery_runtime"]
    if report.get("drr_candidate_rows"):
        return tuple(dict(row) for row in report["drr_candidate_rows"])
    protected = [
        candidate
        for row in report.get("scientist_agent_counterfactual_backbones", ())
        for candidate in row.get("backbone_candidates", ())]
    return drr_adapter.candidate_rows(
        report, report["best_expression"], n_features, protected)


def _correlation(rows):
    x = np.asarray([
        query["predicted_lower_bound"]
        for row in rows if row["condition"] == "full_scientist_v6"
        and row["policy"] == "decision_risk"
        for query in row["trajectory"]["queries"]])
    y = np.asarray([
        query["realized_risk_reduction"]
        for row in rows if row["condition"] == "full_scientist_v6"
        and row["policy"] == "decision_risk"
        for query in row["trajectory"]["queries"]])
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and args.continuation is None:
        raise ValueError("realized DRR output must be new")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if args.continuation is None:
        _verify(freeze)
        rows = []
        args.output_dir.mkdir(parents=True)
    else:
        continuation = json.loads(
            args.continuation.read_text(encoding="utf-8"))
        if (continuation.get("schema") !=
                "scientific-aistats-realized-drr-continuation-v1"
                or _sha(args.freeze) != continuation["source_freeze_sha256"]
                or str(args.output_dir.resolve()) !=
                    continuation["source_output"]
                or _sha(args.output_dir / "REALIZED_ROWS.json") !=
                    continuation["partial_rows_sha256"]
                or verify_clean_git_source(ROOT) !=
                    continuation["benchmark_source"]
                or verify_clean_git_source(
                    ROOT.parent / "hypothesis_mvp") !=
                    continuation["mainline_source"]):
            raise ValueError("invalid realized DRR continuation")
        rows = json.loads((args.output_dir / "REALIZED_ROWS.json").read_text(
            encoding="utf-8"))
        completed = {
            (row["family"], row["seed"], row["label"]) for row in rows}
        if (len(rows) != continuation["completed_trajectory_count"]
                or sorted("/".join(map(str, key)) for key in completed)
                != continuation["completed_trajectory_keys"]):
            raise ValueError("realized DRR continuation rows changed")
        for path, expected in freeze["candidate_artifacts"].items():
            if _sha(path) != expected:
                raise ValueError("realized DRR candidate artifact changed")
        if _sha(freeze["hdf5"]["path"]) != freeze["hdf5"]["sha256"]:
            raise ValueError("realized DRR HDF5 changed")
    completed = {
        (row["family"], row["seed"], row["label"]) for row in rows}
    with h5py.File(freeze["hdf5"]["path"], "r") as handle:
        for family, task in freeze["selected_tasks"].items():
            for seed in freeze["seeds"]:
                X0, y0, action_X, action_y = _roles_with_responses(
                    handle, family, task, seed)
                for condition in freeze["conditions"]:
                    candidates = _candidate_rows(
                        freeze, family, task, seed, condition, X0.shape[1])
                    for policy in ("decision_risk", "random"):
                        label = (
                            ("full" if condition == "full_scientist_v6"
                             else "no_llm") + "_" + (
                                "targeted" if policy == "decision_risk"
                                else "random"))
                        if (family, seed, label) in completed:
                            print(
                                f"resume-preserved {label} "
                                f"{family}/{task} seed={seed}", flush=True)
                            continue
                        print(
                            f"realized {label} {family}/{task} seed={seed}",
                            flush=True)
                        identity = sha256(
                            f"realized-drr-v1:{family}:{task}:{seed}:"
                            f"{condition}".encode()).hexdigest()
                        try:
                            trajectory = run_realized_drr_trajectory(
                                candidates, X0, y0, action_X, action_y,
                                condition=condition,
                                exploration_identity=identity,
                                policy=policy, random_seed=seed)
                            failure = ""
                        except Exception as error:
                            trajectory = {}
                            failure = f"{type(error).__name__}:{error}"
                        rows.append({
                            "family": family, "task": task, "seed": seed,
                            "condition": condition, "policy": policy,
                            "label": label,
                            "normalized_realized_risk_aulc": float(
                                trajectory.get(
                                    "normalized_realized_risk_aulc", 0.0)),
                            "symmetric_realized_risk_aulc": float(
                                trajectory.get(
                                    "symmetric_realized_risk_aulc", 0.0)),
                            "absolute_realized_risk_aulc": float(
                                trajectory.get(
                                    "absolute_realized_risk_aulc", 0.0)),
                            "status": "success" if not failure else "failure",
                            "failure": failure, "trajectory": trajectory,
                        })
                        (args.output_dir / "REALIZED_ROWS.json").write_text(
                            json.dumps(rows, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    by = {(row["family"], row["seed"], row["label"]): row for row in rows}
    synthesis, acquisition = {}, {}
    absolute_synthesis, absolute_acquisition = {}, {}
    for family in freeze["selected_tasks"]:
        synthesis[family] = float(np.mean([
            by[(family, seed, "full_targeted")][
                "symmetric_realized_risk_aulc"]
            - by[(family, seed, "no_llm_targeted")][
                "symmetric_realized_risk_aulc"]
            for seed in freeze["seeds"]]))
        acquisition[family] = float(np.mean([
            by[(family, seed, "full_targeted")][
                "symmetric_realized_risk_aulc"]
            - by[(family, seed, "full_random")][
                "symmetric_realized_risk_aulc"]
            for seed in freeze["seeds"]]))
        absolute_synthesis[family] = float(np.mean([
            by[(family, seed, "full_targeted")][
                "absolute_realized_risk_aulc"]
            - by[(family, seed, "no_llm_targeted")][
                "absolute_realized_risk_aulc"]
            for seed in freeze["seeds"]]))
        absolute_acquisition[family] = float(np.mean([
            by[(family, seed, "full_targeted")][
                "absolute_realized_risk_aulc"]
            - by[(family, seed, "full_random")][
                "absolute_realized_risk_aulc"]
            for seed in freeze["seeds"]]))
    tolerance = freeze["screening_decision"]["numerical_tolerance"]
    synthesis_mean = float(np.mean(list(synthesis.values())))
    acquisition_mean = float(np.mean(list(acquisition.values())))
    correlation = _correlation(rows)
    absolute_synthesis_mean = float(np.mean(
        list(absolute_synthesis.values())))
    absolute_acquisition_mean = float(np.mean(
        list(absolute_acquisition.values())))
    decisions = {
        "all_32_trajectories_accounted_for": len(rows) == 32,
        "zero_failures": all(row["status"] == "success" for row in rows),
        "synthesis_mean_strictly_positive":
            synthesis_mean > tolerance,
        "acquisition_mean_strictly_positive":
            acquisition_mean > tolerance,
        "at_least_two_positive_synthesis_families":
            sum(value > tolerance for value in synthesis.values()) >= 2,
        "at_least_two_positive_acquisition_families":
            sum(value > tolerance for value in acquisition.values()) >= 2,
        "predicted_realized_correlation_strictly_positive":
            correlation > 0.0,
        "absolute_synthesis_mean_is_nonnegative":
            absolute_synthesis_mean >= -tolerance,
        "absolute_acquisition_mean_is_nonnegative":
            absolute_acquisition_mean >= -tolerance,
    }
    result = {
        "schema": "scientific-aistats-realized-drr-pilot-result-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "synthesis_family_effects": synthesis,
        "acquisition_family_effects": acquisition,
        "absolute_synthesis_family_effects": absolute_synthesis,
        "absolute_acquisition_family_effects": absolute_acquisition,
        "synthesis_mean_effect": synthesis_mean,
        "acquisition_mean_effect": acquisition_mean,
        "absolute_synthesis_mean_effect": absolute_synthesis_mean,
        "absolute_acquisition_mean_effect": absolute_acquisition_mean,
        "predicted_realized_correlation": correlation,
        "failure_count": sum(row["status"] != "success" for row in rows),
        "rows": rows,
        "candidate_response_accessed": True,
        "action_response_accessed": True,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": all(decisions.values()),
        "claim_boundary": (
            "Matched-budget realized acquisition development evidence only; "
            "not held-out confirmation or universal superiority."),
    }
    path = args.output_dir / "AISTATS_REALIZED_DRR_PILOT_RESULT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
