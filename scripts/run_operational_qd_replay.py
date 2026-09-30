"""Run response-free BOQD versus legacy candidate-pool replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
from scripts.run_aistats_drr_response_free_replay import _take
from scripts.run_aistats_drr_benchmark import _sha

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


def _verify(freeze):
    if freeze.get("schema") != "scientific-operational-qd-replay-freeze-v1":
        raise ValueError("invalid BOQD replay freeze")
    for path, expected in {
            **freeze["files"], **freeze["candidate_artifacts"]}.items():
        if _sha(path) != expected:
            raise ValueError(f"BOQD replay frozen file changed: {path}")
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("BOQD replay benchmark source changed")
    if verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp") != freeze["mainline_source"]:
        raise ValueError("BOQD replay mainline source changed")


def _dataset(handle, family, task):
    if family == "lsr_transform":
        return handle[f"lsr_transform/{task}/train"]
    return handle[f"lsr_synth/{family}/{task}/train"]


def _roles(handle, family, task, seed):
    dataset = _dataset(handle, family, task)
    order = drr_adapter._ordered_indices(len(dataset), task, seed)
    cuts = (
        int(round(.50 * len(dataset))),
        int(round(.70 * len(dataset))),
        int(round(.90 * len(dataset))),
    )
    indices = {
        "discovery_development": tuple(order[:cuts[0]]),
        "discovery_validation": tuple(order[cuts[0]:cuts[1]]),
        "inference_initial": tuple(order[cuts[1]:cuts[2]]),
        "action_covariates": tuple(order[cuts[2]:]),
    }
    def opened(role):
        rows = _take(dataset, indices[role], slice(None))
        return rows[:, 1:], rows[:, 0]
    development, validation, initial = (
        opened("discovery_development"),
        opened("discovery_validation"),
        opened("inference_initial"))
    actions = _take(
        dataset, indices["action_covariates"], slice(1, None))
    return drr_adapter.DRRSelectionRoles(
        development[0], development[1],
        validation[0], validation[1],
        initial[0], initial[1], actions, indices)


def _artifact(freeze, family, task, seed, condition):
    path = (
        Path(freeze["source_output"]) / "runs" / family / task /
        f"seed{seed}" / condition / "pcpi_artifacts" / f"{task}.json")
    return json.loads(path.read_text(
        encoding="utf-8"))["scientific_discovery_runtime"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("BOQD replay output must be new")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    _verify(freeze)
    args.output_dir.mkdir(parents=True)
    rows = []
    with h5py.File(freeze["hdf5"]["path"], "r") as handle:
        for family, task in freeze["selected_tasks"].items():
            for seed in freeze["seeds"]:
                roles = _roles(handle, family, task, seed)
                for condition in freeze["conditions"]:
                    report = _artifact(
                        freeze, family, task, seed, condition)
                    protected = [
                        candidate
                        for row in report.get(
                            "scientist_agent_counterfactual_backbones", ())
                        for candidate in row.get("backbone_candidates", ())]
                    legacy = drr_adapter.candidate_rows(
                        report, report["best_expression"],
                        roles.X_initial.shape[1], protected)
                    try:
                        boqd, audit = (
                            drr_adapter.candidate_rows_operational_qd(
                                report, report["best_expression"], roles,
                                task_name=task, seed=seed,
                                protected_expressions=protected))
                        legacy_curve = (
                            drr_adapter.evaluate_drr_prefix_candidates(
                                legacy, roles, condition=condition,
                                task_name=task, seed=seed))
                        boqd_curve = (
                            drr_adapter.evaluate_drr_prefix_candidates(
                                boqd, roles, condition=condition,
                                task_name=task, seed=seed))
                        failure = ""
                    except Exception as error:
                        audit, legacy_curve, boqd_curve = {}, None, None
                        failure = f"{type(error).__name__}:{error}"
                    rows.append({
                        "family": family, "task": task, "seed": seed,
                        "condition": condition,
                        "legacy_aulc": (
                            0.0 if legacy_curve is None
                            else legacy_curve.normalized_aulc),
                        "boqd_aulc": (
                            0.0 if boqd_curve is None
                            else boqd_curve.normalized_aulc),
                        "status": "success" if not failure else "failure",
                        "failure": failure, "boqd_audit": audit,
                    })
                    print(
                        f"BOQD replay {condition} "
                        f"{family}/{task} seed={seed}", flush=True)
                    (args.output_dir / "BOQD_ROWS.json").write_text(
                        json.dumps(rows, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    tolerance = freeze["screening_decision"]["numerical_tolerance"]
    family_effects, llm_families = {}, set()
    for family in freeze["selected_tasks"]:
        subset = [row for row in rows if row["family"] == family]
        family_effects[family] = float(np.mean([
            row["boqd_aulc"] - row["legacy_aulc"] for row in subset]))
        if any(
            elite.get("source_family") == "llm"
            for row in subset
            for elite in row.get("boqd_audit", {}).get("elites", ())):
            llm_families.add(family)
    mean = float(np.mean(list(family_effects.values())))
    decisions = {
        "all_16_rows_accounted_for": len(rows) == 16,
        "zero_failures": all(row["status"] == "success" for row in rows),
        "mean_boost_strictly_positive": mean > tolerance,
        "at_least_two_positive_families":
            sum(value > tolerance
                for value in family_effects.values()) >= 2,
        "no_negative_transfer_family":
            all(value >= -tolerance
                for value in family_effects.values()),
        "at_least_two_llm_novel_niche_families":
            len(llm_families) >= 2,
    }
    result = {
        "schema": "scientific-operational-qd-replay-result-v1",
        "passed": all(decisions.values()), "decisions": decisions,
        "family_effects": family_effects, "mean_effect": mean,
        "llm_novel_niche_families": sorted(llm_families),
        "failure_count": sum(row["status"] != "success" for row in rows),
        "rows": rows,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Response-free BOQD candidate-pool development replay; not "
            "realized efficacy, superiority, or confirmation."),
    }
    path = args.output_dir / "OPERATIONAL_QD_REPLAY_RESULT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
