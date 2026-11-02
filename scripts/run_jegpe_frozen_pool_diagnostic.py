"""Run the fixed JEGPE frozen-candidate-pool development comparison."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.candidate_region_expansion import (
    FrozenAxisRegions,
)
from hypothesis_mvp.discovery.joint_evidence_gated_expansion import (
    CandidateArchive,
    admit_joint_bundle,
    extract_subexpression_materials,
    select_fixed_budget_bundle,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY,
    freeze_discovery_target,
    freeze_expanded_formula_model,
)
from hypothesis_mvp.discovery.regional_candidate_admission import (
    admit_global_protected_candidates,
)
from hypothesis_mvp.discovery.staged_candidate_admission import (
    admit_global_protected_with_quarantine,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from run_gap_candidate_marginal_value import (
    DATASET_GROUPS,
    _generation_roles,
    _gap_admission_role,
    _opened,
    _posterior_update_role,
    _reporting_response,
    identity,
)
from run_protected_admission_ablation import _same_data_admission


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError("JEGPE output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def role(hdf5, family, task, indices, name):
    with h5py.File(hdf5, "r") as handle:
        dataset = handle[f"{DATASET_GROUPS[family]}/{task}/train"]
        X, y = _opened(dataset, indices[name], response=True)
    return RoleDataset(DataRole.VALIDATION, X, y)


def density(rows, selection, X, y, identity_value, update=None):
    model = freeze_expanded_formula_model(
        rows, n_features=selection.development.X.shape[1],
        prior=NormalInverseGammaPrior(),
        exploration_identity=identity_value,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    fit = selection.development
    if update is not None:
        fit = RoleDataset(
            DataRole.DEVELOPMENT,
            np.vstack((fit.X, update.X)),
            np.concatenate((fit.y, update.y)))
    target = freeze_discovery_target(
        model, fit, selection.acquisition_pool.X,
        measurement_budget=2,
        expected_model_identity=model.stable_hash)
    values = np.asarray(
        model.engine(model.stable_hash).predictive_logpdf(
            target.initial_posterior, X, y), dtype=float)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("JEGPE predictive density is nonfinite")
    return values


def stacking_weights(lpd):
    matrix = np.asarray(lpd, dtype=float)
    count = matrix.shape[1]
    if count == 1:
        return np.ones(1)
    def objective(weights):
        if np.any(weights <= 0):
            return 1e100
        return -float(np.sum(logsumexp(
            matrix + np.log(weights)[None, :], axis=1)))
    result = minimize(
        objective, np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=[(1e-12, 1.0)] * count,
        constraints=[{
            "type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
        options={"ftol": 1e-10, "maxiter": 1000})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("predictive stacking optimizer failed")
    weights = np.maximum(result.x, 0)
    return weights / np.sum(weights)


def bootstrap(task_values, seed, count):
    values = np.asarray(list(task_values.values()), dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(count)
    for index in range(count):
        samples[index] = float(np.mean(
            values[rng.integers(0, len(values), len(values))]))
    return [
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("JEGPE output root exists")
    freeze = read(args.freeze)
    if (freeze.get("schema")
            != "jegpe-frozen-pool-development-freeze-v1"
            or freeze.get("execution_authorized") is not True
            or freeze.get("provider_calls_authorized") is not False):
        raise ValueError("invalid JEGPE Phase-1 freeze")
    if digest(freeze["hdf5"]) != freeze["hdf5_sha256"]:
        raise ValueError("JEGPE HDF5 identity mismatch")
    for path, expected in freeze["file_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"JEGPE code identity mismatch: {path}")
    for row in freeze["artifact_manifest"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError(
                f"JEGPE source artifact mismatch: {row['path']}")
    args.output_root.mkdir(parents=True)
    write_new(args.output_root / "RUN_FREEZE.json", freeze)
    source = Path(freeze["source_output"])
    hdf5 = Path(freeze["hdf5"])
    arms = tuple(freeze["arms"])
    rows, failures = [], []
    for index, coordinate in enumerate(freeze["coordinates"], start=1):
        family = coordinate["family"]
        task = coordinate["task"]
        seed = int(coordinate["seed"])
        print(f"[{index}/{freeze['coordinate_count']}] "
              f"JEGPE {task} seed={seed}", flush=True)
        if coordinate["coordinate_failure"]:
            failures.append({
                "family": family, "task": task, "seed": seed,
                "reason": "historical-coordinate-failure"})
            rows.append({
                "family": family, "task": task, "seed": seed,
                "coordinate_failure": True,
                "gains": {arm: 0.0 for arm in arms}})
            continue
        try:
            root = Path(coordinate["coordinate_root"])
            shared = read(root / "SHARED_BLOCK.json")
            E0 = [dict(row) for row in shared["shared"]["E0_rows"]]
            candidate_rows = []
            for condition in freeze["candidate_conditions"]:
                generation = read(
                    root / condition / "GENERATION_RESULT.json")
                candidate_rows.extend(generation["optional_rows"])
            roles = _generation_roles(hdf5, family, task, seed)
            selection = roles["selection"]
            calibration = role(
                hdf5, family, task, roles["indices"],
                "decision_calibration")
            admission = _gap_admission_role(
                hdf5, family, task, roles["indices"])
            update = _posterior_update_role(
                hdf5, family, task, roles["indices"])
            y_report = _reporting_response(
                hdf5, family, task, roles["indices"])
            regions = FrozenAxisRegions(
                selection.development.X.shape[1], 0,
                (float(np.median(
                    selection.acquisition_pool.X[:, 0])),))
            archive = CandidateArchive.freeze(
                candidate_rows,
                n_features=selection.development.X.shape[1],
                coefficient_policy=EXPANDED_FORMULA_POLICY)
            candidates = [
                candidate.posterior_row()
                for candidate in archive.candidates]
            base_cal = float(np.mean(density(
                E0, selection, calibration.X, calibration.y,
                identity({"task": task, "bank": "E0-calibration"}))))
            def bundle_calibration_gain(bundle):
                bank = [
                    *E0,
                    *[candidate.posterior_row()
                      for candidate in bundle]]
                return float(np.mean(density(
                    bank, selection, calibration.X, calibration.y,
                    identity({
                        "task": task,
                        "bundle": [row.structural_hash
                                   for row in bundle]}))) - base_cal)
            selected_joint, joint_selection = (
                select_fixed_budget_bundle(
                    archive, bundle_calibration_gain,
                    maximum_size=int(
                        freeze["bundle_selection"]["maximum_size"]),
                    evaluation_budget=int(
                        freeze["bundle_selection"]["evaluation_budget"]))
                if archive.candidates else (
                    (), {"selected_size": 0, "records": []}))
            selected_single, singleton_selection = (
                select_fixed_budget_bundle(
                    archive, bundle_calibration_gain,
                    maximum_size=1,
                    evaluation_budget=min(
                        len(archive.candidates),
                        int(freeze[
                            "bundle_selection"]["evaluation_budget"])))
                if archive.candidates else (
                    (), {"selected_size": 0, "records": []}))
            all_regions = tuple(range(len(regions.cuts) + 1))
            original, original_audit = (
                admit_global_protected_candidates(
                    E0, candidates, selection.development,
                    roles["gap_audit"], admission,
                    selection.acquisition_pool.X,
                    regions, all_regions,
                    NormalInverseGammaPrior(), 2, alpha=.025,
                    coefficient_policy=EXPANDED_FORMULA_POLICY)
                if candidates else ([], {"attempts": 0}))
            staged, staged_audit = (
                admit_global_protected_with_quarantine(
                    E0, candidates, selection.development,
                    roles["gap_audit"], admission, calibration,
                    selection.acquisition_pool.X,
                    regions, all_regions,
                    NormalInverseGammaPrior(), 2, alpha=.025,
                    coefficient_policy=EXPANDED_FORMULA_POLICY)
                if candidates else ([], {"attempts": 0}))
            joint, joint_audit = (
                admit_joint_bundle(
                    E0, selected_joint, selection.development,
                    roles["gap_audit"], admission,
                    selection.acquisition_pool.X,
                    regions, all_regions,
                    NormalInverseGammaPrior(), 2, alpha=.025,
                    coefficient_policy=EXPANDED_FORMULA_POLICY,
                    tested_bundle_count=1)
                if selected_joint else (
                    [], {"passed": False,
                         "reason": "training-selection-empty"}))
            same_data, same_data_audit = _same_data_admission(
                E0, candidates, selection.development,
                roles["gap_audit"], selection.acquisition_pool.X,
                regions, all_regions)
            banks = {
                "original_protected": [*E0, *original],
                "staged_protected": [*E0, *staged],
                "jegpe_joint": [*E0, *joint],
                "accept_all": [*E0, *candidates],
                "same_data": [*E0, *same_data],
            }
            base_report = density(
                E0, selection, roles["reporting_X"], y_report,
                identity({"task": task, "bank": "E0-report"}),
                update=update)
            base_score = float(np.mean(base_report))
            gains = {}
            for name, bank in banks.items():
                gains[name] = float(np.mean(density(
                    bank, selection, roles["reporting_X"], y_report,
                    identity({"task": task, "bank": name}),
                    update=update)) - base_score)
            component_banks = [E0] + [
                [*E0, candidate.posterior_row()]
                for candidate in archive.candidates]
            cal_lpd = np.column_stack([
                density(
                    bank, selection, calibration.X, calibration.y,
                    identity({"task": task, "stack-cal": idx}))
                for idx, bank in enumerate(component_banks)])
            weights = stacking_weights(cal_lpd)
            report_lpd = np.column_stack([
                density(
                    bank, selection, roles["reporting_X"], y_report,
                    identity({"task": task, "stack-report": idx}),
                    update=update)
                for idx, bank in enumerate(component_banks)])
            stack_score = float(np.mean(logsumexp(
                report_lpd + np.log(weights)[None, :], axis=1)))
            gains["predictive_stacking"] = stack_score - base_score
            def reporting_gain(bundle):
                if not bundle:
                    return 0.0
                bank = [
                    *E0,
                    *[candidate.posterior_row()
                      for candidate in bundle]]
                return float(np.mean(density(
                    bank, selection, roles["reporting_X"], y_report,
                    identity({
                        "task": task, "diagnostic-bundle":
                        [row.structural_hash for row in bundle]}),
                    update=update)) - base_score)
            selected_joint_reporting = reporting_gain(selected_joint)
            selected_single_reporting = reporting_gain(selected_single)
            records = joint_selection.get("records", [])
            best_pair_cal = max(
                (row["score"] for row in records if row["size"] == 2),
                default=0.0)
            best_single_cal = max(
                (row["score"] for row in records if row["size"] == 1),
                default=0.0)
            rows.append({
                "family": family, "task": task, "seed": seed,
                "coordinate_failure": False,
                "candidate_count": len(archive.candidates),
                "archive_identity": archive.identity,
                "subexpression_material_count": len(
                    extract_subexpression_materials(archive)),
                "selected_joint_size": len(selected_joint),
                "selected_singleton_size": len(selected_single),
                "calibration_best_pair_minus_singleton":
                    best_pair_cal - best_single_cal,
                "selected_joint_reporting_gain":
                    selected_joint_reporting,
                "selected_singleton_reporting_gain":
                    selected_single_reporting,
                "selected_joint_minus_singleton_reporting":
                    selected_joint_reporting
                    - selected_single_reporting,
                "gains": gains,
                "stacking_weights": weights.tolist(),
                "joint_selection": joint_selection,
                "singleton_selection": singleton_selection,
                "admission_audits": {
                    "original_protected": original_audit,
                    "staged_protected": staged_audit,
                    "jegpe_joint": joint_audit,
                    "same_data": same_data_audit},
            })
        except Exception as error:
            failures.append({
                "family": family, "task": task, "seed": seed,
                "reason": f"{type(error).__name__}:{error}"})
            rows.append({
                "family": family, "task": task, "seed": seed,
                "coordinate_failure": True,
                "gains": {arm: 0.0 for arm in arms}})
    task_arm = {}
    for row in rows:
        task_arm.setdefault(row["task"], {})
        for arm in arms:
            task_arm[row["task"]].setdefault(arm, []).append(
                float(row["gains"][arm]))
    task_means = {
        task: {
            arm: float(np.mean(values))
            for arm, values in by_arm.items()}
        for task, by_arm in task_arm.items()}
    arm_summaries = {}
    for arm in arms:
        values = {
            task: by_arm[arm] for task, by_arm in task_means.items()}
        arm_summaries[arm] = {
            "task_mean_gain": float(np.mean(list(values.values()))),
            "ci95": bootstrap(
                values, int(freeze["analysis"]["bootstrap_seed"]),
                int(freeze["analysis"]["resamples"])),
            "positive_task_count": sum(
                value > 0 for value in values.values()),
            "negative_task_count": sum(
                value < 0 for value in values.values())}
    contrasts = {}
    for right in ("original_protected", "predictive_stacking"):
        values = {
            task: by_arm["jegpe_joint"] - by_arm[right]
            for task, by_arm in task_means.items()}
        contrasts[f"jegpe_joint_minus_{right}"] = {
            "task_mean": float(np.mean(list(values.values()))),
            "ci95": bootstrap(
                values, int(freeze["analysis"]["bootstrap_seed"]) + 1,
                int(freeze["analysis"]["resamples"]))}
    successful = [row for row in rows if not row["coordinate_failure"]]
    result = {
        "schema": "jegpe-frozen-pool-development-result-v1",
        "protocol_complete": True,
        "provider_calls": 0,
        "coordinate_count": len(rows),
        "coordinate_failure_count": len(failures),
        "failures": failures,
        "arm_summaries": arm_summaries,
        "contrasts": contrasts,
        "mechanism": {
            "coordinates_with_at_least_two_candidates": sum(
                row["candidate_count"] >= 2 for row in successful),
            "training_pair_advantage_count": sum(
                row["calibration_best_pair_minus_singleton"] > 0
                for row in successful),
            "selected_pair_count": sum(
                row["selected_joint_size"] == 2
                for row in successful),
            "selected_pair_reporting_advantage_count": sum(
                row["selected_joint_minus_singleton_reporting"] > 0
                for row in successful),
            "subexpression_material_count": sum(
                row["subexpression_material_count"]
                for row in successful),
            "joint_admitted_coordinate_count": sum(
                row["admission_audits"]["jegpe_joint"].get(
                    "passed") is True for row in successful),
        },
        "task_arm_means": task_means,
        "rows": rows,
        "post_reporting_development_only": True,
        "independent_confirmation_claim_authorized": False,
        "anytime_valid_claimed": False,
        "novelty_status": "NEW_METHOD_CANDIDATE_UNVERIFIED",
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "claim_boundary": freeze["claim_boundary"],
    }
    write_new(args.output_root / "JEGPE_FROZEN_POOL_ABLATION.json", result)
    print(json.dumps({
        "arm_summaries": arm_summaries,
        "contrasts": contrasts,
        "mechanism": result["mechanism"],
        "coordinate_failure_count": len(failures),
        "protocol_complete": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
