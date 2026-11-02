"""Attribute predictive-stacking gains on frozen candidate pools."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.candidate_region_expansion import (
    FrozenAxisRegions,
)
from hypothesis_mvp.discovery.joint_evidence_gated_expansion import (
    CandidateArchive,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY,
    freeze_discovery_target,
    freeze_expanded_formula_model,
)
from hypothesis_mvp.discovery.regional_candidate_admission import (
    admit_global_protected_candidates,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from run_gap_candidate_marginal_value import (
    _generation_roles,
    _gap_admission_role,
    _posterior_update_role,
    _reporting_response,
    identity,
)
from run_jegpe_frozen_pool_diagnostic import role
from stacking_diagnostics import (
    fit_stacking_weights,
    mixture_log_density,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError("stacking attribution output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def density(rows, selection, X, y, identity_value, update=None,
            minimum_supports=2):
    model = freeze_expanded_formula_model(
        rows, n_features=selection.development.X.shape[1],
        prior=NormalInverseGammaPrior(),
        exploration_identity=identity_value,
        coefficient_policy=EXPANDED_FORMULA_POLICY,
        minimum_supports=minimum_supports)
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
        raise FloatingPointError("stacking attribution density is nonfinite")
    return values


def stacking_score(component_banks, selection, calibration, report_X,
                   report_y, namespace, *, calibration_update,
                   reporting_update, seed, minimum_supports=None):
    cal_columns, report_columns = [], []
    for index, bank in enumerate(component_banks):
        floor = (
            minimum_supports[index] if minimum_supports is not None else 2)
        cal_columns.append(density(
            bank, selection, calibration.X, calibration.y,
            identity({"namespace": namespace, "component": index,
                      "phase": "calibration"}),
            update=calibration_update, minimum_supports=floor))
        report_columns.append(density(
            bank, selection, report_X, report_y,
            identity({"namespace": namespace, "component": index,
                      "phase": "reporting"}),
            update=reporting_update, minimum_supports=floor))
    cal_matrix = np.column_stack(cal_columns)
    report_matrix = np.column_stack(report_columns)
    weights, audit = fit_stacking_weights(
        cal_matrix, seed=seed, random_starts=4)
    score = float(np.mean(mixture_log_density(report_matrix, weights)))
    return score, weights, audit


def bootstrap(values, seed, count):
    data = np.asarray(list(values.values()), dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(count)
    for index in range(count):
        samples[index] = float(np.mean(
            data[rng.integers(0, len(data), len(data))]))
    return [
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5))]


def classify(value, tolerance):
    if value > tolerance:
        return "practical_positive"
    if value < -tolerance:
        return "practical_negative"
    if value > 0:
        return "near_zero_positive"
    if value < 0:
        return "near_zero_negative"
    return "exact_zero"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("stacking attribution output root exists")
    freeze = read(args.freeze)
    if (freeze.get("schema") != "stacking-gain-attribution-freeze-v1"
            or freeze.get("execution_authorized") is not True
            or freeze.get("provider_calls_authorized") is not False):
        raise ValueError("invalid stacking attribution freeze")
    if digest(freeze["hdf5"]) != freeze["hdf5_sha256"]:
        raise ValueError("stacking attribution HDF5 mismatch")
    for path, expected in freeze["file_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"stacking attribution code mismatch: {path}")
    for row in freeze["artifact_manifest"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError(
                f"stacking attribution artifact mismatch: {row['path']}")
    args.output_root.mkdir(parents=True)
    write_new(args.output_root / "RUN_FREEZE.json", freeze)
    source = Path(freeze["source_output"])
    hdf5 = Path(freeze["hdf5"])
    rows, failures = [], []
    for index, coordinate in enumerate(freeze["coordinates"], start=1):
        family = coordinate["family"]
        task = coordinate["task"]
        seed = int(coordinate["seed"])
        print(f"[{index}/{freeze['coordinate_count']}] "
              f"stacking attribution {task} seed={seed}", flush=True)
        if coordinate["coordinate_failure"]:
            failures.append({
                "task": task, "seed": seed,
                "reason": "historical-coordinate-failure"})
            rows.append({
                "task": task, "seed": seed,
                "coordinate_failure": True,
                "gains": {
                    arm: 0.0 for arm in freeze["arms"]}})
            continue
        try:
            root = Path(coordinate["coordinate_root"])
            shared = read(root / "SHARED_BLOCK.json")
            E0 = [dict(row) for row in shared["shared"]["E0_rows"]]
            candidates = []
            for condition in freeze["candidate_conditions"]:
                generation = read(
                    root / condition / "GENERATION_RESULT.json")
                candidates.extend(generation["optional_rows"])
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
                candidates,
                n_features=selection.development.X.shape[1],
                coefficient_policy=EXPANDED_FORMULA_POLICY)
            candidate_rows = [
                candidate.posterior_row()
                for candidate in archive.candidates]
            admitted, admission_audit = (
                admit_global_protected_candidates(
                    E0, candidate_rows, selection.development,
                    roles["gap_audit"], admission,
                    selection.acquisition_pool.X,
                    regions, (0, 1),
                    NormalInverseGammaPrior(), 2, alpha=.025,
                    coefficient_policy=EXPANDED_FORMULA_POLICY)
                if candidate_rows else ([], {"candidates": []}))
            admitted_hashes = {
                sha256(json.dumps(
                    row, sort_keys=True, default=str).encode()).hexdigest()
                for row in admitted}
            rejected = [
                row for row in candidate_rows
                if sha256(json.dumps(
                    row, sort_keys=True,
                    default=str).encode()).hexdigest()
                not in admitted_hashes]
            base_updated = density(
                E0, selection, roles["reporting_X"], y_report,
                identity({"task": task, "bank": "E0-updated"}),
                update=update)
            base_frozen = density(
                E0, selection, roles["reporting_X"], y_report,
                identity({"task": task, "bank": "E0-frozen"}))
            base_updated_score = float(np.mean(base_updated))
            base_frozen_score = float(np.mean(base_frozen))

            full_components = [E0] + [
                [*E0, row] for row in candidate_rows]
            admitted_components = [E0] + [
                [*E0, row] for row in admitted]
            rejected_components = [E0] + [
                [*E0, row] for row in rejected]
            engine_components = [E0] + [[row] for row in E0]
            engine_floors = [2] + [1] * len(E0)

            full_matched, full_weights, full_audit = stacking_score(
                full_components, selection, calibration,
                roles["reporting_X"], y_report,
                f"{task}/full/matched", calibration_update=update,
                reporting_update=update, seed=seed)
            full_cross, cross_weights, cross_audit = stacking_score(
                full_components, selection, calibration,
                roles["reporting_X"], y_report,
                f"{task}/full/cross", calibration_update=None,
                reporting_update=update, seed=seed)
            full_frozen, frozen_weights, frozen_audit = stacking_score(
                full_components, selection, calibration,
                roles["reporting_X"], y_report,
                f"{task}/full/frozen", calibration_update=None,
                reporting_update=None, seed=seed)
            admitted_stack, admitted_weights, admitted_stack_audit = (
                stacking_score(
                    admitted_components, selection, calibration,
                    roles["reporting_X"], y_report,
                    f"{task}/admitted", calibration_update=update,
                    reporting_update=update, seed=seed)
                if len(admitted_components) > 1 else (
                    base_updated_score, np.ones(1),
                    {"component_count": 1, "objective_range": 0.0}))
            rejected_stack, rejected_weights, rejected_stack_audit = (
                stacking_score(
                    rejected_components, selection, calibration,
                    roles["reporting_X"], y_report,
                    f"{task}/rejected", calibration_update=update,
                    reporting_update=update, seed=seed)
                if len(rejected_components) > 1 else (
                    base_updated_score, np.ones(1),
                    {"component_count": 1, "objective_range": 0.0}))
            engine_stack, engine_weights, engine_audit = stacking_score(
                engine_components, selection, calibration,
                roles["reporting_X"], y_report,
                f"{task}/engine", calibration_update=update,
                reporting_update=update, seed=seed,
                minimum_supports=engine_floors)
            full_bma = float(np.mean(density(
                [*E0, *candidate_rows], selection,
                roles["reporting_X"], y_report,
                identity({"task": task, "bank": "full-bma"}),
                update=update))) if candidate_rows else base_updated_score
            protected_bma = float(np.mean(density(
                [*E0, *admitted], selection,
                roles["reporting_X"], y_report,
                identity({"task": task, "bank": "protected-bma"}),
                update=update))) if admitted else base_updated_score
            report_matrix = np.column_stack([
                density(
                    bank, selection, roles["reporting_X"], y_report,
                    identity({"task": task, "uniform": idx}),
                    update=update)
                for idx, bank in enumerate(full_components)])
            uniform_weights = np.full(
                len(full_components), 1.0 / len(full_components))
            uniform_score = float(np.mean(mixture_log_density(
                report_matrix, uniform_weights)))
            gains = {
                "e0": 0.0,
                "protected_bma": protected_bma - base_updated_score,
                "full_bma": full_bma - base_updated_score,
                "full_stacking_matched":
                    full_matched - base_updated_score,
                "full_stacking_cross":
                    full_cross - base_updated_score,
                "full_stacking_frozen":
                    full_frozen - base_frozen_score,
                "admitted_only_stacking":
                    admitted_stack - base_updated_score,
                "rejected_only_stacking":
                    rejected_stack - base_updated_score,
                "engine_only_stacking":
                    engine_stack - base_updated_score,
                "uniform_full_mixture":
                    uniform_score - base_updated_score,
            }
            candidate_weight_rows = []
            for candidate, weight in zip(
                    archive.candidates, full_weights[1:]):
                admitted_flag = candidate.posterior_row() in admitted
                candidate_weight_rows.append({
                    "structural_hash": candidate.structural_hash,
                    "source": candidate.source,
                    "admitted_by_protected": admitted_flag,
                    "weight": float(weight),
                    "nonzero_weight": bool(weight > 1e-8),
                })
            rows.append({
                "task": task, "seed": seed,
                "coordinate_failure": False,
                "candidate_count": len(candidate_rows),
                "admitted_candidate_count": len(admitted),
                "rejected_candidate_count": len(rejected),
                "gains": gains,
                "decomposition": {
                    "candidate_coverage_effect":
                        gains["full_stacking_matched"]
                        - gains["admitted_only_stacking"],
                    "weight_optimization_effect":
                        gains["admitted_only_stacking"]
                        - gains["protected_bma"],
                    "state_mismatch_effect":
                        gains["full_stacking_cross"]
                        - gains["full_stacking_matched"],
                    "stacking_over_full_bma":
                        gains["full_stacking_matched"]
                        - gains["full_bma"],
                    "rejected_candidate_recovery":
                        gains["rejected_only_stacking"],
                },
                "weights": {
                    "full": full_weights.tolist(),
                    "cross": cross_weights.tolist(),
                    "frozen": frozen_weights.tolist(),
                    "admitted_only": admitted_weights.tolist(),
                    "rejected_only": rejected_weights.tolist(),
                    "engine_only": engine_weights.tolist(),
                },
                "optimizer_audits": {
                    "full": full_audit, "cross": cross_audit,
                    "frozen": frozen_audit,
                    "admitted_only": admitted_stack_audit,
                    "rejected_only": rejected_stack_audit,
                    "engine_only": engine_audit},
                "candidate_weights": candidate_weight_rows,
                "admission_audit": admission_audit,
            })
        except Exception as error:
            failures.append({
                "task": task, "seed": seed,
                "reason": f"{type(error).__name__}:{error}"})
            rows.append({
                "task": task, "seed": seed,
                "coordinate_failure": True,
                "gains": {
                    arm: 0.0 for arm in freeze["arms"]}})

    task_arms, task_decomp = {}, {}
    for row in rows:
        task_arms.setdefault(row["task"], {})
        for arm in freeze["arms"]:
            task_arms[row["task"]].setdefault(arm, []).append(
                float(row["gains"][arm]))
        if not row["coordinate_failure"]:
            task_decomp.setdefault(row["task"], {})
            for name, value in row["decomposition"].items():
                task_decomp[row["task"]].setdefault(name, []).append(
                    float(value))
    task_means = {
        task: {
            arm: float(np.mean(values))
            for arm, values in by_arm.items()}
        for task, by_arm in task_arms.items()}
    task_decomposition = {
        task: {
            name: float(np.mean(values))
            for name, values in by_name.items()}
        for task, by_name in task_decomp.items()}
    arm_summaries = {}
    tolerance = float(freeze["practical_zero_tolerance"])
    for arm in freeze["arms"]:
        values = {
            task: by_arm[arm] for task, by_arm in task_means.items()}
        classifications = [classify(value, tolerance)
                           for value in values.values()]
        arm_summaries[arm] = {
            "task_mean_gain": float(np.mean(list(values.values()))),
            "ci95": bootstrap(
                values, int(freeze["analysis"]["bootstrap_seed"]),
                int(freeze["analysis"]["resamples"])),
            "exact_positive_count": sum(
                value > 0 for value in values.values()),
            "exact_negative_count": sum(
                value < 0 for value in values.values()),
            "practical_positive_count":
                classifications.count("practical_positive"),
            "practical_negative_count":
                classifications.count("practical_negative"),
            "near_zero_count": sum(
                name.startswith("near_zero") or name == "exact_zero"
                for name in classifications),
            "minimum_task_gain": float(min(values.values())),
            "maximum_task_gain": float(max(values.values())),
        }
    stack_diff = {
        task: arms["full_stacking_matched"]
        - arms["protected_bma"]
        for task, arms in task_means.items()}
    ordered = sorted(
        stack_diff, key=lambda task: stack_diff[task], reverse=True)
    removed = ordered[:int(
        freeze["analysis"]["large_task_sensitivity_count"])]
    trimmed = {
        task: value for task, value in stack_diff.items()
        if task not in removed}
    leave_one_out = {
        omitted: float(np.mean([
            value for task, value in stack_diff.items()
            if task != omitted]))
        for omitted in stack_diff}
    recovery_rows = [
        {
            "task": row["task"], "seed": row["seed"],
            "rejected_candidate_count": row.get(
                "rejected_candidate_count", 0),
            "rejected_only_stacking_gain": row["gains"].get(
                "rejected_only_stacking", 0.0),
            "rejected_nonzero_weight_count": sum(
                candidate["nonzero_weight"]
                and not candidate["admitted_by_protected"]
                for candidate in row.get("candidate_weights", [])),
            "rejected_weight_mass": float(sum(
                candidate["weight"]
                for candidate in row.get("candidate_weights", [])
                if not candidate["admitted_by_protected"])),
            "candidate_weights": row.get("candidate_weights", []),
        }
        for row in rows if not row["coordinate_failure"]]
    result = {
        "schema": "stacking-gain-attribution-result-v1",
        "protocol_complete": True,
        "provider_calls": 0,
        "coordinate_count": len(rows),
        "coordinate_failure_count": len(failures),
        "failures": failures,
        "arm_summaries": arm_summaries,
        "task_arm_means": task_means,
        "task_decomposition": task_decomposition,
        "stacking_minus_protected": {
            "task_mean": float(np.mean(list(stack_diff.values()))),
            "ci95": bootstrap(
                stack_diff,
                int(freeze["analysis"]["bootstrap_seed"]) + 1,
                int(freeze["analysis"]["resamples"])),
            "top_two_tasks": removed,
            "trimmed_task_mean": float(np.mean(list(trimmed.values()))),
            "leave_one_task_out_min": float(min(leave_one_out.values())),
            "leave_one_task_out_max": float(max(leave_one_out.values())),
        },
        "rejected_candidate_recovery": {
            "coordinate_count_with_nonzero_rejected_weight": sum(
                row["rejected_nonzero_weight_count"] > 0
                for row in recovery_rows),
            "coordinate_count_with_positive_rejected_stack_gain": sum(
                row["rejected_only_stacking_gain"] > 0
                for row in recovery_rows),
            "mean_rejected_weight_mass": float(np.mean([
                row["rejected_weight_mass"] for row in recovery_rows])),
            "rows": recovery_rows,
        },
        "rows": rows,
        "practical_zero_tolerance": tolerance,
        "post_reporting_development_only": True,
        "independent_confirmation_claim_authorized": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "claim_boundary": freeze["claim_boundary"],
    }
    write_new(args.output_root / "STACKING_GAIN_ATTRIBUTION.json", result)
    write_new(
        args.output_root / "STACKING_REJECTED_CANDIDATE_RECOVERY.json",
        result["rejected_candidate_recovery"])
    with (args.output_root / "STACKING_TASK_LEVEL_CONTRASTS.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "task", *freeze["arms"],
            "stacking_minus_protected",
            "candidate_coverage_effect",
            "weight_optimization_effect",
            "state_mismatch_effect"])
        for task in sorted(task_means):
            arms = task_means[task]
            decomp = task_decomposition.get(task, {})
            writer.writerow([
                task, *[arms[arm] for arm in freeze["arms"]],
                arms["full_stacking_matched"]
                - arms["protected_bma"],
                decomp.get("candidate_coverage_effect", 0.0),
                decomp.get("weight_optimization_effect", 0.0),
                decomp.get("state_mismatch_effect", 0.0)])
    print(json.dumps({
        "arm_summaries": arm_summaries,
        "stacking_minus_protected":
            result["stacking_minus_protected"],
        "rejected_candidate_recovery": {
            key: value for key, value in
            result["rejected_candidate_recovery"].items()
            if key != "rows"},
        "coordinate_failure_count": len(failures),
        "protocol_complete": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
