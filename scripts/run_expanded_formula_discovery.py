"""Run frozen Gap generation, three admission projections, and formula recovery."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.pcpi_adapter import (
    freeze_discovery_model, structural_terms,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from scripts.expanded_formula_discovery_harness import (
    exact_formula_recovery, project_admission_arms,
    structural_formula_recovery,
)
from scripts.run_aistats_drr_benchmark import _sha
from scripts.run_aistats_three_arm_predictive_gate import (
    _artifact, _llm_rows,
)


GROUPS = {
    "bio_pop_growth": "lsr_synth/bio_pop_growth",
    "chem_react": "lsr_synth/chem_react",
    "matsci": "lsr_synth/matsci",
    "phys_osc": "lsr_synth/phys_osc",
    "lsr_transform": "lsr_transform",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _opened(dataset, indices):
    ordered = sorted(indices)
    rows = np.asarray(dataset[ordered, :])
    lookup = dict(zip(ordered, rows, strict=True))
    return np.asarray([lookup[index] for index in indices])


def _roles(dataset, task, seed):
    frozen = drr_adapter.split_three_arm_training_samples(
        dataset, task_name=task, seed=seed)
    idx = frozen.role_row_indices
    opened = {
        name: _opened(dataset, idx[name])
        for name in (
            "discovery_development", "gap_audit", "gap_admission",
            "decision_selector_update", "inference_initial",
            "action_covariates")
    }
    return {
        "fit": RoleDataset(
            DataRole.DEVELOPMENT,
            opened["discovery_development"][:, 1:],
            opened["discovery_development"][:, 0]),
        "gap": RoleDataset(
            DataRole.VALIDATION, opened["gap_audit"][:, 1:],
            opened["gap_audit"][:, 0]),
        "admission": RoleDataset(
            DataRole.VALIDATION, opened["gap_admission"][:, 1:],
            opened["gap_admission"][:, 0]),
        "selector": RoleDataset(
            DataRole.VALIDATION,
            opened["decision_selector_update"][:, 1:],
            opened["decision_selector_update"][:, 0]),
        "posterior": RoleDataset(
            DataRole.DEVELOPMENT,
            opened["inference_initial"][:, 1:],
            opened["inference_initial"][:, 0]),
        "action_domain": opened["action_covariates"][:, 1:],
    }


def _run_child(freeze, item):
    run_dir = Path(item["run_dir"])
    completed = run_dir / "THREE_ARM_CHILD_RESULT.json"
    if completed.is_file():
        return
    if run_dir.exists() and any(path.is_file()
                                for path in run_dir.rglob("*")):
        raise ValueError("incomplete generation contains materialized files")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "scripts/run_aistats_three_arm_child.py"),
        "--family", item["family"], "--task", item["task"],
        "--seed", str(item["seed"]),
        "--condition", "three_arm_l_gap_v1",
        "--config", str(ROOT / "configs/aistats_three_arm_l_gap_v1.yaml"),
        "--hdf5", freeze["hdf5"]["path"],
        "--provider-env", freeze["provider_env"],
        "--metadata-parquet", freeze["metadata_files"][item["family"]],
        "--output-dir", str(run_dir),
    ]
    env = os.environ.copy()
    env["THREE_ARM_PROMPT_BYTES"] = str(
        freeze["llm_user_prompt_utf8_bytes"])
    with Path(str(run_dir) + ".stdout.log").open(
            "w", encoding="utf-8") as stdout:
        process = subprocess.run(
            command, cwd=ROOT, env=env, stdout=stdout,
            stderr=subprocess.STDOUT, check=False,
            timeout=float(freeze["wall_time_seconds_per_run"]))
    if process.returncode:
        raise RuntimeError(
            f"Gap generation failed: {item['family']}/"
            f"{item['task']}/{item['seed']}")


def _map_expression(bank, posterior):
    member = max(
        posterior.members,
        key=lambda value: (value.probability,
                           value.structure.structure_id))
    matches = sorted(
        expression for _, expression, identifier in bank.candidate_bindings
        if identifier == member.structure.structure_id)
    if not matches:
        raise ValueError("posterior MAP support has no candidate binding")
    return matches[0], float(member.probability)


def _frozen_baseline_rows(report, n_features):
    """Read the artifact's downstream frozen bank, not its audit candidate log."""
    source_rows = list(report.get("drr_candidate_rows") or ())
    if not source_rows:
        source_rows = list(report.get("evaluated_hypothesis_bank") or ())
    rows, seen, rejected = [], set(), []
    for raw in source_rows:
        if not raw.get("expression") or raw.get("origin") == "llm":
            continue
        row = {
            "expression": str(raw["expression"]),
            "source": str(raw.get("source") or "engine:unknown"),
            "origin": str(raw.get("origin") or "deterministic"),
            "lineage_id": str(raw.get("lineage_id") or ""),
        }
        try:
            support = structural_terms(row["expression"], n_features)
        except Exception as error:
            rejected.append({
                "source": row["source"],
                "expression_sha256": sha256(
                    row["expression"].encode()).hexdigest(),
                "diagnostic": str(error),
            })
            continue
        if support in seen:
            continue
        seen.add(support)
        rows.append(row)
    if len(rows) < 2:
        raise ValueError("frozen downstream bank has fewer than two supports")
    return rows, {
        "source": ("drr_candidate_rows" if report.get("drr_candidate_rows")
                   else "evaluated_hypothesis_bank"),
        "input_count": len(source_rows), "retained_count": len(rows),
        "adapter_rejections": rejected,
    }


def _metadata_rows(paths, selected):
    output = {}
    for family, path in paths.items():
        columns = [
            "name", "symbols", "symbol_descs", "symbol_properties",
            "expression"]
        table = pq.read_table(path, columns=columns)
        names = [str(value) for value in table.column("name").to_pylist()]
        for task in selected[family]:
            if names.count(task) != 1:
                raise ValueError("development equation metadata missing")
            index = names.index(task)
            output[(family, task)] = {
                name: table.column(name)[index].as_py()
                for name in columns[1:]}
    return output


def _evaluate(admissions, metadata):
    rows = []
    for row in admissions:
        meta = metadata[(row["family"], row["task"])]
        symbols = [str(value) for value in meta["symbols"]]
        feature_count = int(row["feature_count"])
        offset = 1 if len(symbols) >= feature_count + 1 else 0
        inputs = symbols[offset:offset + feature_count]
        truth = str(meta["expression"])
        arm_results = {}
        for arm, arm_row in row["arms"].items():
            bank_expressions = arm_row["bank_expressions"]
            arm_results[arm] = {
                **arm_row,
                "top1_exact_recovery": exact_formula_recovery(
                    arm_row["map_expression"], truth, inputs),
                "top1_structural_recovery": structural_formula_recovery(
                    arm_row["map_expression"], truth, inputs),
                "bank_exact_recall": any(exact_formula_recovery(
                    expression, truth, inputs)
                    for expression in bank_expressions),
                "bank_structural_recall": any(structural_formula_recovery(
                    expression, truth, inputs)
                    for expression in bank_expressions),
            }
        rows.append({**{key: row[key] for key in (
            "family", "task", "seed", "candidate_bank_identity")},
            "arms": arm_results})
    return rows


def _summary(rows, expected_rows):
    arms = ("independent_protected", "same_data", "accept_all")
    rates = {}
    for arm in arms:
        rates[arm] = {
            metric: float(np.mean([
                int(row["arms"][arm][metric]) for row in rows]))
            for metric in (
                "top1_exact_recovery", "top1_structural_recovery",
                "bank_exact_recall", "bank_structural_recall")
        }
    independent = rates["independent_protected"]
    decisions = {
        "all_registered_rows_accounted_for": len(rows) == expected_rows,
        "candidate_bank_identity_shared_across_arms": all(
            row["candidate_bank_identity"] for row in rows),
        "independent_exact_top1_exceeds_same_data":
            independent["top1_exact_recovery"]
            > rates["same_data"]["top1_exact_recovery"],
        "independent_exact_top1_exceeds_accept_all":
            independent["top1_exact_recovery"]
            > rates["accept_all"]["top1_exact_recovery"],
    }
    return rates, decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    freeze = _read(args.freeze)
    if (freeze.get("schema") !=
            "scientific-expanded-formula-discovery-freeze-v1"
            or freeze.get("execution_authorized") is not True):
        raise ValueError("invalid formula-discovery freeze")
    for path, expected in freeze["files"].items():
        if _sha(path) != expected:
            raise ValueError(f"frozen formula-discovery file changed: {path}")
    if _sha(freeze["hdf5"]["path"]) != freeze["hdf5"]["sha256"]:
        raise ValueError("frozen formula-discovery HDF5 changed")
    if not Path(freeze["provider_env"]).is_file():
        raise ValueError("frozen formula-discovery provider env is unavailable")
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("formula-discovery benchmark source changed")
    if verify_clean_git_source(
            Path(freeze["mainline_root"])) != freeze["mainline_source"]:
        raise ValueError("formula-discovery statistical core changed")
    if args.output_dir.exists() and not args.resume:
        raise ValueError("formula-discovery output must be new")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    plan = [
        {"family": family, "task": task, "seed": seed,
         "run_dir": str((args.output_dir / "generation" / family / task
                         / f"seed{seed}").resolve())}
        for family, tasks in freeze["development_tasks"].items()
        for task in tasks for seed in freeze["seeds"]]
    if args.preflight_only:
        result = {
            "schema": "scientific-expanded-formula-discovery-run-preflight-v1",
            "passed": len(plan) == freeze["gap_generation_run_count"],
            "planned_gap_generation_runs": len(plan),
            "planned_admission_rows": freeze["admission_row_count"],
            "candidate_bank_accessed": False,
            "ground_truth_expression_values_opened": False,
            "hdf5_arrays_opened": False,
            "llm_called": False, "test_or_ood_accessed": False,
            "confirmation_responses_opened": False,
            "confirmation_results_opened": False,
            "execution_started": False,
        }
        (args.output_dir / "PREFLIGHT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    for index, item in enumerate(plan, 1):
        print(f"[{index}/{len(plan)}] Gap generation "
              f"{item['family']}/{item['task']} seed={item['seed']}",
              flush=True)
        _run_child(freeze, item)

    admission_path = args.output_dir / "FROZEN_ADMISSIONS.json"
    admission_complete_path = args.output_dir / "ADMISSION_COMPLETE.json"
    if admission_path.is_file():
        admissions = _read(admission_path)
        if not admission_complete_path.is_file():
            raise ValueError(
                "admission rows exist without completion certificate")
    else:
        admissions = []
        with h5py.File(freeze["hdf5"]["path"], "r") as handle:
            for item in plan:
                dataset = handle[
                    f"{GROUPS[item['family']]}/{item['task']}/train"]
                roles = _roles(dataset, item["task"], item["seed"])
                report = _artifact(Path(item["run_dir"]), item["task"])
                core, baseline_audit = _frozen_baseline_rows(
                    report, roles["fit"].X.shape[1])
                optional, _, _ = _llm_rows(
                    report, core, roles["fit"].X.shape[1],
                    freeze["llm_materialization_limit"])
                identity = sha256(
                    f"expanded-formula-admission-v1:{item['family']}:"
                    f"{item['task']}:{item['seed']}".encode()).hexdigest()
                projection = project_admission_arms(
                    core, optional, roles["fit"], roles["gap"],
                    roles["admission"], roles["selector"],
                    roles["action_domain"],
                    n_features=roles["fit"].X.shape[1],
                    identity=identity)
                arm_rows = {}
                for arm, candidates in projection["arms"].items():
                    model = freeze_discovery_model(
                        candidates, n_features=roles["fit"].X.shape[1],
                        prior=NormalInverseGammaPrior(),
                        exploration_identity=identity,
                        coefficient_policy=(
                            "discard-fitted-coefficients-refit-closed-basis"))
                    posterior = model.engine(model.stable_hash).fit_batch(
                        roles["posterior"].X, roles["posterior"].y)
                    expression, probability = _map_expression(
                        model, posterior)
                    arm_rows[arm] = {
                        "bank_expressions": [
                            str(row["expression"]) for row in candidates],
                        "bank_size": len(candidates),
                        "model_identity": model.stable_hash,
                        "map_expression": expression,
                        "map_probability": probability,
                    }
                admissions.append({
                    "family": item["family"], "task": item["task"],
                    "seed": item["seed"],
                    "feature_count": roles["fit"].X.shape[1],
                    "candidate_bank_identity":
                        projection["candidate_bank_identity"],
                    "arms": arm_rows, "audits": projection["audits"],
                    "baseline_bank_audit": baseline_audit,
                    "generation_cost": _read(
                        Path(item["run_dir"])
                        / "THREE_ARM_CHILD_RESULT.json").get(
                            "provider_cost", []),
                })
        admission_path.write_text(json.dumps(
            admissions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        admission_complete_path.write_text(
            json.dumps({
                "schema": "expanded-formula-admission-complete-v1",
                "row_count": len(admissions),
                "admission_sha256": _sha(admission_path),
                "ground_truth_expression_opened": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completion = _read(admission_complete_path)
    if (completion.get("schema") !=
            "expanded-formula-admission-complete-v1"
            or completion.get("ground_truth_expression_opened") is not False
            or completion.get("row_count") != len(admissions)
            or completion.get("admission_sha256") != _sha(admission_path)):
        raise ValueError(
            "admission completion identity failed before ground truth")
    metadata = _metadata_rows(
        freeze["metadata_files"], freeze["development_tasks"])
    rows = _evaluate(admissions, metadata)
    rates, decisions = _summary(
        rows, freeze["development_task_count"] * len(freeze["seeds"]))
    result = {
        "schema": "scientific-expanded-formula-discovery-result-v1",
        "passed": all(decisions.values()), "decisions": decisions,
        "rates": rates, "row_count": len(rows), "rows": rows,
        "ground_truth_opened_after_admission_freeze": True,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "confirmation_results_opened": False,
        "claim_boundary": freeze["claim_boundary"],
    }
    (args.output_dir / "EXPANDED_FORMULA_DISCOVERY_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
