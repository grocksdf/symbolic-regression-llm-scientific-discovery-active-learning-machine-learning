"""Run frozen Gap generation, three admission projections, and formula recovery."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
sys.path.insert(0, str(ROOT / "scripts"))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from expanded_formula_admission_v2 import project_expanded_admission_arms
from formula_bank_materialization_v2 import (
    expanded_bank_rows, proposal_stage_audit, read_generator_artifact,
)
from formula_recovery_contract import assess_formula_recovery
from provider_health_contract import require_healthy_generation
from run_aistats_drr_benchmark import _sha


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
        require_healthy_generation(_read(completed).get("provider_cost"))
        return
    if run_dir.exists() and any(path.is_file()
                                for path in run_dir.rglob("*")):
        raise ValueError("incomplete generation contains materialized files")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "scripts/run_aistats_three_arm_formula_child.py"),
        "--family", item["family"], "--task", item["task"],
        "--seed", str(item["seed"]),
        "--condition", "three_arm_l_gap_v1",
        "--config", str(ROOT / "configs/aistats_three_arm_formula_prospective.yaml"),
        "--hdf5", freeze["hdf5"]["path"],
        "--provider-env", freeze["provider_env"],
        "--metadata-parquet", freeze["development_metadata_files"][item["family"]],
        "--input-symbols-json", json.dumps(freeze["input_symbols_by_task"][f"{item['family']}/{item['task']}"], ensure_ascii=True),
        "--output-dir", str(run_dir),
    ]
    env = os.environ.copy()
    env["THREE_ARM_PROMPT_BYTES"] = str(
        freeze["llm_user_prompt_utf8_bytes"])
    env["FORMULA_PROVIDER_LOCK"] = item["provider_lock_path"]
    env["FORMULA_PROVIDER_LOCK_TIMEOUT"] = str(
        freeze["provider_lock_timeout_seconds"])
    env["FORMULA_ENGINE_PROCESS_ISOLATION"] = "1"
    with Path(str(run_dir) + ".stdout.log").open(
            "w", encoding="utf-8") as stdout:
        process = subprocess.run(
            command, cwd=ROOT, env=env, stdout=stdout,
            stderr=subprocess.STDOUT, check=False,
            timeout=float(freeze["wall_time_seconds_per_run"]))
    if process.returncode:
        raise RuntimeError(
            "Gap generation stopped before admission; preserve the child "
            "failure log and do not retry under this protocol")
    require_healthy_generation(_read(completed).get("provider_cost"))


def _run_generation_plan(freeze, plan):
    """Run deterministic two-task batches; never exceed one provider request."""
    pending = []
    for index, item in enumerate(plan, 1):
        if item["reused"]:
            print(f"[{index}/{len(plan)}] reused "
                  f"{item['family']}/{item['task']} seed={item['seed']}",
                  flush=True)
        else:
            pending.append((index, item))
    parallelism = int(freeze["task_parallelism"])
    if parallelism not in {1, 2} or freeze["provider_concurrency"] != 1:
        raise ValueError("formula scheduling identity changed")
    for start in range(0, len(pending), parallelism):
        batch = pending[start:start + parallelism]
        for index, item in batch:
            print(f"[{index}/{len(plan)}] Gap generation "
                  f"{item['family']}/{item['task']} seed={item['seed']}",
                  flush=True)
        errors = []
        if parallelism == 1:
            try:
                _run_child(freeze, batch[0][1])
            except BaseException as error:
                errors.append((batch[0], error))
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = {
                    executor.submit(_run_child, freeze, item): (index, item)
                    for index, item in batch}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BaseException as error:
                        errors.append((futures[future], error))
        if errors:
            (_, item), error = errors[0]
            raise RuntimeError(
                "parallel generation batch stopped after terminal child "
                f"failure: {item['family']}/{item['task']}/{item['seed']}"
            ) from error


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


def _metadata_rows(paths, selected):
    output = {}
    for family, path in paths.items():
        columns = [
            "name", "symbols", "symbol_descs", "symbol_properties",
            "expression"]
        table = pq.read_table(path, columns=columns)
        names = [str(value) for value in table.column("name").to_pylist()]
        if len(names) != len(selected[family]) or set(names) != set(selected[family]):
            raise ValueError("metadata file contains tasks outside frozen development")
        for task in selected[family]:
            if names.count(task) != 1:
                raise ValueError("development equation metadata missing")
            index = names.index(task)
            output[(family, task)] = {
                name: table.column(name)[index].as_py()
                for name in columns[1:]}
    return output


def _evaluate(admissions, metadata, input_symbols_by_task):
    rows = []
    for row in admissions:
        meta = metadata[(row["family"], row["task"])]
        symbols = [str(value) for value in meta["symbols"]]
        feature_count = int(row["feature_count"])
        inputs = input_symbols_by_task[f"{row['family']}/{row['task']}"]
        if (len(inputs) != feature_count or len(set(inputs)) != len(inputs)
                or any(value not in symbols for value in inputs)):
            raise ValueError("frozen input symbol mapping disagrees with metadata")
        truth = str(meta["expression"])
        arm_results = {}
        for arm, arm_row in row["arms"].items():
            bank_expressions = arm_row["bank_expressions"]
            top = assess_formula_recovery(
                arm_row["map_expression"], truth, inputs)
            bank = [assess_formula_recovery(expression, truth, inputs)
                    for expression in bank_expressions]
            def some(key):
                values = [item[key] for item in bank]
                return (True if True in values else
                        None if None in values else False)
            arm_results[arm] = {
                **arm_row,
                "top1_assessment": top,
                "top1_literal_exact": top["literal_exact"],
                "top1_structural_topology": top["structural_topology"],
                "bank_literal_exact": some("literal_exact"),
                "bank_structural_topology": some("structural_topology"),
            }
        rows.append({**{key: row[key] for key in (
            "family", "task", "seed", "candidate_bank_identity",
            "generation_provenance")},
            "arms": arm_results})
    return rows


def _summary(rows, expected_rows):
    arms = ("independent_protected", "same_data", "accept_all")
    rates, denominators = {}, {}
    for arm in arms:
        rates[arm], denominators[arm] = {}, {}
        for metric in ("top1_literal_exact", "top1_structural_topology",
                       "bank_literal_exact", "bank_structural_topology"):
            values = [row["arms"][arm][metric] for row in rows]
            applicable = [value for value in values if value is not None]
            denominators[arm][metric] = len(applicable)
            rates[arm][metric] = (float(np.mean(applicable))
                                  if applicable else None)
    independent = rates["independent_protected"]
    decisions = {
        "all_registered_rows_accounted_for": len(rows) == expected_rows,
        "candidate_bank_identity_shared_across_arms": all(
            row["candidate_bank_identity"] for row in rows),
        "structural_top1_evaluable_for_all_rows": all(
            denominators[arm]["top1_structural_topology"] == len(rows)
            for arm in arms),
        "independent_structural_top1_exceeds_same_data": bool(
            independent["top1_structural_topology"] is not None
            and rates["same_data"]["top1_structural_topology"] is not None
            and independent["top1_structural_topology"]
                > rates["same_data"]["top1_structural_topology"]),
        "independent_structural_top1_exceeds_accept_all": bool(
            independent["top1_structural_topology"] is not None
            and rates["accept_all"]["top1_structural_topology"] is not None
            and independent["top1_structural_topology"]
                > rates["accept_all"]["top1_structural_topology"]),
    }
    return rates, denominators, decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    freeze = _read(args.freeze)
    if (freeze.get("schema") not in {
            "scientific-expanded-formula-discovery-freeze-v2.1",
            "scientific-expanded-formula-discovery-freeze-v2.2",
            "scientific-expanded-formula-discovery-freeze-v2.3",
            "scientific-expanded-formula-discovery-freeze-v2.4",
            "scientific-expanded-formula-discovery-freeze-v2.5",
            "scientific-expanded-formula-discovery-freeze-v2.6",
            "scientific-expanded-formula-discovery-freeze-v2.7",
            "scientific-expanded-formula-discovery-freeze-v2.8",
            "scientific-expanded-formula-discovery-freeze-v2.9",
            "scientific-expanded-formula-discovery-freeze-v2.10",
            "scientific-expanded-formula-discovery-freeze-v2.11"}
            or freeze.get("execution_authorized") is not True):
        raise ValueError("v2.1/v2.2 formula-discovery continuation freeze required")
    minimum = freeze.get("minimum_paired_bank_contrasts")
    if (type(minimum) is not int or minimum < 1
            or minimum > freeze.get("development_task_count", 0)):
        raise ValueError("predeclared informative-task minimum required")
    input_symbols_by_task = freeze.get("input_symbols_by_task")
    expected_tasks = {f"{family}/{task}"
        for family, tasks in freeze["development_tasks"].items()
        for task in tasks}
    if (not isinstance(input_symbols_by_task, dict)
            or set(input_symbols_by_task) != expected_tasks
            or any(not isinstance(value, list) or not value
                   or len(set(value)) != len(value)
                   for value in input_symbols_by_task.values())):
        raise ValueError("prospective input symbol mapping must be frozen")
    metadata_files = freeze.get("development_metadata_files")
    if (freeze.get("metadata_scope") != "physically-development-only-v1"
            or not isinstance(metadata_files, dict)
            or set(metadata_files) != set(freeze["development_tasks"])):
        raise ValueError("development-only metadata files must be frozen")
    if any(path not in freeze.get("files", {})
           for path in metadata_files.values()):
        raise ValueError("development metadata hashes absent from freeze")
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
    identity_path = args.output_dir / "PROTOCOL_IDENTITY.json"
    protocol_identity = {"schema": "prospective-formula-protocol-identity-v2",
                         "freeze_sha256": _sha(args.freeze)}
    if args.resume:
        if not identity_path.is_file() or _read(identity_path) != protocol_identity:
            raise ValueError("resume crossed prospective protocol identity")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("prospective output contains preexisting files")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    if not identity_path.is_file():
        identity_path.write_text(json.dumps(protocol_identity,
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan = [
        {"family": family, "task": task, "seed": seed,
         "key": f"{family}/{task}/{seed}",
         "run_dir": str(Path(
             freeze.get("reused_generation_runs", {}).get(
                 f"{family}/{task}/{seed}", {}).get("run_dir")
             or (args.output_dir / "generation" / family / task
                 / f"seed{seed}")).resolve()),
         "provider_lock_path": str(
             (args.output_dir / "FORMULA_PROVIDER.lock").resolve()),
         "reused": f"{family}/{task}/{seed}" in
             freeze.get("reused_generation_runs", {})}
        for family, tasks in freeze["development_tasks"].items()
        for task in tasks for seed in freeze["seeds"]]
    for key, reused in freeze.get("reused_generation_runs", {}).items():
        run_dir = Path(reused["run_dir"])
        child = run_dir / "THREE_ARM_CHILD_RESULT.json"
        artifact = run_dir / "pcpi_artifacts" / f"{reused['task']}.json"
        if (_sha(child) != reused["child_sha256"]
                or _sha(artifact) != reused["artifact_sha256"]):
            raise ValueError(
                f"reused v2 candidate artifact changed: {key}")
    if args.preflight_only:
        result = {
            "schema": "scientific-expanded-formula-discovery-run-preflight-v2",
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
    _run_generation_plan(freeze, plan)

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
                report = read_generator_artifact(
                    Path(item["run_dir"]), item["task"])
                core, optional, baseline_audit = expanded_bank_rows(
                    report, roles["fit"].X.shape[1],
                    freeze["llm_materialization_limit"])
                generation_cost = _read(Path(item["run_dir"])
                    / "THREE_ARM_CHILD_RESULT.json").get("provider_cost")
                stages = proposal_stage_audit(
                    report, generation_cost, baseline_audit)
                if stages["transport_failed"]:
                    raise ValueError("transport failed before admission freeze")
                identity = sha256(
                    f"expanded-formula-admission-v2:{item['family']}:"
                    f"{item['task']}:{item['seed']}".encode()).hexdigest()
                projection = project_expanded_admission_arms(
                    core, optional, roles["fit"], roles["gap"],
                    roles["admission"], roles["selector"],
                    roles["action_domain"],
                    n_features=roles["fit"].X.shape[1],
                    identity=identity)
                arm_rows = {}
                for arm, candidates in projection["arms"].items():
                    model = freeze_expanded_formula_model(
                        candidates, n_features=roles["fit"].X.shape[1],
                        prior=NormalInverseGammaPrior(),
                        exploration_identity=identity,
                        coefficient_policy=EXPANDED_FORMULA_POLICY)
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
                    "generation_provenance": (
                        "v2-pre-typed-output-repair" if item["reused"]
                        else "v2.1-post-typed-output-repair"),
                    "feature_count": roles["fit"].X.shape[1],
                    "candidate_bank_identity":
                        projection["candidate_bank_identity"],
                    "arms": arm_rows, "audits": projection["audits"],
                    "baseline_bank_audit": baseline_audit,
                    "proposal_stage_audit": stages,
                    "generation_cost": generation_cost,
                })
        admission_path.write_text(json.dumps(
            admissions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        admission_complete_path.write_text(
            json.dumps({
                "schema": "expanded-formula-admission-complete-v2",
                "row_count": len(admissions),
                "admission_sha256": _sha(admission_path),
                "ground_truth_expression_opened": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completion = _read(admission_complete_path)
    if (completion.get("schema") !=
            "expanded-formula-admission-complete-v2"
            or completion.get("ground_truth_expression_opened") is not False
            or completion.get("row_count") != len(admissions)
            or completion.get("admission_sha256") != _sha(admission_path)):
        raise ValueError(
            "admission completion identity failed before ground truth")
    contrast = {control: len({(row["family"], row["task"])
            for row in admissions if row["arms"]["independent_protected"][
                "bank_expressions"] != row["arms"][control][
                    "bank_expressions"]})
        for control in ("same_data", "accept_all")}
    design_gate = {
        "schema": "prospective-formula-identifiability-gate-v2",
        "minimum_paired_bank_contrasts": minimum,
        "distinct_task_counts_by_control": contrast,
        "admission_sha256": _sha(admission_path),
        "passed": all(value >= minimum for value in contrast.values()),
        "ground_truth_expression_opened": False,
    }
    gate_path = args.output_dir / "DESIGN_GATE.json"
    if gate_path.exists() and _read(gate_path) != design_gate:
        raise ValueError("prospective design gate changed on replay")
    if not gate_path.exists():
        gate_path.write_text(json.dumps(design_gate,
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not design_gate["passed"]:
        raise ValueError("admission contrast underpowered; truth remains closed")
    metadata = _metadata_rows(
        metadata_files, freeze["development_tasks"])
    rows = _evaluate(admissions, metadata, input_symbols_by_task)
    rates, denominators, decisions = _summary(
        rows, freeze["development_task_count"] * len(freeze["seeds"]))
    result = {
        "schema": "scientific-expanded-formula-discovery-result-v2",
        "passed": all(decisions.values()), "decisions": decisions,
        "rates": rates, "applicable_denominators": denominators,
        "row_count": len(rows), "rows": rows,
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
