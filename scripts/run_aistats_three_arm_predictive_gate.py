"""Run E / E+L_blind / E+L_gap with independent admission and reporting."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.pcpi_adapter import structural_terms
from hypothesis_mvp.discovery.common_class_projection import (
    freeze_common_class_projection,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    freeze_discovery_model, freeze_discovery_target,
)
from hypothesis_mvp.discovery.source_stacking import (
    filter_fold_safe_source_candidates,
)
from hypothesis_mvp.discovery.system_run import audit_decision_risk_target
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.acquisition import EXACT_CLASS_EIG_EPSABS
from scripts.run_aistats_drr_benchmark import _sha
from scripts.run_aistats_synthesis_only_smoke_gate import _engine_identity


CONDITIONS = (
    "three_arm_e_v1",
    "three_arm_l_blind_v1",
    "three_arm_l_gap_v1",
)
ARM_NAMES = {
    "three_arm_e_v1": "E",
    "three_arm_l_blind_v1": "E+L_blind",
    "three_arm_l_gap_v1": "E+L_gap",
}
CONFIGS = {
    "three_arm_e_v1": "configs/aistats_three_arm_e_v1.yaml",
    "three_arm_l_blind_v1": "configs/aistats_three_arm_l_blind_v1.yaml",
    "three_arm_l_gap_v1": "configs/aistats_three_arm_l_gap_v1.yaml",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(freeze: dict) -> None:
    if (
        freeze.get("schema")
        != "scientific-aistats-three-arm-predictive-freeze-v1"
        or tuple(freeze.get("conditions", ())) != CONDITIONS
        or freeze.get("execution_authorized") is not True
    ):
        raise ValueError("invalid three-arm predictive freeze")
    for path, expected in {
        **freeze["files"], **freeze["exclusion_freezes"]
    }.items():
        if _sha(path) != expected:
            raise ValueError(f"three-arm frozen file changed: {path}")
    if _sha(freeze["hdf5"]["path"]) != freeze["hdf5"]["sha256"]:
        raise ValueError("three-arm HDF5 changed")
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("three-arm benchmark source changed")
    if verify_clean_git_source(
        Path(freeze["mainline_root"])
    ) != freeze["mainline_source"]:
        raise ValueError("three-arm mainline source changed")


def _artifact(run_dir, task):
    path = run_dir / "pcpi_artifacts" / f"{task}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read(path)["scientific_discovery_runtime"]


def _run_generation(freeze, item):
    run_dir = Path(item["run_dir"])
    completed = run_dir / "THREE_ARM_CHILD_RESULT.json"
    if completed.is_file():
        payload = _read(completed)
        if (payload.get("family") != item["family"]
                or payload.get("task") != item["task"]
                or payload.get("seed") != item["seed"]
                or payload.get("condition") != item["condition"]):
            raise ValueError("three-arm child resume identity changed")
        return
    if run_dir.exists() and any(path.is_file()
                                for path in run_dir.rglob("*")):
        raise ValueError("unfinished three-arm child has materialized files")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/run_aistats_three_arm_child.py"),
        "--family", item["family"],
        "--task", item["task"],
        "--seed", str(item["seed"]),
        "--condition", item["condition"],
        "--config", str(ROOT / CONFIGS[item["condition"]]),
        "--hdf5", freeze["hdf5"]["path"],
        "--provider-env", freeze["provider_env"],
        "--output-dir", str(run_dir),
    ]
    env = os.environ.copy()
    env["THREE_ARM_PROMPT_BYTES"] = str(
        freeze["llm_user_prompt_utf8_bytes"])
    with (Path(str(run_dir) + ".stdout.log")).open(
            "w", encoding="utf-8") as stdout:
        process = subprocess.run(
            command, cwd=ROOT, check=False,
            timeout=float(freeze["wall_time_seconds_per_run"]),
            stdout=stdout, stderr=subprocess.STDOUT,
            env=env)
    if process.returncode != 0:
        raise RuntimeError(
            f"three-arm child failed: {item['condition']}/"
            f"{item['task']}/{item['seed']}")


def _execute_generation_plan(freeze, plan, *, resume):
    for index, item in enumerate(plan, start=1):
        if (resume
                and (Path(item["run_dir"])
                     / "THREE_ARM_CHILD_RESULT.json").is_file()):
            continue
        print(f"[{index}/{len(plan)}] {item['condition']} "
              f"{item['family']}/{item['task']} seed={item['seed']}",
              flush=True)
        _run_generation(freeze, item)


def _dataset(handle, family, task):
    return (
        handle[f"lsr_transform/{task}/train"]
        if family == "lsr_transform"
        else handle[f"lsr_synth/{family}/{task}/train"]
    )


def _opened(dataset, indices):
    ordered = sorted(indices)
    rows = np.asarray(dataset[ordered, :])
    lookup = dict(zip(ordered, rows, strict=True))
    return np.asarray([lookup[index] for index in indices])


def _analysis_roles(dataset, task, seed):
    frozen = drr_adapter.split_three_arm_training_samples(
        dataset, task_name=task, seed=seed)
    indices = frozen.role_row_indices
    initial = _opened(dataset, indices["inference_initial"])
    admission = _opened(dataset, indices["gap_admission"])
    selector = _opened(dataset, indices["decision_selector_update"])
    calibration = _opened(dataset, indices["decision_calibration"])
    report = _opened(dataset, indices["reporting"])
    return {
        "X_initial": initial[:, 1:], "y_initial": initial[:, 0],
        "X_admission": admission[:, 1:], "y_admission": admission[:, 0],
        "X_selector": selector[:, 1:], "y_selector": selector[:, 0],
        "X_calibration": calibration[:, 1:],
        "y_calibration": calibration[:, 0],
        "X_report": report[:, 1:], "y_report": report[:, 0],
        "X_actions": frozen.X_actions,
        "role_row_indices": {
            key: list(value) for key, value in indices.items()},
    }


def _support(row, n_features):
    return tuple(structural_terms(str(row["expression"]), n_features))


def _baseline_rows(report):
    rows, seen = [], set()
    for row in report.get("evaluated_hypothesis_bank", ()):
        if not row.get("expression") or row.get("origin") == "llm":
            continue
        candidate = {
            "expression": str(row["expression"]),
            "source": str(row.get("source") or "engine:unknown"),
            "origin": str(row.get("origin") or "deterministic"),
            "lineage_id": str(row.get("lineage_id") or ""),
        }
        support = _support(
            candidate,
            int(report["task_context_audit"][
                "variable_description_count"]))
        if support in seen:
            continue
        seen.add(support)
        rows.append(candidate)
    if len(rows) < 2:
        raise ValueError("engine arm has fewer than two frozen supports")
    return rows


def _llm_rows(report, baseline, n_features, limit):
    occupied = {_support(row, n_features) for row in baseline}
    selected, seen, rejected = [], set(), 0
    candidates = [
        dict(row) for row in report.get("evaluated_hypothesis_bank", ())
        if row.get("origin") == "llm" and row.get("expression")]
    candidates.sort(key=lambda row: sha256(json.dumps(
        row, sort_keys=True, default=str).encode()).hexdigest())
    for row in candidates:
        try:
            support = _support(row, n_features)
        except Exception:
            rejected += 1
            continue
        if support in occupied or support in seen:
            rejected += 1
            continue
        seen.add(support)
        selected.append({
            "expression": str(row["expression"]),
            "source": str(row.get("source") or "llm"),
            "origin": "llm",
            "lineage_id": str(row.get("lineage_id") or ""),
        })
        if len(selected) == limit:
            break
    return selected, rejected, len(candidates)


def _admit(rows, optional, roles, *, identity, n_features):
    initial = RoleDataset(
        DataRole.DEVELOPMENT, roles["X_initial"], roles["y_initial"])
    admission = RoleDataset(
        DataRole.VALIDATION, roles["X_admission"], roles["y_admission"])
    selector = RoleDataset(
        DataRole.VALIDATION, roles["X_selector"], roles["y_selector"])
    kwargs = {
        "n_features": n_features,
        "prior": NormalInverseGammaPrior(),
        "exploration_identity": identity,
        "coefficient_policy":
            "discard-fitted-coefficients-refit-closed-basis",
        "measurement_budget": 2,
        "action_domain": roles["X_actions"],
    }
    protected = [{
        **dict(row),
        "source": (
            "protected_counterfactual_backbone:"
            f"{row.get('source') or 'engine:unknown'}"),
    } for row in rows]
    first, admission_report = filter_fold_safe_source_candidates(
        [*protected, *optional], initial, admission, **kwargs)
    admitted_first = [
        dict(row) for row in first if row.get("origin") == "llm"]
    second, selector_report = filter_fold_safe_source_candidates(
        [*protected, *admitted_first], initial, selector, **kwargs)
    admitted_second = [
        dict(row) for row in second if row.get("origin") == "llm"]
    return [*[dict(row) for row in rows], *admitted_second], {
        "admission": admission_report,
        "selector_update": selector_report,
        "admission_identity": admission.fingerprint,
        "selector_update_identity": selector.fingerprint,
        "protected_engine_core_count": len(rows),
        "admitted_llm_count": len(admitted_second),
    }


def _cost(report, run_dir, max_tokens):
    child = _read(run_dir / "THREE_ARM_CHILD_RESULT.json")
    provider_cost = list(child.get("provider_cost") or ())
    return {
        "logical_llm_calls": int(
            report.get("scientist_agent_logical_llm_call_count") or 0),
        "provider_attempts": int(
            report.get("scientist_agent_provider_attempt_count") or 0),
        "registered_max_completion_tokens_per_call": int(max_tokens),
        "registered_max_completion_token_budget": int(
            (report.get("scientist_agent_logical_llm_call_count") or 0)
            * max_tokens),
        "provider_request_count": len(provider_cost),
        "actual_prompt_tokens": (
            None if any(row["prompt_tokens"] is None for row in provider_cost)
            else sum(row["prompt_tokens"] for row in provider_cost)),
        "actual_completion_tokens": (
            None if any(row["completion_tokens"] is None
                        for row in provider_cost)
            else sum(row["completion_tokens"] for row in provider_cost)),
        "actual_total_tokens": (
            None if any(row["total_tokens"] is None for row in provider_cost)
            else sum(row["total_tokens"] for row in provider_cost)),
        "provider_elapsed_seconds": float(sum(
            row["elapsed_seconds"] for row in provider_cost)),
        "user_prompt_utf8_bytes": [
            int(row["user_prompt_utf8_bytes"]) for row in provider_cost],
        "candidate_evaluation_budget_used": int(
            report.get("evaluation_budget_used") or 0),
        "candidate_evaluation_budget_limit": int(
            report.get("evaluation_budget_limit") or 0),
        "search_time_seconds": float(child["wall_time_seconds"]),
        "candidate_bank_size_before_admission": int(len(
            report.get("drr_candidate_rows") or ())),
    }


def _score_complete_arm(rows, roles, *, identity, n_features):
    started = time.monotonic()
    initial = RoleDataset(
        DataRole.DEVELOPMENT, roles["X_initial"], roles["y_initial"])
    model = freeze_discovery_model(
        rows, n_features=n_features, prior=NormalInverseGammaPrior(),
        exploration_identity=identity,
        coefficient_policy=
            "discard-fitted-coefficients-refit-closed-basis")
    target = freeze_discovery_target(
        model, initial, roles["X_actions"], measurement_budget=2,
        expected_model_identity=model.stable_hash)
    engine = model.engine(model.stable_hash)
    posterior = target.initial_posterior
    predictions = np.zeros(len(roles["y_report"]), dtype=float)
    for member in posterior.members:
        mean, _ = engine.predictive_moments(member, roles["X_report"])
        predictions += member.probability * np.asarray(mean, dtype=float)
    log_density = np.asarray(engine.predictive_logpdf(
        posterior, roles["X_report"], roles["y_report"]), dtype=float)
    if (not np.all(np.isfinite(predictions))
            or not np.all(np.isfinite(log_density))):
        raise FloatingPointError("three-arm report score is nonfinite")
    utility = audit_decision_risk_target(
        model, target, roles["X_actions"], EXACT_CLASS_EIG_EPSABS)
    public = {
        "report_mse": float(np.mean(
            (predictions - roles["y_report"]) ** 2)),
        "report_log_score": float(np.mean(log_density)),
        "posterior_member_count": len(posterior.members),
        "model_identity": model.stable_hash,
        "target_identity": target.stable_hash,
        "decision_risk_utility": utility,
        "decision_status": (
            "certified" if utility["passed"] else "uncertified"),
        "inference_wall_seconds": time.monotonic() - started,
    }
    return public, target


def _pair_record(family, task, seed, reports, run_dirs, roles, freeze):
    baseline = _baseline_rows(reports["three_arm_e_v1"])
    n_features = roles["X_initial"].shape[1]
    engine_identity = {
        condition: _engine_identity(reports["three_arm_e_v1"])
        == _engine_identity(reports[condition])
        for condition in CONDITIONS[1:]
    }
    banks, funnels = {}, {}
    for condition in CONDITIONS:
        if condition == "three_arm_e_v1":
            bank = list(baseline)
            funnel = {"offered": 0, "rejected_before_admission": 0,
                      "source_candidates": 0}
        else:
            optional, rejected, source_count = _llm_rows(
                reports[condition], baseline, n_features,
                freeze["llm_materialization_limit"])
            identity = sha256(
                f"three-arm-admission-v1:{family}:{task}:{seed}:"
                f"{condition}".encode()).hexdigest()
            bank, audit = _admit(
                baseline, optional, roles, identity=identity,
                n_features=n_features)
            funnel = {
                "offered": len(optional),
                "rejected_before_admission": rejected,
                "source_candidates": source_count,
                "independent_admission": audit,
            }
        banks[ARM_NAMES[condition]] = bank
        funnels[ARM_NAMES[condition]] = funnel

    arms, targets = {}, {}
    for name, bank in banks.items():
        score_identity = sha256(
            f"three-arm-score-v1:{family}:{task}:{seed}:{name}".encode()
        ).hexdigest()
        arms[name], targets[name] = _score_complete_arm(
            bank, roles, identity=score_identity, n_features=n_features)
    projections = {}
    for name in ("E+L_blind", "E+L_gap"):
        projection = freeze_common_class_projection(
            targets["E"], targets[name])
        projections[name] = {
            "identity": projection.stable_hash,
            "union_class_count": len(projection.union_class_ids),
            "common_class_loss_assessed": False,
            "common_class_projection_constructed": True,
        }

    costs = {
        ARM_NAMES[condition]: _cost(
            reports[condition], run_dirs[condition],
            freeze["maximum_completion_tokens_per_call"])
        for condition in CONDITIONS
    }
    blind_cost, gap_cost = costs["E+L_blind"], costs["E+L_gap"]
    knowledge = {
        ARM_NAMES[condition]: {
            "namespace": reports[condition].get("knowledge_namespace"),
            "started_empty": reports[condition].get(
                "knowledge_namespace_started_empty"),
            "cross_arm_read_enabled": reports[condition].get(
                "knowledge_cross_arm_read_enabled"),
        }
        for condition in CONDITIONS
    }
    namespaces = [row["namespace"] for row in knowledge.values()]
    checks = {
        "engine_frontier_identical": all(engine_identity.values()),
        "llm_call_count_equal": (
            blind_cost["logical_llm_calls"]
            == gap_cost["logical_llm_calls"]),
        "llm_token_cap_equal": (
            blind_cost["registered_max_completion_token_budget"]
            == gap_cost["registered_max_completion_token_budget"]),
        "llm_prompt_byte_cap_equal": (
            blind_cost["user_prompt_utf8_bytes"]
            == gap_cost["user_prompt_utf8_bytes"]),
        "candidate_budget_equal": (
            blind_cost["candidate_evaluation_budget_limit"]
            == gap_cost["candidate_evaluation_budget_limit"]),
        "knowledge_namespaces_distinct": (
            len(namespaces) == len(set(namespaces))),
        "knowledge_namespaces_started_empty": all(
            row["started_empty"] is True for row in knowledge.values()),
        "cross_arm_knowledge_reads_disabled": all(
            row["cross_arm_read_enabled"] is False
            for row in knowledge.values()),
        "report_scores_finite": all(np.isfinite(
            arm["report_log_score"]) for arm in arms.values()),
        "numerical_status_explicit": all(
            isinstance(arm["decision_risk_utility"].get("passed"), bool)
            for arm in arms.values()),
        "common_class_projection_constructed": all(
            row["common_class_projection_constructed"]
            for row in projections.values()),
    }
    return {
        "family": family, "task": task, "seed": seed,
        "status": "success" if all(checks.values()) else "failure",
        "checks": checks, "engine_identity": engine_identity,
        "knowledge_isolation": knowledge, "costs": costs,
        "funnels": funnels, "arms": arms,
        "common_class_projections": projections,
        "decision_calibration_identity": sha256(np.ascontiguousarray(
            np.column_stack((
                roles["X_calibration"], roles["y_calibration"]))
        ).tobytes()).hexdigest(),
        "paired": {
            "blind_minus_engine_log_score": float(
                arms["E+L_blind"]["report_log_score"]
                - arms["E"]["report_log_score"]),
            "gap_minus_engine_log_score": float(
                arms["E+L_gap"]["report_log_score"]
                - arms["E"]["report_log_score"]),
            "gap_minus_blind_log_score": float(
                arms["E+L_gap"]["report_log_score"]
                - arms["E+L_blind"]["report_log_score"]),
        },
        "report_response_accessed_for_scoring_only": True,
        "action_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }


def _aggregate(rows, freeze):
    tolerance = freeze["decision"]["numerical_tolerance"]
    gap_blind = [row["paired"]["gap_minus_blind_log_score"] for row in rows]
    gap_engine = [row["paired"]["gap_minus_engine_log_score"] for row in rows]
    blind_engine = [
        row["paired"]["blind_minus_engine_log_score"] for row in rows]
    tasks = sorted({row["task"] for row in rows})
    task_effects = {
        task: float(np.mean([
            row["paired"]["gap_minus_blind_log_score"]
            for row in rows if row["task"] == task]))
        for task in tasks
    }
    decisions = {
        "all_pairs_accounted_for": len(rows) == freeze["pair_count"],
        "zero_pair_failures": all(
            row["status"] == "success" for row in rows),
        "global_gap_minus_blind_strictly_positive":
            float(np.mean(gap_blind)) > tolerance,
        "minimum_positive_tasks": sum(
            value > tolerance for value in task_effects.values())
            >= freeze["decision"]["minimum_positive_tasks"],
        "global_gap_minus_engine_strictly_positive":
            float(np.mean(gap_engine)) > tolerance,
        "all_uncertified_decisions_remain_explicit": all(
            isinstance(arm["decision_risk_utility"].get("passed"), bool)
            for row in rows for arm in row["arms"].values()),
    }
    return {
        "task_effects_gap_minus_blind_log_score": task_effects,
        "global_gap_minus_blind_log_score": float(np.mean(gap_blind)),
        "global_gap_minus_engine_log_score": float(np.mean(gap_engine)),
        "global_blind_minus_engine_log_score": float(np.mean(blind_engine)),
        "decisions": decisions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    freeze = _read(args.freeze)
    _verify(freeze)
    if args.output_dir.exists() and not args.resume:
        raise ValueError("three-arm output must be new")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    plan = [
        {"family": family, "task": task, "seed": seed,
         "condition": condition,
         "run_dir": str((args.output_dir / "runs" / family / task
                         / f"seed{seed}" / condition).resolve())}
        for family, task in freeze["selected_tasks"].items()
        for seed in freeze["seeds"] for condition in CONDITIONS]
    plan_path = args.output_dir / "PLAN.json"
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if plan_path.is_file() and plan_path.read_text("utf-8") != text:
        raise ValueError("three-arm plan identity changed")
    plan_path.write_text(text, encoding="utf-8")
    if args.preflight_only:
        result = {
            "schema": "scientific-aistats-three-arm-preflight-v1",
            "passed": True, "planned_child_runs": len(plan),
            "planned_pairs": freeze["pair_count"],
            "task_arrays_accessed": False, "llm_called": False,
            "candidate_response_accessed": False,
            "test_or_ood_accessed": False, "heldout_opened": False,
            "execution_started": False}
        (args.output_dir / "PREFLIGHT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    _execute_generation_plan(freeze, plan, resume=args.resume)

    pairs_path = args.output_dir / "THREE_ARM_PAIRS.json"
    rows = _read(pairs_path) if pairs_path.is_file() else []
    completed = {(row["family"], row["task"], row["seed"]) for row in rows}
    with h5py.File(freeze["hdf5"]["path"], "r") as handle:
        for family, task in freeze["selected_tasks"].items():
            for seed in freeze["seeds"]:
                if (family, task, seed) in completed:
                    continue
                reports, dirs = {}, {}
                for condition in CONDITIONS:
                    run_dir = (args.output_dir / "runs" / family / task
                               / f"seed{seed}" / condition)
                    reports[condition] = _artifact(run_dir, task)
                    dirs[condition] = run_dir
                roles = _analysis_roles(
                    _dataset(handle, family, task), task, seed)
                row = _pair_record(
                    family, task, seed, reports, dirs, roles, freeze)
                rows.append(row)
                pairs_path.write_text(
                    json.dumps(rows, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    analysis = _aggregate(rows, freeze)
    result = {
        "schema": "scientific-aistats-three-arm-predictive-result-v1",
        "passed": all(analysis["decisions"].values()),
        **analysis, "pair_count": len(rows), "rows": rows,
        "predictive_gain_assessed": True,
        "decision_path_assessed": True,
        "decision_gain_assessed": False,
        "action_response_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": freeze["claim_boundary"]}
    (args.output_dir / "AISTATS_THREE_ARM_PREDICTIVE_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
