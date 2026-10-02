"""Run the frozen MatSci quality-first predictive-increment comparison.

Registered step one of the quality-first program: compare paired report-set
predictive log scores of three banks built from identical matched-engine
evidence — 48E baseline, 48E+L_LLM (typed LLM proposals), and 48E+L_control
(equal-size non-LLM engine proposals).  No acquisition trajectory, no action
response reveal, no test/OOD access, no held-out access.  Fresh discovery
runs are spawned only through the frozen benchmark runner path.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.bank_selection import (
    DECISION_RISK_CAPACITY_METHOD,
    select_operational_capacity_bank,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    freeze_discovery_model,
    freeze_discovery_target,
)
from hypothesis_mvp.discovery.source_stacking import DIVERSITY_METHOD
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.acquisition import EXACT_CLASS_EIG_EPSABS

from methods.hypothesis_mvp_pcpi.quality_first_augmentation import (
    build_quality_first_augmentation,
)
from scripts.run_aistats_drr_benchmark import RunSpec, _resume_rows, _run_one


EXPLORATION_NAMESPACE = "aistats-matsci-quality-first-increment-v1"
COEFFICIENT_POLICY = "discard-fitted-coefficients-refit-closed-basis"


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _verify(freeze: dict) -> None:
    if (
        freeze.get("schema")
        != "scientific-aistats-matsci-quality-first-increment-freeze-v1"
        or freeze.get("conditions") != ["full_scientist_v6", "no_llm_v6"]
        or freeze.get("seeds") != [81, 82]
        or freeze.get("baseline_evaluations") != 48
        or freeze.get("llm_proposal_limit") != 12
        or freeze.get("measurement_budget") != 2
        or freeze.get("test_or_ood_accessed") is not False
        or freeze.get("heldout_opened") is not False
        or freeze.get("execution_authorized") is not True
    ):
        raise ValueError("invalid MatSci quality-first increment freeze")
    for path, expected in freeze["files"].items():
        if _sha(Path(path)) != expected:
            raise ValueError(f"quality-first frozen file changed: {path}")
    for path, expected in freeze["exclusion_freezes"].items():
        if _sha(Path(path)) != expected:
            raise ValueError(f"quality-first exclusion freeze changed: {path}")
    if _sha(Path(freeze["hdf5"]["path"])) != freeze["hdf5"]["sha256"]:
        raise ValueError("quality-first HDF5 changed")
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("quality-first benchmark source changed")
    if verify_clean_git_source(
        Path(freeze["mainline_root"])
    ) != freeze["mainline_source"]:
        raise ValueError("quality-first mainline source changed")


def report_slice(order, cuts, *, initial_rows: int, report_rows: int):
    """Carve the frozen initial and report index slices; fail on overlap."""
    inference = tuple(order[cuts[1]:cuts[2]])
    initial = inference[:initial_rows]
    report = inference[initial_rows:initial_rows + report_rows]
    if (
        not initial
        or len(report) != report_rows
        or set(initial) & set(report)
        or set(initial) | set(report) > set(inference)
    ):
        raise ValueError("report slice overlaps the initial data")
    return tuple(initial), tuple(report)


def _matrix_rows(dataset, indices):
    ordered = sorted(indices)
    rows = np.asarray(dataset[ordered, :])
    lookup = dict(zip(ordered, rows, strict=True))
    return np.asarray([lookup[index] for index in indices])


def ordered_roles(dataset, task: str, seed: int):
    """Mirror the DRR adapter split and add the untouched report slice."""
    values_count = dataset.shape[0]
    order = drr_adapter._ordered_indices(values_count, task, seed)
    cuts = (
        int(round(0.50 * values_count)),
        int(round(0.70 * values_count)),
        int(round(0.90 * values_count)),
    )
    initial_ids, report_ids = report_slice(
        order, cuts,
        initial_rows=32,
        report_rows=64,
    )
    action_ids = tuple(order[cuts[2]:])
    if set(action_ids) & (set(initial_ids) | set(report_ids)):
        raise ValueError("action segment overlaps initial or report rows")
    initial = _matrix_rows(dataset, initial_ids)
    report = _matrix_rows(dataset, report_ids)
    actions = _matrix_rows(dataset, action_ids)
    return {
        "X_initial": initial[:, 1:],
        "y_initial": initial[:, 0],
        "X_actions": actions[:, 1:],
        "X_report": report[:, 1:],
        "y_report": report[:, 0],
        "initial_row_indices": sorted(initial_ids),
        "report_row_indices": sorted(report_ids),
    }


def structural_support(expression: str, n_features: int) -> tuple[str, ...]:
    pcpi_adapter = drr_adapter._adapter
    return tuple(
        pcpi_adapter.structural_terms(str(expression), n_features)
    )


def adaptable_proposals(rows, n_features: int):
    """Split proposals into adaptable rows and recorded compile failures."""
    kept, failed = [], []
    for row in rows:
        try:
            structural_support(row["expression"], n_features)
        except Exception as error:  # noqa: BLE001 - recorded, never retried
            failed.append({
                "expression": str(row["expression"]),
                "reason": f"{type(error).__name__}",
            })
            continue
        kept.append(dict(row))
    return kept, failed


def control_candidates(no_llm_report, taken_supports, n_features: int,
                        limit: int):
    """Deterministically pick novel-support engine proposals as the control.

    Candidates come from the frozen No-LLM engine evidence only.  Their
    supports must not appear in the evaluated bank, the production pool, or
    the LLM proposals, so the control receives the same *new-candidate*
    evaluation opportunity without any provider involvement.
    """
    rows, seen, compile_failures = [], set(), 0
    for cycle in no_llm_report.get("scientist_agent_engine_reports", ()):
        for row in cycle.get("all_results", ()):
            expression = str(row.get("expression") or "")
            if not expression:
                compile_failures += 1
                continue
            try:
                support = structural_support(expression, n_features)
            except Exception:  # noqa: BLE001 - recorded, never retried
                compile_failures += 1
                continue
            if (support in taken_supports or support in seen):
                continue
            seen.add(support)
            rows.append({
                "expression": expression,
                "source": f"engine:{row.get('engine') or 'unknown'}",
                "origin": "deterministic",
                "lineage_id": str(row.get("lineage_id") or ""),
                "support": list(support),
            })
    rows.sort(key=lambda row: sha256(row["expression"].encode()).hexdigest())
    return rows[:limit], len(rows) - min(len(rows), max(limit, 0)), (
        compile_failures
    )


def score_arm(rows, roles, *, n_features: int, exploration_identity: str,
              measurement_budget: int = 2):
    """Run the production capacity bank and score it on the report rows."""
    initial = RoleDataset(
        DataRole.DEVELOPMENT,
        np.ascontiguousarray(roles["X_initial"], dtype=float),
        np.ascontiguousarray(roles["y_initial"], dtype=float),
    )
    actions = np.ascontiguousarray(roles["X_actions"], dtype=float)
    selected, selection = select_operational_capacity_bank(
        rows, initial, actions,
        n_features=n_features,
        prior=NormalInverseGammaPrior(),
        exploration_identity=exploration_identity,
        coefficient_policy=COEFFICIENT_POLICY,
        measurement_budget=measurement_budget,
        maximum_candidates=2 * measurement_budget,
        source_safety_roles=(),
        source_safety_folds=2,
        source_stacking_method=DIVERSITY_METHOD,
        selection_method=DECISION_RISK_CAPACITY_METHOD,
        exact_eig_epsabs=EXACT_CLASS_EIG_EPSABS,
    )
    model = freeze_discovery_model(
        selected,
        n_features=n_features,
        prior=NormalInverseGammaPrior(),
        exploration_identity=exploration_identity,
        coefficient_policy=COEFFICIENT_POLICY,
        source_prior_weights=selection["source_prior_weights"],
    )
    target = freeze_discovery_target(
        model, initial, actions,
        measurement_budget=measurement_budget,
        expected_model_identity=model.stable_hash,
    )
    engine = model.engine(model.stable_hash)
    posterior = target.initial_posterior
    X_report = np.ascontiguousarray(roles["X_report"], dtype=float)
    y_report = np.ascontiguousarray(roles["y_report"], dtype=float)
    predictions = np.zeros(len(y_report), dtype=float)
    for member in posterior.members:
        mean, _ = engine.predictive_moments(member, X_report)
        predictions = predictions + member.probability * np.asarray(
            mean, dtype=float)
    log_density = np.asarray(
        engine.predictive_logpdf(posterior, X_report, y_report), dtype=float)
    if (predictions.shape != y_report.shape
            or log_density.shape != y_report.shape
            or not np.all(np.isfinite(predictions))
            or not np.all(np.isfinite(log_density))):
        raise FloatingPointError("report predictive score is invalid")
    anchor = {}
    for row in rows:
        anchor.setdefault(
            " + ".join(structural_support(row["expression"], n_features)),
            row,
        )
    bank, family_mass = [], {}
    for member in posterior.members:
        expression = str(member.structure.expression)
        row = anchor.get(expression, {})
        entry = {
            "expression": expression,
            "source": str(row.get("source") or ""),
            "origin": str(row.get("origin") or ""),
            "posterior_probability": float(member.probability),
        }
        bank.append(entry)
        family = str(drr_adapter._stacking.source_family(row))
        family_mass[family] = family_mass.get(family, 0.0) + entry[
            "posterior_probability"]
    bank.sort(key=lambda entry: -entry["posterior_probability"])
    return {
        "report_mse": float(np.mean((predictions - y_report) ** 2)),
        "report_log_score": float(np.mean(log_density)),
        "predictive_mean": predictions.tolist(),
        "posterior_probability_sum": float(posterior.probability_sum),
        "capacity_bank": bank,
        "family_posterior_mass": dict(sorted(family_mass.items())),
        "source_prior_weights": {
            str(key): float(value)
            for key, value in selection["source_prior_weights"].items()
        },
        "selected_candidate_count": int(len(selected)),
        "posterior_member_count": int(len(posterior.members)),
        "decision_risk_utility": _plain(
            selection.get("decision_risk_utility") or {}),
        "selected_bank_identity": str(selection["target"]),
        "model_identity": str(model.stable_hash),
        "target_identity": str(target.stable_hash),
    }


def _baseline_pool(no_llm_report, n_features: int):
    pool = no_llm_report.get("drr_candidate_rows") or ()
    if pool:
        return [dict(row) for row in pool]
    protected = [
        candidate
        for row in no_llm_report.get(
            "scientist_agent_counterfactual_backbones", ())
        for candidate in row.get("backbone_candidates", ())
    ]
    return [
        dict(row)
        for row in drr_adapter.candidate_rows(
            no_llm_report,
            no_llm_report["best_expression"],
            n_features,
            protected,
        )
    ]


def pair_record(task: str, seed: int, full_report, no_llm_report, roles):
    """Build the three arms and the paired report-set comparison record."""
    failure = ""
    record = {"schema": "scientific-aistats-matsci-quality-first-pair-v1",
              "task": task, "seed": seed}
    try:
        n_features = int(
            no_llm_report["task_context_audit"]["variable_description_count"])
        if n_features != int(roles["X_initial"].shape[1]):
            raise ValueError("paired feature counts disagree")
        construction = build_quality_first_augmentation(
            full_report, no_llm_report, n_features=n_features,
            baseline_evaluations=48, llm_proposal_limit=12,
        )
        baseline_rows = _baseline_pool(no_llm_report, n_features)
        if not baseline_rows:
            raise ValueError("production candidate pool is empty")
        bank_supports = {
            tuple(structural_support(row["expression"], n_features))
            for row in no_llm_report["evaluated_hypothesis_bank"]
        }
        pool_supports = {
            tuple(structural_support(row["expression"], n_features))
            for row in baseline_rows
        }
        llm_rows, llm_failed = adaptable_proposals(
            construction["additional_llm_proposals"], n_features)
        llm_supports = {
            tuple(structural_support(row["expression"], n_features))
            for row in llm_rows
        }
        control_rows, control_overflow, control_failures = (
            control_candidates(
                no_llm_report,
                bank_supports | pool_supports | llm_supports,
                n_features,
                limit=len(llm_rows),
            )
        )
        llm_arm_rows = [dict(row) for row in baseline_rows] + [
            dict(row) for row in llm_rows
        ]
        control_arm_rows = [dict(row) for row in baseline_rows] + [
            dict(row) for row in control_rows
        ]
        if (llm_arm_rows[:len(baseline_rows)] != baseline_rows
                or control_arm_rows[:len(baseline_rows)] != baseline_rows):
            raise ValueError("augmented arm altered the frozen baseline rows")
        exploration_identity = sha256(
            f"{EXPLORATION_NAMESPACE}:{task}:{seed}".encode()
        ).hexdigest()
        arms = {}
        for name, rows in (
            ("baseline_48e", baseline_rows),
            ("llm_augmented_48e_plus_l", llm_arm_rows),
            ("control_augmented_48e_plus_l", control_arm_rows),
        ):
            arms[name] = score_arm(
                rows, roles, n_features=n_features,
                exploration_identity=exploration_identity,
            )
        audits = full_report.get("scientist_agent_synthesis_audits") or []
        record.update({
            "n_features": n_features,
            "llm_slots": {
                "compiled": int(construction["llm_compiled_proposal_count"]),
                "novel_offered": int(
                    construction["llm_novel_proposal_count"]),
                "duplicate_support_excluded": int(
                    len(construction["excluded_llm_proposals"])),
                "not_adaptable_excluded": int(len(llm_failed)),
                "not_adaptable_records": llm_failed,
                "entering_evaluation": int(len(llm_rows)),
                "limit": 12,
            },
            "control_slots": {
                "requested": int(len(llm_rows)),
                "selected": int(len(control_rows)),
                "shortfall": int(len(llm_rows) - len(control_rows)),
                "novel_available_beyond_limit": int(control_overflow),
                "engine_compile_failures": int(control_failures),
            },
            "accounting": {
                "provider_abstention_count": int(len(
                    full_report.get("scientist_agent_provider_abstentions")
                    or ())),
                "rejected_candidate_count": int(
                    full_report.get("rejected_candidate_count") or 0),
                "synthesis_directive_total": int(sum(
                    int(audit.get("directive_count") or 0)
                    for audit in audits)),
                "synthesis_rejected_directive_total": int(sum(
                    int(audit.get("rejected_directive_count") or 0)
                    for audit in audits)),
                "synthesis_unavailable_cycle_count": int(sum(
                    bool(audit.get("synthesis_unavailable"))
                    for audit in audits)),
                "full_logical_llm_call_count": int(
                    full_report.get(
                        "scientist_agent_logical_llm_call_count") or 0),
                "full_evaluation_budget_used": int(
                    full_report.get("evaluation_budget_used") or 0),
                "no_llm_evaluation_budget_used": int(
                    no_llm_report.get("evaluation_budget_used") or 0),
            },
            "arms": arms,
            "hard_checks": {
                "engine_evidence_identical": bool(
                    construction["engine_evidence_identical"]),
                "no_llm_completed_registered_baseline_evaluations":
                    int(no_llm_report.get("evaluation_budget_used") or 0)
                    == 48,
                "full_completed_registered_baseline_evaluations":
                    int(full_report.get("evaluation_budget_used") or 0)
                    == 48,
                "augmented_arms_preserve_frozen_baseline_rows": True,
                "llm_proposals_displaced_no_baseline_candidate": False,
                "report_slice_disjoint_from_initial": not set(
                    roles["initial_row_indices"]) & set(
                        roles["report_row_indices"]),
            },
            "paired": {
                "llm_minus_baseline_log_score": float(
                    arms["llm_augmented_48e_plus_l"]["report_log_score"]
                    - arms["baseline_48e"]["report_log_score"]),
                "llm_minus_baseline_mse": float(
                    arms["llm_augmented_48e_plus_l"]["report_mse"]
                    - arms["baseline_48e"]["report_mse"]),
                "control_minus_baseline_log_score": float(
                    arms["control_augmented_48e_plus_l"]["report_log_score"]
                    - arms["baseline_48e"]["report_log_score"]),
                "control_minus_baseline_mse": float(
                    arms["control_augmented_48e_plus_l"]["report_mse"]
                    - arms["baseline_48e"]["report_mse"]),
                "llm_minus_control_log_score": float(
                    arms["llm_augmented_48e_plus_l"]["report_log_score"]
                    - arms["control_augmented_48e_plus_l"][
                        "report_log_score"]),
            },
            "report_slice": {
                "initial_row_count": int(
                    len(roles["initial_row_indices"])),
                "report_row_count": int(len(roles["report_row_indices"])),
                "report_matrix_sha256": sha256(
                    np.ascontiguousarray(np.column_stack([
                        roles["X_report"], roles["y_report"],
                    ])).tobytes()
                ).hexdigest(),
            },
        })
    except Exception as error:  # noqa: BLE001 - recorded, never retried
        failure = f"{type(error).__name__}:{error}"
    record["status"] = "success" if not failure else "failure"
    record["failure"] = failure
    return _plain(record)


def aggregate(pair_rows, freeze):
    by = {(row["task"], row["seed"]): row for row in pair_rows}
    tasks = freeze["selected_tasks"]["matsci"]
    seeds = freeze["seeds"]
    effects_llm, effects_control, effects_mse = {}, {}, {}
    for task in tasks:
        deltas, control, mse = [], [], []
        for seed in seeds:
            row = by[(task, seed)]
            if row["status"] != "success":
                raise ValueError(
                    f"aggregation requires successful pairs: {task}/{seed}")
            deltas.append(row["paired"]["llm_minus_baseline_log_score"])
            control.append(
                row["paired"]["control_minus_baseline_log_score"])
            mse.append(row["paired"]["llm_minus_baseline_mse"])
        effects_llm[task] = float(np.mean(deltas))
        effects_control[task] = float(np.mean(control))
        effects_mse[task] = float(np.mean(mse))
    global_llm = float(np.mean(list(effects_llm.values())))
    global_control = float(np.mean(list(effects_control.values())))
    shortfall = sum(
        1 for row in pair_rows
        if not (row.get("hard_checks") or {}).get(
            "full_completed_registered_baseline_evaluations", True))
    tolerance = freeze["increment_decision"]["numerical_tolerance"]
    minimum_positive = int(
        freeze["increment_decision"]["minimum_strictly_positive_tasks"])
    decisions = {
        "all_pre_registered_pairs_accounted_for": len(pair_rows) == len(
            tasks) * len(seeds),
        "zero_pair_failures": all(
            row["status"] == "success" for row in pair_rows),
        "global_llm_log_score_gain_strictly_positive":
            global_llm > tolerance,
        f"minimum_{minimum_positive}_tasks_strictly_positive": sum(
            value > tolerance for value in effects_llm.values()
        ) >= minimum_positive,
        "llm_gain_exceeds_equal_size_control_gain":
            global_llm - global_control > tolerance,
    }
    return {
        "task_effects_llm_minus_baseline_log_score": effects_llm,
        "task_effects_control_minus_baseline_log_score": effects_control,
        "task_effects_llm_minus_baseline_mse": effects_mse,
        "global_llm_log_score_gain": global_llm,
        "global_control_log_score_gain": global_control,
        "global_llm_minus_control_log_score": global_llm - global_control,
        "full_budget_shortfall_pair_count": int(shortfall),
        "decisions": decisions,
    }


def _registration(freeze):
    return {
        "benchmark_data_root": freeze["benchmark_data_root"],
        "benchmark_cache_root": freeze["benchmark_cache_root"],
        "mainline_root": freeze["mainline_root"],
        "provider_env": freeze["provider_env"],
    }


def _protocol(freeze):
    return {"budget": {
        "evaluated_hypotheses_per_condition_task_seed":
            freeze["discovery_budget"],
        "wall_time_seconds_per_condition_task_seed":
            freeze["wall_time_seconds_per_discovery"],
        "maximum_llm_calls_per_condition_task_seed":
            freeze["maximum_llm_calls_per_discovery"],
        "maximum_provider_attempts_per_condition_task_seed":
            freeze["maximum_provider_attempts_per_discovery"],
    }}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    freeze = _read(args.freeze)
    _verify(freeze)
    if args.output_dir.exists() and not args.resume:
        raise ValueError("quality-first increment output must be new")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    plan = []
    for task in freeze["selected_tasks"]["matsci"]:
        for seed in freeze["seeds"]:
            for condition in freeze["conditions"]:
                plan.append({
                    "task": task, "seed": seed, "condition": condition,
                    "run_dir": str((
                        args.output_dir / "runs" / task /
                        f"seed{seed}" / condition
                    ).resolve()),
                })
    plan_path = args.output_dir / "PLAN.json"
    plan_text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if plan_path.is_file() and plan_path.read_text("utf-8") != plan_text:
        raise ValueError("quality-first plan identity changed")
    plan_path.write_text(plan_text, encoding="utf-8")
    if args.preflight_only:
        preflight = {
            "schema": "scientific-aistats-matsci-quality-first-preflight-v1",
            "passed": True,
            "planned_discovery_runs": len(plan),
            "planned_pairs": len(plan) // len(freeze["conditions"]),
            "task_arrays_accessed": False,
            "candidate_response_accessed": False,
            "llm_called": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
            "execution_started": False,
        }
        (args.output_dir / "PREFLIGHT.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    registration, protocol = _registration(freeze), _protocol(freeze)
    for index, item in enumerate(plan, start=1):
        spec = RunSpec(
            "matsci", "matsci", item["task"], int(item["seed"]),
            item["condition"], Path(item["run_dir"]),
        )
        if args.resume and (rows := _resume_rows(spec)) is not None:
            print(f"[{index}/{len(plan)}] resume-preserved "
                  f"{item['condition']} matsci/{item['task']} "
                  f"seed={item['seed']}", flush=True)
            continue
        print(f"[{index}/{len(plan)}] {item['condition']} "
              f"matsci/{item['task']} seed={item['seed']}", flush=True)
        _run_one(spec, registration, protocol)

    import h5py

    pairs_path = args.output_dir / "QUALITY_FIRST_PAIRS.json"
    pair_rows = _read(pairs_path) if pairs_path.is_file() else []
    completed = {(row["task"], row["seed"]) for row in pair_rows}
    with h5py.File(freeze["hdf5"]["path"], "r") as handle:
        for task in freeze["selected_tasks"]["matsci"]:
            for seed in freeze["seeds"]:
                if (task, seed) in completed:
                    continue
                full_path = (args.output_dir / "runs" / task /
                              f"seed{seed}" / "full_scientist_v6" /
                              "pcpi_artifacts" / f"{task}.json")
                no_path = (args.output_dir / "runs" / task /
                           f"seed{seed}" / "no_llm_v6" /
                           "pcpi_artifacts" / f"{task}.json")
                full_report = _read(full_path)[
                    "scientific_discovery_runtime"]
                no_llm_report = _read(no_path)[
                    "scientific_discovery_runtime"]
                dataset = handle[f"lsr_synth/matsci/{task}/train"]
                roles = ordered_roles(dataset, task, seed)
                row = pair_record(
                    task, seed, full_report, no_llm_report, roles)
                pair_rows.append(row)
                pairs_path.write_text(
                    json.dumps(pair_rows, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    analysis = aggregate(pair_rows, freeze)
    decisions = analysis["decisions"]
    result = {
        "schema": "scientific-aistats-matsci-quality-first-increment-v1",
        "passed": all(decisions.values()),
        **analysis,
        "pair_failure_count": sum(
            row["status"] != "success" for row in pair_rows),
        "pair_row_count": len(pair_rows),
        "rows": pair_rows,
        "primary_metric": freeze["primary_metric"],
        "protocol_mode": freeze.get("protocol_mode", "fresh-formal"),
        "independent_tasks": bool(freeze.get("independent_tasks", True)),
        "candidate_response_accessed": True,
        "report_response_accessed_for_scoring_only": True,
        "action_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": all(decisions.values()),
        "claim_boundary": freeze["claim_boundary"],
    }
    path = args.output_dir / "AISTATS_MATSCI_QUALITY_FIRST_INCREMENT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
