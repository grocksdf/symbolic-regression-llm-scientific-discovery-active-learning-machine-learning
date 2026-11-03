"""Static paired-edit search adapters; no experiment execution on import."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
import time

import numpy as np


def _array_identity(value):
    array = np.asarray(value)
    if array.dtype.hasobject or not np.all(np.isfinite(array)):
        raise ValueError("search identity requires finite numerical arrays")
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def search_scoring_identity(E0, selection, left, right, seed, freeze):
    """Bind the actual scorer inputs, deliberately accepting no evaluation role."""
    value = {
        "schema": "cbsi-paired-search-scoring-v1", "E0": E0,
        "roles": {name: {"X": _array_identity(role.X),
                         "y": _array_identity(role.y)}
                  for name, role in (("fit", selection.development),
                                     ("left", left), ("right", right))},
        "pool_X": _array_identity(selection.acquisition_pool.X),
        "n_features": int(selection.development.X.shape[1]),
        "scorer_seed": int(seed), "operators": list(freeze["operators"]),
        "budget": dict(freeze["budget"]),
    }
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _enforce_compute_ceiling(scorer, ceilings):
    for name, maximum in ceilings.items():
        if hasattr(scorer, name) and getattr(scorer, name) > maximum:
            raise ValueError(f"paired search compute ceiling exceeded: {name}")


def run_paired_arm(arm, freeze, coordinate, archive, E0, selection,
                   left, right, evaluation, context, domains,
                   shared_opportunities):
    """Two materialization slots per opportunity; both candidates can win.

    Matched contrasts train once per identified pair. The greedy comparator
    consumes the same scored candidate stream and observes both raw gains.
    Invalid or missing candidates never become numerical zero observations.
    """
    import run_cbsi_zero_provider_experiment as runner
    from hypothesis_mvp.discovery.paired_edit_policy import (
        PairedEditPolicy,
    )

    started = time.perf_counter()
    if arm not in {"cbsi_paired", "bank_greedy_paired"}:
        raise ValueError("unknown paired search arm")
    evaluations = freeze["budget"]["edit_evaluations_per_arm"]
    if type(evaluations) is not int or evaluations <= 0 or evaluations % 2:
        raise ValueError("paired search budget must be positive and even")
    if len(shared_opportunities) != evaluations // 2:
        raise ValueError("paired search needs one frozen opportunity per block")
    if float(freeze["budget"].get("complexity_cost_weight", 0)) != 0:
        raise ValueError("paired search models unpenalized bank gain")
    operators = tuple(op for op in freeze["operators"]
                      if op in runner.INVENTION_OPERATORS)
    seed = int(coordinate["seed"]) + int(sha256(arm.encode()).hexdigest()[:8], 16)
    scorer_seed = int(coordinate["seed"]) + int(sha256(
        f"{coordinate['family']}:{coordinate['task']}:shared-scorer".encode()
    ).hexdigest()[:8], 16)
    scoring_identity = search_scoring_identity(
        E0, selection, left, right, scorer_seed, freeze)
    editor = runner.StructuralEditingEngine(selection.development.X.shape[1])
    scorer = runner.CrossfitBankScorer(
        E0, selection, left, right, evaluation,
        f"{coordinate['task']}/{coordinate['seed']}/shared-scorer", scorer_seed)
    paired = arm == "cbsi_paired"
    policy = (PairedEditPolicy(operators=operators, seed=seed,
                              scoring_identity=scoring_identity) if paired
              else runner.BayesianOperatorPolicy(operators=operators, seed=seed))
    seen_structures, seen_canonicals, seen_supports = set(), set(), set()
    for original in archive.candidates:
        dag = editor.runtime.dag(original.expression)
        seen_structures.add(dag.structural_hash)
        seen_canonicals.add(dag.canonical_hash)
        support = tuple(runner.compile_fixed_formula_support(
            original.expression, editor.n_features))
        seen_supports.add(sha256(json.dumps(
            support, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    records, eligible = [], []
    for block, opportunity in enumerate(shared_opportunities):
        parent, donor = opportunity["parent"], opportunity["donor"]
        site = opportunity["site_path"]
        site_context = runner.FeasibilityAwareContextualPolicy.site_context(
            parent.expression, site)
        allowed = editor.applicable_operators(
            parent.expression, donor, site, seed=opportunity["opportunity_seed"],
            operators=operators, known_support_hashes=seen_supports)
        record = {
            "block": block, "opportunity_id": opportunity["opportunity_id"],
            "shared_opportunity": True, "used_frontier_parent": False,
            "site_context": site_context, "planning_eligible_operators": list(allowed),
            "planning_no_eligible_action": not bool(allowed),
            "primary": None, "control": None,
            "primary_request": None, "control_request": None,
            "primary_request_identity": None, "control_request_identity": None,
            "control_matched_primary_identity": None,
            "primary_fold_gains": None, "control_fold_gains": None,
            "primary_scoring_failure": None, "control_scoring_failure": None,
            "control_planning_failure": None, "pair_identity_failure": None,
            "counterfactual_pair_valid": False, "credit": None,
            "paired_evidence_id": None,
            "counterfactual_pair_exclusion_reason": "no_eligible_action",
            "scoring_identity": scoring_identity,
            "parent_generation": int(getattr(parent, "generation", 0)),
            "candidate_generation": int(getattr(parent, "generation", 0)) + 1,
        }
        if not allowed:
            records.append(record)
            continue
        if paired:
            operator = policy.choose(block, context=site_context,
                                     eligible_operators=allowed)
        else:
            operator = min(allowed, key=lambda op: (
                -policy.arms[op].posterior_mean(), op))
            policy.selections.append(operator)
        primary_request = editor.deterministic_request(
            operator, parent.expression, donor,
            seed=opportunity["opportunity_seed"], site_path=site,
            task_identity=f"{coordinate['task']}:{coordinate['seed']}:"
                          f"{opportunity['opportunity_id']}:{block}")
        control_request = None
        try:
            control_request = editor.matched_control_request(primary_request)
            if control_request.operator not in operators:
                raise ValueError("matched control is outside declared operator set")
        except ValueError as error:
            record["control_planning_failure"] = str(error)
            control_request = None
        results, folds = {}, {}
        for role, request in (("primary", primary_request),
                              ("control", control_request)):
            if request is None:
                continue
            record[f"{role}_request"] = asdict(request)
            record[f"{role}_request_identity"] = request.identity
            result = editor.apply(
                request, registered_domains=domains,
                known_structural_hashes=seen_structures,
                known_canonical_hashes=seen_canonicals,
                known_support_hashes=seen_supports)
            results[role] = result
            record[role] = result.row()
            if not result.valid or not result.novel:
                continue
            try:
                values = np.asarray(scorer.crossfit_gain(str(result.expression)), dtype=float)
                if values.shape != (2,) or not np.all(np.isfinite(values)):
                    raise ValueError("paired search requires two finite fold gains")
                gain = float(np.mean(values))
                if not math.isfinite(gain):
                    raise ValueError("paired search mean gain is nonfinite")
            except (ValueError, FloatingPointError, ArithmeticError) as error:
                record[f"{role}_scoring_failure"] = f"{type(error).__name__}:{error}"
                continue
            folds[role] = tuple(float(v) for v in values)
            record[f"{role}_fold_gains"] = list(folds[role])
            eligible.append({"result": result, "search_gain": gain, "role": role,
                             "block": block})
            if not paired and result.operator in policy.arms:
                policy.update(result.operator, gain, reward_kind="raw_bank_gain",
                              audit={"block": block, "role": role})
        # Both members see the same pre-block novelty set. Publish both after
        # resolving the pair, including a usable control after primary failure.
        for result in results.values():
            if result.valid and result.novel:
                seen_structures.add(str(result.structural_hash))
                seen_canonicals.add(str(result.canonical_hash))
                seen_supports.add(str(result.support_hash))
        record["counterfactual_pair_exclusion_reason"] = "missing_scored_pair"
        if control_request is not None:
            record["control_matched_primary_identity"] = control_request.matched_control_of
        if "primary" in folds and "control" in folds:
            try:
                runner.validate_matched_edit_pair(
                    primary_request, control_request, results["primary"], results["control"])
            except ValueError as error:
                record["pair_identity_failure"] = str(error)
                record["counterfactual_pair_exclusion_reason"] = "invalid_pair_identity"
            else:
                record["counterfactual_pair_valid"] = True
                record["counterfactual_pair_exclusion_reason"] = None
                record["paired_evidence_id"] = (
                    f"{scoring_identity}:{opportunity['opportunity_id']}:{block}")
                if paired:
                    policy.observe_pair(
                        primary_request=primary_request, control_request=control_request,
                        primary_result=results["primary"], control_result=results["control"],
                        primary_folds=folds["primary"], control_folds=folds["control"],
                        context=site_context,
                        evidence_id=record["paired_evidence_id"],
                        scoring_identity=scoring_identity)
        records.append(record)
    search_ceilings = {"candidate_density_evaluations": 2 * evaluations,
                       "density_evaluations": 2 * evaluations + 2,
                       "optimizer_fits": 2 * evaluations}
    _enforce_compute_ceiling(scorer, search_ceilings)
    selected = max(eligible, key=lambda row: (
        row["search_gain"], str(row["result"].structural_hash))) if eligible else None
    if selected is not None and selected["search_gain"] <= 0:
        selected = None
    expression = str(selected["result"].expression) if selected else None
    scorer.freeze_selection(expression)
    if selected:
        evaluation_gain, weights, optimizer = scorer.evaluation_gain(expression)
    else:
        evaluation_gain, weights, optimizer = 0.0, [1.0], {"component_count": 1}
    ceilings = {"candidate_density_evaluations": 2 * evaluations + 1,
                "density_evaluations": 2 * evaluations + 4,
                "optimizer_fits": 2 * evaluations + 1}
    _enforce_compute_ceiling(scorer, ceilings)
    return {
        "arm": arm, "mode": "paired_contrast" if paired else "paired_greedy",
        "implementation_contract": runner.IMPLEMENTATION_CONTRACT,
        "shared_scorer_seed": scorer_seed, "scoring_identity": scoring_identity,
        "edit_evaluation_budget": evaluations,
        "primary_attempt_count": len(records),
        "actual_materialization_attempts": sum(
            int(r["primary"] is not None) + int(r["control"] is not None)
            for r in records),
        "reserved_primary_slots": len(records), "reserved_control_slots": len(records),
        "attempted_control_count": sum(r["control"] is not None for r in records),
        "materialized_control_count": sum(r["control"] is not None for r in records),
        "abstention_count": sum(r["planning_no_eligible_action"] for r in records),
        "matched_control_count": sum(r["counterfactual_pair_valid"] for r in records),
        "invalid_pair_count": sum(not r["counterfactual_pair_valid"] for r in records),
        "shared_opportunities": True,
        "valid_primary_count": sum(bool(r["primary"] and r["primary"]["valid"]) for r in records),
        "novel_primary_count": sum(bool(r["primary"] and r["primary"]["novel"]) for r in records),
        "valid_control_count": sum(bool(r["control"] and r["control"]["valid"]) for r in records),
        "frontier_parent_reuses": 0, "frontier_max_observed_depth": 0,
        "max_attempted_candidate_generation": max((r["candidate_generation"] for r in records), default=0),
        "final_frontier_size": 0,
        "selected": selected["result"].row() if selected else None,
        "selected_candidate_role": selected["role"] if selected else None,
        "selected_block": selected["block"] if selected else None,
        "selected_search_gain": selected["search_gain"] if selected else 0.0,
        "selected_counterfactual_credit": None,
        "independent_development_evaluation_gain": evaluation_gain,
        "evaluation_stacking_weights": weights, "evaluation_optimizer": optimizer,
        "score_calls": scorer.score_calls,
        "compute": {name: getattr(scorer, name, 0) for name in (
            "density_evaluations", "baseline_density_evaluations",
            "candidate_density_evaluations", "optimizer_fits", "cache_hits")},
        "search_compute_ceilings": search_ceilings, "compute_ceilings": ceilings,
        "selection_identity": getattr(scorer, "selection_identity", None),
        "operator_preflight_count": len(operators) * len(records),
        "wall_time_seconds": time.perf_counter() - started,
        "provider_calls": 0, "pysr_calls": 0,
        "policy": policy.audit(), "records": records,
    }
