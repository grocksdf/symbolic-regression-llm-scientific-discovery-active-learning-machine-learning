"""Explicit Bank/residual/blind generation comparison; no experiment on import.

The generation signal is computed on fitting rows only. Candidate ranking
uses existing search crossfit scores. All arms finish and freeze selection
before independent evaluation is allowed. Historical CB-SI runners are intact.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from hypothesis_mvp.data.roles import DataRole
from hypothesis_mvp.discovery.bank_search_feedback import feedback_from_posterior, rank_materials
from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
    INVENTION_OPERATORS, StructuralEditingEngine,
)
from hypothesis_mvp.discovery.joint_evidence_gated_expansion import extract_subexpression_materials
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model
from hypothesis_mvp.discovery.search_budget import SearchBudget, SearchBudgetTracker, validate_comparison_budgets
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import compile_fixed_formula_support
from scripts.frozen_predictive_expansion import _publish, _posterior_id


CONTRACT = "bank-location-guided-search-v1"
MODES = ("bank", "residual", "blind")


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _manifest(selection, left, right, row_ids):
    if set(row_ids) != {"fit", "left", "right", "evaluation"}:
        raise ValueError("complete generation/search/evaluation row manifest required")
    seen = set()
    for name, ids in row_ids.items():
        if (not ids or len(ids) != len(set(ids)) or seen.intersection(ids)
                or any(not isinstance(i, str) or not i for i in ids)):
            raise ValueError("source row identities must be unique and disjoint")
        seen.update(ids)
    for name, data, role in (("fit", selection.development, DataRole.DEVELOPMENT),
                             ("left", left, DataRole.VALIDATION),
                             ("right", right, DataRole.VALIDATION)):
        if data.role is not role or len(data.X) != len(row_ids[name]):
            raise ValueError("registered role identity/count mismatch")
    visible = (selection.development, left, right)
    if any(a.row_fingerprints & b.row_fingerprints for i, a in enumerate(visible) for b in visible[i + 1:]):
        raise ValueError("opened generation/search observations overlap")


def prepare_generation(E0, archive, selection, *, row_ids, forbidden_row_ids):
    """Fit E0 once; construct dense feedback and executable donor pool."""
    fit = selection.development
    model = freeze_expanded_formula_model(
        E0, n_features=fit.X.shape[1], prior=NormalInverseGammaPrior(),
        exploration_identity=_hash(E0), coefficient_policy=EXPANDED_FORMULA_POLICY,
        minimum_supports=1)
    engine = model.engine(model.stable_hash)
    posterior = engine.fit_batch(fit.X, fit.y)
    bank_id = _posterior_id(model, posterior, _hash({"rows": row_ids, "data": fit.fingerprint}))
    feedback = feedback_from_posterior(engine, posterior, fit, search_row_ids=row_ids,
                                       forbidden_row_ids=forbidden_row_ids, bank_identity=bank_id)
    editor = StructuralEditingEngine(fit.X.shape[1])
    materials = {row["expression"] for row in extract_subexpression_materials(archive)}
    materials.update(f"x{i}" for i in range(fit.X.shape[1]))
    values, failures = {}, []
    for expression in sorted(materials):
        try:
            pred = np.asarray(editor.runtime.predict(expression, fit.X), dtype=float)
            if pred.shape != fit.y.shape or not np.all(np.isfinite(pred)):
                raise ValueError("nonfinite material")
            values[expression] = pred
        except (ValueError, ArithmeticError, TypeError) as error:
            failures.append({"expression": expression, "error_type": type(error).__name__})
    parents = []
    for candidate in sorted(archive.candidates, key=lambda c: c.structural_hash):
        paths = tuple(editor.editable_paths(candidate.expression))
        if paths:
            parents.append((candidate, paths))
    if not parents or not values:
        raise ValueError("no executable parent/material for generation")
    return feedback, values, parents, failures


def generate_arm(mode, budget, *, feedback, material_values, parents, archive,
                 E0, selection, left, right, evaluation, seed, namespace, operators,
                 record_callback=None):
    """All attempts are counted; learning a sparse operator policy is unnecessary.

    Parent and AST-site coverage are deterministic. Least-attempted feasible
    operators are chosen first. Donor rankings change with the registered
    information condition; round-robin donor coverage prevents one material
    monopolizing a long quality-first search. No outcome-dependent retries.
    """
    from scripts.run_cbsi_zero_provider_experiment import CrossfitBankScorer

    if mode not in MODES:
        raise ValueError("unknown generation information condition")
    started = time.perf_counter()
    tracker = SearchBudgetTracker(budget)
    ranked = rank_materials(feedback, material_values, mode=mode)
    if mode == "blind":
        ranked.sort(key=lambda row: _hash([seed, row["expression"]]))
    editor = StructuralEditingEngine(selection.development.X.shape[1])
    scorer = CrossfitBankScorer(E0, selection, left, right, evaluation, namespace, seed)
    scorer_setup_wall_seconds = time.perf_counter() - started
    seen_s, seen_c, seen_support = set(), set(), set()
    for candidate in archive.candidates:
        dag = editor.runtime.dag(candidate.expression)
        seen_s.add(dag.structural_hash)
        seen_c.add(dag.canonical_hash)
        seen_support.add(_hash(tuple(compile_fixed_formula_support(candidate.expression, editor.n_features))))
    domains = (selection.development.X, left.X, right.X, selection.acquisition_pool.X)
    records, selected, usages = [], None, dict.fromkeys(operators, 0)
    counter_names = ("density_evaluations", "candidate_density_evaluations", "optimizer_fits")
    setup_compute = {name: getattr(scorer, name) for name in counter_names}
    while tracker.can_attempt:
        attempt_started = time.perf_counter()
        before_compute = {name: getattr(scorer, name) for name in counter_names}
        index = len(records)
        parent, paths = parents[index % len(parents)]
        site = paths[(index // len(parents)) % len(paths)]
        # Separate donor rounds from parent coverage to avoid permanently
        # coupling one parent to one donor when cardinalities share a divisor.
        donor_round = (index // len(parents)) // len(paths)
        donor = ranked[donor_round % len(ranked)]
        opportunity_seed = seed + index
        allowed = editor.applicable_operators(parent.expression, donor["expression"], site,
                                              seed=opportunity_seed, operators=operators,
                                              known_support_hashes=seen_support)
        record = {"attempt": index, "parent": parent.expression, "site_path": list(site),
                  "donor": donor["expression"], "donor_alignment": donor["alignment"],
                  "feedback_identity": feedback.identity, "eligible_operators": list(allowed)}
        work = {"operator_preflights": len(operators), "materializations": 0}
        def measured_work():
            return {**work, **{name: getattr(scorer, name) - before_compute[name] for name in counter_names},
                    "wall_time_seconds": time.perf_counter() - attempt_started}
        if not allowed:
            tracker.record_attempt(attempt_id=str(index), outcome="invalid", failure_reason="no_eligible_operator", work=measured_work())
            record.update(status="invalid", reason="no_eligible_operator")
            records.append(record)
            if record_callback is not None:
                record_callback(record)
            continue
        operator = min(allowed, key=lambda op: (usages[op], _hash([seed, index, op])))
        usages[operator] += 1
        request = editor.deterministic_request(operator, parent.expression, donor["expression"],
                                               seed=opportunity_seed, site_path=site,
                                               task_identity=f"{namespace}:{index}")
        result = editor.apply(request, registered_domains=domains,
                              known_structural_hashes=seen_s, known_canonical_hashes=seen_c,
                              known_support_hashes=seen_support)
        work["materializations"] = 1
        record.update(request=asdict(request), candidate=result.row())
        if not result.valid or not result.novel:
            tracker.record_attempt(attempt_id=str(index), outcome="invalid",
                                   failure_reason=result.failure_reason or "duplicate_support", work=measured_work())
            record.update(status="invalid", reason=result.failure_reason or "duplicate_support")
        else:
            seen_s.add(result.structural_hash)
            seen_c.add(result.canonical_hash)
            seen_support.add(result.support_hash)
            try:
                gains = np.asarray(scorer.crossfit_gain(result.expression), dtype=float)
                if gains.shape != (2,) or not np.all(np.isfinite(gains)):
                    raise FloatingPointError("two finite search fold gains required")
                gain = float(np.sum(gains / 2))
                if not math.isfinite(gain):
                    raise FloatingPointError("nonfinite search mean")
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                tracker.record_attempt(attempt_id=str(index), outcome="failed",
                                       failure_reason=type(error).__name__, work=measured_work())
                record.update(status="failed", reason=type(error).__name__)
            else:
                tracker.record_attempt(attempt_id=str(index), outcome="evaluated", search_gain=gain, work=measured_work())
                record.update(status="evaluated", search_gain=gain, fold_gains=gains.tolist())
                if gain > 0 and (selected is None or (gain, result.structural_hash) >
                                (selected["gain"], selected["candidate"]["structural_hash"])):
                    selected = {"gain": gain, "candidate": result.row()}
        records.append(record)
        if record_callback is not None:
            record_callback(record)
    expression = selected["candidate"]["expression"] if selected else None
    selection_id = scorer.freeze_selection(expression)
    audit = {"schema": CONTRACT, "mode": mode, "budget": tracker.audit(),
             "selection_identity": selection_id, "selected": selected,
             "feedback_identity": feedback.identity, "bank_identity": feedback.bank_identity,
             "material_ranking": ranked, "operator_attempts": usages, "records": records,
             "generation_wall_seconds": time.perf_counter() - started,
             "search_compute": {key: getattr(scorer, key) for key in
                                ("density_evaluations", "optimizer_fits", "score_calls")},
             "scorer_setup_compute": setup_compute,
             "scorer_setup_wall_seconds": scorer_setup_wall_seconds,
             "reporting_used_for_generation": False, "provider_calls": 0}
    return scorer, expression, audit


class _ClosedEvaluation:
    def __init__(self):
        self._data = None

    @property
    def X(self):
        if self._data is None:
            raise ValueError("evaluation remains closed before all generation freezes")
        return self._data.X

    @property
    def y(self):
        if self._data is None:
            raise ValueError("evaluation remains closed before all generation freezes")
        return self._data.y


def run_guided_comparison(*, implementation_contract, purpose, budgets, E0, archive,
                          selection, left, right, row_ids, read_evaluation, seed,
                          output_root, operators=INVENTION_OPERATORS):
    """Explicit callable harness; caller supplies opened development roles only.

    This is not a matrix registration or authorization to read confirmation.
    New source/protocol freeze required; old CB-SI execution flags do not apply.
    """
    if implementation_contract != CONTRACT or set(budgets) != set(MODES):
        raise ValueError("explicit Bank-guidance contract and all three conditions required")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative integer")
    operators = tuple(operators)
    if (not operators or len(set(operators)) != len(operators)
            or any(op not in INVENTION_OPERATORS for op in operators)):
        raise ValueError("registered semantic operator set required")
    comparison = validate_comparison_budgets(purpose=purpose, arm_budgets=budgets)
    _manifest(selection, left, right, row_ids)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    stage = "generation"
    try:
        _publish(root / "PLAN.json", {"contract": CONTRACT, "comparison": comparison,
                 "row_ids": row_ids, "seed": seed, "operators": operators,
                 "feedback_rows": "fit-only", "candidate_selection_rows": ["left", "right"],
                 "data_role": "opened-development-only"})
        preparation_started = time.perf_counter()
        feedback, values, parents, failures = prepare_generation(
            E0, archive, selection, row_ids=row_ids["fit"],
            forbidden_row_ids=(*row_ids["left"], *row_ids["right"], *row_ids["evaluation"]))
        common_preparation = {"wall_time_seconds": time.perf_counter() - preparation_started,
                              "bank_model_fits": 1, "material_prediction_evaluations": len(values) + len(failures)}
        _publish(root / "GENERATION_FEEDBACK.json", {"bank": feedback.prompt_payload("bank"),
                 "residual": feedback.prompt_payload("residual"), "invalid_materials": failures,
                 "common_preparation_compute": common_preparation})
        evaluation = _ClosedEvaluation()
        generated = {}
        for mode in MODES:
            with (root / f"{mode}_ATTEMPTS.jsonl").open("x", encoding="utf-8") as trace:
                def record_attempt(record):
                    trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    trace.flush()
                    os.fsync(trace.fileno())
                scorer, expression, audit = generate_arm(
                    mode, budgets[mode], feedback=feedback, material_values=values, parents=parents,
                    archive=archive, E0=E0, selection=selection, left=left, right=right,
                    evaluation=evaluation, seed=seed, namespace="shared-bank-guidance-scorer",
                    operators=operators, record_callback=record_attempt)
            generated[mode] = (scorer, expression, audit)
            _publish(root / f"{mode}_GENERATION.json", audit)
        _publish(root / "GENERATION_FROZEN.json", {mode: _hash(value[2]) for mode, value in generated.items()})
        stage = "evaluation"
        ids, data = read_evaluation()
        if (tuple(ids) != tuple(row_ids["evaluation"]) or len(ids) != len(data.X)
                or data.role is not DataRole.VALIDATION):
            raise ValueError("independent development evaluation identity/role mismatch")
        if any(data.row_fingerprints & role.row_fingerprints for role in (selection.development, left, right)):
            raise ValueError("independent evaluation overlaps opened search observations")
        evaluation._data = data
        results = {}
        for mode, (scorer, expression, audit) in generated.items():
            evaluation_started = time.perf_counter()
            gain, weights, optimizer = scorer.evaluation_gain(expression) if expression else (0., [1.], None)
            results[mode] = {"gain": gain, "weights": weights, "optimizer": optimizer,
                             "selection_identity": audit["selection_identity"], "budget": audit["budget"],
                             "density_evaluations": scorer.density_evaluations,
                             "optimizer_fits": scorer.optimizer_fits,
                             "generation_wall_seconds": audit["generation_wall_seconds"],
                             "evaluation_wall_seconds": time.perf_counter() - evaluation_started}
        result = {"schema": CONTRACT, "results": results, "comparison": comparison,
                  "common_preparation_compute": common_preparation,
                  "protocol_complete": True, "provider_calls": 0, "pysr_calls": 0,
                  "claim_boundary": "Development Bank-location guidance comparison only. "
                                    "Local derivative is a proposal surrogate, not an expansion certificate. "
                                    "Quality mode does not isolate equal-compute mechanism effects."}
        _publish(root / "RESULT.json", result)
        return result
    except Exception as error:
        _publish(root / "FAILURE.json", {"stage": stage, "error_type": type(error).__name__,
                 "message": str(error), "protocol_complete": False, "retry_authorized": False})
        raise
