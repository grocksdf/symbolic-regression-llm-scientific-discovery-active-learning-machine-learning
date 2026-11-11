"""Shared LLM/program plan pool -> selection -> unchanged independent admission.

An opt-in callable adapter, not a new experiment authorization or a resume path.
All proposal collection precedes selection. Each policy has its own search state
and scores, while both consume the same immutable pool and evaluation budget.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from hypothesis_mvp.data.roles import DataRole
from hypothesis_mvp.discovery.bank_search_feedback import feedback_from_posterior
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model
from hypothesis_mvp.discovery.predictive_bank_expansion import PredictiveExpansionSession
from hypothesis_mvp.discovery.predictive_plan_policy import PredictivePlanPolicy, PlanAction, PlanOutcome
from hypothesis_mvp.discovery.region_site_attribution import digest
from hypothesis_mvp.discovery.structural_plan_pool import freeze_plan_pool, candidate_domain_status
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from scripts.bank_guided_search import _manifest, _ClosedEvaluation
from scripts.cbsi_paired_search import search_scoring_identity
from scripts.frozen_predictive_expansion import FrozenBankComponents, DIRECT_CANDIDATE_POLICY, _posterior_id, _publish
from scripts.regional_structural_search import DirectCrossfitBankScorer
from scripts.stacking_diagnostics import fit_stacking_weights


CONTRACT = "shared-plan-paired-bayesian-selection-v1"
MODES = ("random", "thompson")


def run_predictive_plan_comparison(*, implementation_contract, E0, plans, proposal_audits,
        selection, left, right, row_ids, generation_ancestry_ids, design, regions,
        task_identity, seed, blocks_per_arm, read_calibration, read_admission, read_reporting,
        output_root, frozen_prior=None):
    """Freeze once; use real typed proposals or deterministic program proposals.

    No LLM/provider is called here. Use structural_plan_pool's production
    ProposalBatch adapter after collection. LLM cost belongs to the shared
    proposal stage; it is not counted as free or charged twice across arms.
    """
    if (implementation_contract != CONTRACT or type(seed) is not int or seed < 0
            or type(blocks_per_arm) is not int or blocks_per_arm <= 0
            or not task_identity or design.tested_mixture_count < len(MODES)):
        raise ValueError("explicit plan contract, budgets, identity and multiplicity required")
    _manifest(selection, left, right, {**row_ids, "evaluation": design.reporting_ids})
    visible_ids = tuple(row_ids["fit"]) + tuple(row_ids["left"]) + tuple(row_ids["right"])
    if (not generation_ancestry_ids or not set(visible_ids).issubset(design.training_ids)
            or not set(generation_ancestry_ids).issubset(design.training_ids)):
        raise ValueError("all upstream/adaptive rows must be registered training")
    plans = tuple(plans)
    if not plans: raise ValueError("empty proposal pool")
    proposal_audits = json.loads(json.dumps(proposal_audits, allow_nan=False))
    audit_ids = [p for a in proposal_audits for p in a["plan_ids"]]
    llm_plans = [p for p in plans if p.source == "llm"]
    if sorted(audit_ids) != sorted(p.identity for p in llm_plans):
        raise ValueError("each LLM plan needs exactly one preserved proposal audit")
    for plan in llm_plans:
        audit = next(a for a in proposal_audits if plan.identity in a["plan_ids"])
        if (audit["batch_prompt_hash"] != plan.prompt_hash or audit["batch_response_hash"] != plan.response_hash
                or audit.get("provider_attempts", 0) < 1 or audit.get("logical_calls", 0) < 1):
            raise ValueError("LLM source identity/cost mismatch")
    # Snapshot caller-owned mutable inputs before generating any scores.
    E0 = json.loads(json.dumps(E0, allow_nan=False))
    regions = json.loads(json.dumps(regions, allow_nan=False))
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    stage, started = "pool", time.perf_counter()
    try:
        forbidden = design.calibration_ids + design.admission_ids + design.reporting_ids
        _publish(root / "PLAN.json", {"schema": CONTRACT, "design": asdict(design),
            "row_ids": row_ids, "generation_ancestry_ids": generation_ancestry_ids,
            "task_identity": task_identity, "seed": seed, "blocks_per_arm": blocks_per_arm,
            "scoring_slots_per_arm": blocks_per_arm * 2, "modes": MODES,
            "regions": regions, "frontier_enabled": False,
            "reward": "standalone-predictive-stacking-crossfit-logscore-v1",
            "no_expansion_semantics": "invalid/duplicate candidate executes KEEP_E0",
            "infrastructure_failure_is_zero": False,
            "proposal_audits": proposal_audits, "code_performs_provider_calls": False,
            "prior_identity": frozen_prior.get("identity") if frozen_prior else None})
        fit = selection.development
        model = freeze_expanded_formula_model(E0, n_features=fit.X.shape[1], prior=NormalInverseGammaPrior(),
            exploration_identity=digest(E0), coefficient_policy=EXPANDED_FORMULA_POLICY, minimum_supports=1)
        engine = model.engine(model.stable_hash)
        posterior = engine.fit_batch(fit.X, fit.y)
        bank_id = _posterior_id(model, posterior, digest({"data": fit.fingerprint, "rows": row_ids["fit"]}))
        feedback = feedback_from_posterior(engine, posterior, fit, search_row_ids=row_ids["fit"],
                                          forbidden_row_ids=forbidden, bank_identity=bank_id)
        locations = np.column_stack([engine.predictive_location_scale(member, fit.X)[0]
                                     for member in posterior.members])
        pool = freeze_plan_pool(plans, fit=fit, fit_row_ids=row_ids["fit"], forbidden_row_ids=forbidden,
            feedback=feedback, bank_locations=locations, core_rows=E0, regions=regions, task_identity=task_identity)
        _publish(root / "PROPOSAL_POOL_FROZEN.json", pool)
        scoring_id = search_scoring_identity(E0, selection, left, right, seed,
            {"operators": sorted({p.operation for p in plans}),
             "budget": {"contract": CONTRACT, "pool": pool["identity"], "blocks_per_arm": blocks_per_arm,
                        "no_expansion_semantics": "KEEP_E0", "component_policy": DIRECT_CANDIDATE_POLICY}})
        actions = tuple(PlanAction(r["plan_id"], pool["context_identity"], tuple(r["features"])) for r in pool["rows"])
        by_id = {r["plan_id"]: r for r in pool["rows"]}
        domains = (fit.X, left.X, right.X, selection.acquisition_pool.X)
        generated = {}
        stage = "search"
        for mode in MODES:
            policy = PredictivePlanPolicy(seed=seed, task_identity=task_identity, bank_identity=bank_id,
                scoring_identity=scoring_id, feature_context_identity=pool["context_identity"], frozen_prior=frozen_prior)
            scorer = DirectCrossfitBankScorer(E0, selection, left, right, _ClosedEvaluation(), scoring_id, seed)
            candidates, records, used = {}, [], set()
            with (root / f"{mode}_ATTEMPTS.jsonl").open("x", encoding="utf-8") as trace:
                for block in range(blocks_per_arm):
                    pair = policy.choose_pair(block, [a for a in actions if a.plan_id not in used], mode=mode)
                    if pair is None:
                        break  # Finite pool exhaustion; no replacement proposals/retries.
                    outcomes, slots = [], []
                    for action in pair:
                        row = by_id[action.plan_id]
                        expression = row["plan"]["expression"]
                        if row["core_duplicate"]:
                            outcome = PlanOutcome(action.plan_id, "duplicate", reason="same_fixed_support_as_core")
                        else:
                            defined, reason = candidate_domain_status(expression, n_features=fit.X.shape[1], domains=domains)
                            if not defined:
                                reason_code = {"fixed formula undefined on registered domain": "formula_undefined_on_registered_domain",
                                               "nonfinite fixed formula design column": "nonfinite_formula_design"}[reason]
                                outcome = PlanOutcome(action.plan_id, "invalid", reason=reason_code)
                            else:
                                # Any scorer/fit/optimizer exception is terminal, never zero-imputed.
                                gains = tuple(float(v) for v in scorer.crossfit_gain(expression))
                                outcome = PlanOutcome(action.plan_id, "scored", gains)
                                candidates[row["support_id"]] = {"expression": expression,
                                    "source": "plan:" + row["plan"]["source"], "lineage_id": row["plan"]["lineage_id"]}
                        outcomes.append(outcome)
                        slots.append({"plan_id": action.plan_id, "outcome": asdict(outcome)})
                    evidence_id = digest([scoring_id, mode, block, slots])
                    policy.observe_pair(block, primary_outcome=outcomes[0], control_outcome=outcomes[1],
                                        evidence_id=evidence_id, scoring_identity=scoring_id)
                    used.update(a.plan_id for a in pair)
                    record = {"block": block, "slots": slots, "evidence_id": evidence_id}
                    records.append(record)
                    trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    trace.flush()
                    os.fsync(trace.fileno())
            generated[mode] = {"pool_identity": pool["identity"], "candidates": list(candidates.values()),
                "records": records, "policy": policy.audit(), "requested_slots": blocks_per_arm * 2,
                "executed_slots": len(used), "pool_exhausted": len(actions) - len(used) < 2,
                "density_evaluations": scorer.density_evaluations, "optimizer_fits": scorer.optimizer_fits,
                "model_fits": scorer.model_fits, "component_policy": DIRECT_CANDIDATE_POLICY}
            _publish(root / f"{mode}_GENERATION.json", generated[mode])
        _publish(root / "GENERATION_FROZEN.json", {m: digest(a) for m, a in generated.items()})
        # Search/selection has ended. Independent responses never train the policy.
        stage = "component_fit"
        components = {m: FrozenBankComponents(E0, a["candidates"], fit,
            fit_row_ids=row_ids["fit"], design=design, component_policy=DIRECT_CANDIDATE_POLICY) for m, a in generated.items()}
        sessions = {m: PredictiveExpansionSession(design, c.component_ids) for m, c in components.items()}
        visible = [fit, left, right]
        def read_role(reader, name):
            ids, data = reader()
            if (tuple(ids) != getattr(design, name + "_ids") or len(ids) != len(data.X)
                    or data.role is not DataRole.VALIDATION
                    or any(data.row_fingerprints & d.row_fingerprints for d in visible)):
                raise ValueError("independent role identity/overlap mismatch")
            visible.append(data)
            return ids, data
        stage = "calibration"
        ids, data = read_role(read_calibration, stage)
        for mode, component in components.items():
            values = component.densities(data)
            weights, optimizer = fit_stacking_weights(values, seed=seed, random_starts=4)
            frozen = sessions[mode].freeze(values, weights, row_ids=ids, component_ids=component.component_ids)
            _publish(root / f"{mode}_CALIBRATION_FROZEN.json", {**frozen, "optimizer": optimizer})
        stage = "admission"
        ids, data = read_role(read_admission, stage)
        for mode, component in components.items():
            admission = sessions[mode].admit(component.densities(data), row_ids=ids, component_ids=component.component_ids)
            _publish(root / f"{mode}_ADMISSION_FROZEN.json", admission)
        stage = "reporting"
        ids, data = read_role(read_reporting, stage)
        reports = {m: sessions[m].report(c.densities(data), row_ids=ids, component_ids=c.component_ids)
                   for m, c in components.items()}
        result = {"schema": CONTRACT, "reports": reports, "protocol_complete": True,
            "pool_identity": pool["identity"], "pool_shared": True, "feature_dimension": 8,
            "candidate_scoring_slots_equal": generated["random"]["executed_slots"] == generated["thompson"]["executed_slots"],
            "pool_smaller_than_budget": len(actions) < blocks_per_arm * 2,
            "frontier_enabled": False, "provider_calls_in_selector": 0,
            "shared_proposal_logical_calls": sum(a["logical_calls"] for a in proposal_audits),
            "wall_seconds": time.perf_counter() - started,
            "claim_boundary": "Conditional shared-pool selection adapter; correctness is not efficacy or calibrated utility evidence."}
        _publish(root / "RESULT.json", result)
        return result
    except Exception as error:
        _publish(root / "FAILURE.json", {"stage": stage, "type": type(error).__name__, "message": str(error),
            "protocol_complete": False, "retry_authorized": False})
        raise
