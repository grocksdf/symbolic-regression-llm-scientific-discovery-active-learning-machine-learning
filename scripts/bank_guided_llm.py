"""Opt-in LLM Bank composition with unchanged independent mixture admission.

This callable adapter is not an experiment launcher. It reads no dataset or
credential file. All modes share E0, grammar, parents, request slots and data
roles. Only the precomputed generation feedback differs. No result-based
prompt repair, threshold change, memory sharing or new scientific authority
is granted to the provider.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import re
import time
from urllib.parse import urlsplit

import numpy as np

from hypothesis_mvp.data.roles import DataRole
from hypothesis_mvp.discovery.bank_diagnostic_context import validate_diagnostic_axes
from hypothesis_mvp.discovery.bank_search_feedback import attach_feedback, feedback_from_posterior
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model
from hypothesis_mvp.discovery.predictive_bank_expansion import (
    ExpansionDesign, PredictiveExpansionSession, digest,
)
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime, ProviderSettings
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.reference.basis import design_matrix
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import compile_fixed_formula_support
from scripts.frozen_predictive_expansion import (
    FrozenBankComponents, _posterior_id, _publish, NESTED_BANK_POLICY, DIRECT_CANDIDATE_POLICY,
)
from scripts.bank_composition_preflight import validate_generation_ancestry
from scripts.stacking_diagnostics import fit_stacking_weights


CONTRACT = "llm-bank-composition-v1"
MODES = ("bank", "residual", "blind")
MULTISCALE_CONTRACT = "llm-bank-multiscale-composition-v1"
DIRECT_CONTRACT = "llm-bank-multiscale-direct-v1"
MULTISCALE_MODES = ("blind", "global", "local", "joint")


@dataclass(frozen=True)
class LLMCompositionBudget:
    calls_per_arm: int
    candidates_per_call: int
    new_skeleton_quota: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in
               (self.calls_per_arm, self.candidates_per_call)):
            raise ValueError("positive integer request/candidate ceilings required")
        if (type(self.new_skeleton_quota) is not int
                or not 0 <= self.new_skeleton_quota <= self.candidates_per_call):
            raise ValueError("new skeleton quota must fit the common candidate ceiling")


def _provider_identity(settings):
    if not isinstance(settings, ProviderSettings) or len(settings.routes) != 1:
        raise ValueError("one explicit shared provider route required")
    route = settings.routes[0]
    route.validate()
    url = urlsplit(route.endpoint)
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("provider endpoint must not contain embedded credentials or query parameters")
    if (type(settings.attempts) is not int or settings.attempts <= 0
            or type(settings.max_tokens) is not int or settings.max_tokens <= 0):
        raise ValueError("finite registered provider request/token limits required")
    for name in ("connect_timeout_s", "read_timeout_s", "retry_backoff_s",
                 "rate_limit_backoff_s", "min_request_interval_s", "temperature"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid provider time/sampling policy")
    if settings.connect_timeout_s == 0 or settings.read_timeout_s == 0:
        raise ValueError("provider timeouts must be positive")
    # Never serialize/hash the credential itself. A formal runner must bind
    # the authorized credential file outside this low-level adapter.
    public = {k: v for k, v in settings.__dict__.items() if k != "routes"}
    public["route"] = {k: v for k, v in route.__dict__.items() if k != "api_key"}
    return public


def _public_transport(runtime):
    """Preserve attempt accounting without persisting provider error bodies."""
    rows = []
    for logical in runtime.telemetry:
        for attempt in logical.get("provider_outcomes", ()):
            status = attempt.get("provider_http_status")
            error = attempt.get("provider_error", "")
            if status is None and error:
                match = re.search(r'"provider_http_status":\s*(\d+)', error)
                status = int(match.group(1)) if match else None
            row = {"http_status": status,
                   "failed": bool(error) or status is None or not 200 <= status < 300,
                   "prompt_hash": logical.get("prompt_hash"),
                   "attempt": attempt.get("provider_attempt_index"),
                   "elapsed_seconds": attempt.get("provider_elapsed_s")}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = attempt.get("provider_" + key)
                row[key] = value if type(value) is int and value >= 0 else None
            rows.append(row)
    return rows


def _cost(runtime, elapsed):
    attempts = _public_transport(runtime)
    result = {"logical_calls": runtime.call_count, "provider_attempts": runtime.attempt_count,
              "provider_failures": sum(row["failed"] for row in attempts),
              "transport_records": attempts, "wall_seconds": elapsed}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [row[key] for row in attempts]
        result[key] = sum(values) if values and all(v is not None for v in values) else None
        result[key + "_known_subtotal"] = sum(v for v in values if v is not None)
    return result


def _generate(mode, runtime, feedback, core, fit, budget, root, *,
              implementation_contract=CONTRACT, diagnostic_scope=None):
    started = time.perf_counter()
    n_features = fit.X.shape[1]
    supports = tuple(tuple(compile_fixed_formula_support(r["expression"], n_features)) for r in core)
    seen = set(supports)
    candidates, records = [], []
    parent_rows = sorted(core, key=lambda r: (digest(r["expression"]), r["expression"]))
    context_base = {
        "expanded_formula_discovery": True,
        "new_skeleton_quota": budget.new_skeleton_quota,
        "posterior_gap_existing_supports": supports,
        "proposal_domain_context": {"source": "fitting-covariates-only",
            "feature_ranges": [{"symbol": f"x{i}", "min": float(fit.X[:, i].min()),
                                "max": float(fit.X[:, i].max())} for i in range(n_features)]},
    }
    root.mkdir(exist_ok=False)
    try:
        for slot in range(budget.calls_per_arm):
            parent = parent_rows[slot % len(parent_rows)]["expression"]
            parent_hash = runtime.equation_runtime.dag(parent).canonical_hash
            context = {**context_base, "current_expression": parent}
            if diagnostic_scope is None:
                context = attach_feedback(context, feedback, mode=mode)
            else:
                signal_mode = "blind" if mode == "blind" else "bank"
                context = attach_feedback(context, feedback, mode=signal_mode,
                                          diagnostic_scope=diagnostic_scope)
            envelope_options = ({"envelope_policy": "request-bound-semantic-v1"}
                                if implementation_contract == DIRECT_CONTRACT else {})
            batch = runtime.propose(
                task_name="registered-symbolic-discovery", task_desc="Propose complementary symbolic structures.",
                round_id=slot, island="composition", parent_hash=parent_hash,
                island_context=context, library_rows=(), ephemeral_refinements=(),
                max_candidates=budget.candidates_per_call, allow_protocol_repair=False, **envelope_options)
            record = {"slot": slot, "parent": parent, "parent_hash": parent_hash,
                      "feedback": context["bank_predictive_search_feedback"],
                      "protocol_valid": batch.protocol_valid, "protocol_reason": batch.reason,
                      "prompt_hash": batch.prompt_hash, "response_hash": batch.response_hash,
                      "parsed_candidate_count": len(batch.candidates), "candidates": []}
            for candidate in batch.candidates:
                row = {"expression": candidate.equation, "source": "llm:bank-composition",
                       "lineage_id": candidate.lineage_id, "action": candidate.action,
                       "prompt_hash": candidate.prompt_hash, "response_hash": candidate.response_hash}
                try:
                    support = tuple(compile_fixed_formula_support(candidate.equation, n_features))
                    design_matrix(fit.X, support)
                except (ValueError, ArithmeticError) as error:
                    record["candidates"].append({**row, "status": "invalid-on-fit-domain",
                                                 "error_type": type(error).__name__})
                    continue
                if support in seen:
                    record["candidates"].append({**row, "status": "duplicate-support"})
                    continue
                seen.add(support)
                candidates.append(row)
                record["candidates"].append({**row, "status": "materialized-novel"})
            # Typed parser rejection is a generation outcome, never imputed
            # to an admission gain. Infrastructure failure is terminal.
            record["rejected_candidates"] = batch.telemetry.get("candidate_validation_rejections", [])
            if envelope_options:
                record["envelope_policy"] = envelope_options["envelope_policy"]
                record["request_bindings"] = [request["request_binding"]
                    for request in batch.telemetry.get("provider_requests", [])
                    if "request_binding" in request]
            record["cost_so_far"] = _cost(runtime, time.perf_counter() - started)
            _publish(root / f"CALL_{slot:04d}.json", record)
            records.append(record)
            transport = record["cost_so_far"]["transport_records"]
            if (batch.reason == "provider_or_protocol_failure" or not transport
                    or transport[-1]["failed"]):
                raise RuntimeError("LLM provider/protocol failed; see preserved request audit")
        audit = {"schema": implementation_contract, "mode": mode, "candidates": candidates,
                 "records": records, "cost": _cost(runtime, time.perf_counter() - started),
                 "new_skeleton_quota": budget.new_skeleton_quota,
                 "provider_repair_calls": 0, "memory_reads": 0, "memory_writes": 0,
                 "admission_used_for_generation": False}
        if diagnostic_scope is not None:
            audit["diagnostic_scope"] = diagnostic_scope
        _publish(root / "GENERATION.json", audit)
        return candidates, audit
    except Exception as error:
        _publish(root / "FAILURE.json", {"error_type": type(error).__name__,
                 "cost": _cost(runtime, time.perf_counter() - started), "retry_authorized": False})
        raise


def run_llm_comparison(*, implementation_contract, budget, design, core, fit,
                       provider_settings, read_calibration, read_admission,
                       read_reporting, output_root, seed, diagnostic_axes=None,
                       fit_row_ids=None, core_generation_row_ids=None):
    """Generate -> freeze ALL banks -> calibrate ALL -> admit ALL -> report.

    Supply opened development fitting data only. No historical archive, truth,
    free-form task hints, source dataset or environment is read by this code.
    A formal runner still needs task/provenance/provider authorization freezes.
    """
    if implementation_contract not in {CONTRACT, MULTISCALE_CONTRACT, DIRECT_CONTRACT} or not isinstance(budget, LLMCompositionBudget):
        raise ValueError("explicit LLM composition contract/budget required")
    direct = implementation_contract == DIRECT_CONTRACT
    multiscale = implementation_contract in {MULTISCALE_CONTRACT, DIRECT_CONTRACT}
    modes = MULTISCALE_MODES if multiscale else MODES
    scopes = {"blind": "none", "global": "global", "local": "local", "joint": "joint"}
    if not isinstance(design, ExpansionDesign) or design.tested_mixture_count < len(modes):
        raise ValueError("register every tested mixture in the evidence policy")
    if (fit.role is not DataRole.DEVELOPMENT
            or type(seed) is not int or seed < 0 or not core):
        raise ValueError("registered fitting role, initial Bank and seed required")
    ancestry = None
    if direct:
        ancestry = validate_generation_ancestry(design, fit_row_ids, core_generation_row_ids, len(fit.X))
    else:
        if fit_row_ids is not None or core_generation_row_ids is not None or len(fit.X) != len(design.training_ids):
            raise ValueError("legacy fitting IDs must equal the registered training role")
        fit_row_ids = design.training_ids
    component_policy = DIRECT_CANDIDATE_POLICY if direct else NESTED_BANK_POLICY
    if multiscale:
        validate_diagnostic_axes(diagnostic_axes, fit.X.shape[1])
    elif diagnostic_axes is not None:
        raise ValueError("diagnostic axes require the explicit multiscale contract")
    provider = _provider_identity(provider_settings)
    core = json.loads(json.dumps(core, allow_nan=False))
    # Resolve structural validity before any provider call.
    for row in core:
        design_matrix(fit.X, compile_fixed_formula_support(row["expression"], fit.X.shape[1]))
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    stage = "generation"
    stage_started = time.perf_counter()
    phase_wall_seconds = {}
    try:
        plan = {"contract": implementation_contract, "budget": asdict(budget),
                 "design": asdict(design), "provider": provider, "core_identity": digest(core),
                 "fitting_identity": fit.fingerprint, "seed": seed, "modes": modes,
                 "actual_compute_matching_is_performance_gate": False,
                 "provider_seed_guaranteed": False, "historical_results_reused": False}
        if diagnostic_axes is not None:
            plan.update(diagnostic_axes=diagnostic_axes, diagnostic_scopes=scopes,
                        arm_execution_order=modes)
        if direct:
            plan.update(component_policy=component_policy, generation_ancestry=ancestry,
                        envelope_policy="request-bound-semantic-v1")
        _publish(root / "PLAN.json", plan)
        model = freeze_expanded_formula_model(
            core, n_features=fit.X.shape[1], prior=NormalInverseGammaPrior(),
            exploration_identity=digest(core), coefficient_policy=EXPANDED_FORMULA_POLICY,
            minimum_supports=1)
        engine = model.engine(model.stable_hash)
        posterior = engine.fit_batch(fit.X, fit.y)
        bank_id = _posterior_id(model, posterior, fit.fingerprint)
        feedback_options = {"include_bank_inventory": True}
        if diagnostic_axes is not None:
            feedback_options["diagnostic_axes"] = diagnostic_axes
        feedback = feedback_from_posterior(
            engine, posterior, fit, search_row_ids=fit_row_ids,
            forbidden_row_ids=(*design.calibration_ids, *design.admission_ids, *design.reporting_ids),
            bank_identity=bank_id, **feedback_options)
        generated, costs = {}, {}
        # New runtime per mode: no inherited counters, disabled route or memory.
        for mode in modes:
            runtime = ProposalRuntime(EquationRuntime(fit.X.shape[1], refit_policy="pcpi-expanded-fixed-inner-v1"),
                                      fit.X.shape[1], provider_settings, budget.candidates_per_call)
            scope = scopes[mode] if diagnostic_axes is not None else None
            candidates, audit = _generate(mode, runtime, feedback, core, fit, budget, root / mode,
                                          implementation_contract=implementation_contract,
                                          diagnostic_scope=scope)
            generated[mode] = candidates
            costs[mode] = audit["cost"]
        _publish(root / "GENERATION_FROZEN.json", {"bank_identity": bank_id,
                 "candidate_banks": {mode: {"identity": digest(rows), "count": len(rows)}
                                     for mode, rows in generated.items()},
                 "generation_feedback_identity": feedback.identity})
        phase_wall_seconds[stage] = time.perf_counter() - stage_started
        stage = "component_fitting"
        stage_started = time.perf_counter()
        components = {mode: FrozenBankComponents(core, rows, fit, fit_row_ids=fit_row_ids,
                                                 design=design, component_policy=component_policy)
                      for mode, rows in generated.items()}
        sessions = {mode: PredictiveExpansionSession(design, value.component_ids)
                    for mode, value in components.items()}
        _publish(root / "COMPONENTS_FROZEN.json", {mode: value.component_ids for mode, value in components.items()})
        phase_wall_seconds[stage] = time.perf_counter() - stage_started
        visible = [fit]

        def read_role(reader, name):
            ids, data = reader()
            if (tuple(ids) != getattr(design, name + "_ids") or len(data.X) != len(ids)
                    or data.role is not DataRole.VALIDATION
                    or any(data.row_fingerprints & old.row_fingerprints for old in visible)):
                raise ValueError("independent development row/role identity or overlap violation")
            visible.append(data)
            return ids, data

        stage = "calibration"
        stage_started = time.perf_counter()
        ids, data = read_role(read_calibration, stage)
        frozen = {}
        for mode, component in components.items():
            density = component.densities(data)
            weights, optimizer = fit_stacking_weights(density, seed=seed, random_starts=4)
            frozen[mode] = {**sessions[mode].freeze(density, weights, row_ids=ids,
                            component_ids=component.component_ids), "optimizer": optimizer}
            if direct:
                frozen[mode]["component_diagnostics"] = _density_diagnostics(density)
        _publish(root / "CALIBRATION_FROZEN.json", frozen)
        phase_wall_seconds[stage] = time.perf_counter() - stage_started
        stage = "admission"
        stage_started = time.perf_counter()
        ids, data = read_role(read_admission, stage)
        decisions = {mode: sessions[mode].admit(component.densities(data), row_ids=ids,
                         component_ids=component.component_ids) for mode, component in components.items()}
        _publish(root / "ADMISSION_FROZEN.json", decisions)
        phase_wall_seconds[stage] = time.perf_counter() - stage_started
        stage = "reporting"
        stage_started = time.perf_counter()
        ids, data = read_role(read_reporting, stage)
        results = {}
        for mode, component in components.items():
            density = component.densities(data)
            results[mode] = sessions[mode].report(density, row_ids=ids, component_ids=component.component_ids)
            if direct:
                results[mode]["component_diagnostics"] = _density_diagnostics(density)
        phase_wall_seconds[stage] = time.perf_counter() - stage_started
        result = {"schema": implementation_contract, "protocol_complete": True, "results": results,
                  "phase_wall_seconds": phase_wall_seconds,
                  "cost": costs, "candidate_counts": {m: len(v) for m, v in generated.items()},
                  "component_counts": {m: len(v.component_ids) for m, v in components.items()},
                  "density_evaluations": {m: v.evaluation_count for m, v in components.items()},
                  "statistical_core_changed": False, "independent_efficacy_established": False,
                  "claim_boundary": "Opened-development Bank-conditioned LLM generation comparison. "
                    "No innovation, universal risk improvement or LLM reasoning attribution is established."}
        if direct:
            result.pop("statistical_core_changed")
            result.update(component_policy=component_policy, predictive_component_family_changed=True,
                          posterior_solver_changed=False, admission_rule_changed=False,
                          weights_are_posterior_probabilities=False)
        _publish(root / "RESULT.json", result)
        return result
    except Exception as error:
        _publish(root / "FAILURE.json", {"stage": stage, "error_type": type(error).__name__,
                 "protocol_complete": False, "retry_authorized": False})
        raise


def _density_diagnostics(density):
    """Per-component audit only; never passed to generation or admission."""
    with np.errstate(over="raise", invalid="raise"):
        differences = density - density[:, :1]
        means = np.sum(differences / len(differences), axis=0)
        maximum = np.max(np.abs(differences), axis=0)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(maximum)):
        raise FloatingPointError("nonfinite component diagnostic")
    return {"mean_log_score_minus_core": means.tolist(),
            "maximum_absolute_log_density_difference": maximum.tolist(),
            "component_order_matches_frozen_ids": True, "audit_only": True}
