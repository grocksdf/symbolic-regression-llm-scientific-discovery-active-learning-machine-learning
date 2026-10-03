"""Matched exploration ablations; no measured-pool or held-out access.

Uses the production DiscoveryAgent, not a parallel discovery implementation.
Interrupted unfinished exploration is not silently repeated: a started marker
blocks replay of provider/engine calls. Completed variants may be reused only
under the byte-identical experiment contract.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time
import math
import numpy as np

from .agent import DiscoveryAgent, DiscoveryAgentConfig
from .system_run import analyze_system_contract
from .resource_limits import run_bounded
from hypothesis_mvp.pcpi.discovery_transaction import _publish
from hypothesis_mvp.symbolic.registry import registered_engine_names
from .pcpi_adapter import (
    freeze_discovery_model, freeze_discovery_target, structural_terms,
)
from .initializer import generic_deterministic_candidates
from .source_stacking import source_family
from .regional_candidate_admission import admit_regional_candidates
from .regional_decision_audit import (
    FrozenFiniteActionReference, audit_admitted_candidate_action,
    _covariate_rows,
)
from .candidate_region_expansion import FrozenAxisRegions
from .common_class_projection import freeze_common_class_projection
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.hypotheses import EvidenceEventType, EvidenceRegistry


class ExplorationProtocolError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.public_diagnostic = str(code)


def audit_usage(config, result, elapsed, compute_ceiling, provider_attempt_ceiling):
    jobs = sum(len(c.engine_report["run_records"]) for c in result.cycles)
    evaluations = sum(c.candidate_evaluations for c in result.cycles)
    attempts = sum(c.provider_attempts for c in result.cycles)
    if (not math.isfinite(elapsed) or elapsed < 0
            or any(type(v) is not int or v < 0 for v in (jobs, evaluations, attempts))):
        raise ValueError("invalid measured exploration usage")
    actual_cycles = len(result.cycles)
    if actual_cycles < 1 or actual_cycles > config.cycles:
        raise ValueError("invalid exploration cycle count")
    if actual_cycles < config.cycles:
        raise ValueError("unregistered early exploration termination")
    expected_jobs = config.engine_budget * actual_cycles
    if (jobs != expected_jobs or any(c.engine_report["failures"] for c in result.cycles)
            or any(r["status"] != "succeeded" for c in result.cycles
                   for r in c.engine_report["run_records"])):
        raise ValueError("incomplete or failed engine execution")
    if (evaluations > config.discovery_budget * actual_cycles
            or attempts > provider_attempt_ceiling or elapsed > compute_ceiling):
        raise ValueError("registered exploration ceiling exceeded")
    return {"engine_jobs_used": jobs, "candidate_evaluations_used": evaluations,
            "provider_attempts_used": attempts, "elapsed_seconds": elapsed,
            "cycles_completed": actual_cycles,
            "scientist_stop_requested": any(
                getattr(c, "scientist_review", {}).get("stop") is True
                for c in result.cycles)}


def _run_variant(config, provider_settings, selection, workspace, compute_ceiling,
                 provider_attempt_ceiling, scientific_context,
                 gap_audit=None, gap_prior=None, gap_measurement_budget=0):
    start = time.monotonic()
    agent = DiscoveryAgent(config, provider_settings)
    context = dict(scientific_context)
    result = agent.run(selection=selection, task_name=context["task_name"],
        task_description=context["task_description"], output_dir=workspace,
        knowledge_dir=workspace / "knowledge", variable_metadata=context,
        gap_audit=gap_audit, gap_prior=gap_prior,
        gap_measurement_budget=gap_measurement_budget)
    usage = audit_usage(config, result, time.monotonic() - start,
                        compute_ceiling, provider_attempt_ceiling)
    report = result.discovery.report
    if report.get("llm_error_count", 0) or any(
            getattr(c, "provider_errors", 0) for c in result.cycles):
        raise ExplorationProtocolError(
            "provider-or-protocol-failure-blocks-exploration")
    gap_abstained = (config.posterior_gap_directed
        and result.cycles
        and all(not getattr(c, "posterior_gap_brief", {}).get(
            "propose_allowed", False) for c in result.cycles))
    if provider_settings is not None and not gap_abstained and (
            sum(c.provider_calls for c in result.cycles) < 1
            or sum(c.provider_attempts for c in result.cycles) < 1):
        raise ExplorationProtocolError("llm-enabled-exploration-had-zero-provider-attempts")
    # Retain every candidate that is both discovery-valid and representable by
    # the frozen PCPI closed basis.  The latter is a response-free protocol
    # compatibility check, not an efficacy/result filter; doing it here keeps
    # malformed LLM structures from failing much later during bank freezing.
    topk = report.get("evaluated_hypothesis_bank", report.get("final_topk", []))
    if any(not str(row.get("source", "")).strip() for row in topk):
        raise ExplorationProtocolError("retained-hypothesis-missing-source-provenance")
    candidates = []
    adapter_rejections = []
    n_features = int(selection.development.X.shape[1])
    for row in topk:
        candidate = {"expression": row["expression"], "source": str(row["source"]),
                   "origin": str(row.get("origin", "unknown")),
                   "lineage_id": str(row.get("lineage_id", ""))}
        try:
            structural_terms(candidate["expression"], n_features)
        except Exception as error:
            adapter_rejections.append({
                "expression": candidate["expression"],
                "source": candidate["source"],
                "origin": candidate["origin"],
                "error_type": type(error).__name__,
                "error": str(error),
            })
            continue
        candidates.append(candidate)
    if not candidates:
        raise ExplorationProtocolError("no-pcpi-adaptable-hypotheses-retained")
    provenance = {
        "schema": "scientific-hypothesis-provenance-audit-v1",
        "candidate_count": len(candidates),
        "distinct_sources": sorted({row["source"] for row in candidates}),
        "origin_counts": {origin: sum(row["origin"] == origin for row in candidates)
                          for origin in sorted({row["origin"] for row in candidates})},
        "llm_retained_candidate_count": sum(row["origin"] == "llm" for row in candidates),
        "engine_sources": sorted({row["source"] for row in candidates
                                  if row["source"].startswith("engine:")}),
        "all_candidates_source_bound": bool(candidates),
        "pcpi_adapter_rejections": adapter_rejections,
        "raw_engine_candidates": [record for cycle in result.cycles
                                  for record in cycle.engine_report.get("all_results", [])],
        "discovery_candidate_registry": report.get("candidate_registry", []),
        "evaluation_rejections": report.get("rejected_candidates", []),
        "llm_candidate_lifecycle": report.get("llm_rounds", []),
        "heldout_accessed": False,
    }
    if config.posterior_gap_directed:
        provenance["gap_knowledge_stage_ids"] = list(
            report.get("knowledge_stage_ids", ()))
    policy_trace = [{
        "cycle": getattr(cycle, "cycle", index),
        "research_plan": dict(getattr(cycle, "research_plan", {})),
        "scientist_review": dict(getattr(cycle, "scientist_review", {})),
        "scientist_state_before": dict(getattr(
            cycle, "scientist_state_before", {})),
        "scientist_state_after": dict(getattr(
            cycle, "scientist_state_after", {})),
        "engine_allocations": {
            str(call.get("engine")): int(call.get("jobs", 0))
            for call in getattr(
                cycle, "research_plan", {}).get("engine_calls", [])},
        "provider_calls": cycle.provider_calls,
        "provider_attempts": cycle.provider_attempts,
        "task_local_probe_allocation": dict(getattr(
            cycle, "probe_allocation", {})),
        "candidate_response_accessed": False,
        "heldout_opened": False,
    } for index, cycle in enumerate(result.cycles)]
    return {"best_val_nmse": report["best_val_nmse"], "usage": usage,
        "provider_calls": sum(c.provider_calls for c in result.cycles),
        **({"posterior_gap": [{"audit": dict(c.posterior_gap_audit),
                           "brief": dict(c.posterior_gap_brief)}
                          for c in result.cycles
                          if getattr(c, "posterior_gap_audit", None)]}
           if config.posterior_gap_directed else {}),
        "candidates": candidates,
        "hypothesis_provenance": provenance,
        "scientist_policy_trace": policy_trace,
        "scientist_policy_provider_configured": getattr(
            result, "provider_configured", provider_settings is not None),
        "evidence_registry_path": str(workspace / "evidence_registry.jsonl")}


def run_exploration_ablations(root, selection, *, dataset, config,
                             provider_settings, single_engine, compute_ceiling,
                             provider_attempt_ceiling, source_identity,
                             scientific_context, gap_audit=None,
                             gap_prior=None, gap_measurement_budget=0,
                             gap_admission=None, decision_reference=None,
                             decision_calibration=None,
                             decision_selector_update=None):
    if getattr(config, "iterative_posterior_refinement", False):
        raise ValueError("iterative posterior refinement has no paired ablation role forwarding")
    if (not isinstance(config, DiscoveryAgentConfig) or config.cycles < 1
            or len(config.engines) < 2 or len(set(config.engines)) != len(config.engines)
            or single_engine not in config.engines or config.engine_repeats < 1
            or config.engine_retries != 0 or config.use_knowledge
            or config.acquisition_enabled or config.engine_workers != 1
            or any(engine not in set(registered_engine_names())
                   for engine in config.engines)
            or provider_settings is None or not provider_settings.routes or not source_identity
            or not isinstance(scientific_context, dict)
            or not scientific_context.get("task_name")
            or not scientific_context.get("task_description")
            or not math.isfinite(compute_ceiling) or compute_ceiling <= 0
            or type(provider_attempt_ceiling) is not int or provider_attempt_ceiling < 1):
        raise ValueError("invalid matched exploration contract")
    for route in provider_settings.routes:
        route.validate()
    total_jobs = config.engine_budget
    if config.posterior_gap_directed and (
            not config.typed_inner_augmentation
            or not isinstance(gap_audit, RoleDataset)
            or gap_audit.role is not DataRole.VALIDATION
            or not isinstance(gap_prior, NormalInverseGammaPrior)
            or type(gap_measurement_budget) is not int
            or gap_measurement_budget < 1
            or not isinstance(gap_admission, RoleDataset)
            or gap_admission.role is not DataRole.VALIDATION
            or config.task_local_memory
            or selection.acquisition_pool is None
            or config.engine_budget != len(config.engines) * config.engine_repeats
            or config.refit_policy != "pcpi-closed-basis-amplitudes"):
        raise ValueError("gap-directed exploration needs distinct audit/admission roles and no unadmitted memory")
    if config.posterior_gap_directed and (
            gap_audit.row_fingerprints & (
                selection.development.row_fingerprints
                | selection.validation.row_fingerprints
                | gap_admission.row_fingerprints)
            or gap_admission.row_fingerprints & (
                selection.development.row_fingerprints
                | selection.validation.row_fingerprints)):
        raise ValueError("gap-directed independent roles overlap discovery roles")
    if any(item is not None for item in (decision_reference,
            decision_calibration, decision_selector_update)) and not all(
            item is not None for item in (decision_reference,
                decision_calibration, decision_selector_update)):
        raise ValueError("finite decision reference needs distinct selector and calibration roles")
    if decision_reference is not None and (
            not config.posterior_gap_directed
            or not isinstance(decision_reference, FrozenFiniteActionReference)
            or not isinstance(decision_calibration, RoleDataset)
            or decision_calibration.role is not DataRole.VALIDATION
            or not isinstance(decision_selector_update, RoleDataset)
            or decision_selector_update.role is not DataRole.VALIDATION
            or decision_reference.calibration_identity
                != decision_calibration.fingerprint
            or any(decision_calibration.row_fingerprints & role.row_fingerprints
                   for role in (selection.development, selection.validation,
                                gap_audit, gap_admission, decision_selector_update))
            or any(decision_selector_update.row_fingerprints & role.row_fingerprints
                   for role in (selection.development, selection.validation,
                                gap_audit, gap_admission))
            or any(_covariate_rows(a.X) & _covariate_rows(b.X)
                   for i, a in enumerate((selection.development,
                       selection.validation, gap_audit, gap_admission,
                       decision_calibration, decision_selector_update))
                   for b in (selection.development, selection.validation,
                       gap_audit, gap_admission, decision_calibration,
                       decision_selector_update)[i + 1:])
            or not np.array_equal(decision_reference.action_covariates,
                                  selection.acquisition_pool.X)
            or not np.array_equal(decision_reference.target_covariates,
                                  selection.acquisition_pool.X)):
        raise ValueError("finite decision reference crosses independent roles or target")
    if total_jobs < len(config.engines):
        raise ValueError("engine budget must cover every registered skill")
    if config.scientist_orchestration:
        return _run_scientist_ablations(
            root, selection, dataset, config, provider_settings, single_engine,
            compute_ceiling, provider_attempt_ceiling, source_identity,
            scientific_context, total_jobs, gap_audit, gap_prior,
            gap_measurement_budget, gap_admission, decision_reference,
            decision_calibration, decision_selector_update)
    if config.typed_inner_augmentation:
        raise ValueError("augmentation requires Scientist source projection")
    optional_engines = tuple(engine for engine in config.engines if engine != single_engine)
    if not optional_engines:
        raise ValueError("source-first exploration requires an optional engine")
    deterministic_budget = config.discovery_budget - config.llm_evaluation_reserve
    if deterministic_budget < len(config.engines):
        raise ValueError("source-first discovery budget cannot cover registered engines")
    per_engine_budget, remainder = divmod(deterministic_budget, len(config.engines))
    core_discovery_budget = (per_engine_budget + remainder
                             + config.llm_evaluation_reserve)
    optional_discovery_budget = per_engine_budget * len(optional_engines)
    core_config = replace(config, engines=(single_engine,),
        engine_budget=config.engine_repeats, discovery_budget=core_discovery_budget)
    optional_config = replace(config, engines=optional_engines,
        engine_budget=len(optional_engines) * config.engine_repeats,
        discovery_budget=optional_discovery_budget, llm_evaluation_reserve=0)
    public_provider = asdict(provider_settings)
    for route in public_provider["routes"]:
        route.pop("api_key", None)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contract = {"schema": "scientific-source-first-exploration-ablation-v2", "dataset": dataset,
        "config": asdict(config), "single_engine": single_engine,
        "generation_contract": {
            "method": "immutable-independent-source-generation-then-projection-v1",
            "core_engines": [single_engine], "optional_engines": list(optional_engines),
            "llm_parent_context": "core-only", "llm_generation_count": 1,
            "total_candidate_evaluation_budget": config.discovery_budget,
            "core_llm_candidate_evaluation_budget": core_discovery_budget,
            "optional_engine_candidate_evaluation_budget": optional_discovery_budget,
            "variant_projection": {
                "full": ["core", *[f"engine:{name}" for name in optional_engines], "llm"],
                "no_llm": ["core", *[f"engine:{name}" for name in optional_engines]],
                "single_engine": ["core", "llm"],
            },
        },
        "development": selection.development.fingerprint,
        "validation": selection.validation.fingerprint,
        "compute_ceiling": compute_ceiling, "provider_attempt_ceiling": provider_attempt_ceiling,
        "source": source_identity, "heldout_opened": False,
        "scientific_context": scientific_context,
        "provider_settings_identity": sha256(json.dumps(public_provider,
            sort_keys=True, default=str).encode()).hexdigest(),
        "formal_experiment_authorized": False}
    contract = json.loads(json.dumps(contract, allow_nan=False))
    _publish(root / "ABLATION_CONTRACT.json", contract)
    source_root = root / "_source_generation"
    source_root.mkdir(exist_ok=True)
    core = _execute_source(source_root / "core_llm", "core_llm", core_config,
        provider_settings, selection, compute_ceiling, provider_attempt_ceiling,
        scientific_context)
    optional = _execute_source(source_root / "optional_engines", "optional_engines",
        optional_config, None, selection, compute_ceiling, provider_attempt_ceiling,
        scientific_context)
    rows = _materialize_source_first_variants(
        root, dataset, config, contract, core, optional, optional_engines,
        compute_ceiling, provider_attempt_ceiling, selection)
    _complete_anchor_banks(rows, selection)
    _assert_source_first_projection(rows)
    analysis = analyze_system_contract(rows)
    _publish(root / "ANALYSIS.json", analysis)
    return analysis


def _scientist_contract(dataset, config, single_engine, selection,
                        compute_ceiling, provider_attempt_ceiling,
                        source_identity, scientific_context, provider_settings):
    public_provider = asdict(provider_settings)
    for route in public_provider["routes"]:
        route.pop("api_key", None)
    return {"schema": "scientific-llm-skill-orchestration-ablation-v1",
        "dataset": dataset, "config": asdict(config),
        "single_engine": single_engine,
        "development": selection.development.fingerprint,
        "validation": selection.validation.fingerprint,
        "compute_ceiling": compute_ceiling,
        "provider_attempt_ceiling": provider_attempt_ceiling,
        "source": source_identity, "heldout_opened": False,
        "scientific_context": scientific_context,
        "provider_settings_identity": sha256(json.dumps(
            public_provider, sort_keys=True, default=str).encode()).hexdigest(),
        "full_policy": "llm-research-plan-engine-dispatch-review-synthesis-v1",
        "candidate_budget_contract": (
            {"mode": "engine-plus-typed-plus-inner-v1",
             "full": config.discovery_budget,
             "engine_reference": config.discovery_budget
                 - config.synthesis_evaluation_reserve
                 - config.llm_evaluation_reserve,
             "typed_synthesis": config.synthesis_evaluation_reserve,
             "inner_proposal": config.llm_evaluation_reserve,
             "compute_matched": False}
            if config.typed_inner_augmentation else
            {"mode": "legacy-matched-budget-v1",
             "compute_matched": True}),
        "single_engine_policy": (
            "llm-plan-review-synthesis-fixed-singleton-engine-and-jobs-v1"),
        "formal_experiment_authorized": False}


def _run_scientist_variant(root, variant, variant_config, provider, selection,
                           compute_ceiling, provider_attempt_ceiling,
                           scientific_context, contract, dataset, total_jobs,
                           gap_audit=None, gap_prior=None, gap_measurement_budget=0):
    workspace = Path(root) / variant
    workspace.mkdir(exist_ok=True)
    completed = workspace / "RESULT.json"
    if completed.exists():
        return json.loads(completed.read_text(encoding="utf-8"))
    if (workspace / "STARTED.json").exists():
        raise ValueError("unfinished scientist exploration cannot repeat calls")
    _publish(workspace / "STARTED.json", {"variant": variant})
    print(f"scientist exploration variant started: {variant} hard_limit={compute_ceiling}s",
          flush=True)
    try:
        summary, enforcement = run_bounded(
            _run_variant, args=(variant_config, provider, selection, workspace,
                compute_ceiling, provider_attempt_ceiling, scientific_context,
                gap_audit if variant_config.posterior_gap_directed else None,
                gap_prior if variant_config.posterior_gap_directed else None,
                gap_measurement_budget if variant_config.posterior_gap_directed else 0),
            seconds=compute_ceiling,
            provider_attempts=(provider_attempt_ceiling if provider else 0))
    except Exception as error:
        diagnostic = str(error)
        safe_diagnostic = (
            diagnostic if diagnostic.startswith("isolated stage failed:")
            else type(error).__name__
        )
        _publish(workspace / "FAILURE.json", {
            "schema": "scientific-llm-skill-orchestration-failure-v1",
            "variant": variant,
            "error_type": type(error).__name__,
            "diagnostic": safe_diagnostic,
            "candidate_response_accessed": False,
            "heldout_opened": False,
            "efficacy_demonstrated": False,
            "formal_experiment_authorized": False,
        })
        raise
    usage = summary["usage"]
    row = {"dataset": dataset, "seed": variant_config.random_seed,
        "variant": variant, "status": "succeeded", "heldout_opened": False,
        "selection_used_heldout": False, "best_val_nmse": summary["best_val_nmse"],
        "development_fingerprint": contract["development"],
        "validation_fingerprint": contract["validation"], "measurement_budget": 0,
        "engine_job_budget": total_jobs * variant_config.cycles,
        "candidate_evaluation_budget": variant_config.discovery_budget * variant_config.cycles,
        "compute_ceiling": compute_ceiling, "provider_calls": summary["provider_calls"],
        **usage, "resource_enforcement": enforcement,
        "candidates": summary["candidates"],
        **({"posterior_gap": summary["posterior_gap"]}
           if variant_config.posterior_gap_directed else {}),
        "hypothesis_provenance": summary["hypothesis_provenance"],
        "scientist_policy_trace": summary["scientist_policy_trace"],
        "scientist_policy_provider_configured": summary[
            "scientist_policy_provider_configured"],
        "evidence_registry_path": summary["evidence_registry_path"]}
    _publish(completed, row)
    return row


def _run_scientist_ablations(root, selection, dataset, config, provider_settings,
                             single_engine, compute_ceiling,
                             provider_attempt_ceiling, source_identity,
                             scientific_context, total_jobs,
                             gap_audit=None, gap_prior=None,
                             gap_measurement_budget=0, gap_admission=None,
                             decision_reference=None, decision_calibration=None,
                             decision_selector_update=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    contract = _scientist_contract(
        dataset, config, single_engine, selection, compute_ceiling,
        provider_attempt_ceiling, source_identity, scientific_context,
        provider_settings)
    if config.posterior_gap_directed:
        contract["candidate_budget_contract"] = {
            "mode": "quality-first-engine-intact-posterior-gap-v1",
            "engine_reference": config.discovery_budget,
            "full": config.discovery_budget
                + config.synthesis_evaluation_reserve
                + config.llm_evaluation_reserve,
            "typed_synthesis": config.synthesis_evaluation_reserve,
            "inner_proposal": config.llm_evaluation_reserve,
            "compute_matched": False}
        contract["gap_audit_identity"] = gap_audit.fingerprint
        contract["gap_admission_identity"] = gap_admission.fingerprint
        contract["gap_measurement_budget"] = gap_measurement_budget
        contract["optional_candidate_attempt_ceiling"] = (
            config.synthesis_evaluation_reserve
            + config.llm_evaluation_reserve) * config.cycles
        contract["gap_prior"] = gap_prior.to_dict()
        if decision_reference is not None:
            contract["finite_decision_reference_identity"] = decision_reference.stable_hash
            contract["finite_decision_calibration_identity"] = (
                decision_calibration.fingerprint)
            contract["finite_decision_selector_update_identity"] = (
                decision_selector_update.fingerprint)
    contract = json.loads(json.dumps(contract, allow_nan=False))
    _publish(root / "ABLATION_CONTRACT.json", contract)
    no_llm_budget = (
        config.discovery_budget if config.posterior_gap_directed else
        config.discovery_budget - config.llm_evaluation_reserve
        - config.synthesis_evaluation_reserve
        if config.typed_inner_augmentation else config.discovery_budget)
    variants = {
        "full": (replace(config, discovery_budget=(config.discovery_budget
            + config.synthesis_evaluation_reserve
            + config.llm_evaluation_reserve)) if config.posterior_gap_directed
            else config, provider_settings),
        "no_llm": (replace(
            config, scientist_orchestration=False,
            typed_evidence_synthesis=False,
            typed_inner_augmentation=False,
            posterior_gap_directed=False,
            synthesis_evaluation_reserve=0,
            discovery_budget=no_llm_budget,
            llm_evaluation_reserve=(0 if config.typed_inner_augmentation
                                    else config.llm_evaluation_reserve)), None),
        "single_engine": (replace(
            config, engines=(single_engine,), engine_repeats=total_jobs,
            engine_budget=total_jobs,
            discovery_budget=(config.discovery_budget
                + config.synthesis_evaluation_reserve
                + config.llm_evaluation_reserve
                if config.posterior_gap_directed else config.discovery_budget),
            posterior_gap_directed=False), provider_settings),
    }
    rows = [_run_scientist_variant(
        root, variant, variant_config, provider, selection, compute_ceiling,
        provider_attempt_ceiling, scientific_context, contract, dataset, total_jobs,
        gap_audit, gap_prior, gap_measurement_budget)
        for variant, (variant_config, provider) in variants.items()]
    if rows[1]["provider_calls"] or rows[1]["provider_attempts_used"]:
        raise ValueError("provider-free scientist ablation attempted provider calls")
    if config.posterior_gap_directed:
        _restore_intact_engine_bank(rows, selection)
    _complete_anchor_banks(rows, selection)
    if config.posterior_gap_directed:
        _screen_gap_candidates(rows, selection, gap_audit, gap_admission,
                               gap_prior, gap_measurement_budget, root,
                               decision_reference, decision_calibration,
                               decision_selector_update,
                               optional_candidate_attempt_ceiling=contract[
                                   "optional_candidate_attempt_ceiling"])
    analysis = analyze_system_contract(
        rows, augmentation_total=(
            (config.synthesis_evaluation_reserve
             + config.llm_evaluation_reserve) * config.cycles
            if config.typed_inner_augmentation else 0))
    _publish(root / "ANALYSIS.json", analysis)
    return analysis


def _screen_gap_candidates(rows, selection, gap_audit, gap_admission,
                           prior, measurement_budget, root,
                           decision_reference=None, decision_calibration=None,
                           decision_selector_update=None, *,
                           optional_candidate_attempt_ceiling=None):
    started = time.monotonic()
    if (gap_admission.row_fingerprints & (
            gap_audit.row_fingerprints
            | selection.development.row_fingerprints
            | selection.validation.row_fingerprints)
            or gap_audit.row_fingerprints & (
                selection.development.row_fingerprints
                | selection.validation.row_fingerprints)):
        raise ExplorationProtocolError("posterior-gap-data-roles-overlap")
    indexed = {row["variant"]: row for row in rows}
    full = indexed["full"]
    final_gap = full["posterior_gap"][-1] if full["posterior_gap"] else None
    allowed = tuple(sorted({int(item["region"])
        for item in (final_gap["brief"].get("regions", ())
                     if final_gap is not None else ())}))
    domain = selection.acquisition_pool.X
    regions = FrozenAxisRegions(selection.development.X.shape[1], 0,
                                (float(np.median(domain[:, 0])),))
    features = selection.development.X.shape[1]
    baseline = {}
    baseline_duplicates = 0
    for row in indexed["no_llm"]["candidates"]:
        support = structural_terms(row["expression"], features)
        if support in baseline:
            baseline_duplicates += 1
        else:
            baseline[support] = dict(row)
    optional = [row for row in full["candidates"]
                if row.get("origin") == "llm"]
    if (type(optional_candidate_attempt_ceiling) is not int
            or optional_candidate_attempt_ceiling < 0
            or len(optional) > optional_candidate_attempt_ceiling):
        raise ExplorationProtocolError("quality-first-optional-candidate-budget")
    if optional and allowed:
        retained, report = admit_regional_candidates(
            list(baseline.values()), optional,
            selection.development, gap_audit, gap_admission, domain,
            regions, allowed, prior, measurement_budget)
    else:
        retained = []
        report = {"schema": "candidate-regional-admission-v1",
                  "attempts": len(optional), "candidates": [
                      {"lineage_id": str(candidate.get("lineage_id") or ""),
                       "composed_expression": candidate["expression"],
                       "attempted_candidate_count": len(optional),
                       "region_identity": regions.identity,
                       "admitted": False,
                       "reason": "no-independent-gap"}
                      for candidate in optional],
                  "admission_identity": gap_admission.fingerprint,
                  "reason": "no-independent-gap-or-no-llm-candidate",
                  "decision_contribution_assessed": False,
                  "candidate_response_accessed": False,
                  "heldout_opened": False}
    report["baseline_duplicate_support_rows"] = baseline_duplicates
    report["optional_candidate_attempt_ceiling"] = (
        optional_candidate_attempt_ceiling)
    full_supports = {structural_terms(row["expression"], features): row
                     for row in full["candidates"]}
    if any(support not in full_supports for support in baseline):
        # The Full discovery top-k may prune a core row, so restore the exact
        # provider-free baseline before projecting the independent candidates.
        report["full_topk_missing_baseline_supports"] = sum(
            support not in full_supports for support in baseline)
    admitted = {}
    for row in retained:
        support = structural_terms(row["expression"], features)
        if support in baseline:
            raise ExplorationProtocolError("quality-first-admission-duplicates-baseline")
        admitted.setdefault(support, dict(row))
    for record in report["candidates"]:
        if record.get("admitted") is not True:
            record["bank_retained"] = False
            continue
        support = structural_terms(record["composed_expression"], features)
        chosen = admitted.get(support)
        record["bank_retained"] = bool(chosen is not None
            and record.get("lineage_id") == chosen.get("lineage_id")
            and record["composed_expression"] == chosen["expression"])
        if not record["bank_retained"]:
            record["reason"] = "independently-predictive-duplicate-llm-support"
    full["candidates"] = [*baseline.values(), *admitted.values()]
    indexed["no_llm"]["candidates"] = list(baseline.values())
    bank_report = _freeze_quality_first_expanded_banks(
        baseline, admitted, selection, prior, measurement_budget)
    if decision_reference is not None:
        if decision_calibration is None or decision_selector_update is None:
            raise ExplorationProtocolError("finite-decision-independent-roles-absent")
        for candidate in admitted.values():
            record = next((item for item in report["candidates"]
                if item.get("admitted") is True and item.get("bank_retained") is True
                and item.get("lineage_id") == candidate.get("lineage_id")
                and item.get("composed_expression") == candidate["expression"]), None)
            if record is None:
                raise ExplorationProtocolError("finite-decision-candidate-identity-missing")
            record["conditional_finite_law_action_audit"] = (
                audit_admitted_candidate_action(
                    list(baseline.values()), candidate,
                    selection.development, gap_audit, gap_admission,
                    decision_selector_update,
                    decision_calibration, domain, regions, prior,
                    measurement_budget, decision_reference,
                    candidate_identity=record["candidate_identity"]))
        report["finite_law_action_contribution_assessed"] = bool(admitted)
        report["finite_decision_reference_identity"] = decision_reference.stable_hash
        report["finite_decision_selector_update_identity"] = (
            decision_selector_update.fingerprint)
    else:
        report["finite_law_action_contribution_assessed"] = False
    full["hypothesis_provenance"]["quality_first_expanded_bank"] = bank_report
    indexed["no_llm"]["hypothesis_provenance"]["quality_first_expanded_bank"] = {
        "role": "engine_only", "bank_identity": bank_report["engine_bank_identity"],
        "target_identity": bank_report["engine_target_identity"]}
    full["hypothesis_provenance"]["regional_candidate_admission"] = report
    full["hypothesis_provenance"]["llm_retained_candidate_count"] = sum(
        row.get("origin") == "llm" for row in full["candidates"])
    full["hypothesis_provenance"]["candidate_count"] = len(full["candidates"])
    from .knowledge_runtime import KnowledgeRuntime
    knowledge = KnowledgeRuntime(
        Path(root) / "full" / "knowledge" / "structure_library.jsonl",
        Path(root) / "full" / "knowledge" / "runtime_ledger.jsonl")
    stage_results = []
    target_identity = (final_gap["audit"].get("target_identity")
                       if final_gap is not None else "")
    for stage_id in full["hypothesis_provenance"].get(
            "gap_knowledge_stage_ids", ()):
        stage_results.append(knowledge.finalize_gap_stage(
            stage_id, admission_identity=gap_admission.fingerprint,
            target_identity=target_identity,
            candidate_records=report["candidates"]))
    full["hypothesis_provenance"]["gap_knowledge_stages"] = [
        {"stage_id": row["stage_id"], "status": row["status"],
         "decision_effect_assessed": False} for row in stage_results]
    report["admission_wall_seconds"] = time.monotonic() - started
    _publish(Path(root) / "REGIONAL_CANDIDATE_ADMISSION.json", report)
    _publish(Path(root) / "QUALITY_FIRST_EXPANDED_BANKS.json", bank_report)


def _freeze_quality_first_expanded_banks(
        baseline, admitted, selection, prior, measurement_budget):
    """Freeze both uncapped exploratory banks without using the old K selector.

    Their class partitions can differ. The shared identity below is the
    covariate target only; it is not a common class-loss certificate.
    """
    rows = (list(baseline.values()), [*baseline.values(), *admitted.values()])
    frozen = []
    for candidates in rows:
        digest = sha256(json.dumps(candidates, sort_keys=True,
            default=str).encode()).hexdigest()
        model = freeze_discovery_model(candidates,
            n_features=selection.development.X.shape[1], prior=prior,
            exploration_identity=digest,
            coefficient_policy="discard-fitted-coefficients-refit-closed-basis")
        target = freeze_discovery_target(model, selection.development,
            selection.acquisition_pool.X, measurement_budget=measurement_budget,
            expected_model_identity=model.stable_hash)
        frozen.append((model, target))
    core, expanded = frozen
    if (core[1].action_domain_identity != expanded[1].action_domain_identity
            or core[1].initial_data_identity != expanded[1].initial_data_identity):
        raise ExplorationProtocolError("quality-first-frozen-bank-identity-crossed")
    common_classes = freeze_common_class_projection(core[1], expanded[1])
    if (not np.isclose(common_classes.common_probabilities(
            core[1].initial_posterior, bank="core").sum(), 1.)
            or not np.isclose(common_classes.common_probabilities(
                expanded[1].initial_posterior, bank="expanded").sum(), 1.)):
        raise ExplorationProtocolError("quality-first-common-class-mass-invalid")
    return {"schema": "quality-first-expanded-operational-banks-v1",
            "capacity_policy": "exploration-only-no-capacity-pruning",
            "engine_support_count": len(baseline),
            "admitted_llm_support_count": len(admitted),
            "full_support_count": len(baseline) + len(admitted),
            "engine_bank_identity": core[0].stable_hash,
            "full_bank_identity": expanded[0].stable_hash,
            "engine_target_identity": core[1].stable_hash,
            "full_target_identity": expanded[1].stable_hash,
            "common_covariate_domain_identity": core[1].action_domain_identity,
            "common_class_projection_identity": common_classes.stable_hash,
            "common_class_union_partition_identity": (
                common_classes.union_partition_identity),
            "common_class_count": len(common_classes.union_class_ids),
            "core_unsupported_class_count": (len(common_classes.union_class_ids)
                - len(common_classes.core_class_positions)),
            "common_class_projection_constructed": True,
            "common_class_loss_assessed": False,
            "measured_runner_authorized": False,
            "candidate_response_accessed": False,
            "heldout_opened": False}


def _restore_intact_engine_bank(rows, selection):
    """Preserve all identical engine supports in both quality-first arms."""
    n_features = selection.development.X.shape[1]
    indexed = {row["variant"]: row for row in rows}
    if set(indexed) != {"full", "no_llm", "single_engine"}:
        raise ExplorationProtocolError("quality-first variants incomplete")
    def engine_rows(row):
        result = {}
        for record in row["hypothesis_provenance"]["raw_engine_candidates"]:
            try:
                support = structural_terms(str(record["expression"]), n_features)
            except (SyntaxError, TypeError, ValueError):
                continue
            result.setdefault(support, {
                "expression": str(record["expression"]),
                "source": f"engine:{record['engine']}",
                "origin": "deterministic",
                "lineage_id": str(record.get("lineage_id", ""))})
        return result
    full, control = engine_rows(indexed["full"]), engine_rows(indexed["no_llm"])
    if not full or full != control:
        raise ExplorationProtocolError("quality-first-engine-frontiers-not-paired")
    for variant in ("full", "no_llm"):
        row = indexed[variant]
        present = {structural_terms(item["expression"], n_features)
                   for item in row["candidates"]}
        for support, candidate in full.items():
            if support not in present:
                row["candidates"].append(candidate)
                present.add(support)
        row["hypothesis_provenance"]["intact_engine_support_count"] = len(full)
        row["hypothesis_provenance"]["quality_first_engine_identity"] = sha256(
            json.dumps([list(support) for support in sorted(full)]).encode()
        ).hexdigest()


def _execute_source(workspace, name, config, provider_settings, selection,
                    compute_ceiling, provider_attempt_ceiling, scientific_context):
    workspace = Path(workspace)
    workspace.mkdir(exist_ok=True)
    completed = workspace / "RESULT.json"
    if completed.exists():
        return json.loads(completed.read_text(encoding="utf-8"))
    if (workspace / "STARTED.json").exists():
        raise ValueError("unfinished source generation cannot silently repeat engine/provider calls")
    _publish(workspace / "STARTED.json", {"source_generation": name})
    print(f"exploration source started: {name} hard_limit={compute_ceiling}s", flush=True)
    try:
        summary, enforcement = run_bounded(_run_variant,
            args=(config, provider_settings, selection, workspace, compute_ceiling,
                  provider_attempt_ceiling, scientific_context),
            seconds=compute_ceiling,
            provider_attempts=provider_attempt_ceiling if provider_settings is not None else 0)
        result = {**summary, "source_generation": name,
                  "resource_enforcement": enforcement, "heldout_opened": False}
        _publish(completed, result)
        return result
    except Exception as error:
        _publish(workspace / "FAILURE.json", {"source_generation": name,
            "error_type": type(error).__name__, "message": str(error),
            "heldout_opened": False, "efficacy_demonstrated": False})
        raise


def _support_deduplicate(candidates, feature_count):
    retained, supports = [], set()
    for candidate in candidates:
        support = tuple(structural_terms(candidate["expression"], feature_count))
        if support not in supports:
            retained.append(dict(candidate)); supports.add(support)
    return retained


def _source_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _projection_provenance(variant, candidates, core, optional, identities,
                           bank_identity):
    relevant = [core["hypothesis_provenance"]]
    if variant != "single_engine":
        relevant.append(optional["hypothesis_provenance"])
    return {
        "schema": "scientific-source-first-hypothesis-provenance-v2",
        "candidate_count": len(candidates),
        "distinct_sources": sorted({row["source"] for row in candidates}),
        "origin_counts": {origin: sum(row["origin"] == origin for row in candidates)
            for origin in sorted({row["origin"] for row in candidates})},
        "llm_retained_candidate_count": sum(row["origin"] == "llm" for row in candidates),
        "engine_sources": sorted({row["source"] for row in candidates
                                  if row["source"].startswith("engine:")}),
        "all_candidates_source_bound": bool(candidates),
        "pcpi_adapter_rejections": [item for audit in relevant
                                     for item in audit["pcpi_adapter_rejections"]],
        "raw_engine_candidates": [item for audit in relevant
                                  for item in audit["raw_engine_candidates"]],
        "discovery_candidate_registry": [item for audit in relevant
                                         for item in audit["discovery_candidate_registry"]],
        "evaluation_rejections": [item for audit in relevant
                                  for item in audit["evaluation_rejections"]],
        "llm_candidate_lifecycle": core["hypothesis_provenance"]["llm_candidate_lifecycle"]
            if variant != "no_llm" else [],
        "source_generation_registry_sha256": identities,
        "source_first_bank_identity": bank_identity,
        "llm_parent_context": "core-only", "heldout_accessed": False,
    }


def _materialize_source_first_variants(root, dataset, config, contract, core, optional,
                                       optional_engines, compute_ceiling,
                                       provider_attempt_ceiling, selection):
    feature_count = len(contract["scientific_context"].get("feature_names", []))
    if feature_count < 1:
        raise ValueError("source-first projection requires registered feature names")
    core_candidates = [row for row in core["candidates"] if row.get("origin") != "llm"]
    for row in generic_deterministic_candidates(
            selection.development.X, selection.development.y):
        try:
            structural_terms(row["expression"], feature_count)
        except Exception:
            continue
        core_candidates.append({**row, "origin": "deterministic"})
    llm_candidates = [row for row in core["candidates"] if row.get("origin") == "llm"]
    optional_sources = {f"engine:{name}" for name in optional_engines}
    optional_candidates = [row for row in optional["candidates"]
                           if row.get("source") in optional_sources]
    banks = {
        "full": [*core_candidates, *optional_candidates, *llm_candidates],
        "no_llm": [*core_candidates, *optional_candidates],
        "single_engine": [*core_candidates, *llm_candidates],
    }
    source_paths = {
        "core_llm": Path(core["evidence_registry_path"]),
        "optional_engines": Path(optional["evidence_registry_path"]),
    }
    for path in source_paths.values():
        verification = EvidenceRegistry(path).verify()
        if not verification.valid or verification.event_count < 1:
            raise ExplorationProtocolError("invalid-source-generation-evidence-chain")
    identities = {name: _source_digest(path) for name, path in source_paths.items()}
    bank_identity = sha256(json.dumps({"sources": identities, "banks": banks},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    rows = []
    for variant, candidates in banks.items():
        workspace = Path(root) / variant
        workspace.mkdir(exist_ok=True)
        completed = workspace / "RESULT.json"
        if completed.exists():
            rows.append(json.loads(completed.read_text(encoding="utf-8"))); continue
        _publish(workspace / "STARTED.json", {"variant": variant,
            "source_first_bank_identity": bank_identity})
        candidates = _support_deduplicate(candidates, feature_count)
        provenance = _projection_provenance(
            variant, candidates, core, optional, identities, bank_identity)
        registry_path = workspace / "evidence_registry.jsonl"
        registry = EvidenceRegistry(registry_path)
        registry.append(hypothesis_id=f"source-first-{bank_identity[:20]}-{variant}",
            event_type=EvidenceEventType.MERGED, payload={
                "schema": "scientific-source-first-projection-evidence-v1",
                "variant": variant, "bank_identity": bank_identity,
                "source_registry_sha256": identities,
                "candidate_supports": [list(structural_terms(row["expression"], feature_count))
                                       for row in candidates],
                "candidate_response_accessed": False, "heldout_opened": False})
        usage = {
            "engine_jobs_used": (core["usage"]["engine_jobs_used"] +
                (0 if variant == "single_engine" else optional["usage"]["engine_jobs_used"])),
            "candidate_evaluations_used": (core["usage"]["candidate_evaluations_used"] +
                (0 if variant == "single_engine" else optional["usage"]["candidate_evaluations_used"])),
            "provider_attempts_used": 0 if variant == "no_llm" else core["usage"]["provider_attempts_used"],
            "elapsed_seconds": (core["usage"]["elapsed_seconds"] +
                (0 if variant == "single_engine" else optional["usage"]["elapsed_seconds"])),
        }
        row = {"dataset": dataset, "seed": config.random_seed, "variant": variant,
            "status": "succeeded", "heldout_opened": False, "selection_used_heldout": False,
            "best_val_nmse": core["best_val_nmse"],
            "development_fingerprint": contract["development"],
            "validation_fingerprint": contract["validation"], "measurement_budget": 0,
            "engine_job_budget": len(config.engines) * config.engine_repeats * config.cycles,
            "candidate_evaluation_budget": config.discovery_budget * config.cycles,
            "compute_ceiling": compute_ceiling,
            "provider_calls": 0 if variant == "no_llm" else core["provider_calls"], **usage,
            "resource_enforcement": {"source_first_projection": True},
            "candidates": candidates, "hypothesis_provenance": provenance,
            "evidence_registry_path": str(registry_path),
            "source_first_bank_identity": bank_identity}
        _publish(completed, row); rows.append(row)
    return rows


def _assert_source_first_projection(rows):
    indexed = {row["variant"]: row for row in rows}
    if set(indexed) != {"full", "no_llm", "single_engine"}:
        raise ExplorationProtocolError("incomplete-source-first-projection")
    identities = {row.get("source_first_bank_identity") for row in rows}
    if len(identities) != 1 or None in identities:
        raise ExplorationProtocolError("variant-source-bank-identities-differ")
    def source_supports(variant, family):
        return {row["expression"] for row in indexed[variant]["candidates"]
                if source_family(row) == family}
    if source_supports("full", "llm") != source_supports("single_engine", "llm"):
        raise ExplorationProtocolError("llm-proposals-differ-across-ablations")
    if (source_supports("full", "core") != source_supports("no_llm", "core")
            or source_supports("full", "core") != source_supports("single_engine", "core")):
        raise ExplorationProtocolError("core-proposals-differ-across-ablations")
    if any(row.get("origin") == "llm" for row in indexed["no_llm"]["candidates"]):
        raise ExplorationProtocolError("no-llm-projection-retained-llm")
    for source in indexed["no_llm"]["hypothesis_provenance"]["engine_sources"]:
        if source_supports("full", source) != source_supports("no_llm", source):
            raise ExplorationProtocolError("engine-proposals-differ-across-ablations")


def _complete_anchor_banks(rows, selection):
    # Every ablation inherits the same task-independent generic anchor bank.
    # Provider/evaluation budgeting may otherwise evict deterministic seeds
    # from an LLM-enabled top-k and leave a one-support bank.  Restoring these
    # development-only anchors is response-free and keeps ablations comparable.
    anchors = generic_deterministic_candidates(
        selection.development.X, selection.development.y)
    feature_count = selection.development.X.shape[1]
    for row in rows:
        supports = {tuple(structural_terms(item["expression"], feature_count))
                    for item in row["candidates"]}
        for anchor in anchors:
            try:
                support = tuple(structural_terms(anchor["expression"], feature_count))
            except Exception:
                continue
            if support not in supports:
                row["candidates"].append({**anchor, "origin": "deterministic"})
                supports.add(support)
        if len(supports) < 2:
            raise ExplorationProtocolError(
                f"{row['variant']}-retained-fewer-than-two-pcpi-supports")
        provenance = row["hypothesis_provenance"]
        provenance["candidate_count"] = len(row["candidates"])
        provenance["distinct_sources"] = sorted({item["source"] for item in row["candidates"]})
        provenance["engine_sources"] = sorted({item["source"] for item in row["candidates"]
                                                if item["source"].startswith("engine:")})
        provenance["origin_counts"] = {
            origin: sum(item["origin"] == origin for item in row["candidates"])
            for origin in sorted({item["origin"] for item in row["candidates"]})}
