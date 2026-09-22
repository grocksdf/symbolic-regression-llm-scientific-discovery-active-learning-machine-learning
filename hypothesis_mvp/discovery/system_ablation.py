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

from .agent import DiscoveryAgent, DiscoveryAgentConfig
from .system_run import analyze_system_contract
from .resource_limits import run_bounded
from hypothesis_mvp.pcpi.discovery_transaction import _publish
from hypothesis_mvp.symbolic.registry import registered_engine_names
from .pcpi_adapter import structural_terms
from .initializer import generic_deterministic_candidates
from .source_stacking import source_family
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
    early_stop = (0 < actual_cycles < config.cycles
                  and result.cycles[-1].scientist_review.get("stop") is True
                  and result.cycles[-1].acquisition.get("reason")
                  == "scientist_stop_condition")
    if actual_cycles < 1 or actual_cycles > config.cycles:
        raise ValueError("invalid exploration cycle count")
    if actual_cycles < config.cycles and not early_stop:
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
            "scientist_early_stop": early_stop}


def _run_variant(config, provider_settings, selection, workspace, compute_ceiling,
                 provider_attempt_ceiling, scientific_context):
    start = time.monotonic()
    agent = DiscoveryAgent(config, provider_settings)
    context = dict(scientific_context)
    result = agent.run(selection=selection, task_name=context["task_name"],
        task_description=context["task_description"], output_dir=workspace,
        knowledge_dir=workspace / "knowledge", variable_metadata=context)
    usage = audit_usage(config, result, time.monotonic() - start,
                        compute_ceiling, provider_attempt_ceiling)
    report = result.discovery.report
    if report.get("llm_error_count", 0) or any(
            getattr(c, "provider_errors", 0) for c in result.cycles):
        raise ExplorationProtocolError(
            "provider-or-protocol-failure-blocks-exploration")
    if provider_settings is not None and (
            int(report.get("llm_call_count", 0)) < 1
            or int(report.get("llm_attempt_count", 0)) < 1):
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
        "candidate_response_accessed": False,
        "heldout_opened": False,
    } for index, cycle in enumerate(result.cycles)]
    return {"best_val_nmse": report["best_val_nmse"], "usage": usage,
        "provider_calls": sum(c.provider_calls for c in result.cycles),
        "candidates": candidates,
        "hypothesis_provenance": provenance,
        "scientist_policy_trace": policy_trace,
        "scientist_policy_provider_configured": getattr(
            result, "provider_configured", provider_settings is not None),
        "evidence_registry_path": str(workspace / "evidence_registry.jsonl")}


def run_exploration_ablations(root, selection, *, dataset, config,
                             provider_settings, single_engine, compute_ceiling,
                             provider_attempt_ceiling, source_identity,
                             scientific_context):
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
    if total_jobs < len(config.engines):
        raise ValueError("engine budget must cover every registered skill")
    if config.scientist_orchestration:
        return _run_scientist_ablations(
            root, selection, dataset, config, provider_settings, single_engine,
            compute_ceiling, provider_attempt_ceiling, source_identity,
            scientific_context, total_jobs)
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
        "single_engine_policy": (
            "llm-plan-review-synthesis-fixed-singleton-engine-and-jobs-v1"),
        "formal_experiment_authorized": False}


def _run_scientist_variant(root, variant, variant_config, provider, selection,
                           compute_ceiling, provider_attempt_ceiling,
                           scientific_context, contract, dataset, total_jobs):
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
                compute_ceiling, provider_attempt_ceiling, scientific_context),
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
                             scientific_context, total_jobs):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    contract = _scientist_contract(
        dataset, config, single_engine, selection, compute_ceiling,
        provider_attempt_ceiling, source_identity, scientific_context,
        provider_settings)
    contract = json.loads(json.dumps(contract, allow_nan=False))
    _publish(root / "ABLATION_CONTRACT.json", contract)
    variants = {
        "full": (config, provider_settings),
        "no_llm": (replace(config, scientist_orchestration=False), None),
        "single_engine": (replace(
            config, engines=(single_engine,), engine_repeats=total_jobs,
            engine_budget=total_jobs), provider_settings),
    }
    rows = [_run_scientist_variant(
        root, variant, variant_config, provider, selection, compute_ceiling,
        provider_attempt_ceiling, scientific_context, contract, dataset, total_jobs)
        for variant, (variant_config, provider) in variants.items()]
    if rows[1]["provider_calls"] or rows[1]["provider_attempts_used"]:
        raise ValueError("provider-free scientist ablation attempted provider calls")
    _complete_anchor_banks(rows, selection)
    analysis = analyze_system_contract(rows)
    _publish(root / "ANALYSIS.json", analysis)
    return analysis


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
