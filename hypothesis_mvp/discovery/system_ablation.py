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
    expected_jobs = len(config.engines) * config.engine_repeats * config.cycles
    if (jobs != expected_jobs or any(c.engine_report["failures"] for c in result.cycles)
            or any(r["status"] != "succeeded" for c in result.cycles
                   for r in c.engine_report["run_records"])):
        raise ValueError("incomplete or failed engine execution")
    if (evaluations > config.discovery_budget * config.cycles
            or attempts > provider_attempt_ceiling or elapsed > compute_ceiling):
        raise ValueError("registered exploration ceiling exceeded")
    return {"engine_jobs_used": jobs, "candidate_evaluations_used": evaluations,
            "provider_attempts_used": attempts, "elapsed_seconds": elapsed}


def _run_variant(config, provider_settings, selection, workspace, compute_ceiling, provider_attempt_ceiling):
    start = time.monotonic()
    agent = DiscoveryAgent(config, provider_settings)
    result = agent.run(selection=selection, task_name="registered-development",
        task_description="", output_dir=workspace,
        knowledge_dir=workspace / "knowledge", variable_metadata={})
    usage = audit_usage(config, result, time.monotonic() - start,
                        compute_ceiling, provider_attempt_ceiling)
    report = result.discovery.report
    if report.get("llm_error_count", 0) or any(getattr(c, "provider_errors", 0) for c in result.cycles):
        raise ExplorationProtocolError("provider-infrastructure-failure-blocks-exploration")
    if provider_settings is not None and (
            int(report.get("llm_call_count", 0)) < 1
            or int(report.get("llm_attempt_count", 0)) < 1):
        raise ExplorationProtocolError("llm-enabled-exploration-had-zero-provider-attempts")
    # All retained top-k proposals are passed, including LLM descendants.
    # No outcome-conditioned filtering for adapter compatibility is permitted.
    candidates = [{"expression": row["expression"], "source": str(row.get("source", "runtime-retained"))}
                  for row in report.get("final_topk", [])]
    return {"best_val_nmse": report["best_val_nmse"], "usage": usage,
        "provider_calls": sum(c.provider_calls for c in result.cycles),
        "candidates": candidates,
        "evidence_registry_path": str(workspace / "evidence_registry.jsonl")}


def run_exploration_ablations(root, selection, *, dataset, config,
                             provider_settings, single_engine, compute_ceiling,
                             provider_attempt_ceiling, source_identity):
    if (not isinstance(config, DiscoveryAgentConfig) or config.cycles < 1
            or len(config.engines) < 2 or len(set(config.engines)) != len(config.engines)
            or single_engine not in config.engines or config.engine_repeats < 1
            or config.engine_retries != 0 or config.use_knowledge
            or config.acquisition_enabled or config.engine_workers != 1
            or any(engine not in {"polynomial_lasso", "mcts"} for engine in config.engines)
            or provider_settings is None or not provider_settings.routes or not source_identity
            or not math.isfinite(compute_ceiling) or compute_ceiling <= 0
            or type(provider_attempt_ceiling) is not int or provider_attempt_ceiling < 1):
        raise ValueError("invalid matched exploration contract")
    for route in provider_settings.routes:
        route.validate()
    total_jobs = len(config.engines) * config.engine_repeats
    if config.engine_budget != total_jobs:
        raise ValueError("engine budget must cover exactly the registered jobs")
    variants = {"full": config, "no_llm": config,
        "single_engine": replace(config, engines=(single_engine,), engine_repeats=total_jobs)}
    public_provider = asdict(provider_settings)
    for route in public_provider["routes"]:
        route.pop("api_key", None)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contract = {"schema": "scientific-exploration-ablation-v1", "dataset": dataset,
        "config": asdict(config), "single_engine": single_engine,
        "development": selection.development.fingerprint,
        "validation": selection.validation.fingerprint,
        "compute_ceiling": compute_ceiling, "provider_attempt_ceiling": provider_attempt_ceiling,
        "source": source_identity, "heldout_opened": False,
        "provider_settings_identity": sha256(json.dumps(public_provider,
            sort_keys=True, default=str).encode()).hexdigest(),
        "formal_experiment_authorized": False}
    contract = json.loads(json.dumps(contract, allow_nan=False))
    _publish(root / "ABLATION_CONTRACT.json", contract)
    rows = []
    for variant, variant_config in variants.items():
        workspace = root / variant
        workspace.mkdir(exist_ok=True)
        completed = workspace / "RESULT.json"
        if completed.exists():
            rows.append(json.loads(completed.read_text(encoding="utf-8")))
            continue
        if (workspace / "STARTED.json").exists():
            raise ValueError("unfinished exploration cannot silently repeat engine/provider calls")
        _publish(workspace / "STARTED.json", {"variant": variant})
        print(f"exploration variant started: {variant} hard_limit={compute_ceiling}s", flush=True)
        try:
            summary, enforcement = run_bounded(_run_variant,
                args=(variant_config, None if variant == "no_llm" else provider_settings,
                      selection, workspace, compute_ceiling, provider_attempt_ceiling),
                seconds=compute_ceiling,
                provider_attempts=0 if variant == "no_llm" else provider_attempt_ceiling)
            usage = summary["usage"]
            row = {"dataset": dataset, "seed": config.random_seed, "variant": variant,
                "status": "succeeded", "heldout_opened": False, "selection_used_heldout": False,
                "best_val_nmse": summary["best_val_nmse"],
                "development_fingerprint": contract["development"],
                "validation_fingerprint": contract["validation"], "measurement_budget": 0,
                "engine_job_budget": total_jobs * config.cycles,
                "candidate_evaluation_budget": config.discovery_budget * config.cycles,
                "compute_ceiling": compute_ceiling,
                "provider_calls": summary["provider_calls"], **usage,
                "resource_enforcement": enforcement,
                "candidates": summary["candidates"],
                "evidence_registry_path": summary["evidence_registry_path"]}
            if variant == "no_llm" and (row["provider_calls"] or usage["provider_attempts_used"]):
                raise ValueError("provider-free ablation attempted provider calls")
            _publish(completed, row)
            rows.append(row)
        except Exception as error:
            _publish(workspace / "FAILURE.json", {"variant": variant,
                "error_type": type(error).__name__, "message": str(error),
                "heldout_opened": False, "efficacy_demonstrated": False})
            raise
    analysis = analyze_system_contract(rows)
    _publish(root / "ANALYSIS.json", analysis)
    return analysis
