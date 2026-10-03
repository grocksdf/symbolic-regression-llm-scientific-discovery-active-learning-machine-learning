"""Freeze one source-first E/Blind/Gap coordinate before truth evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"),
                str(ROOT.parent / "hypothesis_mvp")]

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()

from hypothesis_mvp.data import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from hypothesis_mvp.discovery.agent import (
    DiscoveryAgent, DiscoveryAgentConfig, _engine_evidence,
)
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.evidence_synthesis import (
    compile_evidence_synthesis,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model,
)
from hypothesis_mvp.discovery.proposal_runtime import (
    ProposalRuntime, ProviderInfrastructureError, ProviderSettings,
)
from hypothesis_mvp.discovery.scientist_policy import deterministic_plan
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from expanded_formula_admission_v2 import project_expanded_admission_arms
from formula_round2_source_first import freeze_source_first_banks


GROUPS = {
    "chem_react": "lsr_synth/chem_react",
    "lsr_transform": "lsr_transform",
    "phys_osc": "lsr_synth/phys_osc",
}


def _read_env(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip("\"'")
    return result


def _opened(dataset, indices):
    ordered = sorted(indices)
    rows = np.asarray(dataset[ordered, :], dtype=float)
    lookup = dict(zip(ordered, rows, strict=True))
    return np.asarray([lookup[index] for index in indices])


def _roles(dataset, task: str, seed: int):
    count = int(dataset.shape[0])
    if count < 160:
        raise ValueError("round-two role split requires at least 160 rows")
    order = sorted(range(count), key=lambda index:
        sha256(f"{task}:{seed}:{index}".encode()).digest())
    cuts = tuple(int(round(fraction * count)) for fraction in (
        .40, .55, .65, .75, .80, .85, .90, .95))
    names = (
        "discovery_development", "discovery_validation", "gap_audit",
        "gap_admission", "decision_selector_update",
        "decision_calibration", "inference_initial", "reporting",
        "action_covariates")
    bounds = (0, *cuts, count)
    idx = {
        name: tuple(order[bounds[index]:bounds[index + 1]])
        for index, name in enumerate(names)}
    if len(set().union(*(set(rows) for rows in idx.values()))) != count:
        raise ValueError("round-two role split overlaps or omits rows")
    opened = {name: _opened(dataset, idx[name]) for name in (
        "discovery_development", "discovery_validation", "gap_audit",
        "gap_admission", "decision_selector_update", "inference_initial",
        "action_covariates")}
    fit = RoleDataset(
        DataRole.DEVELOPMENT, opened["discovery_development"][:, 1:],
        opened["discovery_development"][:, 0])
    validation = RoleDataset(
        DataRole.VALIDATION, opened["discovery_validation"][:, 1:],
        opened["discovery_validation"][:, 0])
    action_domain = opened["action_covariates"][:, 1:]
    selection = SelectionData(
        fit, validation, AcquisitionCovariates(
            DataRole.ACQUISITION_POOL, action_domain,
            covariate_fingerprint(action_domain)), ())
    return {
        "selection": selection,
        "gap": RoleDataset(
            DataRole.VALIDATION, opened["gap_audit"][:, 1:],
            opened["gap_audit"][:, 0]),
        "admission": RoleDataset(
            DataRole.VALIDATION, opened["gap_admission"][:, 1:],
            opened["gap_admission"][:, 0]),
        "selector": RoleDataset(
            DataRole.VALIDATION, opened["decision_selector_update"][:, 1:],
            opened["decision_selector_update"][:, 0]),
        "posterior": RoleDataset(
            DataRole.DEVELOPMENT, opened["inference_initial"][:, 1:],
            opened["inference_initial"][:, 0]),
        "action_domain": action_domain,
        "role_row_indices": {
            key: list(value) for key, value in idx.items()},
    }


def _provider(config, env_path: Path):
    values = _read_env(env_path)
    key = (values.get("OPENAI_API_KEY")
           or values.get("HYPOTHESIS_LLM_API_KEY")
           or values.get("API_KEY"))
    if not key:
        raise ValueError("provider key missing")
    return ProviderSettings.from_environment(
        base_url=str(config["llm_api_url"]),
        model=str(config["llm_model"]),
        api_key=key,
        attempts=3,
        connect_timeout_s=15.0,
        read_timeout_s=float(config.get("llm_timeout_s", 300.0)),
        retry_backoff_s=4.0,
        temperature=float(config.get("llm_temperature", .25)),
        max_tokens=int(config.get("llm_max_tokens", 1800)),
        thinking_type=str(config.get("llm_thinking_type") or ""),
        reasoning_effort=str(config.get("llm_reasoning_effort") or ""),
        do_sample=config.get("llm_do_sample"))


def _agent_configs(config, seed: int):
    agent_values = dict(config["agent_config"])
    agent_values["engines"] = tuple(agent_values["engines"])
    agent_values["discovery_islands"] = tuple(
        agent_values["discovery_islands"])
    agent_values["random_seed"] = seed
    composition = replace(
        DiscoveryAgentConfig(**agent_values),
        cycles=1, scientist_orchestration=True,
        posterior_gap_directed=True,
        typed_evidence_synthesis=True,
        typed_inner_augmentation=True,
        expanded_formula_synthesis=True,
        refit_policy="pcpi-expanded-fixed-inner-v1")
    engine = replace(
        composition,
        scientist_orchestration=False,
        posterior_gap_directed=False,
        typed_evidence_synthesis=False,
        typed_inner_augmentation=False,
        expanded_formula_synthesis=False,
        synthesis_evaluation_reserve=0,
        llm_evaluation_reserve=0)
    return engine, composition


def _engine_rows(evidence):
    return [{
        "expression": str(row["expression"]),
        "source": "engine:" + str(row["engine"]),
        "origin": "deterministic",
        "lineage_id": str(row["lineage_id"]),
    } for row in evidence]


def _review(planner, plan, evidence, context=None):
    review, telemetry = planner.review_engine_evidence(
        plan=plan, engine_evidence=evidence,
        require_typed_synthesis=True, allow_interactions=True,
        expanded_formula_synthesis=True,
        scientific_context=context)
    candidates, compiler = compile_evidence_synthesis(
        review.synthesis_directives, evidence, planner.n_features,
        allow_interactions=True,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    return candidates, {
        "review": review.to_dict(),
        "review_identity": review.stable_hash,
        "provider_telemetry": telemetry,
        "compiler": compiler,
        "logical_call_count": planner.call_count,
        "provider_attempt_count": planner.attempt_count,
    }


def _map(bank, posterior):
    member = max(
        posterior.members,
        key=lambda row: (row.probability, row.structure.structure_id))
    expressions = sorted(
        expression for _, expression, identifier in bank.candidate_bindings
        if identifier == member.structure.structure_id)
    if not expressions:
        raise ValueError("MAP support has no expression binding")
    return expressions[0], float(member.probability)


def _write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")


def _admit(core, optional, roles, identity, n_features):
    projection = project_expanded_admission_arms(
        core, optional, roles["selection"].development, roles["gap"],
        roles["admission"], roles["selector"], roles["action_domain"],
        n_features=n_features, identity=identity,
        requested_arms=("independent_protected",))
    candidates = projection["arms"]["independent_protected"]
    model = freeze_expanded_formula_model(
        candidates, n_features=n_features, prior=NormalInverseGammaPrior(),
        exploration_identity=identity,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    posterior = model.engine(model.stable_hash).fit_batch(
        roles["posterior"].X, roles["posterior"].y)
    expression, probability = _map(model, posterior)
    return {
        "bank": [dict(row) for row in candidates],
        "bank_expressions": [str(row["expression"]) for row in candidates],
        "bank_identity": projection["candidate_bank_identity"],
        "model_identity": model.stable_hash,
        "map_expression": expression,
        "map_probability": probability,
        "audit": projection["audits"]["independent_protected"],
        "core_input_count": projection["core_input_count"],
        "eligible_core_count": projection["eligible_core_count"],
        "core_domain_rejections": projection["core_domain_rejections"],
        "optional_domain_rejections":
            projection["optional_domain_rejections"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=GROUPS, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine-stage-source", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    stage_path = args.output_dir / "ENGINE_STAGE.json"
    if args.output_dir.exists() and not args.resume:
        raise ValueError("source-first child output exists; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.engine_stage_source is not None and not stage_path.exists():
        imported = json.loads(args.engine_stage_source.read_text(
            encoding="utf-8"))
        if (imported.get("family"), imported.get("task"),
                imported.get("seed")) != (
                args.family, args.task, args.seed):
            raise ValueError("imported engine checkpoint coordinate changed")
        if (imported.get("ground_truth_expression_opened") is not False
                or imported.get("provider_called") is not False
                or imported.get("engine_executions_per_coordinate") != 1):
            raise ValueError("imported engine checkpoint boundary invalid")
        _write(stage_path, {
            **imported,
            "imported_from": str(args.engine_stage_source.resolve()),
            "imported_sha256": sha256(
                args.engine_stage_source.read_bytes()).hexdigest(),
        })
    if args.resume and not stage_path.is_file():
        raise ValueError("source-first child resume requires engine checkpoint")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    engine_config, composition_config = _agent_configs(
        config, args.seed)
    with h5py.File(args.hdf5, "r") as handle:
        roles = _roles(
            handle[f"{GROUPS[args.family]}/{args.task}/train"],
            args.task, args.seed)
    plan = deterministic_plan(
        engine_config.engines, engine_config.engine_budget)
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if (stage.get("family"), stage.get("task"), stage.get("seed")) != (
                args.family, args.task, args.seed):
            raise ValueError("engine checkpoint coordinate changed")
        evidence = stage["engine_evidence"]
        gap = stage["gap"]
        engine_execution = stage["engine_execution"]
    else:
        engine_agent = DiscoveryAgent(engine_config)
        engine_agent._active_research_plan = plan
        engines = engine_agent._run_engines(roles["selection"], 0)
        evidence = _engine_evidence(engines)
        if not evidence:
            raise ValueError(
                "source-first engine execution produced no evidence")
        provider = _provider(config, args.provider_env)
        composition_agent = DiscoveryAgent(
            composition_config, provider_settings=provider)
        gap = composition_agent._independent_gap_brief(
            roles["selection"], engines, roles["gap"],
            NormalInverseGammaPrior(), 2)
        engine_execution = {
            "best": asdict(engines.best),
            "all_results": [asdict(row) for row in engines.all_results],
            "run_records": [asdict(row) for row in engines.run_records],
            "failures": list(engines.failures),
            "evaluation_budget": engines.evaluation_budget,
            "evaluations_used": engines.evaluations_used,
        }
        _write(stage_path, {
            "schema": "formula-round2-engine-stage-v1",
            "family": args.family, "task": args.task, "seed": args.seed,
            "research_plan_identity": plan.stable_hash,
            "engine_evidence": evidence,
            "engine_execution": engine_execution,
            "gap": gap,
            "engine_executions_per_coordinate": 1,
            "provider_called": False,
            "ground_truth_expression_opened": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
        })
    provider = _provider(config, args.provider_env)
    n_features = roles["selection"].development.X.shape[1]
    blind_planner = ProposalRuntime(
        EquationRuntime(
            n_features, refit_policy="pcpi-expanded-fixed-inner-v1"),
        n_features, provider, 1)
    gap_planner = ProposalRuntime(
        EquationRuntime(
            n_features, refit_policy="pcpi-expanded-fixed-inner-v1"),
        n_features, provider, 1)
    try:
        blind_rows, blind_audit = _review(
            blind_planner, plan, evidence)
    except ProviderInfrastructureError as error:
        _write(args.output_dir / "PROVIDER_FAILURE.json", {
            "schema": "formula-round2-provider-failure-v1",
            "family": args.family, "task": args.task, "seed": args.seed,
            "phase": "blind",
            "error_type": type(error).__name__,
            "diagnostic": error.public_diagnostic,
            "engine_stage_sha256": sha256(
                stage_path.read_bytes()).hexdigest(),
            "candidate_bank_frozen": False,
            "admission_completed": False,
            "ground_truth_expression_opened": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
        })
        raise
    gap_context = {
        "posterior_gap_brief": gap["prompt"],
        "posterior_gap_existing_supports": gap["existing_supports"],
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }
    if gap["prompt"]["propose_allowed"]:
        try:
            gap_rows, gap_audit = _review(
                gap_planner, plan, evidence, gap_context)
        except ProviderInfrastructureError as error:
            _write(args.output_dir / "PROVIDER_FAILURE.json", {
                "schema": "formula-round2-provider-failure-v1",
                "family": args.family, "task": args.task,
                "seed": args.seed, "phase": "gap",
                "error_type": type(error).__name__,
                "diagnostic": error.public_diagnostic,
                "engine_stage_sha256": sha256(
                    stage_path.read_bytes()).hexdigest(),
                "candidate_bank_frozen": False,
                "admission_completed": False,
                "ground_truth_expression_opened": False,
                "test_or_ood_accessed": False,
                "heldout_opened": False,
            })
            raise
    else:
        gap_rows, gap_audit = [], {
            "algorithmic_abstention": True,
            "reason": "independent gap screen did not authorize proposal",
            "logical_call_count": 0,
            "provider_attempt_count": 0,
        }
    core = _engine_rows(evidence)
    source = freeze_source_first_banks(
        core, blind_rows, gap_rows, n_features=n_features)
    operational_core = source["banks"]["multi_engine"]
    blind_admitted = _admit(
        operational_core, blind_rows, roles,
        sha256(f"round2:blind:{args.family}:{args.task}:{args.seed}".encode()
               ).hexdigest(), n_features)
    gap_admitted = _admit(
        operational_core, gap_rows, roles,
        sha256(f"round2:gap:{args.family}:{args.task}:{args.seed}".encode()
               ).hexdigest(), n_features)
    result = {
        "schema": "formula-round2-source-first-child-v1",
        "family": args.family, "task": args.task, "seed": args.seed,
        "feature_count": n_features,
        "research_plan": plan.to_dict(),
        "research_plan_identity": plan.stable_hash,
        "engine_execution": engine_execution,
        "engine_evidence": evidence,
        "engine_evidence_identity": sha256(json.dumps(
            evidence, sort_keys=True, default=str).encode()).hexdigest(),
        "source_first_banks": source,
        "blind_generation": blind_audit,
        "gap_generation": gap_audit,
        "gap_context": gap_context,
        "gap_audit": gap["audit"],
        "admissions": {
            "blind": blind_admitted,
            "gap": gap_admitted,
        },
        "role_row_indices": roles["role_row_indices"],
        "engine_executions_per_coordinate": 1,
        "ground_truth_expression_opened": False,
        "test_or_ood_accessed": False,
        "confirmation_accessed": False,
        "heldout_opened": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "SOURCE_FIRST_CHILD.json", result)
    print(json.dumps({
        "family": args.family, "task": args.task, "seed": args.seed,
        "engine_support_count": len(core),
        "blind_novel_count": source["blind_novel_count"],
        "gap_novel_count": source["gap_novel_count"],
        "ground_truth_expression_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
