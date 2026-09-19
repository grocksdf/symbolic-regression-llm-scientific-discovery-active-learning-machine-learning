"""Registered user-only end-to-end development coordinator.

Reuse production exploration, restricted model freeze and PCPI transactions.
All ablations remain visible; unsupported formulas or uncertified rankings stop
the protocol rather than selecting a convenient surviving subset.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import numpy as np

from hypothesis_mvp.data.system_protocol import load_registered_system_data, validate_data_registration
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.hypotheses import EvidenceRegistry, EvidenceEventType
from hypothesis_mvp.pcpi.discovery_transaction import DiscoveryScoringControls, _publish
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from .agent import DiscoveryAgentConfig
from .proposal_runtime import ProviderSettings
from .resource_limits import run_bounded
from .system_ablation import run_exploration_ablations
from .system_freeze import verify_system_freeze
from .system_run import run_frozen_system_comparison, audit_frozen_hypothesis_bank
from .marginal_influence import (
    audit_marginal_influence, initial_eig_interval_profile,
    leave_one_source_out_candidates, predictive_quality_profile,
)
from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target
from .bank_selection import select_operational_capacity_bank


class HypothesisBankNotViable(RuntimeError):
    def __init__(self):
        super().__init__("response-free-hypothesis-bank-not-viable-no-measurement-authorized")
        self.public_diagnostic = str(self)


class MarginalDecisionInfluenceNotCertified(RuntimeError):
    def __init__(self):
        super().__init__(
            "response-free-marginal-decision-influence-not-certified-no-measurement-authorized"
        )
        self.public_diagnostic = str(self)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def validate_system_registration(config):
    required = {"schema", "data", "seeds", "agent", "single_engine", "prior", "scoring",
        "measurement_budget", "exploration_seconds", "policy_seconds", "data_loading_seconds",
        "provider_attempt_ceiling", "provider_public_identity", "coefficient_policy",
        "user_execution_authorized", "hypothesis_bank_gate", "marginal_influence_gate"}
    if set(config) != required or config["schema"] != "scientific-system-development-registration-v1":
        raise ValueError("unknown or incomplete scientific system registration")
    if (not config["data"] or not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"])
            or any(type(seed) is not int or seed < 0 for seed in config["seeds"])
            or type(config["user_execution_authorized"]) is not bool
            or type(config["measurement_budget"]) is not int or config["measurement_budget"] < 1
            or type(config["provider_attempt_ceiling"]) is not int or config["provider_attempt_ceiling"] < 1
            or config["coefficient_policy"] != "discard-fitted-coefficients-refit-closed-basis"):
        raise ValueError("invalid system seeds/budgets/refit authorization")
    for key in ("exploration_seconds", "policy_seconds", "data_loading_seconds"):
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("invalid registered wall-time ceiling")
    datasets = []
    for registration in config["data"]:
        validate_data_registration(registration)
        datasets.append(registration["dataset"])
        if registration["counts"]["acquisition_pool"] < config["measurement_budget"]:
            raise ValueError("candidate pool smaller than registered measurement budget")
    if len(set(datasets)) != len(datasets):
        raise ValueError("duplicate registered dataset")
    agent = DiscoveryAgentConfig(**config["agent"])
    if (agent.engine_workers != 1 or agent.engine_retries != 0 or agent.acquisition_enabled
            or agent.use_knowledge or agent.cycles < 1 or agent.engine_repeats < 1
            or len(set(agent.engines)) != len(agent.engines) or len(agent.engines) < 2
            or any(e not in {"polynomial_lasso", "mcts"} for e in agent.engines)
            or config["single_engine"] not in agent.engines
            or agent.engine_budget != len(agent.engines) * agent.engine_repeats
            or agent.discovery_budget < 1
            or type(agent.mcts_frontier_size) is not int
            or not 2 <= agent.mcts_frontier_size <= 8
            or agent.mcts_score_folds != 2
            or type(agent.llm_evaluation_reserve) is not int
            or agent.llm_evaluation_reserve < 0
            or agent.llm_evaluation_reserve >= agent.discovery_budget):
        raise ValueError("unmatched or unsupported internal-engine registration")
    if (not agent.discovery_islands or len(set(agent.discovery_islands)) != len(agent.discovery_islands)
            or any(value not in {"balanced", "low_complexity", "nmse", "tail", "novelty"}
                   for value in agent.discovery_islands)):
        raise ValueError("invalid registered discovery objectives")
    NormalInverseGammaPrior(**config["prior"])
    DiscoveryScoringControls(**config["scoring"])
    gate = config["hypothesis_bank_gate"]
    if (set(gate) != {"schema", "exact_eig_epsabs", "require_all_variants",
                      "maximum_candidates"}
            or gate["schema"] != "scientific-hypothesis-bank-gate-v1"
            or gate["exact_eig_epsabs"] != 1e-10
            or gate["maximum_candidates"] != 2 * config["measurement_budget"]
            or gate["require_all_variants"] is not True):
        raise ValueError("invalid hypothesis-bank viability registration")
    influence = config["marginal_influence_gate"]
    if (set(influence) != {"schema", "exact_eig_epsabs", "required_contributions",
                           "decision_rule", "quality_rule", "arbitration_fraction",
                           "require_all_contributions"}
            or influence["schema"] != "scientific-dual-channel-source-contribution-gate-v3"
            or influence["exact_eig_epsabs"] != gate["exact_eig_epsabs"]
            or influence["required_contributions"] != ["llm", "engine:mcts"]
            or influence["decision_rule"] != "full-target-certified-regret-v1"
            or influence["quality_rule"] != "positive-paired-cumulative-log-predictive-ratio-v1"
            or influence["arbitration_fraction"] != 0.5
            or influence["require_all_contributions"] is not True):
        raise ValueError("invalid marginal decision influence registration")
    identity = config["provider_public_identity"]
    if identity is None and config["user_execution_authorized"] is False:
        return config
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("public provider identity must be registered SHA-256")
    return config


def public_provider_identity(settings):
    public = asdict(settings)
    for route in public["routes"]:
        route.pop("api_key", None)
    return _digest(public)


def registered_provider_settings(project_root=None):
    if project_root is not None:
        local_path = Path(project_root) / "config" / "bigmodel.local.json"
        if local_path.is_file():
            return ProviderSettings.from_file(local_path)
        path = Path(project_root) / "config" / "bigmodel_glm_5_2.json"
        if path.is_file():
            # Preserve the production file transport (including thinking,
            # reasoning and retries); malformed files never fall back silently.
            return ProviderSettings.from_file(path)
    # Do not appropriate generic API credentials injected by unrelated apps.
    names = ("HYPOTHESIS_LLM_API_BASE", "HYPOTHESIS_LLM_MODEL", "HYPOTHESIS_LLM_API_KEY")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("missing project LLM settings: " + ", ".join(missing))
    return ProviderSettings.from_environment(base_url=os.environ[names[0]],
        model=os.environ[names[1]], api_key=os.environ[names[2]])


def verify_registered_provider(project_root, config):
    provider = registered_provider_settings(project_root)
    if not provider.routes or public_provider_identity(provider) != config["provider_public_identity"]:
        raise ValueError("provider settings differ from frozen public identity")
    return provider


def _variant_composition(variant, candidates):
    origins = [str(candidate.get("origin", "")) for candidate in candidates]
    engines = {str(candidate.get("source", "")) for candidate in candidates
               if str(candidate.get("source", "")).startswith("engine:")}
    return {
        "llm_enabled_variant_retains_llm_hypothesis": (
            "llm" in origins if variant != "no_llm" else "llm" not in origins
        ),
        "full_variant_retains_multiple_engine_sources": (
            len(engines) >= 2 if variant == "full" else True
        ),
        "no_llm_variant_retains_no_llm_hypothesis": (
            "llm" not in origins if variant == "no_llm" else True
        ),
    }


def _prepare_hypothesis_bank_viability(workspace, exploration, data, config):
    viability = {}
    for row in exploration["rows"]:
        variant = row["variant"]
        selected, selection = select_operational_capacity_bank(
            row["candidates"], data.initial, data.pool.X_pool,
            n_features=data.initial.X.shape[1],
            prior=NormalInverseGammaPrior(**config["prior"]),
            exploration_identity=_digest(row),
            coefficient_policy=config["coefficient_policy"],
            measurement_budget=config["measurement_budget"],
            maximum_candidates=config["hypothesis_bank_gate"]["maximum_candidates"])
        row["candidates"] = list(selected)
        row["hypothesis_provenance"] = {
            **row["hypothesis_provenance"],
            "candidate_count": len(selected),
            "distinct_sources": sorted({item["source"] for item in selected}),
            "engine_sources": sorted({item["source"] for item in selected
                                      if item["source"].startswith("engine:")}),
            "llm_retained_candidate_count": sum(
                item.get("origin") == "llm" for item in selected),
            "operational_capacity_selection": selection,
        }
        _publish(workspace / "exploration" / variant / "FROZEN_BANK.json", {
            "variant": variant, "candidates": list(selected),
            "hypothesis_provenance": row["hypothesis_provenance"],
            "selection": selection})
        audit = audit_frozen_hypothesis_bank(
            row["candidates"], data.initial, data.pool.X_pool,
            n_features=data.initial.X.shape[1],
            prior=NormalInverseGammaPrior(**config["prior"]),
            exploration_identity=_digest(row),
            coefficient_policy=config["coefficient_policy"],
            measurement_budget=config["measurement_budget"],
            exact_eig_epsabs=config["hypothesis_bank_gate"]["exact_eig_epsabs"],
        )
        composition = _variant_composition(variant, row["candidates"])
        audit["composition_decisions"] = composition
        audit["passed"] = bool(audit["passed"] and all(composition.values()))
        viability[variant] = audit
    _publish(workspace / "HYPOTHESIS_BANK_VIABILITY.json", {
        "schema": "scientific-hypothesis-bank-family-gate-v1",
        "variants": viability, "passed": all(row["passed"] for row in viability.values()),
        "candidate_response_accessed": False, "heldout_opened": False,
    })
    if not all(row["passed"] for row in viability.values()):
        raise HypothesisBankNotViable()


def _split_source_arbitration(evaluation, fraction):
    if (evaluation.role is not DataRole.VALIDATION or fraction != 0.5
            or len(evaluation.X) < 4 or len(evaluation.X) % 2):
        raise ValueError("invalid registered source-arbitration split")
    cut = len(evaluation.X) // 2
    arbitration = RoleDataset(DataRole.VALIDATION,
                              evaluation.X[:cut], evaluation.y[:cut])
    reporting = RoleDataset(DataRole.VALIDATION,
                            evaluation.X[cut:], evaluation.y[cut:])
    if arbitration.row_fingerprints & reporting.row_fingerprints:
        raise ValueError("source arbitration and reporting evaluation overlap")
    return arbitration, reporting


def _prepare_marginal_decision_influence(workspace, exploration, data, config,
                                         arbitration):
    gate = config["marginal_influence_gate"]
    prior = NormalInverseGammaPrior(**config["prior"])
    rows = {row["variant"]: row for row in exploration["rows"]}
    if set(rows) != {"full", "no_llm", "single_engine"}:
        raise ValueError("unexpected exploration variants for marginal influence")
    full = rows["full"]
    candidates = {"full": full["candidates"]}
    for contribution in gate["required_contributions"]:
        candidates[f"full_without_{contribution.replace(':', '_')}"] = (
            leave_one_source_out_candidates(full["candidates"], contribution))
    profiles, quality_profiles = {}, {}
    for variant, bank in candidates.items():
        model = freeze_discovery_model(bank, n_features=data.initial.X.shape[1],
            prior=prior, exploration_identity=_digest(full),
            coefficient_policy=config["coefficient_policy"])
        target = freeze_discovery_target(model, data.initial, data.pool.X_pool,
            measurement_budget=config["measurement_budget"],
            expected_model_identity=model.stable_hash)
        profiles[variant] = initial_eig_interval_profile(
            variant, model, target, data.pool.X_pool,
            exact_eig_epsabs=gate["exact_eig_epsabs"])
        quality_profiles[variant] = predictive_quality_profile(
            variant, model, target, arbitration)
    report = audit_marginal_influence(
        profiles, quality_profiles=quality_profiles,
        required_contributions=tuple(gate["required_contributions"]))
    _publish(workspace / "MARGINAL_DECISION_INFLUENCE.json", report)
    if report["passed"] is not True:
        raise MarginalDecisionInfluenceNotCertified()
    return report


def _run_comparison_variant(project_root, workspace, row, data, config, provider,
                            source_identity, seed, expected_freeze, coordinate,
                            evaluation_data):
    variant = row["variant"]
    verify_system_freeze(project_root, config, expected_freeze)
    if not row["candidates"]:
        raise ValueError("discovery retained no registered hypotheses")
    print(f"system measured comparison: {coordinate}:{variant}", flush=True)
    comparison = run_frozen_system_comparison(
        workspace / "measured" / variant, row["candidates"], data.initial, data.pool,
        np.arange(len(data.pool.X_pool)), n_features=data.initial.X.shape[1],
        prior=NormalInverseGammaPrior(**config["prior"]),
        exploration_identity=_digest(row), coefficient_policy=config["coefficient_policy"],
        measurement_budget=config["measurement_budget"],
        controls=DiscoveryScoringControls(**config["scoring"]),
        source_identity=source_identity, random_seed=seed,
        policy_wall_time_seconds=config["policy_seconds"], evaluation_data=evaluation_data)
    registry = EvidenceRegistry(workspace / "exploration" / variant / "evidence_registry.jsonl")
    if not registry.verify().valid or not registry.events():
        raise ValueError("missing or invalid discovery evidence chain")
    payload = {"stage": "system_measured_development", "comparison": comparison,
        "freeze_identity": source_identity, "data_manifest_identity": _digest(data.manifest),
        "heldout_opened": False, "independent_confirmation": False}
    if not any(event.to_dict()["payload"] == payload for event in registry.events()):
        registry.append(hypothesis_id=registry.events()[-1].hypothesis_id,
            event_type=EvidenceEventType.EVIDENCE_ATTACHED, payload=payload)
    if not registry.verify().valid:
        raise ValueError("system evidence export verification failed")
    return comparison


def execute_registered_system(project_root, root, config, expected_freeze, *, execution_role):
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_system_registration(config)
    if execution_role != "user" or config["user_execution_authorized"] is not True:
        raise PermissionError("real development execution requires registered user authorization")
    project_root, root = Path(project_root).resolve(), Path(root).resolve()
    if root == project_root or project_root in root.parents:
        raise ValueError("experiment output must be outside the frozen source worktree")
    # Environment-resolved discovery options must not override registered bytes.
    if any(name.startswith("HYPOTHESIS_DISCOVERY_") for name in os.environ):
        raise ValueError("unregistered discovery environment overrides are forbidden")
    verify_system_freeze(project_root, config, expected_freeze)
    provider = verify_registered_provider(project_root, config)
    source_identity = _digest(expected_freeze)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "TERMINAL_FAILURE.json").exists():
        raise ValueError("terminally failed system protocol cannot resume")
    _publish(root / "SYSTEM_CONTRACT.json", {"registration": config, "freeze": expected_freeze})
    results = []
    coordinate = None
    try:
        for registration in config["data"]:
            dataset = registration["dataset"]
            print(f"system data loading: {dataset} (opened development only)", flush=True)
            data, _ = run_bounded(load_registered_system_data, args=(registration,),
                seconds=config["data_loading_seconds"], provider_attempts=0)
            for seed in config["seeds"]:
                coordinate = f"{dataset}:{seed}"
                verify_system_freeze(project_root, config, expected_freeze)
                workspace = root / dataset / str(seed)
                workspace.mkdir(parents=True, exist_ok=True)
                _publish(workspace / "DATA_MANIFEST.json", data.manifest)
                agent_config = replace(DiscoveryAgentConfig(**config["agent"]), random_seed=seed)
                print(f"system exploration: {coordinate}", flush=True)
                exploration = run_exploration_ablations(workspace / "exploration", data.selection,
                    dataset=dataset, config=agent_config, provider_settings=provider,
                    single_engine=config["single_engine"], compute_ceiling=config["exploration_seconds"],
                    provider_attempt_ceiling=config["provider_attempt_ceiling"], source_identity=source_identity,
                    scientific_context=data.manifest["scientific_context"])
                _prepare_hypothesis_bank_viability(workspace, exploration, data, config)
                arbitration, reporting_evaluation = _split_source_arbitration(
                    data.evaluation, config["marginal_influence_gate"]["arbitration_fraction"])
                _prepare_marginal_decision_influence(
                    workspace, exploration, data, config, arbitration)
                comparisons = {row["variant"]: _run_comparison_variant(
                    project_root, workspace, row, data, config, provider,
                    source_identity, seed, expected_freeze, coordinate,
                    reporting_evaluation
                ) for row in exploration["rows"]}
                results.append({"dataset": dataset, "seed": seed, "family": data.manifest["family"],
                    "exploration": exploration, "comparisons": comparisons})
        verify_system_freeze(project_root, config, expected_freeze)
        result = {"schema": "scientific-system-development-terminal-v1", "results": results,
            "protocol_complete": True, "heldout_opened": False, "superiority_demonstrated": False,
            "claim_boundary": "conditional opened-development evidence; no confirmatory claim"}
        _publish(root / "SYSTEM_MANIFEST.json", result)
        return result
    except BaseException as error:
        _publish(root / "TERMINAL_FAILURE.json", {"coordinate": coordinate,
            "error_type": type(error).__name__, "completed_coordinates": len(results),
            "protocol_complete": False, "heldout_opened": False, "superiority_demonstrated": False})
        raise


def _sha256_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _verify_zero_response_continuation(source_root, config, continuation):
    """Bind one terminal pre-decision failure without mutating its artifacts."""
    source_root = Path(source_root).resolve()
    required = {"schema", "source_output", "artifacts", "claim_boundary"}
    if (set(continuation) != required
            or continuation["schema"] != "scientific-zero-response-continuation-v1"
            or Path(continuation["source_output"]).resolve() != source_root
            or continuation["claim_boundary"] !=
            "resume immutable passed source screen after pre-decision numerical failure only"):
        raise ValueError("invalid scientific continuation registration")
    artifacts = continuation["artifacts"]
    if len(config["data"]) != 1 or len(config["seeds"]) != 1:
        raise ValueError("continuation requires one registered coordinate")
    dataset, seed = config["data"][0]["dataset"], config["seeds"][0]
    coordinate_root = f"{dataset}/{seed}"
    expected_names = {
        "SYSTEM_CONTRACT.json", "TERMINAL_FAILURE.json",
        f"{coordinate_root}/exploration/ANALYSIS.json",
        f"{coordinate_root}/HYPOTHESIS_BANK_VIABILITY.json",
        f"{coordinate_root}/MARGINAL_DECISION_INFLUENCE.json",
        f"{coordinate_root}/measured/full/TERMINAL_FAILURE.json",
    }
    if set(artifacts) != expected_names:
        raise ValueError("continuation artifact set changed")
    for relative, expected_hash in artifacts.items():
        path = source_root / Path(relative)
        if (not path.is_file() or _sha256_file(path) != expected_hash):
            raise ValueError("continuation source artifact identity changed")
    contract = json.loads((source_root / "SYSTEM_CONTRACT.json").read_text(encoding="utf-8"))
    if contract.get("registration") != config:
        raise ValueError("continuation source registration changed")
    terminal = json.loads((source_root / "TERMINAL_FAILURE.json").read_text(encoding="utf-8"))
    comparison_failure = json.loads((source_root /
        coordinate_root / "measured/full/TERMINAL_FAILURE.json").read_text(
            encoding="utf-8"))
    if (terminal != {"completed_coordinates": 0,
            "coordinate": f"{dataset}:{seed}", "error_type": "RuntimeError",
            "heldout_opened": False, "protocol_complete": False,
            "superiority_demonstrated": False}
            or comparison_failure.get("policy") != "class_eig"
            or comparison_failure.get("completed_policies") != {}
            or comparison_failure.get("heldout_opened") is not False):
        raise ValueError("continuation is not the registered pre-decision failure")
    forbidden = {"DECISION-001.json", "RECEIPT-001.json", "RUN_MANIFEST.json",
                 "DEVELOPMENT_CURVE.json", "COMPARISON_MANIFEST.json", "SYSTEM_MANIFEST.json"}
    found = sorted(path.name for path in source_root.rglob("*")
                   if path.is_file() and path.name in forbidden)
    if found:
        raise ValueError("continuation source contains response or completed-run artifacts")
    workspace = source_root / dataset / str(seed)
    viability = json.loads((workspace / "HYPOTHESIS_BANK_VIABILITY.json").read_text(
        encoding="utf-8"))
    influence = json.loads((workspace / "MARGINAL_DECISION_INFLUENCE.json").read_text(
        encoding="utf-8"))
    analysis = json.loads((workspace / "exploration/ANALYSIS.json").read_text(
        encoding="utf-8"))
    if (viability.get("passed") is not True or influence.get("passed") is not True
            or influence.get("candidate_response_accessed") is not False
            or influence.get("acquisition_pool_response_accessed") is not False
            or influence.get("heldout_opened") is not False
            or analysis.get("heldout_opened") is not False
            or {row.get("variant") for row in analysis.get("rows", [])}
                != {"full", "no_llm", "single_engine"}
            or any(row.get("status") != "succeeded" for row in analysis["rows"])):
        raise ValueError("continuation source screen did not pass response-free gates")
    for row in analysis["rows"]:
        registry = EvidenceRegistry(
            workspace / "exploration" / row["variant"] / "evidence_registry.jsonl")
        if not registry.verify().valid or not registry.events():
            raise ValueError("continuation source evidence registry is invalid")
    return source_root, workspace, analysis


def execute_registered_system_continuation(project_root, root, source_root, config,
                                           expected_freeze, continuation, *, execution_role):
    """Continue an immutable zero-response screen without rerunning discovery."""
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_system_registration(config)
    if execution_role != "user" or config["user_execution_authorized"] is not True:
        raise PermissionError("real development continuation requires user authorization")
    project_root, root = Path(project_root).resolve(), Path(root).resolve()
    if root == project_root or project_root in root.parents or root.exists():
        raise ValueError("continuation output must be a new path outside source")
    verify_system_freeze(project_root, config, expected_freeze)
    source_root, source_workspace, exploration = _verify_zero_response_continuation(
        source_root, config, continuation)
    if root == source_root or root in source_root.parents or source_root in root.parents:
        raise ValueError("continuation output and immutable source must be disjoint")
    source_identity = _digest(expected_freeze)
    root.mkdir(parents=True)
    _publish(root / "CONTINUATION_CONTRACT.json", continuation)
    results, coordinate = [], None
    try:
        registration = config["data"][0]
        dataset, seed = registration["dataset"], config["seeds"][0]
        print(f"system continuation data loading: {dataset} (opened development only)", flush=True)
        data, _ = run_bounded(load_registered_system_data, args=(registration,),
            seconds=config["data_loading_seconds"], provider_attempts=0)
        coordinate = f"{dataset}:{seed}"
        workspace = root / dataset / str(seed)
        workspace.mkdir(parents=True)
        source_manifest = json.loads((source_workspace / "DATA_MANIFEST.json").read_text(
            encoding="utf-8"))
        if data.manifest != source_manifest:
            raise ValueError("continuation data manifest changed")
        _publish(workspace / "DATA_MANIFEST.json", data.manifest)
        for name in ("HYPOTHESIS_BANK_VIABILITY.json", "MARGINAL_DECISION_INFLUENCE.json"):
            shutil.copyfile(source_workspace / name, workspace / name)
        (workspace / "exploration").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_workspace / "exploration/ANALYSIS.json",
                        workspace / "exploration/ANALYSIS.json")
        for row in exploration["rows"]:
            source_variant = source_workspace / "exploration" / row["variant"]
            target_variant = workspace / "exploration" / row["variant"]
            source_registry = source_variant / "evidence_registry.jsonl"
            target_registry = target_variant / "evidence_registry.jsonl"
            target_registry.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_registry, target_registry)
            result_payload = json.loads((source_variant / "RESULT.json").read_text(
                encoding="utf-8"))
            result_payload["evidence_registry_path"] = str(target_registry)
            result_payload["zero_response_continuation_source"] = str(source_variant)
            _publish(target_variant / "RESULT.json", result_payload)
        _, reporting_evaluation = _split_source_arbitration(
            data.evaluation, config["marginal_influence_gate"]["arbitration_fraction"])
        comparisons = {row["variant"]: _run_comparison_variant(
            project_root, workspace, row, data, config, None, source_identity, seed,
            expected_freeze, coordinate, reporting_evaluation)
            for row in exploration["rows"]}
        results.append({"dataset": dataset, "seed": seed, "family": data.manifest["family"],
            "exploration": exploration, "comparisons": comparisons,
            "continuation_source": str(source_root)})
        verify_system_freeze(project_root, config, expected_freeze)
        result = {"schema": "scientific-system-development-terminal-v1", "results": results,
            "protocol_complete": True, "heldout_opened": False,
            "superiority_demonstrated": False,
            "zero_response_continuation": True,
            "claim_boundary": "conditional opened-development evidence; no confirmatory claim"}
        _publish(root / "SYSTEM_MANIFEST.json", result)
        return result
    except BaseException as error:
        _publish(root / "TERMINAL_FAILURE.json", {"coordinate": coordinate,
            "error_type": type(error).__name__, "completed_coordinates": len(results),
            "protocol_complete": False, "heldout_opened": False,
            "superiority_demonstrated": False})
        raise
