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

import numpy as np

from hypothesis_mvp.data.system_protocol import load_registered_system_data, validate_data_registration
from hypothesis_mvp.hypotheses import EvidenceRegistry, EvidenceEventType
from hypothesis_mvp.pcpi.discovery_transaction import DiscoveryScoringControls, _publish
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from .agent import DiscoveryAgentConfig
from .proposal_runtime import ProviderSettings
from .resource_limits import run_bounded
from .system_ablation import run_exploration_ablations
from .system_freeze import verify_system_freeze
from .system_run import run_frozen_system_comparison


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def validate_system_registration(config):
    required = {"schema", "data", "seeds", "agent", "single_engine", "prior", "scoring",
        "measurement_budget", "exploration_seconds", "policy_seconds", "data_loading_seconds",
        "provider_attempt_ceiling", "provider_public_identity", "coefficient_policy",
        "user_execution_authorized"}
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
            or agent.discovery_budget < 1):
        raise ValueError("unmatched or unsupported internal-engine registration")
    NormalInverseGammaPrior(**config["prior"])
    DiscoveryScoringControls(**config["scoring"])
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


def registered_provider_settings():
    # Do not appropriate generic API credentials injected by unrelated apps.
    names = ("HYPOTHESIS_LLM_API_BASE", "HYPOTHESIS_LLM_MODEL", "HYPOTHESIS_LLM_API_KEY")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("missing project LLM settings: " + ", ".join(missing))
    return ProviderSettings.from_environment(base_url=os.environ[names[0]],
        model=os.environ[names[1]], api_key=os.environ[names[2]])


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
    provider = registered_provider_settings()
    if not provider.routes or public_provider_identity(provider) != config["provider_public_identity"]:
        raise ValueError("provider settings differ from frozen public identity")
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
                    provider_attempt_ceiling=config["provider_attempt_ceiling"], source_identity=source_identity)
                comparisons = {}
                for row in exploration["rows"]:
                    verify_system_freeze(project_root, config, expected_freeze)
                    variant = row["variant"]
                    if not row["candidates"]:
                        raise ValueError("discovery retained no registered hypotheses")
                    print(f"system measured comparison: {coordinate}:{variant}", flush=True)
                    comparisons[variant] = run_frozen_system_comparison(workspace / "measured" / variant,
                        row["candidates"], data.initial, data.pool, np.arange(len(data.pool.X_pool)),
                        n_features=data.initial.X.shape[1], prior=NormalInverseGammaPrior(**config["prior"]),
                        exploration_identity=_digest(row), coefficient_policy=config["coefficient_policy"],
                        measurement_budget=config["measurement_budget"],
                        controls=DiscoveryScoringControls(**config["scoring"]), source_identity=source_identity,
                        random_seed=seed, policy_wall_time_seconds=config["policy_seconds"],
                        evaluation_data=data.evaluation)
                    registry_path = workspace / "exploration" / variant / "evidence_registry.jsonl"
                    registry = EvidenceRegistry(registry_path)
                    if not registry.verify().valid or not registry.events():
                        raise ValueError("missing or invalid discovery evidence chain")
                    payload = {"stage": "system_measured_development", "comparison": comparisons[variant],
                        "freeze_identity": source_identity, "data_manifest_identity": _digest(data.manifest),
                        "heldout_opened": False, "independent_confirmation": False}
                    # Deterministic complete recovery does not append a duplicate.
                    if not any(event.to_dict()["payload"] == payload for event in registry.events()):
                        registry.append(hypothesis_id=registry.events()[-1].hypothesis_id,
                            event_type=EvidenceEventType.EVIDENCE_ATTACHED, payload=payload)
                    if not registry.verify().valid:
                        raise ValueError("system evidence export verification failed")
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
