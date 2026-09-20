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
from .system_run import (
    run_frozen_system_comparison, audit_frozen_hypothesis_bank,
    audit_frozen_decision_risk_utility,
)
from .marginal_influence import (
    audit_marginal_influence, initial_eig_interval_profile,
    leave_one_source_out_candidates, predictive_quality_profile,
)
from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target
from .bank_selection import select_operational_capacity_bank
from .source_stacking import (
    calibrate_source_admission, filter_fold_safe_source_candidates, source_family,
)


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
    if (not required <= set(config) or set(config) - required > {"targeted_query_policy"}
            or config["schema"] != "scientific-system-development-registration-v1"):
        raise ValueError("unknown or incomplete scientific system registration")
    if config.get("targeted_query_policy", "class_eig") not in {
            "class_eig", "decision_risk"}:
        raise ValueError("invalid targeted query policy")
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
                      "maximum_candidates", "selection_rule", "source_safety_folds",
                      "source_stacking_baseline", "source_stacking_dyadic_depth",
                      "source_stacking_max_optional_mass"}
            or gate["schema"] != "scientific-hypothesis-bank-gate-v4"
            or gate["exact_eig_epsabs"] != 1e-10
            or gate["maximum_candidates"] != 2 * config["measurement_budget"]
            or gate["selection_rule"] !=
                "two-fold-safe-half-core-source-stacking-operational-entropy-v3"
            or gate["source_safety_folds"] != 2
            or gate["source_stacking_baseline"] != "core"
            or gate["source_stacking_dyadic_depth"] != 8
            or gate["source_stacking_max_optional_mass"] != 0.5
            or gate["require_all_variants"] is not True):
        raise ValueError("invalid hypothesis-bank viability registration")
    influence = config["marginal_influence_gate"]
    if (set(influence) != {"schema", "exact_eig_epsabs", "required_contributions",
                           "decision_rule", "quality_rule", "arbitration_fraction",
                           "require_all_contributions", "source_admission_rule",
                           "required_active_contributions", "rejectable_contributions"}
            or influence["schema"] != "scientific-source-admission-influence-gate-v5"
            or influence["exact_eig_epsabs"] != gate["exact_eig_epsabs"]
            or influence["required_contributions"] != ["llm", "engine:mcts"]
            or influence["decision_rule"] != "full-target-certified-regret-v1"
            or influence["quality_rule"] != "positive-paired-cumulative-log-predictive-ratio-v1"
            or influence["arbitration_fraction"] != 0.5
            or influence["source_admission_rule"] !=
                "independent-sourcewise-fold-safe-half-core-log-score-stacking-v2"
            or influence["required_active_contributions"] != ["llm"]
            or influence["rejectable_contributions"] != ["engine:mcts"]
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
            maximum_candidates=config["hypothesis_bank_gate"]["maximum_candidates"],
            source_safety_roles=(),
            source_safety_folds=config["hypothesis_bank_gate"]["source_safety_folds"])
        row["candidates"] = list(selected)
        row["source_prior_weights"] = selection["source_prior_weights"]
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
        _publish(workspace / "exploration" / variant / "H0_CAPACITY_BANK.json", {
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
            source_prior_weights=row["source_prior_weights"],
        )
        composition = _variant_composition(variant, row["candidates"])
        audit["composition_decisions"] = composition
        audit["passed"] = bool(audit["passed"] and all(composition.values())
                               and selection["source_safety_passed"])
        viability[variant] = audit
    _publish(workspace / "H0_HYPOTHESIS_BANK_VIABILITY.json", {
        "schema": "scientific-h0-hypothesis-bank-family-gate-v1",
        "variants": viability, "passed": all(row["passed"] for row in viability.values()),
        "candidate_response_accessed": False, "heldout_opened": False,
    })
    if not all(row["passed"] for row in viability.values()):
        raise HypothesisBankNotViable()


def _prepare_candidate_admission(workspace, exploration, data, config, arbitration):
    reports = {}
    prior = NormalInverseGammaPrior(**config["prior"])
    for row in exploration["rows"]:
        retained, report = filter_fold_safe_source_candidates(
            row["candidates"], data.initial, arbitration,
            n_features=data.initial.X.shape[1], prior=prior,
            exploration_identity=_digest(row),
            coefficient_policy=config["coefficient_policy"],
            measurement_budget=config["measurement_budget"],
            action_domain=data.pool.X_pool)
        row["candidates"] = list(retained)
        reports[row["variant"]] = report
    family = {
        "schema": "scientific-independent-candidatewise-admission-family-v1",
        "variants": reports, "candidate_response_accessed": False,
        "reporting_validation_response_accessed": False, "heldout_opened": False,
    }
    _publish(workspace / "CANDIDATE_ADMISSION.json", family)
    return family


def _required_selection_roles(candidates):
    roles = {str(row["source"]) for row in candidates
             if str(row["source"]).startswith("engine:")}
    if any(str(row.get("origin", "")) == "llm" for row in candidates):
        roles.add("origin:llm")
    return roles


def _selection_safety_roles(variant, candidates, registered_safety):
    """Apply registered leave-source-out safety only to the full production bank."""
    if variant not in {"full", "no_llm", "single_engine"}:
        raise ValueError("unknown scientific-system ablation variant")
    if variant != "full":
        return ()
    available = _required_selection_roles(candidates)
    selected = tuple(role for role in registered_safety if role in available)
    if selected != tuple(registered_safety):
        raise ValueError("full bank lost a registered source-safety role")
    return selected


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


def _renormalized_source_weights(weights, candidates):
    """Condition one frozen hierarchical prior on sources retained by an ablation."""
    available = {source_family(row) for row in candidates}
    selected = {family: float(weights.get(family, 0.0)) for family in available}
    total = float(sum(selected.values()))
    if total <= 0.0:
        raise ValueError("source ablation removed all hierarchical prior mass")
    return {key: value / total for key, value in selected.items()}


def _prepare_source_admission(workspace, exploration, data, config, arbitration):
    """Calibrate production source mass without reporting or pool responses."""
    prior = NormalInverseGammaPrior(**config["prior"])
    reports, viability = {}, {}
    gate = config["marginal_influence_gate"]
    for row in exploration["rows"]:
        certificate, sources = calibrate_source_admission(
            row["candidates"], data.initial, arbitration,
            n_features=data.initial.X.shape[1], prior=prior,
            exploration_identity=_digest(row),
            coefficient_policy=config["coefficient_policy"],
            measurement_budget=config["measurement_budget"],
            action_domain=data.pool.X_pool)
        row["source_prior_weights"] = certificate.source_weights
        variant = row["variant"]
        required = set(gate["required_active_contributions"]) if variant == "full" else set()
        rejected = set(gate["rejectable_contributions"]) if variant == "full" else set()
        decisions = {
            "core_reserve_satisfied": bool(
                certificate.source_weights.get("core", 0.0) >= 0.5 - 2e-12),
            "required_sources_admitted": all(sources.get(name, {}).get("admitted")
                                               for name in required),
            "rejectable_sources_admitted_or_negative_transfer_certified": all(
                sources.get(name, {}).get("admitted")
                or sources.get(name, {}).get("negative_transfer_certified")
                for name in rejected),
        }
        report = {"schema": "scientific-independent-sourcewise-admission-v2",
            "variant": variant, "stacking": certificate.to_dict(),
            "stacking_identity": certificate.stable_hash, "sources": sources,
            "decisions": decisions, "passed": all(decisions.values()),
            "initial_development_response_accessed": True,
            "source_arbitration_validation_responses_accessed": True,
            "reporting_validation_response_accessed": False,
            "candidate_response_accessed": False, "heldout_opened": False}
        reports[variant] = report
        _publish(workspace / "exploration" / variant / "FROZEN_BANK.json", {
            "variant": variant, "candidates": row["candidates"],
            "hypothesis_provenance": row["hypothesis_provenance"],
            "source_admission": report,
            "source_prior_weights": row["source_prior_weights"]})
        audit = audit_frozen_hypothesis_bank(
            row["candidates"], data.initial, data.pool.X_pool,
            n_features=data.initial.X.shape[1], prior=prior,
            exploration_identity=_digest(row),
            coefficient_policy=config["coefficient_policy"],
            measurement_budget=config["measurement_budget"],
            exact_eig_epsabs=config["hypothesis_bank_gate"]["exact_eig_epsabs"],
            source_prior_weights=row["source_prior_weights"])
        audit["source_admission_identity"] = report["stacking_identity"]
        audit["passed"] = bool(audit["passed"] and report["passed"])
        viability[variant] = audit
    family = {"schema": "scientific-independent-sourcewise-admission-family-v2",
        "variants": reports, "passed": all(row["passed"] for row in reports.values()),
        "reporting_validation_response_accessed": False,
        "candidate_response_accessed": False, "heldout_opened": False}
    _publish(workspace / "SOURCE_ADMISSION.json", family)
    _publish(workspace / "HYPOTHESIS_BANK_VIABILITY.json", {
        "schema": "scientific-admitted-hypothesis-bank-family-gate-v1",
        "variants": viability,
        "passed": all(row["passed"] for row in viability.values()),
        "candidate_response_accessed": False, "heldout_opened": False})
    if not family["passed"] or not all(row["passed"] for row in viability.values()):
        raise HypothesisBankNotViable()
    return family


def _bind_source_admission_to_influence(report, admission):
    full = admission["variants"]["full"]
    for name, comparison in report["comparisons"].items():
        source = full["sources"][name]
        comparison["source_admission"] = source
        if not source["admitted"]:
            comparison["pre_admission_influence_passed"] = comparison["passed"]
            comparison["accepted_contribution_role"] = "rejected-negative-transfer"
            comparison["passed"] = bool(source["weight"] <= 2e-12
                                         and source["negative_transfer_certified"])
    report["schema"] = "scientific-source-admission-and-influence-family-gate-v1"
    report["passed"] = all(row["passed"] for row in report["comparisons"].values())
    report["source_admission_identity"] = _digest(full)
    return report


def _prepare_marginal_decision_influence(workspace, exploration, data, config,
                                         arbitration, admission):
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
        source_weights = _renormalized_source_weights(
            full["source_prior_weights"], bank)
        # A leave-one-source-out counterfactual may legitimately condition to
        # one remaining support.  That is a zero-capacity diagnostic model,
        # not a production hypothesis bank.  Keeping it explicit lets the
        # dual-channel audit score predictive quality and report no certified
        # EIG leader instead of crashing or inventing a second hypothesis.
        minimum_supports = 2 if variant == "full" else 1
        model = freeze_discovery_model(bank, n_features=data.initial.X.shape[1],
            prior=prior, exploration_identity=_digest(full),
            coefficient_policy=config["coefficient_policy"],
            source_prior_weights=source_weights,
            minimum_supports=minimum_supports)
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
    report = _bind_source_admission_to_influence(report, admission)
    _publish(workspace / "MARGINAL_DECISION_INFLUENCE.json", report)
    if report["passed"] is not True:
        raise MarginalDecisionInfluenceNotCertified()
    return report


def _prepare_source_gates(workspace, exploration, data, config, arbitration):
    admission = _prepare_source_admission(
        workspace, exploration, data, config, arbitration)
    return _prepare_marginal_decision_influence(
        workspace, exploration, data, config, arbitration, admission)


def _prepare_decision_risk_utility_gate(workspace, exploration, data, config):
    if config.get("targeted_query_policy", "class_eig") != "decision_risk":
        return None
    reports = {}
    for row in exploration["rows"]:
        reports[row["variant"]] = audit_frozen_decision_risk_utility(
            row["candidates"], data.initial, data.pool.X_pool,
            n_features=data.initial.X.shape[1],
            prior=NormalInverseGammaPrior(**config["prior"]),
            exploration_identity=_digest(row),
            coefficient_policy=config["coefficient_policy"],
            measurement_budget=config["measurement_budget"],
            exact_epsabs=config["hypothesis_bank_gate"]["exact_eig_epsabs"],
            source_prior_weights=row["source_prior_weights"])
    family = {
        "schema": "scientific-response-free-decision-risk-utility-family-gate-v1",
        "variants": reports, "passed": all(row["passed"] for row in reports.values()),
        "candidate_response_accessed": False, "heldout_opened": False,
    }
    _publish(workspace / "DECISION_RISK_UTILITY_VIABILITY.json", family)
    if family["passed"] is not True:
        raise HypothesisBankNotViable()
    return family


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
        policy_wall_time_seconds=config["policy_seconds"], evaluation_data=evaluation_data,
        source_prior_weights=row["source_prior_weights"],
        policies=(config.get("targeted_query_policy", "class_eig"), "random"))
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


def execute_registered_system(project_root, root, config, expected_freeze, *, execution_role,
                              measurement_authorized=True):
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_system_registration(config)
    if type(measurement_authorized) is not bool:
        raise TypeError("system measurement authorization must be boolean")
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
                arbitration, reporting_evaluation = _split_source_arbitration(
                    data.evaluation, config["marginal_influence_gate"]["arbitration_fraction"])
                candidate_admission = _prepare_candidate_admission(
                    workspace, exploration, data, config, arbitration)
                _prepare_hypothesis_bank_viability(workspace, exploration, data, config)
                _prepare_source_gates(workspace, exploration, data, config, arbitration)
                utility_gate = _prepare_decision_risk_utility_gate(
                    workspace, exploration, data, config)
                if measurement_authorized is not True:
                    results.append({"dataset": dataset, "seed": seed,
                        "family": data.manifest["family"],
                        "data_manifest": _digest(data.manifest),
                        "candidate_admission": _digest(candidate_admission),
                        "h0_bank_viability": _digest(json.loads((workspace /
                            "H0_HYPOTHESIS_BANK_VIABILITY.json").read_text(encoding="utf-8"))),
                        "source_admission": _digest(json.loads((workspace /
                            "SOURCE_ADMISSION.json").read_text(encoding="utf-8"))),
                        "marginal_influence": _digest(json.loads((workspace /
                            "MARGINAL_DECISION_INFLUENCE.json").read_text(encoding="utf-8"))),
                        **({"decision_risk_utility": _digest(utility_gate)}
                           if utility_gate is not None else {})})
                    continue
                comparisons = {row["variant"]: _run_comparison_variant(
                    project_root, workspace, row, data, config, provider,
                    source_identity, seed, expected_freeze, coordinate,
                    reporting_evaluation
                ) for row in exploration["rows"]}
                results.append({"dataset": dataset, "seed": seed, "family": data.manifest["family"],
                    "exploration": exploration, "comparisons": comparisons})
        verify_system_freeze(project_root, config, expected_freeze)
        if measurement_authorized is not True:
            result = {"schema": "scientific-system-fresh-source-screen-v1",
                "results": results, "protocol_complete": True,
                "measurement_authorized": False, "candidate_response_accessed": False,
                "reporting_validation_response_accessed": False,
                "heldout_opened": False, "superiority_demonstrated": False,
                "claim_boundary": "fresh development source screen only; no acquisition response"}
            _publish(root / "SCREEN_MANIFEST.json", result)
            return result
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


def _passed_screen_artifact_names(config):
    if len(config["data"]) != 1 or len(config["seeds"]) != 1:
        raise ValueError("passed-screen continuation requires one coordinate")
    coordinate = f'{config["data"][0]["dataset"]}/{config["seeds"][0]}'
    names = {"SYSTEM_CONTRACT.json", "SCREEN_MANIFEST.json",
        f"{coordinate}/DATA_MANIFEST.json",
        f"{coordinate}/H0_HYPOTHESIS_BANK_VIABILITY.json",
        f"{coordinate}/HYPOTHESIS_BANK_VIABILITY.json",
        f"{coordinate}/CANDIDATE_ADMISSION.json",
        f"{coordinate}/SOURCE_ADMISSION.json",
        f"{coordinate}/MARGINAL_DECISION_INFLUENCE.json",
        f"{coordinate}/exploration/ANALYSIS.json"}
    if config.get("targeted_query_policy") == "decision_risk":
        names.add(f"{coordinate}/DECISION_RISK_UTILITY_VIABILITY.json")
    for variant in ("full", "no_llm", "single_engine"):
        base = f"{coordinate}/exploration/{variant}"
        names.update({f"{base}/RESULT.json", f"{base}/evidence_registry.jsonl",
                      f"{base}/H0_CAPACITY_BANK.json", f"{base}/FROZEN_BANK.json"})
    return names


def _verify_passed_source_screen(source_root, config, continuation):
    source_root = Path(source_root).resolve()
    if (set(continuation) != {"schema", "source_output", "artifacts", "claim_boundary"}
            or continuation["schema"] != "scientific-passed-source-screen-continuation-v2"
            or Path(continuation["source_output"]).resolve() != source_root
            or continuation["claim_boundary"] !=
            "continue immutable passed sourcewise screen without rerunning exploration"):
        raise ValueError("invalid passed-screen continuation registration")
    if set(continuation["artifacts"]) != _passed_screen_artifact_names(config):
        raise ValueError("passed-screen continuation artifact set changed")
    for relative, expected in continuation["artifacts"].items():
        path = source_root / Path(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("passed-screen continuation artifact identity changed")
    contract = json.loads((source_root / "SYSTEM_CONTRACT.json").read_text(encoding="utf-8"))
    screen = json.loads((source_root / "SCREEN_MANIFEST.json").read_text(encoding="utf-8"))
    if (contract.get("registration") != config
            or screen.get("schema") != "scientific-system-fresh-source-screen-v1"
            or screen.get("protocol_complete") is not True
            or screen.get("measurement_authorized") is not False
            or screen.get("candidate_response_accessed") is not False
            or screen.get("heldout_opened") is not False):
        raise ValueError("source is not a passed measurement-free system screen")
    dataset, seed = config["data"][0]["dataset"], config["seeds"][0]
    workspace = source_root / dataset / str(seed)
    analysis = json.loads((workspace / "exploration/ANALYSIS.json").read_text(encoding="utf-8"))
    gate_names = ["H0_HYPOTHESIS_BANK_VIABILITY.json", "HYPOTHESIS_BANK_VIABILITY.json",
                  "SOURCE_ADMISSION.json", "MARGINAL_DECISION_INFLUENCE.json"]
    if config.get("targeted_query_policy") == "decision_risk":
        gate_names.append("DECISION_RISK_UTILITY_VIABILITY.json")
    for name in gate_names:
        if json.loads((workspace / name).read_text(encoding="utf-8")).get("passed") is not True:
            raise ValueError("passed-screen source Gate is not passed")
    forbidden = {"DECISION-001.json", "RECEIPT-001.json", "SYSTEM_MANIFEST.json"}
    if any(path.name in forbidden or "measured" in path.parts
           for path in source_root.rglob("*") if path.is_file()):
        raise ValueError("passed source screen contains measurement artifacts")
    return source_root, workspace, analysis, "passed-sourcewise-screen"


def _verify_zero_response_continuation(source_root, config, continuation):
    """Bind one terminal pre-decision failure without mutating its artifacts."""
    if continuation.get("schema") == "scientific-passed-source-screen-continuation-v2":
        return _verify_passed_source_screen(source_root, config, continuation)
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
    source_registration = contract.get("registration")
    continuation_mode = "identity-preserving"
    if source_registration != config:
        compatible = json.loads(json.dumps(config))
        capacity = compatible.get("hypothesis_bank_gate", {})
        capacity.pop("maximum_candidates", None)
        capacity.pop("selection_rule", None)
        capacity.pop("source_safety_folds", None)
        capacity.pop("source_stacking_baseline", None)
        capacity.pop("source_stacking_dyadic_depth", None)
        capacity.pop("source_stacking_max_optional_mass", None)
        if capacity.get("schema") in {
                "scientific-hypothesis-bank-gate-v3",
                "scientific-hypothesis-bank-gate-v4"}:
            capacity["schema"] = "scientific-hypothesis-bank-gate-v1"
        influence = compatible.get("marginal_influence_gate", {})
        influence.pop("source_admission_rule", None)
        influence.pop("required_active_contributions", None)
        influence.pop("rejectable_contributions", None)
        if influence.get("schema") in {
                "scientific-source-admission-influence-gate-v4",
                "scientific-source-admission-influence-gate-v5"}:
            influence["schema"] = "scientific-dual-channel-source-contribution-gate-v3"
        if source_registration != compatible:
            raise ValueError("continuation source registration changed beyond bank capacity")
        continuation_mode = "response-free-capacity-rebank"
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
    return source_root, workspace, analysis, continuation_mode


def _copy_continuation_artifacts(source_workspace, workspace, exploration, mode):
    if mode in {"identity-preserving", "passed-sourcewise-screen"}:
        names = ("HYPOTHESIS_BANK_VIABILITY.json", "MARGINAL_DECISION_INFLUENCE.json")
        if mode == "passed-sourcewise-screen":
            names += ("H0_HYPOTHESIS_BANK_VIABILITY.json", "SOURCE_ADMISSION.json",
                      "CANDIDATE_ADMISSION.json")
            if (source_workspace / "DECISION_RISK_UTILITY_VIABILITY.json").is_file():
                names += ("DECISION_RISK_UTILITY_VIABILITY.json",)
        for name in names:
            shutil.copyfile(source_workspace / name, workspace / name)
    (workspace / "exploration").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_workspace / "exploration/ANALYSIS.json",
                    workspace / "exploration/ANALYSIS.json")
    for row in exploration["rows"]:
        source_variant = source_workspace / "exploration" / row["variant"]
        target_variant = workspace / "exploration" / row["variant"]
        target_registry = target_variant / "evidence_registry.jsonl"
        target_registry.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_variant / "evidence_registry.jsonl", target_registry)
        result = json.loads((source_variant / "RESULT.json").read_text(encoding="utf-8"))
        result["evidence_registry_path"] = str(target_registry)
        result["zero_response_continuation_source"] = str(source_variant)
        _publish(target_variant / "RESULT.json", result)
        if mode == "passed-sourcewise-screen":
            for name in ("H0_CAPACITY_BANK.json", "FROZEN_BANK.json"):
                shutil.copyfile(source_variant / name, target_variant / name)
            frozen = json.loads((source_variant / "FROZEN_BANK.json").read_text(
                encoding="utf-8"))
            row["candidates"] = frozen["candidates"]
            row["source_prior_weights"] = frozen["source_prior_weights"]


def execute_registered_system_continuation(project_root, root, source_root, config,
                                           expected_freeze, continuation, *, execution_role,
                                           measurement_authorized=True):
    """Continue an immutable zero-response screen without rerunning discovery."""
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_system_registration(config)
    if type(measurement_authorized) is not bool:
        raise TypeError("continuation measurement authorization must be boolean")
    if execution_role != "user" or config["user_execution_authorized"] is not True:
        raise PermissionError("real development continuation requires user authorization")
    project_root, root = Path(project_root).resolve(), Path(root).resolve()
    if root == project_root or project_root in root.parents or root.exists():
        raise ValueError("continuation output must be a new path outside source")
    verify_system_freeze(project_root, config, expected_freeze)
    source_root, source_workspace, exploration, continuation_mode = _verify_zero_response_continuation(
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
        _copy_continuation_artifacts(
            source_workspace, workspace, exploration, continuation_mode)
        arbitration, reporting_evaluation = _split_source_arbitration(
            data.evaluation, config["marginal_influence_gate"]["arbitration_fraction"])
        if continuation_mode == "response-free-capacity-rebank":
            _prepare_candidate_admission(workspace, exploration, data, config, arbitration)
            _prepare_hypothesis_bank_viability(workspace, exploration, data, config)
            _prepare_source_gates(workspace, exploration, data, config, arbitration)
        if (config.get("targeted_query_policy") == "decision_risk"
                and not (workspace / "DECISION_RISK_UTILITY_VIABILITY.json").is_file()):
            _prepare_decision_risk_utility_gate(workspace, exploration, data, config)
        if measurement_authorized is not True:
            screen = {"schema": "scientific-system-source-admission-screen-v2",
                "coordinate": coordinate, "continuation_source": str(source_root),
                "continuation_mode": continuation_mode,
                "hypothesis_bank_viability": _digest(json.loads((workspace /
                    "HYPOTHESIS_BANK_VIABILITY.json").read_text(encoding="utf-8"))),
                "source_admission": _digest(json.loads((workspace /
                    "SOURCE_ADMISSION.json").read_text(encoding="utf-8"))),
                "marginal_decision_influence": _digest(json.loads((workspace /
                    "MARGINAL_DECISION_INFLUENCE.json").read_text(encoding="utf-8"))),
                **({"decision_risk_utility": _digest(json.loads((workspace /
                    "DECISION_RISK_UTILITY_VIABILITY.json").read_text(encoding="utf-8")))}
                   if config.get("targeted_query_policy") == "decision_risk" else {}),
                "protocol_complete": True, "measurement_authorized": False,
                "candidate_response_accessed": False, "heldout_opened": False,
                "superiority_demonstrated": False,
                "claim_boundary": "H0 and source-arbitration screen only; no acquisition response"}
            _publish(root / "SCREEN_MANIFEST.json", screen)
            return screen
        comparisons = {row["variant"]: _run_comparison_variant(
            project_root, workspace, row, data, config, None, source_identity, seed,
            expected_freeze, coordinate, reporting_evaluation)
            for row in exploration["rows"]}
        results.append({"dataset": dataset, "seed": seed, "family": data.manifest["family"],
            "exploration": exploration, "comparisons": comparisons,
            "continuation_source": str(source_root),
            "continuation_mode": continuation_mode})
        verify_system_freeze(project_root, config, expected_freeze)
        result = {"schema": "scientific-system-development-terminal-v1", "results": results,
            "protocol_complete": True, "heldout_opened": False,
            "superiority_demonstrated": False,
            "zero_response_continuation": True,
            "continuation_mode": continuation_mode,
            "claim_boundary": "conditional opened-development evidence; no confirmatory claim"}
        _publish(root / "SYSTEM_MANIFEST.json", result)
        return result
    except BaseException as error:
        _publish(root / "TERMINAL_FAILURE.json", {"coordinate": coordinate,
            "error_type": type(error).__name__, "completed_coordinates": len(results),
            "protocol_complete": False, "heldout_opened": False,
            "superiority_demonstrated": False})
        raise
