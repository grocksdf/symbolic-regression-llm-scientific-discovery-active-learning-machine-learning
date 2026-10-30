"""Run the frozen shared-E0 one-slot Gap mechanism experiment.

Generation uses development, validation, gap-audit and gap-admission roles.
Reporting responses are opened only after every generation/admission artifact
has been durably frozen.
"""
from __future__ import annotations

import argparse
from base64 import b64encode
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pyarrow.parquet as pq
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
MAINLINE = Path(os.environ.get(
    "HYPOTHESIS_MVP_ROOT", ROOT.parent / "hypothesis_mvp")).resolve()
sys.path[:0] = [str(ROOT), str(MAINLINE)]

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()

from hypothesis_mvp.data.roles import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig
from hypothesis_mvp.discovery.admission_transaction import (
    freeze_admitted_posterior_transaction,
)
from hypothesis_mvp.discovery.contracts import DiscoveryConfig, json_safe
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.factory import build_scientific_discovery_runtime
from hypothesis_mvp.discovery.localized_gap_refinement import (
    certify_localized_refinement, compile_executable_gap_brief,
    executable_target_assignments,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model,
    freeze_discovery_target,
)
from hypothesis_mvp.discovery.candidate_validity import (
    proposal_domain_context,
)
from hypothesis_mvp.discovery.proposal_runtime import ProviderSettings
from hypothesis_mvp.discovery.proposal_runtime import ProposalBatch
from hypothesis_mvp.discovery.regional_candidate_admission import (
    admit_global_protected_candidates, admit_regional_candidates,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from methods.hypothesis_mvp_pcpi.drr_adapter import three_arm_role_indices
from methods.hypothesis_mvp_pcpi.protected_llmsr_generation import (
    run_search_rich_proposals,
)
from scripts.gap_minimal_harness_contract import (
    CONDITIONS, identity, marginal_value, shared_block,
    verify_generation_freeze,
)
from scripts.run_aistats_three_arm_formula_child import (
    DATASET_GROUPS, _provider_key, _require_pysr_runtime, _task_metadata,
)

EXPANDED_REFIT_POLICY = "pcpi-expanded-fixed-inner-v1"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8")


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _git_value(root: Path, *args: str) -> str:
    import subprocess
    process = subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        encoding="utf-8", errors="strict", stdout=subprocess.PIPE)
    return process.stdout.strip()


def _source_identity(root: Path):
    return {
        "root": str(root),
        "commit": _git_value(root, "rev-parse", "HEAD"),
        "tree": _git_value(root, "rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(_git_value(
            root, "status", "--short", "--untracked-files=no")),
    }


def _filter_engine_result_all_domains(
        engine_result, n_features, domains):
    runtime = EquationRuntime(
        n_features, refit_policy=EXPANDED_REFIT_POLICY)
    retained, rejected = [], []
    for row in engine_result.all_results:
        failures = []
        for name, values in domains:
            try:
                prediction = np.asarray(
                    runtime.predict(str(row.expression), values),
                    dtype=float)
                if (prediction.shape != (len(values),)
                        or not np.all(np.isfinite(prediction))):
                    raise ValueError("nonfinite-or-shape-invalid")
            except Exception as error:
                failures.append({
                    "domain": name,
                    "error_type": type(error).__name__,
                    "message": str(error)})
        if failures:
            rejected.append({
                "engine": str(row.engine),
                "expression_sha256": sha256(
                    str(row.expression).encode()).hexdigest(),
                "failures": failures})
        else:
            retained.append(row)
    if not retained:
        raise ValueError(
            "all engine candidates undefined on registered domains")
    result = replace(
        engine_result,
        best=min(retained, key=lambda row: (
            row.score, row.complexity, row.expression)),
        all_results=tuple(retained))
    return result, {
        "schema": "all-domain-engine-definedness-v1",
        "registered_domains": [name for name, _ in domains],
        "input_candidate_count": len(engine_result.all_results),
        "retained_candidate_count": len(retained),
        "rejected_candidate_count": len(rejected),
        "rejections": rejected,
        "response_accessed": False}


def _opened(dataset, indices, *, response: bool):
    original = [int(index) for index in indices]
    ordered = sorted(original)
    values = np.asarray(
        dataset[ordered, :] if response else dataset[ordered, 1:],
        dtype=float)
    lookup = dict(zip(ordered, values, strict=True))
    restored = np.asarray([lookup[index] for index in original])
    if response:
        if not np.all(np.isfinite(restored)):
            raise ValueError("opened generation role contains nonfinite values")
        return restored[:, 1:], restored[:, 0]
    if not np.all(np.isfinite(restored)):
        raise ValueError("opened covariates contain nonfinite values")
    return restored


def _generation_roles(hdf5: Path, family: str, task: str, seed: int):
    with h5py.File(hdf5, "r") as handle:
        dataset = handle[f"{DATASET_GROUPS[family]}/{task}/train"]
        indices = three_arm_role_indices(
            int(dataset.shape[0]), task_name=task, seed=seed)
        development = _opened(
            dataset, indices["discovery_development"], response=True)
        validation = _opened(
            dataset, indices["discovery_validation"], response=True)
        gap_audit = _opened(dataset, indices["gap_audit"], response=True)
        gap_admission_X = _opened(
            dataset, indices["gap_admission"], response=False)
        actions = _opened(
            dataset, indices["action_covariates"], response=False)
        reporting_X = _opened(
            dataset, indices["reporting"], response=False)
        posterior_update_X = _opened(
            dataset, indices["decision_selector_update"], response=False)
    selection = SelectionData(
        RoleDataset(DataRole.DEVELOPMENT, *development),
        RoleDataset(DataRole.VALIDATION, *validation),
        AcquisitionCovariates(
            DataRole.ACQUISITION_POOL, actions,
            covariate_fingerprint(actions)), ())
    return {
        "selection": selection,
        "gap_audit": RoleDataset(DataRole.VALIDATION, *gap_audit),
        "gap_admission_X": gap_admission_X,
        "reporting_X": reporting_X,
        "posterior_update_X": posterior_update_X,
        "indices": {key: list(value) for key, value in indices.items()},
    }


def _gap_admission_role(
        hdf5: Path, family: str, task: str, indices):
    with h5py.File(hdf5, "r") as handle:
        dataset = handle[f"{DATASET_GROUPS[family]}/{task}/train"]
        X, y = _opened(dataset, indices["gap_admission"], response=True)
    return RoleDataset(DataRole.VALIDATION, X, y)


def _reporting_response(
        hdf5: Path, family: str, task: str, indices):
    with h5py.File(hdf5, "r") as handle:
        dataset = handle[f"{DATASET_GROUPS[family]}/{task}/train"]
        _, values = _opened(dataset, indices["reporting"], response=True)
    return values


def _posterior_update_role(
        hdf5: Path, family: str, task: str, indices):
    with h5py.File(hdf5, "r") as handle:
        dataset = handle[f"{DATASET_GROUPS[family]}/{task}/train"]
        X, y = _opened(
            dataset, indices["decision_selector_update"],
            response=True)
    return RoleDataset(DataRole.VALIDATION, X, y)


def _rotate_scalar_leaves(value, namespace: str):
    """Schema-preserving deterministic placebo using only generation data."""
    copied = deepcopy(value)
    leaves = []

    def collect(node, path=()):
        if isinstance(node, dict):
            for key in sorted(node):
                if key in {
                        "schema", "propose_allowed", "candidate_response_accessed",
                        "heldout_opened"}:
                    continue
                collect(node[key], (*path, key))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                collect(child, (*path, index))
        elif type(node) in {int, float, str}:
            leaves.append((path, node))

    def assign(node, path, new_value):
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = new_value

    collect(copied)
    if len(leaves) < 2:
        copied["placebo_binding"] = sha256(namespace.encode()).hexdigest()
        return copied
    order = sorted(range(len(leaves)), key=lambda index: sha256(
        f"{namespace}:{leaves[index][0]}".encode()).digest())
    values = [leaves[index][1] for index in order]
    rotated = values[1:] + values[:1]
    for index, new_value in zip(order, rotated, strict=True):
        assign(copied, leaves[index][0], new_value)
    if copied == value:
        copied["placebo_binding"] = sha256(namespace.encode()).hexdigest()
    return copied


def _neutral_brief(gap_brief):
    def neutral(value):
        if isinstance(value, dict):
            return {key: neutral(child) for key, child in value.items()}
        if isinstance(value, list):
            return [neutral(child) for child in value]
        if type(value) is bool:
            return False
        if type(value) in {int, float}:
            return 0
        if isinstance(value, str):
            return "neutral"
        return value
    result = neutral(gap_brief)
    if isinstance(result, dict):
        result["schema"] = gap_brief.get("schema", "posterior-gap-brief")
    return result


def _placebo_executable_brief(executable):
    value = deepcopy(executable)
    for row in value.get("target_regions", ()):
        direction = row.get("direction")
        row["direction"] = (
            "decrease-parent-prediction"
            if direction == "increase-parent-prediction"
            else "increase-parent-prediction"
            if direction == "decrease-parent-prediction"
            else "undetermined")
    value["source_gap_identity"] = identity({
        "placebo_of": executable.get("source_gap_identity"),
        "target_regions": value.get("target_regions", ())})
    return value


def _provider(config, provider_env: Path):
    return ProviderSettings.from_environment(
        base_url=config["llm_api_url"], model=config["llm_model"],
        api_key=_provider_key(provider_env),
        attempts=1, connect_timeout_s=15.,
        read_timeout_s=float(config.get("llm_timeout_s", 300.)),
        retry_backoff_s=0., temperature=float(config["llm_temperature"]),
        max_tokens=int(config["llm_max_tokens"]),
        thinking_type=str(config.get("llm_thinking_type") or ""),
        reasoning_effort=str(config.get("llm_reasoning_effort") or ""),
        do_sample=config.get("llm_do_sample"))


def _runtime_config(
        seed: int, protected_count: int, *,
        candidates_per_prompt: int = 1,
        candidate_opportunities: int = 1):
    return DiscoveryConfig.from_mapping({
        "evaluation_budget":
            protected_count + candidate_opportunities + 8,
        "llm_evaluation_reserve": candidate_opportunities,
        "synthesis_evaluation_reserve": 0,
        "refit_policy": EXPANDED_REFIT_POLICY,
        "islands": ("nmse",),
        "random_seed": seed,
        "use_library": False,
        "task_local_memory_read": False,
        "task_local_memory_write": False,
        "max_rounds": 1,
        "candidates_per_island": candidates_per_prompt,
        "new_skeleton_quota": 0,
    })


def _capture_one_proposal(runtime, *, context, task_name, task_desc,
                          parent, condition, output_dir,
                          allow_protocol_repair):
    captures = []
    original_post = requests.post

    def capture(url, **kwargs):
        request = dict(kwargs.get("json") or {})
        request_bytes = json.dumps(
            request, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        response = original_post(url, **kwargs)
        captures.append({
            "url": str(url),
            "request_bytes": request_bytes,
            "response_bytes": bytes(response.content),
            "http_status": int(response.status_code),
        })
        return response

    requests.post = capture
    try:
        batch = runtime.proposal.propose(
            task_name=task_name, task_desc=task_desc,
            round_id=1, island="nmse",
            parent_hash=parent.dag.canonical_hash,
            island_context=context, library_rows=(),
            ephemeral_refinements=(), max_candidates=1,
            allow_protocol_repair=allow_protocol_repair)
    finally:
        requests.post = original_post
    maximum_attempts = 2 if allow_protocol_repair else 1
    if not 1 <= len(captures) <= maximum_attempts:
        raise RuntimeError(
            f"{condition} exceeded the registered provider attempts")
    transcripts = []
    for index, capture_row in enumerate(captures, 1):
        suffix = "" if index == 1 else f"_{index}"
        request_path = output_dir / f"PROVIDER_REQUEST{suffix}.json"
        response_path = output_dir / f"PROVIDER_RESPONSE{suffix}.bin"
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_bytes(capture_row["request_bytes"])
        response_path.write_bytes(capture_row["response_bytes"])
        transcripts.append({
            "attempt": index,
            "raw_request_path": str(request_path.resolve()),
            "raw_request_sha256": _digest(request_path),
            "raw_request_bytes": request_path.stat().st_size,
            "raw_response_path": str(response_path.resolve()),
            "raw_response_sha256": _digest(response_path),
            "raw_response_bytes": response_path.stat().st_size,
            "provider_http_status": capture_row["http_status"],
        })
    first = transcripts[0]
    return batch, {
        **first,
        "raw_response_base64": b64encode(
            captures[0]["response_bytes"]).decode("ascii"),
        "provider_attempt_count": len(transcripts),
        "provider_transcripts": transcripts,
        "provider_transcript_identity": identity(transcripts),
    }


def _capture_search_rich_proposals(
        runtime, *, parent, arrays, context, task_name, task_desc,
        condition, output_dir, search_config, candidate_filter):
    captures = []
    original_post = requests.post

    def capture(url, **kwargs):
        request = dict(kwargs.get("json") or {})
        request_bytes = json.dumps(
            request, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        response = original_post(url, **kwargs)
        captures.append({
            "request_bytes": request_bytes,
            "response_bytes": bytes(response.content),
            "http_status": int(response.status_code)})
        return response

    requests.post = capture
    try:
        states, audit = run_search_rich_proposals(
            runtime, parent=parent, arrays=arrays,
            task_name=task_name, task_desc=task_desc,
            island_context=context, condition=condition,
            rounds=int(search_config["rounds"]),
            candidates_per_prompt=int(
                search_config["candidates_per_prompt"]),
            num_islands=int(search_config["num_islands"]),
            functions_per_prompt=int(
                search_config["functions_per_prompt"]),
            temperature=float(search_config["temperature"]),
            seed=int(search_config["seed"]),
            candidate_filter=candidate_filter)
    finally:
        requests.post = original_post
    expected = int(search_config["rounds"])
    if len(captures) != expected:
        raise RuntimeError(
            "search-rich generation did not issue registered calls")
    transcripts = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(captures, 1):
        request_path = output_dir / f"PROVIDER_REQUEST_{index}.json"
        response_path = output_dir / f"PROVIDER_RESPONSE_{index}.bin"
        request_path.write_bytes(row["request_bytes"])
        response_path.write_bytes(row["response_bytes"])
        transcripts.append({
            "attempt": index,
            "raw_request_path": str(request_path.resolve()),
            "raw_request_sha256": _digest(request_path),
            "raw_request_bytes": request_path.stat().st_size,
            "raw_response_path": str(response_path.resolve()),
            "raw_response_sha256": _digest(response_path),
            "raw_response_bytes": response_path.stat().st_size,
            "provider_http_status": row["http_status"]})
    manifest = output_dir / "PROVIDER_TRANSCRIPTS.json"
    manifest.write_text(json.dumps(
        transcripts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    digest = _digest(manifest)
    return states, audit, {
        "raw_request_path": str(manifest.resolve()),
        "raw_request_sha256": digest,
        "raw_request_bytes": manifest.stat().st_size,
        "raw_response_path": str(manifest.resolve()),
        "raw_response_sha256": digest,
        "raw_response_bytes": manifest.stat().st_size,
        "raw_response_base64": "",
        "provider_http_status": 200,
        "provider_attempt_count": len(captures),
        "provider_transcripts": transcripts,
        "provider_transcript_identity": identity(transcripts)}


def _row_from_state(state, condition):
    return {
        "expression": state.dag.expression,
        "source": f"llm_gap_marginal:{condition}",
        "origin": "llm",
        "lineage_id": state.lineage_id,
    }


def _admission_function(freeze):
    method = str(
        freeze.get("admission_method") or "regional-any-region-v1")
    if method == "regional-any-region-v1":
        return admit_regional_candidates
    if method == "global-protected-fixed-expert-v1":
        return admit_global_protected_candidates
    raise ValueError("unregistered candidate admission method")


def _score(
        rows, selection, X_report, y_report, identity_value,
        posterior_update=None):
    model = freeze_expanded_formula_model(
        rows, n_features=selection.development.X.shape[1],
        prior=NormalInverseGammaPrior(),
        exploration_identity=identity_value,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    fit = selection.development
    if posterior_update is not None:
        fit = RoleDataset(
            DataRole.DEVELOPMENT,
            np.vstack((fit.X, posterior_update.X)),
            np.concatenate((fit.y, posterior_update.y)))
    target = freeze_discovery_target(
        model, fit, selection.acquisition_pool.X,
        measurement_budget=2,
        expected_model_identity=model.stable_hash)
    engine = model.engine(model.stable_hash)
    density = np.asarray(engine.predictive_logpdf(
        target.initial_posterior, X_report, y_report), dtype=float)
    if not np.all(np.isfinite(density)):
        raise FloatingPointError("reporting log density is nonfinite")
    return float(np.mean(density)), {
        "model_identity": model.stable_hash,
        "target_identity": target.stable_hash,
        "posterior_member_count": len(target.initial_posterior.members),
    }


def _condition_context(
        base, condition, gap, neutral, placebo,
        instruction_modes=None, localized_executable=None):
    context = dict(base)
    if condition == "B0":
        context.update(
            diagnostic_instruction_mode="refinement-only-neutral",
            candidate_diagnostic=neutral)
        return context
    brief = deepcopy(placebo if condition == "G_placebo0" else gap)
    brief["registered_original_propose_allowed"] = bool(
        brief.get("propose_allowed"))
    brief["propose_allowed"] = True
    modes = dict(instruction_modes or {})
    mode = modes.get(condition) or (
        "historical-hard" if condition == "G_hard0"
        else "non-mandatory-evidence")
    context.update(
        posterior_gap_brief=brief,
        candidate_diagnostic=brief,
        diagnostic_instruction_mode=mode)
    if localized_executable is not None:
        context.update(
            localized_refinement_contract=True,
            executable_gap_brief=localized_executable)
    return context


def _run_block(freeze, family, task, seed, output_dir):
    config = yaml.safe_load(
        Path(freeze["config"]).read_text(encoding="utf-8"))
    roles = _generation_roles(
        Path(freeze["hdf5"]), family, task, seed)
    selection = roles["selection"]
    raw_agent = dict(config["agent_config"])
    raw_agent.update(
        engines=tuple(raw_agent["engines"]),
        discovery_islands=tuple(raw_agent["discovery_islands"]),
        random_seed=seed)
    # The frozen agent configuration enables typed inner augmentation, whose
    # constructor contract requires the registered provider to be present.
    # This object is used only for ``_run_engines`` and
    # ``_independent_gap_brief`` below; neither path performs a provider call.
    # Supplying the same frozen provider therefore satisfies construction
    # without creating a second or hidden proposal route.
    agent = DiscoveryAgent(
        DiscoveryAgentConfig(**raw_agent),
        provider_settings=_provider(
            config, Path(freeze["provider_env"])))
    engines = agent._run_engines(selection, 0)
    domains = (
        ("development", selection.development.X),
        ("validation", selection.validation.X),
        ("gap_audit", roles["gap_audit"].X),
        ("gap_admission", roles["gap_admission_X"]),
        ("posterior_update", roles["posterior_update_X"]),
        ("reporting", roles["reporting_X"]),
        ("action", selection.acquisition_pool.X))
    engines, definedness = _filter_engine_result_all_domains(
        engines, selection.development.X.shape[1], domains)
    gap = agent._independent_gap_brief(
        selection, engines, roles["gap_audit"],
        NormalInverseGammaPrior(), 2)
    E0 = [dict(row) for row in gap["core_rows"]]
    if not E0:
        raise ValueError("shared E0 is empty")
    checker = EquationRuntime(
        selection.development.X.shape[1],
        refit_policy=EXPANDED_REFIT_POLICY)
    for row in E0:
        for name, values in domains:
            try:
                prediction = np.asarray(
                    checker.predict(str(row["expression"]), values),
                    dtype=float)
            except Exception as error:
                raise ValueError(
                    f"frozen E0 undefined on registered domain:{name}"
                ) from error
            if (prediction.shape != (len(values),)
                    or not np.all(np.isfinite(prediction))):
                raise ValueError(
                    f"frozen E0 undefined on registered domain:{name}")
    symbols, descriptions, properties, task_desc, _ = _task_metadata(
        Path(freeze["metadata"]), task,
        selection.development.X.shape[1])
    variable_metadata = {
        "feature_names": symbols[1:],
        "feature_units": properties[1:],
        "target_name": symbols[0],
        "target_unit": properties[0],
    }
    setup_runtime = build_scientific_discovery_runtime(
        n_features=selection.development.X.shape[1],
        config=_runtime_config(seed, len(E0)),
        library_path=output_dir / "setup_knowledge.jsonl",
        ledger_path=output_dir / "setup_ledger.jsonl",
        variable_metadata=variable_metadata)
    setup_runtime.evaluation.registered_covariates = np.vstack((
        selection.development.X, selection.validation.X,
        roles["gap_audit"].X, roles["gap_admission_X"],
        selection.acquisition_pool.X))
    parent, _ = setup_runtime.evaluation.seed_survivor(
        E0, selection.development.X, selection.development.y,
        selection.validation.X, selection.validation.y)
    gap_brief = dict(gap["prompt"])
    neutral = _neutral_brief(gap_brief)
    placebo = _rotate_scalar_leaves(
        gap_brief, f"{family}/{task}/{seed}")
    localized = bool(freeze.get("localized_gap_refinement"))
    executable = (
        compile_executable_gap_brief(gap_brief) if localized else None)
    placebo_executable = (
        _placebo_executable_brief(executable)
        if executable is not None else None)
    block = shared_block(
        task=task, seed=seed, E0_rows=E0,
        parent={"expression": parent.dag.expression,
                "lineage_id": parent.lineage_id,
                "canonical_hash": parent.dag.canonical_hash},
        gap_brief=gap_brief, neutral_brief=neutral,
        placebo_brief=placebo)
    _write_new(output_dir / "SHARED_BLOCK.json", block)
    _write_new(
        output_dir / "ALL_DOMAIN_DEFINEDNESS.json", definedness)
    domain = proposal_domain_context(np.vstack((
        selection.development.X, selection.validation.X,
        selection.acquisition_pool.X)))
    results, pending = [], []
    conditions = tuple(freeze["conditions"])
    allow_repair = bool(freeze.get("format_only_repair_attempts") == 1)
    search_config = dict(freeze.get("search_rich_generation") or {})
    search_rich = bool(search_config)
    candidates_per_prompt = int(
        search_config.get("candidates_per_prompt", 1))
    candidate_opportunities = int(
        search_config.get("rounds", 1)) * candidates_per_prompt
    for condition in conditions:
        condition_dir = output_dir / condition
        runtime = build_scientific_discovery_runtime(
            n_features=selection.development.X.shape[1],
            config=_runtime_config(
                seed, len(E0),
                candidates_per_prompt=candidates_per_prompt,
                candidate_opportunities=candidate_opportunities),
            library_path=condition_dir / "knowledge.jsonl",
            ledger_path=condition_dir / "ledger.jsonl",
            provider_settings=_provider(
                config, Path(freeze["provider_env"])),
            variable_metadata=variable_metadata)
        runtime.evaluation.registered_covariates = (
            setup_runtime.evaluation.registered_covariates)
        condition_parent, _ = runtime.evaluation.seed_survivor(
            E0, selection.development.X, selection.development.y,
            selection.validation.X, selection.validation.y)
        if condition_parent.dag.canonical_hash != parent.dag.canonical_hash:
            raise ValueError("condition parent diverged from shared parent")
        # E0 fitting establishes the shared frozen parent; it is not one of
        # the four registered one-slot candidate opportunities. Start every
        # condition's candidate ledger from the same empty budget state after
        # parent identity has been verified.
        runtime.evaluation.reset()
        runtime.evaluation.registered_covariates = (
            setup_runtime.evaluation.registered_covariates)
        exploration = runtime._explore(
            condition_parent, selection.development.X,
            selection.development.y, "nmse")
        base_context = {
            "objective": "nmse",
            "proposal_domain_context": domain,
            "current_equation_state": condition_parent.compact(),
            "current_expression": condition_parent.dag.expression,
            "executable_exploration_function": exploration.as_prompt_dict(),
            "failure_signature": list(runtime.evaluation.failure_signature(
                condition_parent, exploration)),
            "posterior_gap_existing_supports": gap["existing_supports"],
            "allowed_edits": [
                "ADD", "DELETE", "REPLACE", "REPARAMETERIZE",
                "CHANGE_OPERATOR", "CHANGE_INTERACTION"],
            "new_skeleton_quota": 0,
            "registered_task_context": {
                "name": task, "description": task_desc,
                "feature_names": symbols[1:],
                "feature_descriptions": descriptions[1:]},
            "experiment_condition": condition,
            "experiment_block_identity": block["block_identity"],
        }
        context = _condition_context(
            base_context, condition, gap_brief, neutral, placebo,
            freeze.get("instruction_modes"),
            (executable if condition == "G_hard0"
             else placebo_executable if condition == "G_placebo0"
             else None))
        no_target = bool(
            context.get("localized_refinement_contract")
            and not context.get(
                "executable_gap_brief", {}).get("target_regions"))
        if no_target:
            empty_hash = sha256(b"").hexdigest()
            batch = ProposalBatch(
                (), True, "localized_gap_no_target_abstention",
                telemetry={
                    "semantic_abstention": True,
                    "reason": "no-executable-target-region"})
            transcript = {
                "raw_request_path": "",
                "raw_request_sha256": empty_hash,
                "raw_request_bytes": 0,
                "raw_response_path": "",
                "raw_response_sha256": empty_hash,
                "raw_response_bytes": 0,
                "raw_response_base64": "",
                "provider_http_status": 0,
                "provider_attempt_count": 0,
                "provider_transcripts": [],
                "provider_transcript_identity": identity([]),
            }
            candidates = []
            validity = []
            winner = None
            locality = []
            search_audit = {}
        elif search_rich:
            def locality_filter(search_batch, round_id, island):
                if not context.get("localized_refinement_contract"):
                    return search_batch, []
                assignments, target_regions = (
                    executable_target_assignments(
                        context["executable_gap_brief"],
                        selection.acquisition_pool.X))
                kept, certificates = [], []
                for candidate in search_batch.candidates:
                    certificate = certify_localized_refinement(
                        runtime.equation,
                        condition_parent.dag.expression,
                        candidate.equation, candidate.local_gate,
                        selection.acquisition_pool.X,
                        assignments, target_regions)
                    certificates.append({
                        "candidate_id": candidate.candidate_id,
                        "round": round_id, "island": island,
                        **certificate})
                    if certificate["passed"]:
                        kept.append(candidate)
                return replace(
                    search_batch, candidates=tuple(kept)), certificates

            candidates, search_audit, transcript = (
                _capture_search_rich_proposals(
                    runtime, parent=condition_parent,
                    arrays=(
                        selection.development.X,
                        selection.development.y,
                        selection.validation.X,
                        selection.validation.y),
                    context=context, task_name=task,
                    task_desc=task_desc, condition=condition,
                    output_dir=condition_dir,
                    search_config={
                        **search_config,
                        "seed": seed + 1009 * (
                            conditions.index(condition) + 1)},
                    candidate_filter=locality_filter))
            locality = [
                certificate
                for record in search_audit["records"]
                for certificate in record["filter_audit"]]
            validity = [
                row for record in search_audit["records"]
                for row in record["validity"]]
            winner = None
            batch = ProposalBatch(
                (), True, "search_rich_generation_completed",
                telemetry={"search_rich_generation": search_audit})
        else:
            batch, transcript = _capture_one_proposal(
                runtime, context=context, task_name=task,
                task_desc=task_desc, parent=condition_parent,
                condition=condition, output_dir=condition_dir,
                allow_protocol_repair=allow_repair)
        if (not no_target and not search_rich):
            locality = []
        if (not search_rich and context.get(
                "localized_refinement_contract") and batch.candidates):
            assignments, target_regions = executable_target_assignments(
                context["executable_gap_brief"],
                selection.acquisition_pool.X)
            kept = []
            for candidate in batch.candidates:
                certificate = certify_localized_refinement(
                    runtime.equation, condition_parent.dag.expression,
                    candidate.equation, candidate.local_gate,
                    selection.acquisition_pool.X,
                    assignments, target_regions)
                locality.append({
                    "candidate_id": candidate.candidate_id,
                    **certificate})
                if certificate["passed"]:
                    kept.append(candidate)
            batch = replace(batch, candidates=tuple(kept))
        if not no_target and not search_rich:
            winner, candidates, validity = runtime._evaluate_batch(
                batch, condition_parent, (
                    selection.development.X, selection.development.y,
                    selection.validation.X, selection.validation.y),
                "nmse", 1)
        optional = [
            _row_from_state(state, condition) for state in candidates]
        pending.append({
            "condition": condition, "condition_dir": condition_dir,
            "context": context, "batch": batch,
            "transcript": transcript, "validity": validity,
            "locality": locality, "candidates": candidates,
            "winner": winner, "optional": optional,
            "search_audit": search_audit if search_rich else {}})
    _write_new(output_dir / "ARM_GENERATION_FROZEN.json", {
        "schema": "gap-arm-generation-freeze-v1",
        "admission_response_opened": False,
        "conditions": [{
            "condition": row["condition"],
            "context_identity": identity(row["context"]),
            "prompt_hash": row["batch"].prompt_hash,
            "response_hash": row["batch"].response_hash,
            "candidate_count": len(row["batch"].candidates)}
            for row in pending],
        "candidate_response_accessed": False,
        "heldout_opened": False})
    gap_admission = _gap_admission_role(
        Path(freeze["hdf5"]), family, task, roles["indices"])
    all_regions = tuple(range(len(gap["regions"].cuts) + 1))
    for pending_row in pending:
        optional = pending_row["optional"]
        if optional:
            retained, admission = _admission_function(freeze)(
                E0, optional, selection.development,
                roles["gap_audit"], gap_admission,
                selection.acquisition_pool.X, gap["regions"],
                all_regions, NormalInverseGammaPrior(), 2,
                alpha=.025,
                coefficient_policy=EXPANDED_FORMULA_POLICY)
        else:
            retained, admission = [], {
                "schema": (
                    "candidate-global-protected-admission-v1"
                    if freeze.get("admission_method")
                    == "global-protected-fixed-expert-v1"
                    else "candidate-regional-admission-v1"),
                "attempts": 0, "admitted": 0,
                "reason": "no-valid-materialized-candidate",
                "candidate_response_accessed": False,
                "heldout_opened": False}
        condition = pending_row["condition"]
        batch = pending_row["batch"]
        context = pending_row["context"]
        candidates = pending_row["candidates"]
        row = {
            "schema": "gap-minimal-condition-generation-v1",
            "family": family, "task": task, "seed": seed,
            "condition": condition,
            "block_identity": block["block_identity"],
            "E0_identity": identity(E0),
            "parent_identity": identity(block["shared"]["parent"]),
            "context_identity": block["contexts"][condition][
                "context_identity"],
            "generation_context_sha256": identity(context),
            **pending_row["transcript"],
            "batch": {
                "protocol_valid": batch.protocol_valid,
                "reason": batch.reason,
                "candidate_count": (
                    len(candidates) if search_rich
                    else len(batch.candidates)),
                "prompt_hash": batch.prompt_hash,
                "response_hash": batch.response_hash,
                "telemetry": batch.telemetry,
                "error": batch.error},
            "validity": pending_row["validity"],
            "locality_certificates": pending_row["locality"],
            "search_rich_generation": pending_row["search_audit"],
            "materialized_candidate_count": len(candidates),
            "transition_winner_present":
                pending_row["winner"] is not None,
            "optional_rows": optional,
            "admission": admission,
            "admitted_rows": [dict(value) for value in retained],
            "candidate_response_accessed": False,
            "reporting_response_accessed": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
        }
        _write_new(
            pending_row["condition_dir"] / "GENERATION_RESULT.json", row)
        results.append(row)
    return {
        "family": family, "task": task, "seed": seed,
        "block_identity": block["block_identity"],
        "E0_rows": E0,
        "selection": selection,
        "gap_admission": gap_admission,
        "reporting_X": roles["reporting_X"],
        "indices": roles["indices"],
        "conditions": results,
    }


def _bootstrap(task_rows, contrast, seed, count):
    rng = np.random.default_rng(seed)
    tasks = sorted(task_rows)
    values = np.asarray([task_rows[task][contrast] for task in tasks])
    samples = np.empty(count)
    for index in range(count):
        draw = rng.integers(0, len(values), size=len(values))
        samples[index] = float(np.mean(values[draw]))
    return [
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("output root already exists; resume is forbidden")
    freeze = _read(args.freeze)
    if (freeze.get("schema")
            != "gap-candidate-marginal-value-freeze-v1"
            or freeze.get("execution_authorized") is not True):
        raise ValueError("invalid or unauthorized experiment freeze")
    mainline = _source_identity(MAINLINE)
    benchmark = _source_identity(ROOT)
    if (mainline["tracked_dirty"] or benchmark["tracked_dirty"]
            or mainline["commit"] != freeze["source"]["mainline_commit"]
            or benchmark["commit"] != freeze["source"]["benchmark_commit"]):
        raise ValueError("source identity mismatch")
    for path, digest in freeze["file_hashes"].items():
        if _digest(Path(path)) != digest:
            raise ValueError(f"frozen file identity mismatch: {path}")
    provider_identity = freeze["provider_env_identity"]
    if (_digest(Path(provider_identity["path"]))
            != provider_identity["sha256"]):
        raise ValueError("provider credential-file identity mismatch")
    _require_pysr_runtime(yaml.safe_load(
        Path(freeze["config"]).read_text(encoding="utf-8")))
    args.output_root.mkdir(parents=True, exist_ok=False)
    _write_new(args.output_root / "RUN_FREEZE.json", freeze)
    blocks, coordinate_failures = [], []
    try:
        for coordinate in freeze["coordinates"]:
            family, task, seed = (
                coordinate["family"], coordinate["task"],
                int(coordinate["seed"]))
            coordinate_dir = (
                args.output_root / "generation" / family / task /
                f"seed{seed}")
            try:
                blocks.append(_run_block(
                    freeze, family, task, seed, coordinate_dir))
            except Exception as error:
                if not freeze.get(
                        "outcome", {}).get("all_failures_included"):
                    raise
                failure = {
                    "schema": "gap-coordinate-failure-v1",
                    "family": family, "task": task, "seed": seed,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "conditions": list(freeze["conditions"]),
                    "marginal_increment_per_condition": 0.0,
                    "retry_or_replacement": False,
                    "test_or_ood_accessed": False,
                    "heldout_opened": False}
                _write_new(
                    coordinate_dir / "COORDINATE_FAILURE.json",
                    failure)
                coordinate_failures.append(failure)
        generation_rows = [
            row for block in blocks for row in block["conditions"]]
        failure_generation_rows = []
        for failure in coordinate_failures:
            failure_id = identity(failure)
            for condition in freeze["conditions"]:
                failure_generation_rows.append({
                    "family": failure["family"],
                    "task": failure["task"], "seed": failure["seed"],
                    "condition": condition,
                    "block_identity": failure_id,
                    "E0_identity": failure_id,
                    "parent_identity": failure_id,
                    "context_identity": identity({
                        "failure": failure_id, "condition": condition}),
                    "generation_context_sha256": failure_id,
                    "raw_request_sha256": sha256(b"").hexdigest(),
                    "raw_response_sha256": sha256(b"").hexdigest(),
                    "coordinate_failure": True})
        generation_rows.extend(failure_generation_rows)
        generation_freeze = {
            "schema": "gap-minimal-generation-freeze-v1",
            "reporting_opened": False,
            "condition_count": len(generation_rows),
            "conditions": [{
                key: row[key] for key in (
                    "family", "task", "seed", "condition",
                    "block_identity", "E0_identity", "parent_identity",
                    "context_identity", "generation_context_sha256",
                    "raw_request_sha256",
                    "raw_response_sha256")}
                for row in generation_rows],
            "candidate_response_accessed": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
        }
        conditions = tuple(freeze["conditions"])
        expected_count = len(freeze["coordinates"]) * len(conditions)
        verify_generation_freeze(
            generation_freeze, expected_conditions=conditions,
            expected_count=expected_count)
        _write_new(
            args.output_root / "GENERATION_FROZEN.json",
            generation_freeze)

        updates = {}
        for block in blocks:
            key = (block["family"], block["task"], block["seed"])
            updates[key] = _posterior_update_role(
                Path(freeze["hdf5"]), block["family"], block["task"],
                block["indices"])
        _write_new(
            args.output_root / "POSTERIOR_UPDATE_FROZEN.json", {
                "schema": "gap-posterior-update-freeze-v1",
                "reporting_opened": False,
                "roles": [{
                    "family": family, "task": task, "seed": seed,
                    "update_identity": role.fingerprint,
                    "update_row_count": len(role.X)}
                    for (family, task, seed), role in updates.items()],
                "admission_responses_reused": False,
                "test_or_ood_accessed": False,
                "heldout_opened": False})
        result_rows = [{
            "family": failure["family"], "task": failure["task"],
            "seed": failure["seed"], "condition": condition,
            "baseline_log_score": None, "candidate_log_score": None,
            "marginal_value": 0.0, "materialized": False,
            "valid": False, "admitted": False,
            "coordinate_failure": True,
            "failure_type": failure["error_type"]}
            for failure in coordinate_failures
            for condition in freeze["conditions"]]
        for block in blocks:
            posterior_update = updates[(
                block["family"], block["task"], block["seed"])]
            y_report = _reporting_response(
                Path(freeze["hdf5"]), block["family"], block["task"],
                block["indices"])
            baseline, baseline_audit = _score(
                block["E0_rows"], block["selection"],
                block["reporting_X"], y_report,
                identity({"block": block["block_identity"],
                          "bank": "E0"}),
                posterior_update=posterior_update)
            for generation in block["conditions"]:
                admitted = list(generation["admitted_rows"])
                candidate_score = baseline
                score_audit = baseline_audit
                if admitted:
                    model, target, transaction = (
                        freeze_admitted_posterior_transaction(
                            block["E0_rows"], admitted,
                            block["selection"].development,
                            block["gap_admission"],
                            posterior_update,
                            block["selection"].acquisition_pool.X,
                            prior=NormalInverseGammaPrior(),
                            measurement_budget=2,
                            coefficient_policy=EXPANDED_FORMULA_POLICY))
                    engine = model.engine(model.stable_hash)
                    density = np.asarray(engine.predictive_logpdf(
                        target.initial_posterior,
                        block["reporting_X"], y_report), dtype=float)
                    if not np.all(np.isfinite(density)):
                        raise FloatingPointError(
                            "transaction reporting density is nonfinite")
                    candidate_score = float(np.mean(density))
                    score_audit = {
                        "model_identity": model.stable_hash,
                        "target_identity": target.stable_hash,
                        "posterior_member_count":
                            len(target.initial_posterior.members),
                        "admission_posterior_transaction":
                            transaction.__dict__,
                    }
                increment = marginal_value(
                    baseline_score=baseline,
                    candidate_score=candidate_score,
                    materialized=bool(
                        generation["materialized_candidate_count"]),
                    valid=bool(generation["optional_rows"]),
                    admitted=bool(admitted))
                result_rows.append({
                    "family": block["family"], "task": block["task"],
                    "seed": block["seed"],
                    "condition": generation["condition"],
                    "baseline_log_score": baseline,
                    "candidate_log_score": candidate_score,
                    "marginal_value": increment,
                    "materialized": bool(
                        generation["materialized_candidate_count"]),
                    "valid": bool(generation["optional_rows"]),
                    "admitted": bool(admitted),
                    "score_audit": score_audit,
                })
        grouped = {}
        for row in result_rows:
            grouped.setdefault(row["task"], {}).setdefault(
                row["condition"], []).append(row["marginal_value"])
        task_values = {
            task: {
                condition: float(np.mean(values))
                for condition, values in conditions.items()}
            for task, conditions in grouped.items()}
        contrast_pairs = freeze.get("contrast_pairs") or {
            "gap_vs_blind": ["G_hard0", "B0"],
            "soft_vs_hard": ["G_soft0", "G_hard0"],
            "real_vs_placebo": ["G_soft0", "G_placebo0"],
        }
        contrasts = {
            name: (lambda row, left=pair[0], right=pair[1]:
                   row[left] - row[right])
            for name, pair in contrast_pairs.items()}
        task_contrasts = {
            task: {name: function(values)
                   for name, function in contrasts.items()}
            for task, values in task_values.items()}
        summary = {
            name: {
                "task_mean": float(np.mean([
                    values[name] for values in task_contrasts.values()])),
                "ci95": _bootstrap(
                    task_contrasts, name,
                    int(freeze["analysis"]["bootstrap_seed"]),
                    int(freeze["analysis"]["resamples"])),
            } for name in contrasts}
        result = {
            "schema": "gap-candidate-marginal-value-result-v1",
            "claim_boundary": freeze["claim_boundary"],
            "protocol_complete": True,
            "generation_frozen_before_reporting": True,
            "row_count": len(result_rows),
            "coordinate_failure_count": len(coordinate_failures),
            "coordinate_failures": coordinate_failures,
            "rows": result_rows,
            "task_condition_means": task_values,
            "task_contrasts": task_contrasts,
            "contrasts": summary,
            "pass_fail_gate": None,
            "confirmation_responses_opened": bool(
                freeze.get("confirmation_execution")),
            "test_or_ood_accessed": False,
            "heldout_opened": False,
        }
        decision = freeze.get("confirmation_decision") or {}
        if decision:
            admitted_rows = [
                row for row in result_rows if row["admitted"]]
            gains = [float(row["marginal_value"])
                     for row in admitted_rows]
            assessment = {
                "schema": "gap-protected-final-confirmation-assessment-v1",
                "admitted_candidate_count": len(gains),
                "positive_admitted_count":
                    sum(value > 0 for value in gains),
                "negative_admitted_count":
                    sum(value < 0 for value in gains),
                "zero_admitted_count":
                    sum(value == 0 for value in gains),
                "mean_admitted_reporting_gain": (
                    float(np.mean(gains)) if gains else None),
            }
            assessment["passed"] = bool(
                len(gains) >= int(decision["minimum_admitted_candidates"])
                and assessment["negative_admitted_count"]
                <= int(decision["maximum_negative_admitted_candidates"])
                and assessment["mean_admitted_reporting_gain"] is not None
                and assessment["mean_admitted_reporting_gain"]
                > float(decision["minimum_mean_admitted_gain"]))
            result["confirmation_assessment"] = assessment
        _write_new(args.output_root / "RESULT.json", result)
        print(json.dumps({
            "protocol_complete": True,
            "row_count": len(result_rows),
            "contrasts": summary,
            **({"confirmation_assessment":
                result["confirmation_assessment"]}
               if "confirmation_assessment" in result else {})},
            indent=2, sort_keys=True))
        return 0
    except Exception as error:
        _write_new(args.output_root / "FAILURE.json", {
            "schema": "gap-candidate-marginal-value-failure-v1",
            "error_type": type(error).__name__,
            "message": str(error),
            "reporting_opened": (
                args.output_root / "GENERATION_FROZEN.json").exists(),
            "test_or_ood_accessed": False,
            "heldout_opened": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
