"""Post-reporting diagnostic replay of the 32 frozen provider responses.

This is not prospective efficacy evidence. It repairs only the candidate
evaluation-ledger isolation bug, runs no engine/provider, freezes all replayed
validity/admission outcomes, and then recomputes the development reporting
diagnostic.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from hypothesis_mvp.discovery.contracts import json_safe
from hypothesis_mvp.discovery.factory import build_scientific_discovery_runtime
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY
from hypothesis_mvp.discovery.proposal_runtime import (
    ProposalBatch, ProposalContext, strict_json_loads,
)
from hypothesis_mvp.discovery.regional_candidate_admission import (
    admit_regional_candidates,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from scripts.gap_minimal_harness_contract import CONDITIONS, marginal_value
from scripts.run_aistats_three_arm_formula_child import _task_metadata
from scripts.run_gap_candidate_marginal_reporting_continuation import (
    _load_generation,
)
from scripts.run_gap_candidate_marginal_value import (
    _bootstrap, _generation_roles, _provider, _read, _reporting_response,
    _row_from_state, _runtime_config, _score, _write_new, identity,
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, encoding="utf-8",
        stdout=subprocess.PIPE).stdout.strip()


def _batch_from_transcript(runtime, generation):
    request = json.loads(
        Path(generation["raw_request_path"]).read_text(encoding="utf-8"))
    response = json.loads(
        Path(generation["raw_response_path"]).read_text(encoding="utf-8"))
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("frozen request has no messages")
    payload_text = str(messages[-1].get("content") or "")
    payload = strict_json_loads(payload_text)
    choices = response.get("choices")
    message = (
        choices[0].get("message")
        if isinstance(choices, list) and choices else None)
    content = (
        message.get("content") if isinstance(message, dict) else None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("frozen response has no content")
    scientific = dict(payload["scientific_context"])
    contract = dict(payload["contract"])
    context = ProposalContext(
        int(payload["round_id"]), str(payload["island"]),
        str(payload["parent_hash"]), int(payload["task"]["n_features"]),
        int(contract["max_candidates"]),
        incumbent_expression=str(
            scientific.get("current_expression") or ""),
        existing_supports=tuple(tuple(row) for row in
            scientific.get("posterior_gap_existing_supports", ())),
        gap_directed=bool(
            scientific.get("posterior_gap_brief", {}).get(
                "propose_allowed", False)),
        new_skeleton_quota=int(
            contract.get("new_skeleton_quota") or 0))
    prompt_hash = sha256(payload_text.encode()).hexdigest()
    response_hash = sha256(content.encode()).hexdigest()
    try:
        parsed = strict_json_loads(content)
        candidates, rejections, normalizations = runtime.proposal._validate_batch(
            parsed, context, prompt_hash, response_hash)
    except Exception as error:
        candidates, normalizations = (), ()
        rejections = ({
            "proposal_index": -1, "candidate_id": "",
            "error_type": type(error).__name__, "error": str(error)},)
    return ProposalBatch(
        candidates=tuple(candidates),
        protocol_valid=bool(candidates and not rejections),
        reason=("ok" if candidates and not rejections
                else "partial_candidates_rejected" if candidates
                else "all_candidates_rejected"),
        prompt_hash=prompt_hash, response_hash=response_hash,
        telemetry={
            "offline_frozen_response_replay": True,
            "candidate_validation_rejections": list(rejections),
            "candidate_normalizations": list(normalizations)},
        error="" if candidates else json.dumps(
            list(rejections), sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("frozen-response replay output already exists")
    freeze = _read(args.freeze)
    authorization = _read(args.authorization)
    if (authorization.get("frozen_response_replay_authorized") is not True
            or authorization.get("freeze_sha256") != _digest(args.freeze)
            or authorization.get("source_output")
            != str(args.source_output.resolve())
            or authorization.get("generation_freeze_sha256")
            != _digest(args.source_output / "GENERATION_FROZEN.json")
            or authorization.get("benchmark_commit") != _git_commit(ROOT)
            or authorization.get("mainline_commit")
            != _git_commit(ROOT.parent / "hypothesis_mvp")
            or authorization.get("replay_source_sha256")
            != _digest(Path(__file__))):
        raise ValueError("invalid frozen-response replay authorization")
    _, generation_freeze, frozen_rows = _load_generation(
        args.source_output, freeze)
    args.output_root.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(
        Path(freeze["config"]).read_text(encoding="utf-8"))
    replay_rows = []
    blocks = {}
    for coordinate in freeze["coordinates"]:
        family, task, seed = (
            str(coordinate["family"]), str(coordinate["task"]),
            int(coordinate["seed"]))
        roles = _generation_roles(
            Path(freeze["hdf5"]), family, task, seed)
        block_dir = (
            args.source_output / "generation" / family / task /
            f"seed{seed}")
        shared = _read(block_dir / "SHARED_BLOCK.json")
        E0 = [dict(row) for row in shared["shared"]["E0_rows"]]
        symbols, _, properties, _, _ = _task_metadata(
            Path(freeze["metadata"]), task,
            roles["selection"].development.X.shape[1])
        metadata = {
            "feature_names": symbols[1:],
            "feature_units": properties[1:],
            "target_name": symbols[0],
            "target_unit": properties[0]}
        setup = build_scientific_discovery_runtime(
            n_features=roles["selection"].development.X.shape[1],
            config=_runtime_config(seed, len(E0)),
            library_path=args.output_root / "setup" / task /
            f"seed{seed}" / "knowledge.jsonl",
            ledger_path=args.output_root / "setup" / task /
            f"seed{seed}" / "ledger.jsonl",
            variable_metadata=metadata)
        setup.evaluation.registered_covariates = np.vstack((
            roles["selection"].development.X,
            roles["selection"].validation.X,
            roles["gap_audit"].X, roles["gap_admission"].X,
            roles["selection"].acquisition_pool.X))
        parent, _ = setup.evaluation.seed_survivor(
            E0, roles["selection"].development.X,
            roles["selection"].development.y,
            roles["selection"].validation.X,
            roles["selection"].validation.y)
        for condition in CONDITIONS:
            generation = frozen_rows[(family, task, seed, condition)]
            runtime = build_scientific_discovery_runtime(
                n_features=roles["selection"].development.X.shape[1],
                config=_runtime_config(seed, len(E0)),
                library_path=args.output_root / "replay" / task /
                f"seed{seed}" / condition / "knowledge.jsonl",
                ledger_path=args.output_root / "replay" / task /
                f"seed{seed}" / condition / "ledger.jsonl",
                variable_metadata=metadata)
            runtime.evaluation.registered_covariates = (
                setup.evaluation.registered_covariates)
            condition_parent, _ = runtime.evaluation.seed_survivor(
                E0, roles["selection"].development.X,
                roles["selection"].development.y,
                roles["selection"].validation.X,
                roles["selection"].validation.y)
            if condition_parent.dag.canonical_hash != parent.dag.canonical_hash:
                raise ValueError("replay parent differs across conditions")
            runtime.evaluation.reset()
            runtime.evaluation.registered_covariates = (
                setup.evaluation.registered_covariates)
            batch = _batch_from_transcript(runtime, generation)
            _, candidates, validity = runtime._evaluate_batch(
                batch, condition_parent, (
                    roles["selection"].development.X,
                    roles["selection"].development.y,
                    roles["selection"].validation.X,
                    roles["selection"].validation.y),
                "nmse", 1)
            optional = [
                _row_from_state(state, condition) for state in candidates]
            gap = _read(block_dir / "SHARED_BLOCK.json")
            # The original generation used all frozen axis regions under the
            # support-preserving admission policy. Reconstruct those regions
            # from the same opened action domain through the agent helper is
            # unnecessary: the condition artifact records the exact admission
            # axis via its frozen E0 block. Load it from the original condition
            # by recomputing the registered one-axis median partition.
            from hypothesis_mvp.discovery.candidate_region_expansion import (
                FrozenAxisRegions,
            )
            domain = roles["selection"].acquisition_pool.X
            regions = FrozenAxisRegions(
                domain.shape[1], 0, (float(np.median(domain[:, 0])),))
            eligible = tuple(range(len(regions.cuts) + 1))
            if optional:
                retained, admission = admit_regional_candidates(
                    E0, optional, roles["selection"].development,
                    roles["gap_audit"], roles["gap_admission"],
                    domain, regions, eligible,
                    NormalInverseGammaPrior(), 2, alpha=.025,
                    coefficient_policy=EXPANDED_FORMULA_POLICY)
            else:
                retained, admission = [], {
                    "attempts": 0, "admitted": 0,
                    "reason": "no-valid-materialized-candidate"}
            replay_rows.append({
                "family": family, "task": task, "seed": seed,
                "condition": condition,
                "block_identity": shared["block_identity"],
                "batch_protocol_valid": batch.protocol_valid,
                "batch_reason": batch.reason,
                "candidate_count": len(batch.candidates),
                "validity": validity,
                "materialized_candidate_count": len(candidates),
                "optional_rows": optional,
                "admission": admission,
                "admitted_rows": [dict(row) for row in retained],
            })
        blocks[(family, task, seed)] = (roles, shared, E0)
    replay_freeze = {
        "schema": "gap-frozen-response-evaluation-replay-v1",
        "post_reporting_diagnostic_only": True,
        "row_count": len(replay_rows),
        "rows": replay_rows,
        "source_generation_freeze_sha256": _digest(
            args.source_output / "GENERATION_FROZEN.json"),
        "engine_calls": 0, "provider_calls": 0,
        "test_or_ood_accessed": False, "heldout_opened": False}
    _write_new(args.output_root / "REPLAY_FROZEN.json", replay_freeze)

    scored = []
    for (family, task, seed), (roles, shared, E0) in blocks.items():
        y_report = _reporting_response(
            Path(freeze["hdf5"]), family, task, roles["indices"])
        baseline, baseline_audit = _score(
            E0, roles["selection"], roles["reporting_X"], y_report,
            identity({"block": shared["block_identity"], "bank": "E0"}))
        for row in [
                item for item in replay_rows
                if item["family"] == family and item["task"] == task
                and item["seed"] == seed]:
            admitted = list(row["admitted_rows"])
            candidate_score, audit = baseline, baseline_audit
            if admitted:
                candidate_score, audit = _score(
                    [*E0, *admitted], roles["selection"],
                    roles["reporting_X"], y_report,
                    identity({"block": shared["block_identity"],
                              "condition": row["condition"],
                              "admitted": admitted}))
            scored.append({
                "family": family, "task": task, "seed": seed,
                "condition": row["condition"],
                "baseline_log_score": baseline,
                "candidate_log_score": candidate_score,
                "marginal_value": marginal_value(
                    baseline_score=baseline,
                    candidate_score=candidate_score,
                    materialized=bool(row["materialized_candidate_count"]),
                    valid=bool(row["optional_rows"]),
                    admitted=bool(admitted)),
                "materialized": bool(row["materialized_candidate_count"]),
                "valid": bool(row["optional_rows"]),
                "admitted": bool(admitted), "score_audit": audit})
    grouped = {}
    for row in scored:
        grouped.setdefault(row["task"], {}).setdefault(
            row["condition"], []).append(row["marginal_value"])
    task_values = {
        task: {condition: float(np.mean(values))
               for condition, values in conditions.items()}
        for task, conditions in grouped.items()}
    functions = {
        "gap_vs_blind": lambda row: row["G_hard0"] - row["B0"],
        "soft_vs_hard": lambda row:
            row["G_soft0"] - row["G_hard0"],
        "real_vs_placebo": lambda row:
            row["G_soft0"] - row["G_placebo0"]}
    task_contrasts = {
        task: {name: function(values)
               for name, function in functions.items()}
        for task, values in task_values.items()}
    summary = {
        name: {
            "task_mean": float(np.mean(
                [row[name] for row in task_contrasts.values()])),
            "ci95": _bootstrap(
                task_contrasts, name,
                int(freeze["analysis"]["bootstrap_seed"]),
                int(freeze["analysis"]["resamples"]))}
        for name in functions}
    result = {
        "schema": "gap-frozen-response-replay-result-v1",
        "claim_boundary": (
            "Post-reporting development mechanism diagnostic only; not "
            "prospective efficacy, superiority, confirmation or protocol PASS."),
        "protocol_complete": True,
        "post_reporting_diagnostic_only": True,
        "engine_calls": 0, "provider_calls": 0,
        "rows": scored, "task_condition_means": task_values,
        "task_contrasts": task_contrasts, "contrasts": summary,
        "test_or_ood_accessed": False, "heldout_opened": False}
    _write_new(args.output_root / "RESULT.json", result)
    print(json.dumps({
        "protocol_complete": True,
        "post_reporting_diagnostic_only": True,
        "row_count": len(scored),
        "contrasts": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
