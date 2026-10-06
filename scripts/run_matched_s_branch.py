"""Build the isolated iterative S-gap-matched branch before reporting opens."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()

from hypothesis_mvp.data.roles import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from hypothesis_mvp.discovery.agent import (
    DiscoveryAgent, DiscoveryAgentConfig,
)
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.iterative_refinement_gate import (
    verify_iterative_refinement,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_discovery_target,
    model_factory_for_policy, support_parser_for_policy,
)
from hypothesis_mvp.discovery.regional_candidate_admission import (
    admit_regional_candidates,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.symbolic.scheduler import (
    EngineResult, EngineRunRecord, MultiEngineResult,
)
from scripts.matched_gap_structural_generator import (
    ISLANDS, generate_matched_slot,
)
from scripts.effective_structural_generator import (
    LEGACY_POLICY, POLICY, generate_effective_cycle, policy_contract,
)


GROUPS = {
    "bio_pop_growth": "lsr_synth/bio_pop_growth",
    "chem_react": "lsr_synth/chem_react",
    "lsr_transform": "lsr_transform",
    "phys_osc": "lsr_synth/phys_osc",
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _opened(dataset, indices, *, response):
    original = [int(index) for index in indices]
    ordered = sorted(original)
    values = np.asarray(dataset[ordered, :] if response else (
        dataset[ordered, 1:]), dtype=float)
    lookup = dict(zip(ordered, values, strict=True))
    values = np.asarray([lookup[index] for index in original])
    return (values[:, 1:], values[:, 0]) if response else values


def _roles(report, hdf5: Path, family: str, task: str, seed: int):
    from methods.hypothesis_mvp_pcpi.drr_adapter import (
        iterative_gap_role_indices, three_arm_role_indices,
    )
    indices = report["drr_role_row_indices"]
    pairs = report["iterative_cycle_role_row_indices"]
    with h5py.File(hdf5, "r") as handle:
        data = handle[f"{GROUPS[family]}/{task}/train"]
        expected = three_arm_role_indices(
            data.shape[0], task_name=task, seed=seed)
        if {key: list(value) for key, value in expected.items()} != indices:
            raise ValueError("matched S role indices changed")
        expected_pairs = [
            {"gap_audit": list(a), "gap_admission": list(b)}
            for a, b in iterative_gap_role_indices(expected, 2)]
        if expected_pairs != pairs:
            raise ValueError("matched S cycle role partition changed")
        fit = RoleDataset(
            DataRole.DEVELOPMENT,
            *_opened(data, indices["discovery_development"],
                     response=True))
        validation = RoleDataset(
            DataRole.VALIDATION,
            *_opened(data, indices["discovery_validation"],
                     response=True))
        actions = _opened(
            data, indices["action_covariates"], response=False)
        cycles = tuple((
            RoleDataset(DataRole.VALIDATION,
                *_opened(data, pair["gap_audit"], response=True)),
            RoleDataset(DataRole.VALIDATION,
                *_opened(data, pair["gap_admission"], response=True)),
        ) for pair in pairs)
    return (
        SelectionData(
            fit, validation,
            AcquisitionCovariates(
                DataRole.ACQUISITION_POOL, actions,
                covariate_fingerprint(actions)), ()),
        cycles,
    )


def _engine_result(payload) -> MultiEngineResult:
    def result(row):
        return EngineResult(
            engine=str(row["engine"]),
            expression=str(row["expression"]),
            mse_val=float(row["mse_val"]),
            complexity=float(row["complexity"]),
            score=float(row["score"]),
            diagnostics=dict(row.get("diagnostics") or {}),
            repeats=tuple(dict(value) for value in row.get("repeats", ())),
            lineage_id=str(row.get("lineage_id") or ""),
        )

    def record(row):
        return EngineRunRecord(
            engine=str(row["engine"]), repeat=int(row["repeat"]),
            attempt=int(row["attempt"]), seed=int(row["seed"]),
            status=str(row["status"]),
            elapsed_seconds=float(row["elapsed_seconds"]),
            lineage_id=str(row["lineage_id"]),
            expression=str(row.get("expression") or ""),
            error_type=str(row.get("error_type") or ""),
            error_message=str(row.get("error_message") or ""),
            controls=tuple(row.get("controls") or ()),
        )

    return MultiEngineResult(
        best=result(payload["best"]),
        all_results=tuple(result(row) for row in payload["all_results"]),
        run_records=tuple(record(row) for row in payload.get(
            "run_records", ())),
        failures=tuple(dict(row) for row in payload.get("failures", ())),
        evaluation_budget=int(payload["evaluation_budget"]),
        evaluations_used=int(payload["evaluations_used"]),
    )


def _agent(config_path: Path, seed: int):
    values = yaml.safe_load(config_path.read_text(
        encoding="utf-8"))["agent_config"]
    values = {**values, "engines": tuple(values["engines"]),
              "discovery_islands": tuple(values["discovery_islands"]),
              "random_seed": seed}
    agent = DiscoveryAgent.__new__(DiscoveryAgent)
    agent.config = DiscoveryAgentConfig(**values)
    agent.provider_settings = None
    return agent


def _gap_identity(prompt) -> str:
    return sha256(json.dumps(
        prompt, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _parent_rows(engines, admitted):
    rows = [{
        "expression": row.expression,
        "source": f"engine:{row.engine}",
        "lineage_id": row.lineage_id,
    } for row in engines.all_results]
    rows.extend(dict(row) for row in admitted)
    return rows


def _dispersion(rows, gap, domain, n_features):
    runtime = EquationRuntime(
        n_features, refit_policy="pcpi-expanded-fixed-inner-v1")
    assignments = gap["regions"].assign(domain)
    eligible = {
        int(row["region"]) for row in gap["prompt"].get("regions", ())}
    mask = np.isin(assignments, sorted(eligible))
    values = domain[
        mask if np.count_nonzero(mask) >= 2 else np.ones(
            len(assignments), dtype=bool)]
    output = {}
    for row in rows:
        lineage = str(row.get("lineage_id") or "")
        if not lineage:
            continue
        try:
            prediction = runtime.predict(str(row["expression"]), values)
        except (SyntaxError, TypeError, ValueError):
            continue
        score = float(np.std(prediction))
        if np.isfinite(score):
            output[lineage] = score
    return output


def _materialize(slots, selection):
    runtime = EquationRuntime(
        selection.development.X.shape[1],
        refit_policy="pcpi-expanded-fixed-inner-v1")
    parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    rows, audit, seen = [], [], set()
    for slot in slots:
        if slot["status"] != "materialized":
            audit.append({**slot, "materialization": slot["status"]})
            continue
        raw = slot["candidate"]
        stage = "parameter_fit"
        fit_passed = validation_passed = False
        try:
            fitted = runtime.refit_global_constants(
                raw["expression"], selection.development.X,
                selection.development.y)
            fit_passed = True
            stage = "validation_definedness"
            runtime.predict(
                fitted.expression, selection.validation.X)
            validation_passed = True
            stage = "semantic_retention"
            support = parser(
                fitted.expression, selection.development.X.shape[1])
            if (raw.get("generator_policy") == POLICY
                    and tuple(raw["support"]) != support):
                raise ValueError("effective S structure changed during refit")
            if support in seen:
                raise ValueError("matched materialization duplicated support")
            seen.add(support)
        except (ArithmeticError, RuntimeError, SyntaxError,
                TypeError, ValueError) as error:
            audit.append({
                **slot, "materialization": "rejected",
                "materialization_reason": str(error), "rejection_stage": stage,
                "parameter_fit_passed": fit_passed,
                "validation_definedness_passed": validation_passed})
            continue
        row = {
            **dict(raw), "expression": fitted.expression,
            "support": list(support),
        }
        rows.append(row)
        audit.append({
            **slot, "materialization": "validated",
            "parameter_fit_passed": True, "validation_definedness_passed": True,
            "materialized_lineage_id": row["lineage_id"]})
    return rows, audit


def _structural_slots(policy, parent_rows, brief, dispersion,
                      selection, task_identity, seed, cycle):
    if policy == POLICY:
        points = np.vstack((selection.development.X, selection.validation.X,
                            selection.acquisition_pool.X))
        plan = generate_effective_cycle(
            engine_rows=parent_rows, gap_brief=brief,
            parent_dispersion=dispersion, covariates=points,
            task_identity=task_identity, seed=seed, cycle=cycle)
        return plan["slots"], plan
    if policy != LEGACY_POLICY:
        raise ValueError("unregistered S generator policy")
    return [generate_matched_slot(
        engine_rows=parent_rows, gap_brief=brief,
        parent_dispersion=dispersion,
        n_features=selection.development.X.shape[1],
        task_identity=task_identity, seed=seed, cycle=cycle, island=island)
        for island in ISLANDS], None


def _opportunity_funnel(slots, materialization, admission_report):
    ledger = admission_report.get("candidates", ())
    return {
        "registered_slots": len(slots),
        "gap_activated_slots": sum(row.get("gap_activated", bool(
            row["attempt_count"])) for row in slots),
        "materialized_before_fit": sum(
            row["status"] == "materialized" for row in slots),
        "fit_and_validation_survived": sum(
            row["materialization"] == "validated" for row in materialization),
        "parameter_fit_survived": sum(row.get("parameter_fit_passed") is True
                                      for row in materialization),
        "validation_definedness_survived": sum(row.get(
            "validation_definedness_passed") is True for row in materialization),
        "independent_admission_evaluated": len(ledger),
        "independent_admitted": sum(row.get("admitted") is True for row in ledger),
        "planning_checks": sum(row.get("planning_checks", 0) for row in slots),
        "retry_count": sum(row["retry_count"] for row in slots),
    }


def run(artifact: Path, config_path: Path, hdf5: Path,
        family: str, task: str, seed: int, *, generator_policy=LEGACY_POLICY):
    if generator_policy not in (LEGACY_POLICY, POLICY):
        raise ValueError("unregistered S generator policy")
    frozen = _read(artifact)
    if frozen.get("task_name") != task:
        raise ValueError("matched S source artifact task changed")
    report = frozen["scientific_discovery_runtime"]
    evaluation = report.get("scientist_agent_system_evaluation") or {}
    l_cycles = evaluation.get("cycles") or []
    l_trace = evaluation.get("iterative_feedback_trace") or []
    if (len(l_cycles) != 2 or len(l_trace) != 2
            or report.get("test_or_ood_accessed")
            or report.get("heldout_opened")):
        raise ValueError("matched S requires a complete sealed L artifact")
    selection, cycle_roles = _roles(
        report, hdf5, family, task, seed)
    agent = _agent(config_path, seed)
    prior = NormalInverseGammaPrior()
    parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    factory = model_factory_for_policy(EXPANDED_FORMULA_POLICY)
    admitted, trace, cycle_audits = [], [], []
    expected_bank = expected_target = None
    for cycle in range(2):
        engines = _engine_result(l_cycles[cycle]["engine_report"])
        gap = agent._independent_gap_brief(
            selection, engines, cycle_roles[cycle][0],
            prior, 2, admitted_rows=tuple(admitted))
        if expected_bank is not None and (
                gap["model"].stable_hash != expected_bank
                or gap["target"].stable_hash != expected_target):
            raise ValueError("matched S next gap did not use S admission")
        if cycle == 0 and (
                gap["model"].stable_hash != l_trace[0]["bank_before"]
                or gap["target"].stable_hash != l_trace[0][
                    "posterior_before"]):
            raise ValueError("matched S and L do not share initial E bank")
        parent_rows = _parent_rows(engines, admitted)
        dispersion = _dispersion(
            parent_rows, gap, selection.acquisition_pool.X,
            selection.development.X.shape[1])
        brief = {
            **gap["prompt"], "heldout_opened": False,
            "candidate_response_accessed": False}
        slots, structural_plan = _structural_slots(
            generator_policy, parent_rows, brief, dispersion, selection,
            f"{family}/{task}", seed, cycle)
        optional, materialization = _materialize(slots, selection)
        eligible = tuple(sorted({
            int(row["region"]) for row in gap["prompt"].get("regions", ())}))
        if optional and eligible and gap["prompt"].get("propose_allowed"):
            retained, admission_report = admit_regional_candidates(
                gap["core_rows"], optional, selection.development,
                cycle_roles[cycle][0], cycle_roles[cycle][1],
                selection.acquisition_pool.X, gap["regions"], eligible,
                prior, 2, alpha=.025,
                coefficient_policy=EXPANDED_FORMULA_POLICY)
        else:
            retained, admission_report = [], {
                "schema": "candidate-regional-admission-v1",
                "attempts": len(optional), "candidates": [],
                "admission_identity": cycle_roles[
                    cycle][1].fingerprint,
                "candidate_response_accessed": False,
                "heldout_opened": False,
                "reason": "no-independent-gap-or-no-new-candidate"}
        retained_supports = [
            parser(str(row["expression"]),
                   selection.development.X.shape[1])
            for row in retained]
        if len(set(retained_supports)) != len(retained_supports):
            raise ValueError("matched S admission duplicated support")
        admitted.extend(dict(row) for row in retained)
        after_rows = [*gap["core_rows"], *retained]
        identity = sha256(json.dumps(
            after_rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
        after_model = factory(
            after_rows, n_features=selection.development.X.shape[1],
            prior=prior, exploration_identity=identity,
            coefficient_policy=EXPANDED_FORMULA_POLICY)
        after_target = freeze_discovery_target(
            after_model, selection.development,
            selection.acquisition_pool.X, measurement_budget=2,
            expected_model_identity=after_model.stable_hash)
        expected_bank, expected_target = (
            after_model.stable_hash, after_target.stable_hash)
        support_key = lambda row: json.dumps(parser(
            str(row["expression"]),
            selection.development.X.shape[1]))
        brief_identity = _gap_identity(gap["prompt"])
        trace.append({
            "bank_before": gap["model"].stable_hash,
            "core_rows_before": [dict(row) for row in gap["core_rows"]],
            "posterior_before": gap["target"].stable_hash,
            "gap_bank": gap["model"].stable_hash,
            "gap_posterior": gap["target"].stable_hash,
            "gap_identity": brief_identity,
            "proposal_gap_identity": brief_identity,
            "bank_after": after_model.stable_hash,
            "posterior_after": after_target.stable_hash,
            "bank_supports_after": [support_key(row) for row in after_rows],
            "bank_rows_after": [dict(row) for row in after_rows],
            "admitted_supports": [support_key(row) for row in retained],
            "fit_rows": sorted(
                selection.development.row_fingerprints),
            "selection_rows": sorted(
                selection.validation.row_fingerprints),
            "gap_rows": sorted(
                cycle_roles[cycle][0].row_fingerprints),
            "admission_rows": sorted(
                cycle_roles[cycle][1].row_fingerprints),
            "admission_report": admission_report,
            "attempted_candidate_count": len(optional),
            "proposal_batch_count": sum(
                row["attempt_count"] for row in slots),
            "cumulative_admitted_count": len(admitted),
            "conditional_bank_posterior_only": True,
        })
        cycle_audits.append({
            "cycle": cycle, "gap_prompt": gap["prompt"],
            "gap_audit": gap["audit"],
            "parent_dispersion": dispersion, "slots": slots,
            "structural_plan": structural_plan,
            "opportunity_funnel": _opportunity_funnel(
                slots, materialization, admission_report),
            "materialization": materialization,
            "admission_report": admission_report,
            "bank_before": gap["model"].stable_hash,
            "bank_after": after_model.stable_hash,
        })
    gate = verify_iterative_refinement(trace)
    return {
        "schema": "matched-s-iterative-branch-v1",
        "generator_policy": generator_policy,
        "generator_contract": policy_contract() if generator_policy == POLICY else None,
        "family": family, "task": task, "seed": seed,
        "source_artifact": str(artifact.resolve()),
        "source_artifact_sha256": _digest(artifact),
        "engine_execution_reused_from_source": True,
        "provider_called": False,
        "knowledge_namespace": "isolated-matched-S",
        "system_evaluation": {
            "iterative_feedback_trace": trace,
            "iterative_feedback_gate": gate,
        },
        "cycles": cycle_audits,
        "candidate_response_accessed": False,
        "reporting_responses_opened": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(GROUPS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generator-policy", choices=(LEGACY_POLICY, POLICY),
                        default=LEGACY_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("matched S branch output exists")
    result = run(
        args.artifact, args.config, args.hdf5,
        args.family, args.task, args.seed, generator_policy=args.generator_policy)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "family": args.family, "task": args.task, "seed": args.seed,
        "admitted_counts": [
            len(row["admission_report"].get("candidates", ()))
            for row in result["cycles"]],
        "feedback_gate": result["system_evaluation"][
            "iterative_feedback_gate"],
        "reporting_responses_opened": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
