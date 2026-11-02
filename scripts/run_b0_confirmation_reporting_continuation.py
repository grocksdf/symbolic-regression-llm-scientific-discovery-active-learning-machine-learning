"""Complete B0-only reporting from a frozen terminal generation run."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from hypothesis_mvp.discovery.admission_transaction import (
    freeze_admitted_posterior_transaction,
)
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from run_gap_candidate_marginal_value import (
    _generation_roles,
    _gap_admission_role,
    _posterior_update_role,
    _reporting_response,
    _score,
    identity,
)
from gap_minimal_harness_contract import marginal_value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError("reporting continuation output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("reporting continuation root exists")
    freeze = read(args.freeze)
    source_freeze = read(args.source_output / "RUN_FREEZE.json")
    failure = read(args.source_output / "FAILURE.json")
    generation_freeze = args.source_output / "GENERATION_FROZEN.json"
    update_freeze = args.source_output / "POSTERIOR_UPDATE_FROZEN.json"
    if (failure.get("error_type") != "KeyError"
            or failure.get("message") != "'G_hard0'"
            or not generation_freeze.exists()
            or not update_freeze.exists()
            or freeze != source_freeze
            or freeze.get("conditions") != ["B0"]
            or len(freeze.get("coordinates", ())) != 8):
        raise ValueError("source is not the registered B0 aggregation failure")
    args.output_root.mkdir(parents=True)
    write_new(args.output_root / "REPORTING_CONTINUATION_FREEZE.json", {
        "schema": "b0-reporting-continuation-freeze-v1",
        "source_output": str(args.source_output.resolve()),
        "source_run_freeze_sha256":
            digest(args.source_output / "RUN_FREEZE.json"),
        "generation_freeze_sha256": digest(generation_freeze),
        "posterior_update_freeze_sha256": digest(update_freeze),
        "failure_sha256": digest(args.source_output / "FAILURE.json"),
        "provider_calls_authorized": False,
        "generation_or_admission_rerun": False,
        "reporting_only": True,
    })
    rows = []
    hdf5 = Path(freeze["hdf5"])
    for coordinate in freeze["coordinates"]:
        family = coordinate["family"]
        task = coordinate["task"]
        seed = int(coordinate["seed"])
        root = (args.source_output / "generation" / family / task /
                f"seed{seed}")
        shared = read(root / "SHARED_BLOCK.json")
        generation = read(root / "B0" / "GENERATION_RESULT.json")
        E0 = [dict(row) for row in shared["shared"]["E0_rows"]]
        admitted = [dict(row) for row in generation["admitted_rows"]]
        roles = _generation_roles(hdf5, family, task, seed)
        selection = roles["selection"]
        gap_admission = _gap_admission_role(
            hdf5, family, task, roles["indices"])
        update = _posterior_update_role(
            hdf5, family, task, roles["indices"])
        y_report = _reporting_response(
            hdf5, family, task, roles["indices"])
        baseline, baseline_audit = _score(
            E0, selection, roles["reporting_X"], y_report,
            identity({"task": task, "seed": seed, "bank": "E0"}),
            posterior_update=update)
        candidate_score = baseline
        score_audit = baseline_audit
        if admitted:
            model, target, transaction = (
                freeze_admitted_posterior_transaction(
                    E0, admitted, selection.development,
                    gap_admission, update,
                    selection.acquisition_pool.X,
                    prior=NormalInverseGammaPrior(),
                    measurement_budget=2,
                    coefficient_policy=EXPANDED_FORMULA_POLICY))
            density = np.asarray(
                model.engine(model.stable_hash).predictive_logpdf(
                    target.initial_posterior,
                    roles["reporting_X"], y_report), dtype=float)
            if not np.all(np.isfinite(density)):
                raise FloatingPointError(
                    "continuation reporting density is nonfinite")
            candidate_score = float(np.mean(density))
            score_audit = {
                "model_identity": model.stable_hash,
                "target_identity": target.stable_hash,
                "posterior_member_count":
                    len(target.initial_posterior.members),
                "admission_posterior_transaction": transaction.__dict__,
            }
        increment = marginal_value(
            baseline_score=baseline,
            candidate_score=candidate_score,
            materialized=bool(generation["materialized_candidate_count"]),
            valid=bool(generation["optional_rows"]),
            admitted=bool(admitted))
        rows.append({
            "family": family, "task": task, "seed": seed,
            "condition": "B0",
            "baseline_log_score": baseline,
            "candidate_log_score": candidate_score,
            "marginal_value": increment,
            "materialized": bool(generation["materialized_candidate_count"]),
            "valid": bool(generation["optional_rows"]),
            "admitted": bool(admitted),
            "score_audit": score_audit,
        })
    admitted_rows = [row for row in rows if row["admitted"]]
    gains = [row["marginal_value"] for row in admitted_rows]
    decision = freeze["confirmation_decision"]
    assessment = {
        "schema": "gap-protected-final-confirmation-assessment-v1",
        "admitted_candidate_count": len(admitted_rows),
        "positive_admitted_count": sum(value > 0 for value in gains),
        "negative_admitted_count": sum(value < 0 for value in gains),
        "zero_admitted_count": sum(value == 0 for value in gains),
        "mean_admitted_reporting_gain":
            float(np.mean(gains)) if gains else None,
    }
    assessment["passed"] = bool(
        len(admitted_rows) >= int(
            decision["minimum_admitted_candidates"])
        and assessment["negative_admitted_count"] <= int(
            decision["maximum_negative_admitted_candidates"])
        and assessment["mean_admitted_reporting_gain"] is not None
        and assessment["mean_admitted_reporting_gain"] > float(
            decision["minimum_mean_admitted_gain"]))
    result = {
        "schema": "pmlb-protected-confirmation-result-v1",
        "protocol_complete": True,
        "reporting_only_continuation": True,
        "generation_and_admission_reused": True,
        "provider_calls": 0,
        "row_count": len(rows),
        "task_count": len(rows),
        "task_mean_gain_including_nonadmission_as_zero":
            float(np.mean([row["marginal_value"] for row in rows])),
        "confirmation_assessment": assessment,
        "rows": rows,
        "confirmation_responses_opened": True,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "claim_boundary": freeze["claim_boundary"],
    }
    write_new(args.output_root / "RESULT.json", result)
    print(json.dumps({
        "task_mean_gain_including_nonadmission_as_zero":
            result["task_mean_gain_including_nonadmission_as_zero"],
        "confirmation_assessment": assessment,
        "protocol_complete": True,
        "provider_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
