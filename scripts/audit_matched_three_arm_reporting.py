"""Independent common-report scoring for frozen E, S-matched, and L-gap banks."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()

from hypothesis_mvp.data.roles import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_discovery_target,
    model_factory_for_policy,
)
from hypothesis_mvp.discovery.realized_drr import (
    _independent_predictive_score,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


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
    values = np.asarray(dataset[ordered, :], dtype=float)
    lookup = dict(zip(ordered, values, strict=True))
    values = np.asarray([lookup[index] for index in original])
    return (values[:, 1:], values[:, 0]) if response else values[:, 1:]


def _covariates(data):
    return {
        sha256(np.ascontiguousarray(
            row, dtype=np.float64).tobytes()).hexdigest()
        for row in data.X}


def _freeze(rows, selection, prior):
    identity = sha256(json.dumps(
        rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
    model = model_factory_for_policy(EXPANDED_FORMULA_POLICY)(
        rows, n_features=selection.development.X.shape[1],
        prior=prior, exploration_identity=identity,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    target = freeze_discovery_target(
        model, selection.development, selection.acquisition_pool.X,
        measurement_budget=2,
        expected_model_identity=model.stable_hash)
    return model, target


def score_three_banks(*, engine_rows, matched_rows, llm_rows,
                      selection, reporting):
    prior = NormalInverseGammaPrior()
    scores = {}
    for arm, rows in (
            ("E", engine_rows),
            ("E_plus_S_gap_matched", matched_rows),
            ("E_plus_L_gap", llm_rows)):
        model, target = _freeze(rows, selection, prior)
        mse, log_score = _independent_predictive_score(
            model.engine(model.stable_hash),
            target.initial_posterior, reporting)
        scores[arm] = {
            "model_identity": model.stable_hash,
            "posterior_identity": target.stable_hash,
            "mse": mse, "log_score": log_score,
            "bank_size": len(rows),
        }
    return {
        "scores": scores,
        "effects": {
            "L_minus_S_log_score": (
                scores["E_plus_L_gap"]["log_score"]
                - scores["E_plus_S_gap_matched"]["log_score"]),
            "L_minus_E_log_score": (
                scores["E_plus_L_gap"]["log_score"]
                - scores["E"]["log_score"]),
            "S_minus_E_log_score": (
                scores["E_plus_S_gap_matched"]["log_score"]
                - scores["E"]["log_score"]),
            "E_minus_L_mse": (
                scores["E"]["mse"]
                - scores["E_plus_L_gap"]["mse"]),
            "E_minus_S_mse": (
                scores["E"]["mse"]
                - scores["E_plus_S_gap_matched"]["mse"]),
        },
    }


def run(l_artifact: Path, s_artifact: Path, hdf5: Path,
        family: str, task: str, seed: int):
    l_frozen, s_frozen = _read(l_artifact), _read(s_artifact)
    if (l_frozen.get("task_name") != task
            or s_frozen.get("family") != family
            or s_frozen.get("task") != task
            or s_frozen.get("seed") != seed
            or s_frozen.get("source_artifact_sha256")
            != _digest(l_artifact)
            or s_frozen.get("reporting_responses_opened") is not False):
        raise ValueError("matched reporting artifact identity changed")
    report = l_frozen["scientific_discovery_runtime"]
    l_eval = report["scientist_agent_system_evaluation"]
    l_trace = l_eval["iterative_feedback_trace"]
    s_trace = s_frozen["system_evaluation"][
        "iterative_feedback_trace"]
    if (len(l_trace) != 2 or len(s_trace) != 2
            or l_trace[0]["bank_before"] != s_trace[0]["bank_before"]
            or l_trace[0]["posterior_before"]
            != s_trace[0]["posterior_before"]):
        raise ValueError("matched reporting does not share initial E")
    indices = report["drr_role_row_indices"]
    pairs = report["iterative_cycle_role_row_indices"]
    from methods.hypothesis_mvp_pcpi.drr_adapter import (
        iterative_gap_role_indices, three_arm_role_indices,
    )
    with h5py.File(hdf5, "r") as handle:
        data = handle[f"{GROUPS[family]}/{task}/train"]
        expected = three_arm_role_indices(
            data.shape[0], task_name=task, seed=seed)
        if {key: list(value) for key, value in expected.items()} != indices:
            raise ValueError("matched reporting split changed")
        if [{"gap_audit": list(a), "gap_admission": list(b)}
                for a, b in iterative_gap_role_indices(expected, 2)] != pairs:
            raise ValueError("matched reporting cycle split changed")
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
        cycle_roles = tuple((
            RoleDataset(DataRole.VALIDATION,
                *_opened(data, pair["gap_audit"], response=True)),
            RoleDataset(DataRole.VALIDATION,
                *_opened(data, pair["gap_admission"], response=True)),
        ) for pair in pairs)
        reporting = RoleDataset(
            DataRole.VALIDATION,
            *_opened(data, indices["reporting"], response=True))
    selection = SelectionData(
        fit, validation,
        AcquisitionCovariates(
            DataRole.ACQUISITION_POOL, actions,
            covariate_fingerprint(actions)), ())
    report_rows = _covariates(reporting)
    roles = (fit, validation, selection.acquisition_pool,
             *(role for pair in cycle_roles for role in pair))
    if any(report_rows & _covariates(role) for role in roles):
        raise ValueError("matched reporting overlaps generation roles")
    scored = score_three_banks(
        engine_rows=l_trace[0]["core_rows_before"],
        matched_rows=s_trace[-1]["bank_rows_after"],
        llm_rows=l_trace[-1]["bank_rows_after"],
        selection=selection, reporting=reporting)
    return {
        "schema": "matched-three-arm-common-report-v1",
        "family": family, "task": task, "seed": seed,
        **scored,
        "L_feedback_verified": bool(
            report.get("iterative_conditional_feedback_verified")),
        "S_feedback_verified": bool(
            s_frozen["system_evaluation"][
                "iterative_feedback_gate"].get("passed")),
        "reporting_identity": reporting.fingerprint,
        "generation_artifacts_frozen_before_reporting": True,
        "matched_non_llm_skeleton_attempts": True,
        "formula_recovery_attribution_authorized": False,
        "measured_action_authorized": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l-artifact", type=Path, required=True)
    parser.add_argument("--s-artifact", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(GROUPS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("matched report output exists")
    result = run(
        args.l_artifact, args.s_artifact, args.hdf5,
        args.family, args.task, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "family": args.family, "task": args.task, "seed": args.seed,
        "effects": result["effects"],
        "L_feedback_verified": result["L_feedback_verified"],
        "S_feedback_verified": result["S_feedback_verified"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
