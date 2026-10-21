"""Score one Experiment 2 arm on the frozen common report.

Why this exists instead of reusing audit_matched_three_arm_reporting.run():
that scorer reads `iterative_feedback_trace`, which the agent only emits when
`iterative_posterior_refinement` is true. Arm B turns that off, so the trace is absent
and the existing path cannot score it.

Symmetry rule (frozen): both arms must resolve their bank through the SAME field. The
field is chosen once, from the union of what both arms in the pair expose, and if the two
arms resolve to different fields the pair is rejected rather than scored.

Support-dedup rule (Amendment 004): before freezing, the bank is collapsed to one row
per structural support, keeping the first occurrence in bank order -- applied
identically to every arm, with the dropped-row count recorded in the score output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline  # noqa: E402
ensure_mainline()

from hypothesis_mvp.data.roles import (  # noqa: E402
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from hypothesis_mvp.discovery.pcpi_adapter import (  # noqa: E402
    EXPANDED_FORMULA_POLICY, compile_fixed_formula_support, freeze_discovery_target,
    model_factory_for_policy,
)
from hypothesis_mvp.discovery.realized_drr import (  # noqa: E402
    _independent_predictive_score,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior  # noqa: E402

from scripts.audit_matched_three_arm_reporting import (  # noqa: E402
    GROUPS, _covariates, _digest, _opened, _read,
)

BANK_FIELDS = ("evaluated_hypothesis_bank", "final_topk")


def _dedupe_bank_rows(rows, n_features):
    """Collapse duplicate structural supports: first occurrence in bank order wins.

    The discovery bank records every evaluated candidate, so one support can be
    recorded under several source families -- e.g. the constant evaluated by both
    `core` and `engine:pysr`, or an expression the LLM re-discovered after the
    engine found it. The mainline freeze adapter refuses such banks
    (duplicate-support-crosses-source-families); inside discovery the admission
    path deduplicates before anything reaches a frozen model, so the scorer must
    apply the equivalent resolution deterministically: keep the FIRST occurrence
    in bank order, identically for every arm. No per-arm or per-family privilege.
    """
    seen: set = set()
    kept = []
    dropped = 0
    for row in rows:
        terms = compile_fixed_formula_support(str(row["expression"]), n_features)
        key = tuple(terms)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def _freeze_and_score(rows, selection, reporting):
    identity = __import__("hashlib").sha256(json.dumps(
        rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
    model = model_factory_for_policy(EXPANDED_FORMULA_POLICY)(
        rows, n_features=selection.development.X.shape[1],
        prior=NormalInverseGammaPrior(), exploration_identity=identity,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    target = freeze_discovery_target(
        model, selection.development, selection.acquisition_pool.X,
        measurement_budget=2, expected_model_identity=model.stable_hash)
    mse, log_score = _independent_predictive_score(
        model.engine(model.stable_hash), target.initial_posterior, reporting)
    return {
        "model_identity": model.stable_hash,
        "posterior_identity": target.stable_hash,
        "mse": mse, "log_score": log_score, "bank_size": len(rows),
    }


def _reporting_sets(artifact, hdf5, family, task, seed):
    report = artifact["scientific_discovery_runtime"]
    indices = report["drr_role_row_indices"]
    from methods.hypothesis_mvp_pcpi.drr_adapter import three_arm_role_indices
    with h5py.File(hdf5, "r") as handle:
        data = handle[f"{GROUPS[family]}/{task}/train"]
        expected = three_arm_role_indices(data.shape[0], task_name=task, seed=seed)
        if {key: list(value) for key, value in expected.items()} != indices:
            raise ValueError("ablation reporting split changed")
        fit = RoleDataset(DataRole.DEVELOPMENT,
                          *_opened(data, indices["discovery_development"], response=True))
        validation = RoleDataset(DataRole.VALIDATION,
                                 *_opened(data, indices["discovery_validation"], response=True))
        actions = _opened(data, indices["action_covariates"], response=False)
        cycle_roles = tuple(
            (RoleDataset(DataRole.VALIDATION, *_opened(data, pair["gap_audit"], response=True)),
             RoleDataset(DataRole.VALIDATION, *_opened(data, pair["gap_admission"], response=True)))
            for pair in report["iterative_cycle_role_row_indices"])
        reporting = RoleDataset(DataRole.VALIDATION,
                                *_opened(data, indices["reporting"], response=True))
    selection = SelectionData(
        fit, validation,
        AcquisitionCovariates(DataRole.ACQUISITION_POOL, actions,
                              covariate_fingerprint(actions)), ())
    report_rows = _covariates(reporting)
    roles = (fit, validation, selection.acquisition_pool,
             *(role for pair in cycle_roles for role in pair))
    if any(report_rows & _covariates(role) for role in roles):
        raise ValueError("ablation reporting overlaps generation roles")
    return selection, reporting


def resolve_bank_field(artifact):
    runtime = artifact["scientific_discovery_runtime"]
    for field in BANK_FIELDS:
        rows = runtime.get(field)
        if rows:
            return field
    raise ValueError("arm exposes no scorable bank")


def score_arm(*, arm_artifact: Path, hdf5: Path, family: str, task: str, seed: int,
              bank_field: str | None = None):
    artifact = _read(arm_artifact)
    if (artifact.get("task_name") != task
            or artifact.get("scientific_discovery_runtime", {}).get("test_or_ood_accessed")
            is not False):
        raise ValueError("ablation artifact identity changed")
    runtime = artifact["scientific_discovery_runtime"]
    field = bank_field or resolve_bank_field(artifact)
    rows = runtime.get(field)
    if not rows:
        raise ValueError(f"arm exposes no rows under {field}")
    bank_rows = [{"expression": row["expression"], "source": row["source"]} for row in rows]
    selection, reporting = _reporting_sets(artifact, hdf5, family, task, seed)
    bank_rows, dropped = _dedupe_bank_rows(bank_rows, selection.development.X.shape[1])
    scored = _freeze_and_score(bank_rows, selection, reporting)
    scored["bank_field"] = field
    scored["bank_rows_raw"] = len(rows)
    scored["bank_rows_deduped"] = len(bank_rows)
    scored["duplicate_support_rows_dropped"] = dropped
    scored["bank_field_sha256"] = __import__("hashlib").sha256(json.dumps(
        bank_rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
    scored["artifact_sha256"] = _digest(arm_artifact)
    return scored
