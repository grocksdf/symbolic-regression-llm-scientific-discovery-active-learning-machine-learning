"""Train-only LLM-SRBench adapter for the mainline DRR readiness API."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

from ._mainline import load


_drr = load("drr_readiness")
_prefix = load("drr_prefix")
_bank = load("bank_selection")
_adapter = load("pcpi_adapter")
_stacking = load("source_stacking")
_qd = load("operational_qd")


@dataclass(frozen=True)
class DRRSelectionRoles:
    X_development: np.ndarray
    y_development: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_initial: np.ndarray
    y_initial: np.ndarray
    X_actions: np.ndarray
    role_row_indices: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class ThreeArmSelectionRoles:
    X_development: np.ndarray
    y_development: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_gap_audit: np.ndarray
    y_gap_audit: np.ndarray
    X_gap_admission: np.ndarray
    X_selector_update: np.ndarray
    X_decision_calibration: np.ndarray
    X_initial: np.ndarray
    X_report: np.ndarray
    X_actions: np.ndarray
    role_row_indices: Mapping[str, tuple[int, ...]]


def _ordered_indices(count: int, task_name: str, seed: int):
    return sorted(range(count), key=lambda index:
        sha256(f"{task_name}:{seed}:{index}".encode()).digest())


def split_training_samples(samples, *, task_name: str, seed: int):
    """Partition benchmark training rows; action responses are never returned."""
    values = np.asarray(samples, dtype=float)
    if (values.ndim != 2 or values.shape[1] < 2 or len(values) < 40
            or not np.all(np.isfinite(values))):
        raise ValueError("DRR adapter requires at least 40 finite training rows")
    order = _ordered_indices(len(values), task_name, seed)
    cuts = (
        int(round(.50 * len(values))),
        int(round(.70 * len(values))),
        int(round(.90 * len(values))),
    )
    if not 8 <= cuts[0] < cuts[1] < cuts[2] <= len(values) - 4:
        raise ValueError("DRR train-only role split is infeasible")
    row_indices = {
        "discovery_development": tuple(order[:cuts[0]]),
        "discovery_validation": tuple(order[cuts[0]:cuts[1]]),
        "inference_initial": tuple(order[cuts[1]:cuts[2]]),
        "action_covariates": tuple(order[cuts[2]:]),
    }
    def opened(role):
        rows = values[list(row_indices[role])]
        return rows[:, 1:], rows[:, 0]
    X_development, y_development = opened("discovery_development")
    X_validation, y_validation = opened("discovery_validation")
    X_initial, y_initial = opened("inference_initial")
    action_rows = values[list(row_indices["action_covariates"])]
    return DRRSelectionRoles(
        X_development, y_development, X_validation, y_validation,
        X_initial, y_initial, action_rows[:, 1:], row_indices)


def three_arm_role_indices(count: int, *, task_name: str, seed: int):
    """Reconstruct the frozen nine-role split from row count alone."""
    if type(count) is not int or count < 160:
        raise ValueError("three-arm adapter requires at least 160 training rows")
    order = _ordered_indices(count, task_name, seed)
    cuts = tuple(int(round(fraction * count)) for fraction in (
        .40, .55, .65, .75, .80, .85, .90, .95))
    if (cuts != tuple(sorted(cuts)) or len(set(cuts)) != len(cuts)
            or cuts[0] < 32 or count - cuts[-1] < 8):
        raise ValueError("three-arm role split is infeasible")
    names = (
        "discovery_development", "discovery_validation", "gap_audit",
        "gap_admission", "decision_selector_update",
        "decision_calibration", "inference_initial", "reporting",
        "action_covariates",
    )
    bounds = (0, *cuts, count)
    row_indices = {
        name: tuple(order[bounds[index]:bounds[index + 1]])
        for index, name in enumerate(names)
    }
    if len(set().union(*(set(rows) for rows in row_indices.values()))
            ) != count:
        raise ValueError("three-arm roles overlap or omit rows")
    return row_indices


def split_three_arm_training_samples(samples, *, task_name: str, seed: int):
    """Freeze disjoint discovery, gap, decision, report, and action roles."""
    values = np.asarray(samples, dtype=float)
    if (values.ndim != 2 or values.shape[1] < 2 or len(values) < 160
            or not np.all(np.isfinite(values))):
        raise ValueError(
            "three-arm adapter requires at least 160 finite training rows")
    row_indices = three_arm_role_indices(
        len(values), task_name=task_name, seed=seed)

    def opened(name, *, response):
        indices = row_indices[name]
        ordered = sorted(indices)
        rows = values[ordered, :]
        lookup = dict(zip(ordered, rows, strict=True))
        restored = np.asarray([lookup[index] for index in indices])
        return ((restored[:, 1:], restored[:, 0])
                if response else restored[:, 1:])

    development = opened("discovery_development", response=True)
    validation = opened("discovery_validation", response=True)
    gap = opened("gap_audit", response=True)
    admission = opened("gap_admission", response=False)
    selector = opened("decision_selector_update", response=False)
    calibration = opened("decision_calibration", response=False)
    initial = opened("inference_initial", response=False)
    report = opened("reporting", response=False)
    actions = opened("action_covariates", response=False)
    return ThreeArmSelectionRoles(
        *development, *validation, *gap, admission, selector,
        calibration, initial, report, actions, row_indices)


def iterative_gap_role_indices(role_row_indices, cycles: int):
    """Partition only the registered train-only gap/admission roles.

    This is a new protocol; it never changes the historical nine-role split.
    No responses or reporting rows are accessed while constructing indices.
    """
    if type(cycles) is not int or cycles < 2:
        raise ValueError("iterative role partition needs at least two cycles")
    out = []
    for name in ("gap_audit", "gap_admission"):
        rows = tuple(role_row_indices[name])
        if len(rows) < 4 * cycles or len(set(rows)) != len(rows):
            raise ValueError("insufficient distinct rows for iterative gap roles")
        pieces = tuple(tuple(rows[i * len(rows) // cycles:
                                  (i + 1) * len(rows) // cycles])
                       for i in range(cycles))
        if min(map(len, pieces)) < 4:
            raise ValueError("iterative gap role too small")
        out.append(pieces)
    all_rows = [row for group in out for piece in group for row in piece]
    if len(all_rows) != len(set(all_rows)):
        raise ValueError("iterative gap and admission indices overlap")
    return tuple(zip(*out, strict=True))


def _candidate(row):
    candidate = {
        "expression": str(row["expression"]),
        "source": str(row.get("source") or "discovery_candidate"),
        "origin": str(row.get("origin") or "unknown"),
        "lineage_id": str(row.get("lineage_id") or ""),
    }
    if isinstance(row.get("metrics"), Mapping):
        candidate["metrics"] = dict(row["metrics"])
    return candidate


def _distinct_supports(rows, n_features, limit, parser=None):
    parser = parser or _adapter.structural_terms
    selected, supports = [], set()
    for row in rows:
        try:
            support = tuple(parser(row["expression"], n_features))
        except Exception:
            continue
        if support in supports:
            continue
        supports.add(support)
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


def candidate_rows(
    report: Mapping[str, Any], fallback_expression: str, n_features: int,
    protected_expressions: Sequence[Mapping[str, Any]] = (),
    *, coefficient_policy: str | None = None,
):
    """Build a bounded source-safe DRR pool from the complete discovery bank.

    ``final_topk`` is a predictive leaderboard and may contain only one
    baseline support.  The v5/v6 portfolio contract instead requires a
    protected core backbone.  We therefore recover distinct core supports
    from the already-evaluated response-free bank, then add a bounded,
    family-balanced set of leaderboard proposals.  Identical supports are
    owned by core first, so an optional source cannot claim contribution for
    a structure already supplied by the registered baseline.
    """
    if type(n_features) is not int or n_features < 1:
        raise ValueError("candidate extraction requires a positive feature count")
    parser = (_adapter.support_parser_for_policy(coefficient_policy)
              if coefficient_policy is not None else _adapter.structural_terms)
    protected = {}
    for row in protected_expressions:
        try:
            support = tuple(parser(
                str(row["expression"]), n_features))
        except Exception:
            continue
        protected.setdefault(support, str(
            row.get("engine") or "unknown"))
    evaluated = [
        _candidate(row)
        for row in report.get("evaluated_hypothesis_bank", ())
        if isinstance(row, Mapping) and row.get("expression")
    ]
    top = [
        _candidate(row)
        for row in report.get("final_topk", ())
        if isinstance(row, Mapping) and row.get("expression")
    ]
    if not evaluated:
        evaluated = list(top)
    for row in (*evaluated, *top):
        try:
            support = tuple(parser(
                row["expression"], n_features))
        except Exception:
            continue
        if support in protected:
            row["source"] = (
                "protected_counterfactual_backbone:engine:"
                + protected[support])
            row["origin"] = "deterministic"
    core = [
        row for row in evaluated
        if _stacking.source_family(row) == "core"
    ]
    core.sort(key=lambda row: (
        not str(row["source"]).startswith("engine:"),
        str(row["source"]) != "deterministic_linear_anchor",
        str(row["source"]) != "deterministic_constant_anchor",
    ))
    protected = _distinct_supports(core, n_features, 4, parser)

    optional_by_family = {}
    for row in (*top, *evaluated):
        family = _stacking.source_family(row)
        if family == "core":
            continue
        optional_by_family.setdefault(family, []).append(row)
    optional = []
    for family in sorted(optional_by_family):
        optional.extend(_distinct_supports(
            optional_by_family[family], n_features, 1, parser))
    optional = _distinct_supports(optional, n_features, 4, parser)

    rows = [*protected, *optional]
    rows = _distinct_supports(rows, n_features, 8, parser)
    if not rows:
        rows = [{
            "expression": str(fallback_expression),
            "source": str(report.get("selected_source") or
                          "discovery_final"),
            "origin": str(report.get("selected_origin") or "unknown"),
            "lineage_id": str(report.get("final_lineage_id") or ""),
        }]
    return rows


def candidate_rows_operational_qd(
    report: Mapping[str, Any], fallback_expression: str,
    roles: DRRSelectionRoles, *, task_name: str, seed: int,
    protected_expressions: Sequence[Mapping[str, Any]] = (),
):
    """Replace optional top-k truncation with an operational QD repertoire."""
    from hypothesis_mvp.data.roles import DataRole, RoleDataset
    from hypothesis_mvp.pcpi import NormalInverseGammaPrior

    n_features = int(roles.X_initial.shape[1])
    baseline = candidate_rows(
        report, fallback_expression, n_features, protected_expressions)
    core = tuple(
        row for row in baseline
        if _stacking.source_family(row) == "core")
    evaluated = [
        _candidate(row)
        for row in report.get("evaluated_hypothesis_bank", ())
        if isinstance(row, Mapping) and row.get("expression")]
    if not evaluated:
        evaluated = list(baseline)
    evaluated = _distinct_supports(evaluated, n_features, len(evaluated))
    initial = RoleDataset(
        DataRole.DEVELOPMENT,
        roles.X_initial[:8], roles.y_initial[:8])
    identity = sha256(
        f"llm-srbench-boqd-v1:{task_name}:{seed}:"
        f"{roles.role_row_indices}".encode()).hexdigest()
    try:
        optional, audit = _qd.build_operational_qd_repertoire(
            evaluated, core, initial, roles.X_actions,
            n_features=n_features, prior=NormalInverseGammaPrior(),
            exploration_identity=identity, measurement_budget=2,
            maximum_elites=max(1, 8 - len(core)))
        challenger = _distinct_supports(
            [*core, *optional], n_features, 8)
        if len(challenger) < 3:
            raise ValueError("operational QD produced an infeasible DRR pool")
        legacy_curve = evaluate_drr_prefix_candidates(
            baseline, roles, condition="legacy-repertoire",
            task_name=task_name, seed=seed)
        challenger_curve = evaluate_drr_prefix_candidates(
            challenger, roles, condition="boqd-repertoire",
            task_name=task_name, seed=seed)
        legacy_scores = np.asarray(
            legacy_curve.normalized_lower_bounds)
        challenger_scores = np.asarray(
            challenger_curve.normalized_lower_bounds)
        tolerance = float(256.0 * np.finfo(float).eps * max(
            1.0, abs(legacy_curve.normalized_aulc),
            abs(challenger_curve.normalized_aulc)))
        prefix_noninferior = bool(np.all(
            challenger_scores + tolerance >= legacy_scores))
        strict_gain = bool(
            challenger_curve.normalized_aulc
            > legacy_curve.normalized_aulc + tolerance)
        handover = bool(prefix_noninferior and strict_gain)
        selected = challenger if handover else baseline
        audit = {**audit, "conservative_handover": {
            "schema": "scientific-operational-qd-handover-v1",
            "legacy_normalized_lower_bounds":
                legacy_scores.tolist(),
            "challenger_normalized_lower_bounds":
                challenger_scores.tolist(),
            "legacy_normalized_aulc":
                legacy_curve.normalized_aulc,
            "challenger_normalized_aulc":
                challenger_curve.normalized_aulc,
            "numerical_tolerance": tolerance,
            "prefix_noninferior": prefix_noninferior,
            "strict_aulc_gain": strict_gain,
            "targeted_handover": handover,
            "selected_repertoire": (
                "boqd" if handover else "legacy"),
            "candidate_response_accessed": False,
            "heldout_opened": False,
        }}
    except Exception as error:
        selected, audit = baseline, {
            "schema": "scientific-operational-qd-archive-v1",
            "method": _qd.METHOD,
            "conservative_handover": {
                "schema": "scientific-operational-qd-handover-v1",
                "targeted_handover": False,
                "selected_repertoire": "legacy",
                "fallback_reason":
                    f"{type(error).__name__}:{error}",
                "candidate_response_accessed": False,
                "heldout_opened": False,
            },
            "candidate_response_accessed": False,
            "heldout_opened": False,
        }
    return selected, audit


def evaluate_drr_candidates(
    candidates: Sequence[Mapping[str, Any]],
    roles: DRRSelectionRoles, *, condition: str, task_name: str, seed: int,
    selection_method: str = _bank.DECISION_RISK_CAPACITY_METHOD,
):
    identity = sha256(
        f"llm-srbench-drr-v1:{task_name}:{seed}:{condition}:"
        f"{roles.role_row_indices}".encode()).hexdigest()
    return _drr.evaluate_drr_readiness(
        candidates, roles.X_initial, roles.y_initial, roles.X_actions,
        condition=condition, exploration_identity=identity,
        selection_method=selection_method)


def evaluate_drr_prefix_candidates(
    candidates: Sequence[Mapping[str, Any]],
    roles: DRRSelectionRoles, *, condition: str, task_name: str, seed: int,
    selection_method: str = _bank.DECISION_RISK_CAPACITY_METHOD,
):
    identity = sha256(
        f"llm-srbench-drr-prefix-v1:{task_name}:{seed}:{condition}:"
        f"{roles.role_row_indices}".encode()).hexdigest()
    return _prefix.evaluate_drr_prefix_curve(
        candidates, roles.X_initial, roles.y_initial, roles.X_actions,
        condition=condition, exploration_identity=identity,
        selection_method=selection_method)


__all__ = [
    "DRRSelectionRoles", "candidate_rows", "evaluate_drr_candidates",
    "candidate_rows_operational_qd", "evaluate_drr_prefix_candidates",
    "split_training_samples",
]
