"""Response-free operational quality-diversity archive for scientific laws."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution


METHOD = "bayesian-operational-quality-diversity-v1"
LEADER_BIN_COUNT = 8
CONCENTRATION_BIN_COUNT = 4
SUPPORT_BIN_COUNT = 4


@dataclass(frozen=True, order=True)
class OperationalQDDescriptor:
    leader_action_bin: int
    concentration_bin: int
    support_size_bin: int

    def __post_init__(self):
        if (not 0 <= self.leader_action_bin < LEADER_BIN_COUNT
                or not 0 <= self.concentration_bin < CONCENTRATION_BIN_COUNT
                or not 0 <= self.support_size_bin < SUPPORT_BIN_COUNT):
            raise ValueError("invalid operational QD descriptor")

    def to_dict(self):
        return {
            "leader_action_bin": self.leader_action_bin,
            "concentration_bin": self.concentration_bin,
            "support_size_bin": self.support_size_bin,
        }


def operational_descriptor(
        certified_risk_lower_bounds: Sequence[float],
        nonintercept_support_count: int,
) -> OperationalQDDescriptor:
    values = np.asarray(
        certified_risk_lower_bounds, dtype=float).reshape(-1)
    if (not len(values) or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or type(nonintercept_support_count) is not int
            or nonintercept_support_count < 0):
        raise ValueError("invalid operational descriptor inputs")
    leader = int(np.argmax(values))
    leader_bin = min(
        LEADER_BIN_COUNT - 1,
        leader * LEADER_BIN_COUNT // len(values))
    total = float(np.sum(values))
    concentration = 0.0 if total <= 0.0 else float(np.max(values) / total)
    concentration_bin = min(
        CONCENTRATION_BIN_COUNT - 1,
        int(concentration * CONCENTRATION_BIN_COUNT))
    support_bin = min(
        SUPPORT_BIN_COUNT - 1, nonintercept_support_count)
    return OperationalQDDescriptor(
        leader_bin, concentration_bin, support_bin)


@dataclass(frozen=True)
class OperationalQDElite:
    candidate_identity: str
    expression: str
    source_family: str
    descriptor: OperationalQDDescriptor
    certified_risk_lower_bound: float
    predictive_quality: float
    complexity: float
    verification_passed: bool

    def __post_init__(self):
        values = (
            self.certified_risk_lower_bound,
            self.predictive_quality, self.complexity)
        if (not self.candidate_identity or not self.expression
                or not self.source_family
                or any(not np.isfinite(value) for value in values)
                or self.certified_risk_lower_bound < 0.0
                or self.complexity < 0.0):
            raise ValueError("invalid operational QD elite")

    @property
    def quality_key(self):
        return (
            self.certified_risk_lower_bound,
            self.predictive_quality,
            -self.complexity,
            self.candidate_identity,
        )

    def to_dict(self):
        return {
            "candidate_identity": self.candidate_identity,
            "expression": self.expression,
            "source_family": self.source_family,
            "descriptor": self.descriptor.to_dict(),
            "certified_risk_lower_bound":
                self.certified_risk_lower_bound,
            "predictive_quality": self.predictive_quality,
            "complexity": self.complexity,
            "verification_passed": self.verification_passed,
        }


class OperationalQDArchive:
    """One verified elite per operational behavior cell."""

    def __init__(self):
        self._cells: dict[
            OperationalQDDescriptor, OperationalQDElite] = {}
        self._events: list[dict[str, Any]] = []

    @property
    def elites(self):
        return tuple(
            self._cells[key] for key in sorted(self._cells))

    @property
    def coverage(self):
        return len(self._cells) / (
            LEADER_BIN_COUNT * CONCENTRATION_BIN_COUNT
            * SUPPORT_BIN_COUNT)

    @property
    def qd_score(self):
        return float(sum(
            elite.certified_risk_lower_bound
            for elite in self._cells.values()))

    def add(self, elite: OperationalQDElite):
        previous = self._cells.get(elite.descriptor)
        accepted = bool(
            elite.verification_passed
            and (previous is None
                 or elite.quality_key > previous.quality_key))
        reason = (
            "accepted-empty-cell" if accepted and previous is None
            else "accepted-quality-improvement" if accepted
            else "rejected-verification-failed"
                if not elite.verification_passed
            else "rejected-cell-incumbent-dominates")
        if accepted:
            self._cells[elite.descriptor] = elite
        event = {
            "schema": "scientific-operational-qd-admission-v1",
            "candidate_identity": elite.candidate_identity,
            "descriptor": elite.descriptor.to_dict(),
            "accepted": accepted, "reason": reason,
            "incumbent_identity": (
                None if previous is None
                else previous.candidate_identity),
            "candidate_response_accessed": False,
            "heldout_opened": False,
        }
        self._events.append(event)
        return event

    def empty_cells(self, limit: int = 16):
        if type(limit) is not int or limit < 1:
            raise ValueError("empty-cell limit must be positive")
        occupied = set(self._cells)
        output = []
        for leader in range(LEADER_BIN_COUNT):
            for concentration in range(CONCENTRATION_BIN_COUNT):
                for support in range(SUPPORT_BIN_COUNT):
                    descriptor = OperationalQDDescriptor(
                        leader, concentration, support)
                    if descriptor not in occupied:
                        output.append(descriptor)
                    if len(output) == limit:
                        return tuple(output)
        return tuple(output)

    def certificate(self):
        payload = {
            "schema": "scientific-operational-qd-archive-v1",
            "method": METHOD,
            "cell_count": len(self._cells),
            "total_cell_count": (
                LEADER_BIN_COUNT * CONCENTRATION_BIN_COUNT
                * SUPPORT_BIN_COUNT),
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "elites": [elite.to_dict() for elite in self.elites],
            "events": list(self._events),
            "candidate_response_accessed": False,
            "heldout_opened": False,
        }
        payload["identity"] = sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest()
        return payload


@dataclass(frozen=True)
class EmitterTaskEvidence:
    task_identity: str
    emitter: str
    admitted_new_cell: bool

    def __post_init__(self):
        if not self.task_identity or not self.emitter:
            raise ValueError("invalid operational QD emitter evidence")


def fit_emitter_credit(
        evidence: Sequence[EmitterTaskEvidence], *,
        credible_level: float = 0.9,
) -> dict[str, Mapping[str, float]]:
    if not 0.5 < credible_level < 1.0:
        raise ValueError("invalid emitter credible level")
    output = {}
    for emitter in sorted({row.emitter for row in evidence}):
        rows = [row for row in evidence if row.emitter == emitter]
        success = sum(row.admitted_new_cell for row in rows)
        alpha, beta = success + 0.5, len(rows) - success + 0.5
        output[emitter] = {
            "task_count": len(rows), "successes": success,
            "posterior_alpha": alpha, "posterior_beta": beta,
            "posterior_mean": alpha / (alpha + beta),
            "lower_credible_bound": float(beta_distribution.ppf(
                1.0 - credible_level, alpha, beta)),
        }
    return output


def _candidate_identity(candidate):
    return sha256(json.dumps(
        candidate, sort_keys=True, default=str,
        separators=(",", ":")).encode()).hexdigest()


def _candidate_metrics(candidate, support_count):
    metrics = dict(candidate.get("metrics") or {})
    val_nmse = float(metrics.get("val_nmse", np.finfo(float).max))
    complexity = float(metrics.get("complexity", support_count))
    invalid = int(metrics.get("invalid_predictions", 0))
    verified = bool(
        np.isfinite(val_nmse) and np.isfinite(complexity)
        and complexity >= 0.0 and invalid == 0)
    quality = -val_nmse if np.isfinite(val_nmse) else -np.finfo(float).max
    return quality, max(0.0, complexity), verified


def _candidate_risk_profile(
        rows, initial_data, actions, *, n_features, prior,
        exploration_identity, measurement_budget):
    from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target
    from .system_run import audit_decision_risk_target

    model = freeze_discovery_model(
        rows, n_features=n_features, prior=prior,
        exploration_identity=exploration_identity,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis")
    target = freeze_discovery_target(
        model, initial_data, actions,
        measurement_budget=measurement_budget,
        expected_model_identity=model.stable_hash)
    audit = audit_decision_risk_target(
        model, target, actions, 1e-10)
    return np.asarray(audit["action_lower_bounds"], dtype=float)


def build_operational_qd_repertoire(
        candidates: Sequence[Mapping[str, Any]],
        protected_core: Sequence[Mapping[str, Any]],
        initial_data, actions, *, n_features: int, prior,
        exploration_identity: str, measurement_budget: int = 2,
        maximum_elites: int = 4,
):
    """Build a verified candidate pool keyed by operational risk behavior."""
    from .pcpi_adapter import structural_terms
    from .source_stacking import source_family

    core = tuple(protected_core)
    if (len(core) < 2 or maximum_elites < 1
            or len(exploration_identity) != 64):
        raise ValueError("operational QD requires two protected core supports")
    archive, rows_by_identity = OperationalQDArchive(), {}
    action_values = np.asarray(actions, dtype=float)
    indices = np.unique(np.linspace(
        0, len(action_values) - 1,
        min(32, len(action_values)), dtype=int))
    descriptor_actions = action_values[indices]
    core_supports = {
        tuple(structural_terms(str(row["expression"]), n_features))
        for row in core}
    for candidate in candidates:
        support = tuple(structural_terms(
            str(candidate["expression"]), n_features))
        if support in core_supports:
            continue
        identity = _candidate_identity(candidate)
        qd_identity = sha256(
            f"{exploration_identity}:{identity}".encode()).hexdigest()
        try:
            lower = _candidate_risk_profile(
                (*core[:2], candidate), initial_data, descriptor_actions,
                n_features=n_features, prior=prior,
                exploration_identity=qd_identity,
                measurement_budget=measurement_budget)
            quality, complexity, verified = _candidate_metrics(
                candidate, max(0, len(support) - 1))
        except Exception:
            lower = np.zeros(len(descriptor_actions), dtype=float)
            quality, complexity, verified = (
                -np.finfo(float).max,
                float(max(0, len(support) - 1)), False)
        elite = OperationalQDElite(
            identity, str(candidate["expression"]),
            source_family(candidate),
            operational_descriptor(lower, max(0, len(support) - 1)),
            float(np.max(lower)), quality, complexity, verified)
        archive.add(elite)
        rows_by_identity[identity] = dict(candidate)
    ranked = sorted(
        archive.elites, key=lambda row: row.quality_key, reverse=True)
    selected = tuple(
        rows_by_identity[row.candidate_identity]
        for row in ranked[:maximum_elites])
    certificate = {
        **archive.certificate(),
        "descriptor_action_count": len(descriptor_actions),
        "protected_core_count": len(core),
        "selected_elite_count": len(selected),
        "maximum_elites": maximum_elites,
    }
    return selected, certificate


__all__ = [
    "EmitterTaskEvidence", "METHOD", "OperationalQDArchive",
    "OperationalQDDescriptor", "OperationalQDElite",
    "build_operational_qd_repertoire", "fit_emitter_credit",
    "operational_descriptor",
]
