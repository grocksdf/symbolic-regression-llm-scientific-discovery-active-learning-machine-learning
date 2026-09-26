"""Initial-data-only operational-capacity selection for a finite hypothesis bank."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
import json

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset

from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target, structural_terms
from .source_stacking import (
    DIVERSITY_METHOD,
    METHOD as SOURCE_STACKING_METHOD,
    crossfit_source_log_predictive, safe_source_stacking, source_family,
)


SCHEMA = "scientific-predictive-safe-operational-capacity-bank-v2"
METHOD = "two-fold-safe-half-core-source-stacking-operational-entropy-v3"
DIVERSITY_CAPACITY_METHOD = (
    "two-fold-safe-diversity-preserving-half-core-operational-entropy-v4")
PORTFOLIO_CAPACITY_METHOD = (
    "fold-safe-protected-core-variable-cardinality-operational-entropy-v5")


def _identity(candidate):
    return sha256(json.dumps(candidate, sort_keys=True, default=str).encode()).hexdigest()


def _required_roles(candidates):
    roles = {str(row["source"]) for row in candidates
             if str(row["source"]).startswith("engine:")}
    if any(str(row.get("origin", "")) == "llm" for row in candidates):
        roles.add("origin:llm")
    return tuple(sorted(roles))


def _roles(candidate):
    output = set()
    source = str(candidate["source"])
    if source.startswith("engine:"):
        output.add(source)
    if str(candidate.get("origin", "")) == "llm":
        output.add("origin:llm")
    return output


def _without_role(rows, role):
    if role == "origin:llm":
        return tuple(row for row in rows if str(row.get("origin", "")) != "llm")
    return tuple(row for row in rows if str(row.get("source", "")) != role)


def _candidate_portfolios(pool, target_size, maximum_candidates,
                          selection_method, available_roles, safety_roles):
    if selection_method == PORTFOLIO_CAPACITY_METHOD:
        sizes = range(min(3, target_size), target_size + 1)
        return [
            tuple(rows)
            for size in sizes
            for rows in combinations(pool, size)
            if sum(source_family(row) == "core" for row in rows)
                >= max(2, (size + 1) // 2)
            and set(safety_roles).issubset(
                set().union(*(_roles(row) for row in rows)))
        ], safety_roles
    if maximum_candidates < len(available_roles):
        raise ValueError("bank capacity cannot preserve required provenance roles")
    return [
        tuple(rows) for rows in combinations(pool, target_size)
        if set(available_roles).issubset(
            set().union(*(_roles(row) for row in rows)))
    ], available_roles


def _excluded_families(pool, selected):
    available = {source_family(row) for row in pool}
    retained = {source_family(row) for row in selected}
    return sorted(family for family in available - retained if family != "core")


@dataclass
class _CapacityEvaluator:
    initial_data: RoleDataset
    action_domain: np.ndarray
    n_features: int
    prior: object
    exploration_identity: str
    coefficient_policy: str
    measurement_budget: int
    safety_roles: tuple[str, ...]
    source_stacking_method: str = SOURCE_STACKING_METHOD
    target_cache: dict = field(default_factory=dict)
    profile_cache: dict = field(default_factory=dict)

    def target(self, rows, initial, source_weights=None):
        weight_key = None if source_weights is None else tuple(sorted(source_weights.items()))
        key = (tuple(sorted(_identity(row) for row in rows)), initial.fingerprint, weight_key)
        if key not in self.target_cache:
            model = freeze_discovery_model(rows, n_features=self.n_features,
                prior=self.prior, exploration_identity=self.exploration_identity,
                coefficient_policy=self.coefficient_policy,
                source_prior_weights=source_weights)
            frozen = freeze_discovery_target(model, initial, self.action_domain,
                measurement_budget=self.measurement_budget,
                expected_model_identity=model.stable_hash)
            self.target_cache[key] = (model, frozen)
        return self.target_cache[key]

    def capacity(self, rows, source_weights):
        model, frozen = self.target(rows, self.initial_data, source_weights)
        return (float(frozen.partition.entropy), len(frozen.partition.class_ids),
                model.stable_hash, frozen.stable_hash)

    def crossfit_profile(self, rows):
        key = tuple(sorted(_identity(row) for row in rows))
        if key not in self.profile_cache:
            values = []
            indices = np.arange(len(self.initial_data.X))
            for fold in range(2):
                held = (indices % 2) == fold
                training = RoleDataset(DataRole.DEVELOPMENT,
                    self.initial_data.X[~held], self.initial_data.y[~held])
                model, frozen = self.target(rows, training)
                logpdf = model.engine(frozen.model_identity).predictive_logpdf(
                    frozen.initial_posterior,
                    self.initial_data.X[held], self.initial_data.y[held])
                values.append(np.asarray(logpdf, dtype=float))
            self.profile_cache[key] = np.concatenate(values)
        return self.profile_cache[key]

    def safety(self, rows):
        profile, folds, sources = crossfit_source_log_predictive(
            rows, self.initial_data, self.action_domain,
            n_features=self.n_features, prior=self.prior,
            exploration_identity=self.exploration_identity,
            coefficient_policy=self.coefficient_policy,
            measurement_budget=self.measurement_budget)
        certificate = safe_source_stacking(
            profile, folds, sources, baseline_source="core",
            method=self.source_stacking_method)
        audits = {}
        for role in self.safety_roles:
            family = "llm" if role == "origin:llm" else role
            weight = certificate.source_weights.get(family, 0.0)
            audits[role] = {"source_family": family, "stacking_weight": weight,
                            "passed": bool(weight > 2e-12)}
        return audits, certificate


def select_operational_capacity_bank(candidates, initial_data, action_domain, *,
                                     n_features, prior, exploration_identity,
                                     coefficient_policy, measurement_budget,
                                     maximum_candidates, source_safety_roles=(),
                                     source_safety_folds=2,
                                     source_stacking_method=SOURCE_STACKING_METHOD,
                                     selection_method=None):
    """Select the highest-entropy bank whose registered sources are predictive-safe.

    Safety is a paired two-fold posterior-predictive log score computed only on
    the registered initial data. Acquisition-pool responses, the independent
    source-arbitration split, reporting responses and held-out objects are not
    accepted by this interface.
    """
    if (type(maximum_candidates) is not int or maximum_candidates < 3
            or maximum_candidates != 2 * measurement_budget):
        raise ValueError("bank capacity must equal twice the measurement budget")
    if source_safety_folds != 2:
        raise ValueError("operational bank source safety requires two folds")
    if selection_method is None:
        selection_method = (
            DIVERSITY_CAPACITY_METHOD
            if source_stacking_method == DIVERSITY_METHOD else METHOD)
    if selection_method not in {
            METHOD, DIVERSITY_CAPACITY_METHOD, PORTFOLIO_CAPACITY_METHOD}:
        raise ValueError("unknown operational-capacity selection method")
    if ((source_stacking_method == SOURCE_STACKING_METHOD)
            != (selection_method == METHOD)):
        raise ValueError("source stacking and capacity selection differ")
    if initial_data.role is not DataRole.DEVELOPMENT or len(initial_data.X) < 4:
        raise ValueError("source safety requires registered initial development data")
    unique = {}
    for row in candidates:
        support = tuple(structural_terms(str(row["expression"]), n_features))
        unique.setdefault(support, dict(row))
    pool = tuple(sorted(unique.values(), key=_identity))
    available_roles = _required_roles(pool)
    safety_roles = tuple(sorted(set(source_safety_roles)))
    if (len(pool) < 2 or any(role not in available_roles
                             for role in safety_roles)):
        raise ValueError("bank capacity cannot preserve required provenance roles")

    target_size = min(len(pool), maximum_candidates)
    # In v5 capacity is an upper bound: a scientific-core backbone is
    # protected while optional proposal sources compete for remaining slots.
    candidate_sets, required = _candidate_portfolios(
        pool, target_size, maximum_candidates, selection_method,
        available_roles, safety_roles)
    if not candidate_sets:
        raise ValueError("no capacity bank preserves registered provenance")

    evaluator = _CapacityEvaluator(initial_data, action_domain, n_features, prior,
        exploration_identity, coefficient_policy, measurement_budget,
        safety_roles, source_stacking_method)

    evaluated = []
    for rows in candidate_sets:
        audits, certificate = evaluator.safety(rows)
        passed = all(audit["passed"] for audit in audits.values())
        # A rejected set may collapse to one support family.  Its capacity is
        # diagnostic under the unchanged finite-bank prior; only admitted sets
        # are scored under the hierarchical source prior used in production.
        cap = evaluator.capacity(rows, certificate.source_weights if passed else None)
        minimum_margin = min((audit["stacking_weight"]
                              for audit in audits.values()), default=float("inf"))
        identity = tuple(sorted(_identity(row) for row in rows))
        evaluated.append((passed, cap, minimum_margin, identity, rows, audits, certificate))
    feasible = [item for item in evaluated if item[0]]
    ranked = feasible if feasible else evaluated
    chosen = min(ranked, key=lambda item: (
        -item[1][0] if feasible else -item[2], -item[1][1], item[3]))
    passed, final_score, _, _, selected, audits, certificate = chosen
    capacity_excluded = _excluded_families(pool, selected)
    return tuple(selected), {
        "schema": SCHEMA,
        "selection_method": selection_method,
        "input_support_count": len(pool), "selected_support_count": len(selected),
        "evaluated_capacity_bank_count": len(candidate_sets),
        "maximum_candidates": maximum_candidates,
        "capacity_is_upper_bound": selection_method == PORTFOLIO_CAPACITY_METHOD,
        "protected_core_support_count": sum(
            source_family(row) == "core" for row in selected),
        "available_roles": list(available_roles),
        "required_roles": list(required),
        "capacity_excluded_source_families": capacity_excluded,
        "source_safety_roles": list(safety_roles), "source_safety_folds": 2,
        "source_safety": audits, "source_safety_passed": passed,
        "source_stacking": certificate.to_dict(),
        "source_stacking_identity": certificate.stable_hash,
        "source_prior_weights": certificate.source_weights,
        "class_entropy_nats": final_score[0], "operational_class_count": final_score[1],
        "model": final_score[2], "target": final_score[3],
        "initial_development_response_accessed": True,
        "candidate_response_accessed": False,
        "source_arbitration_validation_response_accessed": False,
        "reporting_validation_response_accessed": False,
        "heldout_opened": False,
    }


__all__ = [
    "DIVERSITY_CAPACITY_METHOD", "PORTFOLIO_CAPACITY_METHOD",
    "select_operational_capacity_bank"]
