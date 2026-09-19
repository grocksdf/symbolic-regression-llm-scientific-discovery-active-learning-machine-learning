"""Initial-data-only operational-capacity selection for a finite hypothesis bank."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
import json

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset

from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target, structural_terms


SCHEMA = "scientific-predictive-safe-operational-capacity-bank-v2"
METHOD = "two-fold-initial-predictive-safe-operational-entropy-v1"


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
    target_cache: dict = field(default_factory=dict)
    profile_cache: dict = field(default_factory=dict)

    def target(self, rows, initial):
        key = (tuple(sorted(_identity(row) for row in rows)), initial.fingerprint)
        if key not in self.target_cache:
            model = freeze_discovery_model(rows, n_features=self.n_features,
                prior=self.prior, exploration_identity=self.exploration_identity,
                coefficient_policy=self.coefficient_policy)
            frozen = freeze_discovery_target(model, initial, self.action_domain,
                measurement_budget=self.measurement_budget,
                expected_model_identity=model.stable_hash)
            self.target_cache[key] = (model, frozen)
        return self.target_cache[key]

    def capacity(self, rows):
        model, frozen = self.target(rows, self.initial_data)
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
        full = self.crossfit_profile(rows)
        audits = {}
        for role in self.safety_roles:
            ablated_rows = _without_role(rows, role)
            if len(ablated_rows) < 2 or len(ablated_rows) == len(rows):
                raise ValueError("source-safety ablation is empty or absent")
            ablated = self.crossfit_profile(ablated_rows)
            delta = float(np.sum(full - ablated))
            scale = max(1.0, float(np.sum(np.abs(full))),
                        float(np.sum(np.abs(ablated))))
            tolerance = float(1024.0 * np.finfo(float).eps * scale)
            audits[role] = {"paired_cumulative_log_predictive_ratio_nats": delta,
                            "numerical_tolerance_nats": tolerance,
                            "passed": bool(delta > tolerance)}
        return audits


def select_operational_capacity_bank(candidates, initial_data, action_domain, *,
                                     n_features, prior, exploration_identity,
                                     coefficient_policy, measurement_budget,
                                     maximum_candidates, source_safety_roles=(),
                                     source_safety_folds=2):
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
    if initial_data.role is not DataRole.DEVELOPMENT or len(initial_data.X) < 4:
        raise ValueError("source safety requires registered initial development data")
    unique = {}
    for row in candidates:
        support = tuple(structural_terms(str(row["expression"]), n_features))
        unique.setdefault(support, dict(row))
    pool = tuple(sorted(unique.values(), key=_identity))
    required = _required_roles(pool)
    safety_roles = tuple(sorted(set(source_safety_roles)))
    if (len(pool) < 2 or maximum_candidates < len(required)
            or any(role not in required for role in safety_roles)):
        raise ValueError("bank capacity cannot preserve required provenance roles")

    target_size = min(len(pool), maximum_candidates)
    candidate_sets = [tuple(rows) for rows in combinations(pool, target_size)
                      if set(required).issubset(set().union(*(_roles(row) for row in rows)))]
    if not candidate_sets:
        raise ValueError("no capacity bank preserves registered provenance")

    evaluator = _CapacityEvaluator(initial_data, action_domain, n_features, prior,
        exploration_identity, coefficient_policy, measurement_budget, safety_roles)

    evaluated = []
    for rows in candidate_sets:
        audits = evaluator.safety(rows)
        passed = all(audit["passed"] for audit in audits.values())
        cap = evaluator.capacity(rows)
        minimum_margin = min((audit["paired_cumulative_log_predictive_ratio_nats"]
                              - audit["numerical_tolerance_nats"]
                              for audit in audits.values()), default=float("inf"))
        identity = tuple(sorted(_identity(row) for row in rows))
        evaluated.append((passed, cap, minimum_margin, identity, rows, audits))
    feasible = [item for item in evaluated if item[0]]
    ranked = feasible if feasible else evaluated
    chosen = min(ranked, key=lambda item: (
        -item[1][0] if feasible else -item[2], -item[1][1], item[3]))
    passed, final_score, _, _, selected, audits = chosen
    return tuple(selected), {
        "schema": SCHEMA,
        "selection_method": METHOD,
        "input_support_count": len(pool), "selected_support_count": len(selected),
        "evaluated_capacity_bank_count": len(candidate_sets),
        "maximum_candidates": maximum_candidates, "required_roles": list(required),
        "source_safety_roles": list(safety_roles), "source_safety_folds": 2,
        "source_safety": audits, "source_safety_passed": passed,
        "class_entropy_nats": final_score[0], "operational_class_count": final_score[1],
        "model": final_score[2], "target": final_score[3],
        "initial_development_response_accessed": True,
        "candidate_response_accessed": False,
        "source_arbitration_validation_response_accessed": False,
        "reporting_validation_response_accessed": False,
        "heldout_opened": False,
    }


__all__ = ["select_operational_capacity_bank"]
