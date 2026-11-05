"""Shared Bayesian working regression for joint site/operator contrasts.

No task-ID feature, dataset-specific rule, success threshold or arm-shared
mutable memory. AST descriptors are a small part of numeric context. Each
valid pair is ONE observation, even when its scorer returns several folds.
Adaptive crossfit reuse does not make this a calibrated scientific posterior.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy

import numpy as np

from .counterfactual_bank_symbolic_invention import INVENTION_OPERATORS
from .paired_edit_policy import _validated_state
from .region_site_attribution import FEATURE_NAMES, digest


SCHEMA = "regional-joint-site-operator-policy-v1"
# Explicit interactions survive contrasts and let site preferences vary with
# the defect. This is a fixed feature map, not learned representation discovery.
INTERACTIONS = tuple((z, h) for z in ("defect_direction", "defect_strength", "score_direction")
                     for h in ("output_mean", "output_spread", "region_axis_dependency"))


@dataclass(frozen=True)
class SiteOperatorAction:
    parent: str
    donor: str
    region_id: str
    site_path: tuple[int, ...]
    operator: str
    context_identity: str
    features: tuple[float, ...]

    def __post_init__(self):
        if (not all(isinstance(v, str) and v for v in (self.parent, self.donor, self.region_id, self.context_identity))
                or type(self.site_path) is not tuple or any(type(i) is not int or i < 0 for i in self.site_path)
                or type(self.features) is not tuple or len(self.features) != len(FEATURE_NAMES)
                or not np.all(np.isfinite(self.features)) or self.operator not in INVENTION_OPERATORS):
            raise ValueError("invalid immutable site/operator action")

    @property
    def identity(self):
        return digest(asdict(self))


class RegionalSiteOperatorPolicy:
    def __init__(self, *, operators, seed, scoring_identity, task_identity, frozen_prior=None,
                 intervention_identity="standalone", utility_identity="paired-bank-logscore-v1"):
        self.operators = tuple(sorted(operators))
        if (not self.operators or len(set(self.operators)) != len(self.operators)
                or any(o not in INVENTION_OPERATORS for o in self.operators)
                or type(seed) is not int or seed < 0
                or not isinstance(scoring_identity, str) or not scoring_identity
                or not isinstance(task_identity, str) or not task_identity
                or not isinstance(intervention_identity, str) or not intervention_identity
                or not isinstance(utility_identity, str) or not utility_identity):
            raise ValueError("explicit operator, task and scoring identities required")
        self.seed, self.scoring_identity, self.task_identity = seed, scoring_identity, task_identity
        self.intervention_identity, self.utility_identity = intervention_identity, utility_identity
        self.feature_schema = digest([SCHEMA, FEATURE_NAMES, INTERACTIONS, self.operators,
                                      intervention_identity, utility_identity])
        self.dimension = (1 + len(FEATURE_NAMES) + len(INTERACTIONS)) * (1 + len(self.operators))
        self._mean, self._precision = np.zeros(self.dimension), np.eye(self.dimension)
        self._alpha, self._beta = 1., 1.
        self._training_tasks, self._inherited_evidence = set(), []
        self._prior_identity = None
        if frozen_prior is not None:
            prior = copy.deepcopy(frozen_prior)
            identity = prior.pop("identity", None)
            if (digest(prior) != identity or prior.get("schema") != SCHEMA
                    or prior.get("feature_schema") != self.feature_schema
                    or task_identity in prior.get("training_tasks", [])):
                raise ValueError("frozen prior identity/schema or task isolation mismatch")
            if not prior.get("training_tasks") or not prior.get("evidence"):
                raise ValueError("transfer requires explicit training provenance")
            self._mean = np.asarray(prior["posterior"]["mean"], dtype=float)
            self._precision = np.asarray(prior["posterior"]["precision"], dtype=float)
            self._alpha, self._beta = prior["posterior"]["alpha"], prior["posterior"]["beta"]
            self._training_tasks = set(prior["training_tasks"])
            self._inherited_evidence = list(prior["evidence"])
            self._prior_identity = identity
        if self._mean.shape != (self.dimension,) or self._precision.shape != (self.dimension, self.dimension):
            raise ValueError("joint prior dimensions mismatch")
        self._cholesky, self._covariance = _validated_state(self._mean, self._precision, self._alpha, self._beta)
        self._decisions, self._observations, self._seen = {}, [], set()

    def feature(self, action):
        if action.operator not in self.operators:
            raise ValueError("operator outside frozen policy")
        z = dict(zip(FEATURE_NAMES, action.features))
        base = np.asarray((1., *action.features, *(z[a] * z[b] for a, b in INTERACTIONS)))
        vector = np.zeros(self.dimension)
        vector[:len(base)] = base
        start = (1 + self.operators.index(action.operator)) * len(base)
        vector[start:start + len(base)] = base
        return vector

    def choose_pair(self, step, actions, *, contrast, mode="thompson"):
        if (type(step) is not int or step < 0 or step in self._decisions
                or contrast not in {"operator", "site"} or mode not in {"thompson", "random"}):
            raise ValueError("invalid or duplicate decision")
        actions = tuple(sorted(actions, key=lambda a: a.identity))
        if not actions or len({a.identity for a in actions}) != len(actions):
            raise ValueError("unique feasible actions required")
        anchor = actions[0]
        if any((a.parent, a.donor, a.region_id, a.context_identity) !=
               (anchor.parent, anchor.donor, anchor.region_id, anchor.context_identity) for a in actions):
            raise ValueError("a pair must share parent, donor, region and frozen context")
        features = {a.identity: self.feature(a) for a in actions}
        def alternatives(a):
            return [b for b in actions if ((a.site_path == b.site_path and a.operator != b.operator)
                    if contrast == "operator" else (a.operator == b.operator and a.site_path != b.site_path))]
        eligible = [a for a in actions if alternatives(a)]
        if not eligible:
            # Caller records unavailable contrast; no different control is substituted.
            return None
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        if mode == "random":
            scores = {a.identity: float(rng.random()) for a in actions}
            sigma2 = None
        else:
            sigma2 = float(1. / rng.gamma(self._alpha, 1. / self._beta))
            theta = self._mean + np.sqrt(sigma2) * np.linalg.solve(
                self._cholesky.T, rng.standard_normal(self.dimension))
            scores = {a.identity: float(features[a.identity] @ theta) for a in actions}
            if not np.isfinite(sigma2) or sigma2 <= 0:
                raise ValueError("invalid joint posterior draw")
        if not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("nonfinite action scores")
        primary = min(eligible, key=lambda a: (-scores[a.identity], a.identity))
        controls = alternatives(primary)
        control = controls[int(rng.integers(len(controls)))]
        row = {"step": step, "contrast": contrast, "mode": mode,
               "primary": asdict(primary), "control": asdict(control),
               "primary_identity": primary.identity, "control_identity": control.identity,
               "scores": scores, "sampled_sigma2": sigma2,
               "eligible_count": len(eligible), "legal_action_count": len(actions),
               "control_probability_given_primary": 1 / len(controls),
               "primary_selection_probability": None, "primary_probability_estimated": False}
        self._decisions[step] = row
        return primary, control

    def observe_pair(self, step, *, primary_request, control_request, primary_result,
                     control_result, primary_folds, control_folds, evidence_id, scoring_identity):
        if step not in self._decisions or scoring_identity != self.scoring_identity:
            raise ValueError("unknown decision or scorer")
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in self._seen or step in self._seen:
            raise ValueError("duplicate or missing evidence")
        record = self._decisions[step]
        actions = [SiteOperatorAction(**{**record[role], "site_path": tuple(record[role]["site_path"]),
                                        "features": tuple(record[role]["features"])}) for role in ("primary", "control")]
        if (primary_request.seed != control_request.seed
                or primary_request.task_identity != control_request.task_identity
                or control_request.matched_control_of != primary_request.identity):
            raise ValueError("counterfactual request identity mismatch")
        for action, request, result in zip(actions, (primary_request, control_request), (primary_result, control_result)):
            if (request.operator != action.operator or request.site_path != action.site_path
                    or request.parent_expression != action.parent or request.donor_expression != action.donor
                    or request.mock_llm_expression is not None
                    or result.request_identity != request.identity or not result.valid or not result.novel
                    or any(getattr(result, k) != getattr(request, k) for k in
                           ("operator", "site_path", "parent_expression", "donor_expression"))):
                raise ValueError("materialized result does not match frozen action")
        if not primary_result.support_hash or not control_result.support_hash or primary_result.support_hash == control_result.support_hash:
            raise ValueError("paired edits do not have distinct predictive supports")
        p, c = (np.asarray(v, dtype=float) for v in (primary_folds, control_folds))
        if p.ndim != 1 or c.shape != p.shape or len(p) < 2 or not np.all(np.isfinite([p, c])):
            raise ValueError("matched finite fold scores required")
        x = self.feature(actions[0]) - self.feature(actions[1])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            y = float(np.mean(p - c))
            u = self._covariance @ x
            denominator = 1 + float(x @ u)
            residual = y - float(x @ self._mean)
            mean = self._mean + u * residual / denominator
            precision = self._precision + np.outer(x, x)
            alpha = self._alpha + .5
            beta = self._beta + .5 * (residual / np.sqrt(denominator)) ** 2
            cholesky, covariance = _validated_state(mean, precision, alpha, beta)
        observation = {"step": step, "evidence_id": evidence_id, "scoring_identity": scoring_identity,
                       "task_identity": self.task_identity, "contrast": record["contrast"],
                       "primary_request": primary_request.identity, "control_request": control_request.identity,
                       "fold_contrasts": (p-c).tolist(), "contrast_mean": y, "observation_weight": 1}
        digest(observation)
        self._mean, self._precision, self._alpha, self._beta = mean, precision, alpha, beta
        self._cholesky, self._covariance = cholesky, covariance
        self._observations.append(observation)
        self._seen.update((step, evidence_id))

    def audit(self):
        return copy.deepcopy({"schema": SCHEMA, "feature_schema": self.feature_schema,
            "feature_names": FEATURE_NAMES, "interactions": INTERACTIONS, "operators": self.operators,
            "intervention_identity": self.intervention_identity, "utility_identity": self.utility_identity,
            "prior_identity": self._prior_identity, "task_identity": self.task_identity,
            "scoring_identity": self.scoring_identity, "observation_count": len(self._observations),
            "posterior": {"mean": self._mean.tolist(), "precision": self._precision.tolist(),
                          "alpha": self._alpha, "beta": self._beta},
            "decisions": list(self._decisions.values()), "observations": self._observations,
            "claim_boundary": "paired conditional working-model regression; no causal, calibrated or efficacy claim"})

    def freeze_transfer_prior(self):
        if not self._observations:
            raise ValueError("no training observations to freeze")
        value = {"schema": SCHEMA, "feature_schema": self.feature_schema,
                 "training_tasks": sorted(self._training_tasks | {self.task_identity}),
                 "evidence": self._inherited_evidence + [digest(o) for o in self._observations],
                 "posterior": self.audit()["posterior"]}
        return {**value, "identity": digest(value)}
