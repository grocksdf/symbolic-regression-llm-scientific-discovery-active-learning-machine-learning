"""Paired working regression for complete structural plans.

The response is the *executed* search-bank increment difference, not a
candidate's standalone fit or a posterior probability that a formula is true.
Explicit invalid/duplicate plans execute KEEP_E0. Infrastructure and scoring
errors are not outcomes and must be raised by the caller. Reusing search folds
adaptively makes this a conditional working model, not a calibrated scientific
posterior. No independent admission or reporting response enters this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy

import numpy as np

from .paired_edit_policy import _validated_state
from .region_site_attribution import digest
from .structural_plan_pool import FEATURE_NAMES


SCHEMA = "predictive-plan-paired-policy-v1"
FEATURE_SCHEMA = "bank-plan-behavior-features-v1"
FEATURE_SCHEMA_HASH = digest([FEATURE_SCHEMA, FEATURE_NAMES])
DIMENSION = len(FEATURE_NAMES)
UTILITY = "standalone-predictive-stacking-crossfit-logscore-v1"
INVALID_REASONS = frozenset(("formula_undefined_on_registered_domain", "nonfinite_formula_design"))
DUPLICATE_REASON = "same_fixed_support_as_core"


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity")
    return value


@dataclass(frozen=True)
class PlanAction:
    plan_id: str
    context_identity: str
    features: tuple[float, ...]

    def __post_init__(self):
        _identity(self.plan_id, "plan")
        _identity(self.context_identity, "context")
        if (type(self.features) is not tuple or len(self.features) != DIMENSION
                or any(not isinstance(v, (int, float, np.integer, np.floating))
                       or isinstance(v, (bool, np.bool_)) for v in self.features)):
            raise ValueError("eight bounded numeric plan features required")
        values = np.asarray(self.features, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1):
            raise ValueError("plan features must be finite and in [-1,1]")

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class PlanOutcome:
    plan_id: str
    status: str
    fold_gains: tuple[float, ...] = ()
    reason: str | None = None

    def __post_init__(self):
        _identity(self.plan_id, "plan outcome")
        if self.status not in {"scored", "invalid", "duplicate"}:
            raise ValueError("unknown outcome; infrastructure failures must propagate")
        if type(self.fold_gains) is not tuple:
            raise ValueError("immutable fold gains required")
        if self.status == "scored":
            if (len(self.fold_gains) != 2 or self.reason is not None
                    or any(not isinstance(v, (int, float, np.integer, np.floating))
                           or isinstance(v, (bool, np.bool_)) for v in self.fold_gains)
                    or not np.all(np.isfinite(np.asarray(self.fold_gains, dtype=float)))):
                raise ValueError("scored plan requires exactly two finite gains")
        elif (self.fold_gains or not isinstance(self.reason, str)
                or (self.status == "invalid" and self.reason not in INVALID_REASONS)
                or (self.status == "duplicate" and self.reason != DUPLICATE_REASON)):
            raise ValueError("KEEP_E0 requires a registered formula rejection, not an infrastructure error")

    @property
    def executed_gains(self):
        return self.fold_gains if self.status == "scored" else (0., 0.)


def _step_state(state, x, y):
    mean, precision, alpha, beta = state
    _, covariance = _validated_state(mean, precision, alpha, beta)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        u = covariance @ x
        denominator = 1. + float(x @ u)
        residual = y - float(x @ mean)
        staged = (mean + u * residual / denominator,
                  precision + np.outer(x, x), alpha + .5,
                  beta + .5 * (residual / np.sqrt(denominator)) ** 2)
        _validated_state(*staged)
    return staged


def _base_state():
    return np.zeros(DIMENSION), np.eye(DIMENSION), 1., 1.


def _posterior(state):
    mean, precision, alpha, beta = state
    return {"mean": mean.tolist(), "precision": precision.tolist(),
            "alpha": float(alpha), "beta": float(beta)}


def _action_from_dict(row):
    return PlanAction(row["plan_id"], row["context_identity"], tuple(row["features"]))


def _outcome_from_dict(row):
    return PlanOutcome(row["plan_id"], row["status"], tuple(row["fold_gains"]), row["reason"])


def _observation_vectors(primary, control, primary_outcome, control_outcome):
    if primary.plan_id != primary_outcome.plan_id or control.plan_id != control_outcome.plan_id:
        raise ValueError("outcome plan does not match selected action")
    x = np.asarray(primary.features, dtype=float) - np.asarray(control.features, dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        contrasts = np.asarray(primary_outcome.executed_gains) - control_outcome.executed_gains
        # Dividing first avoids avoidable summation overflow for large finite gains.
        y = float(np.sum(contrasts / 2.))
    if not np.all(np.isfinite(contrasts)) or not np.isfinite(y):
        raise ValueError("nonfinite paired execution contrast")
    no_information = ("both_keep_e0" if primary_outcome.status != "scored"
                      and control_outcome.status != "scored"
                      else "identical_features" if not np.any(x) else None)
    return x, y, contrasts, no_information


class PredictivePlanPolicy:
    """Choose complete plans; learn once per completed paired intervention.

    Context and source row identities are fixed by the caller before selection.
    A transfer prior is accepted only if its posterior can be rebuilt from
    hashed, task-disjoint training observations under this exact schema.
    """

    def __init__(self, *, seed, task_identity, bank_identity, scoring_identity,
                 feature_context_identity=None, frozen_prior=None):
        if type(seed) is not int or seed < 0:
            raise ValueError("nonnegative integer seed required")
        self.seed = seed
        self.task_identity = _identity(task_identity, "task")
        self.bank_identity = _identity(bank_identity, "bank")
        self.scoring_identity = _identity(scoring_identity, "scoring")
        self.feature_context_identity = _identity(
            scoring_identity if feature_context_identity is None else feature_context_identity, "feature context")
        self.dimension = DIMENSION
        self._state = _base_state()
        self._decisions, self._observations, self._used, self._seen = {}, [], set(), set()
        self._known_actions = {}
        self._pending, self._prior_identity, self._inherited = None, None, []
        if frozen_prior is not None:
            self._load_prior(frozen_prior)

    def _load_prior(self, frozen_prior):
        prior = copy.deepcopy(frozen_prior)
        prior_identity = prior.pop("identity", None)
        if (digest(prior) != prior_identity or prior.get("schema") != SCHEMA
                or prior.get("feature_schema") != FEATURE_SCHEMA
                or prior.get("feature_schema_hash") != FEATURE_SCHEMA_HASH or prior.get("utility") != UTILITY):
            raise ValueError("transfer identity/schema/utility mismatch")
        tasks = prior.get("training_tasks", [])
        if (not tasks or len(set(tasks)) != len(tasks) or self.task_identity in tasks
                or any(not isinstance(t, str) or not t for t in tasks)):
            raise ValueError("transfer requires explicit disjoint training tasks")
        evidence = prior.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("transfer requires complete evidence lineage")
        state, seen, actual_tasks, source_keys, source_plans = _base_state(), set(), set(), set(), set()
        for entry in evidence:
            observation = copy.deepcopy(entry)
            expected = observation.pop("identity", None)
            if digest(observation) != expected or expected in seen:
                raise ValueError("transfer evidence hash mismatch or duplicate")
            seen.add(expected)
            task = _identity(observation["task_identity"], "source task")
            key = (task, _identity(observation["evidence_id"], "source evidence"))
            if key in source_keys:
                raise ValueError("duplicate source evidence in transfer")
            source_keys.add(key)
            actual_tasks.add(task)
            _identity(observation["bank_identity"], "source bank")
            _identity(observation["scoring_identity"], "source scoring")
            primary, control = (_action_from_dict(observation[k]) for k in ("primary", "control"))
            for action in (primary, control):
                if (task, action.plan_id) in source_plans:
                    raise ValueError("source plan repeatedly counted in transfer")
                source_plans.add((task, action.plan_id))
            if (primary.plan_id == control.plan_id or primary.context_identity != control.context_identity
                    or primary.context_identity != observation["feature_context_identity"]):
                raise ValueError("transfer evidence context mismatch")
            x, y, contrasts, no_information = _observation_vectors(primary, control,
                *(_outcome_from_dict(observation[k]) for k in ("primary_outcome", "control_outcome")))
            if (observation["contrast_mean"] != y or observation["fold_contrasts"] != contrasts.tolist()
                    or observation["no_information"] != no_information
                    or observation["updated"] != (no_information is None)):
                raise ValueError("transfer reward reconstruction mismatch")
            if no_information is None:
                state = _step_state(state, x, y)
        if sorted(actual_tasks) != sorted(tasks) or _posterior(state) != prior.get("posterior"):
            raise ValueError("transfer task lineage or posterior reconstruction mismatch")
        if not any(e["updated"] for e in evidence):
            raise ValueError("transfer requires at least one informative observation")
        self._state, self._inherited, self._prior_identity = state, evidence, prior_identity

    def choose_pair(self, step, actions, *, mode="thompson"):
        if (type(step) is not int or step < 0 or step in self._decisions
                or self._pending is not None or mode not in {"thompson", "random"}):
            raise ValueError("invalid/duplicate decision or unresolved previous pair")
        actions = tuple(actions)
        if any(not isinstance(a, PlanAction) for a in actions):
            raise ValueError("typed immutable plan actions required")
        actions = tuple(sorted(actions, key=lambda a: a.plan_id))
        if (len({a.plan_id for a in actions}) != len(actions)
                or any(a.context_identity != self.feature_context_identity for a in actions)):
            raise ValueError("unique plan IDs in the frozen context required")
        if any(a.plan_id in self._known_actions and self._known_actions[a.plan_id] != a.identity for a in actions):
            raise ValueError("frozen plan features changed between decisions")
        # A caller may supply the complete pool each time. Previously consumed
        # plans cannot be resampled or counted again as new training evidence.
        available = tuple(a for a in actions if a.plan_id not in self._used)
        if len(available) < 2:
            return None
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        sigma2 = None
        if mode == "random":
            scores = {a.plan_id: float(rng.random()) for a in available}
        else:
            mean, precision, alpha, beta = self._state
            cholesky, _ = _validated_state(*self._state)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                sigma2 = float(1. / rng.gamma(alpha, 1. / beta))
                theta = mean + np.sqrt(sigma2) * np.linalg.solve(cholesky.T, rng.standard_normal(DIMENSION))
                scores = {a.plan_id: float(np.asarray(a.features) @ theta) for a in available}
            if not np.isfinite(sigma2) or sigma2 <= 0:
                raise ValueError("invalid posterior sample")
        if not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("nonfinite plan ranking")
        primary = min(available, key=lambda a: (-scores[a.plan_id], a.plan_id))
        controls = tuple(a for a in available if a.plan_id != primary.plan_id)
        control = controls[int(rng.integers(len(controls)))]
        record = {"step": step, "mode": mode, "primary": asdict(primary), "control": asdict(control),
                  "available_count": len(available), "scores": scores, "sampled_sigma2": sigma2,
                  "control_probability_given_primary": 1 / len(controls),
                  "primary_selection_probability": 1 / len(available) if mode == "random" else None,
                  "primary_probability_estimated": False}
        digest(record)
        self._decisions[step], self._pending = record, step
        self._known_actions.update((a.plan_id, a.identity) for a in actions)
        self._used.update((primary.plan_id, control.plan_id))
        return primary, control

    def observe_pair(self, step, *, primary_outcome, control_outcome, evidence_id, scoring_identity):
        if (type(step) is not int or self._pending != step or step not in self._decisions
                or scoring_identity != self.scoring_identity):
            raise ValueError("no matching pending decision/scoring identity")
        _identity(evidence_id, "evidence")
        if evidence_id in self._seen:
            raise ValueError("duplicate evidence")
        if not isinstance(primary_outcome, PlanOutcome) or not isinstance(control_outcome, PlanOutcome):
            raise ValueError("typed outcomes required; errors are not zero-gain observations")
        record = self._decisions[step]
        primary, control = (_action_from_dict(record[k]) for k in ("primary", "control"))
        x, y, contrasts, no_information = _observation_vectors(primary, control, primary_outcome, control_outcome)
        staged = self._state if no_information is not None else _step_state(self._state, x, y)
        observation = {"step": step, "evidence_id": evidence_id, "task_identity": self.task_identity,
            "bank_identity": self.bank_identity, "scoring_identity": scoring_identity,
            "feature_context_identity": self.feature_context_identity,
            "primary": asdict(primary), "control": asdict(control),
            "primary_outcome": asdict(primary_outcome), "control_outcome": asdict(control_outcome),
            "fold_contrasts": contrasts.tolist(), "contrast_mean": y,
            "updated": no_information is None, "no_information": no_information,
            "observation_weight": 1 if no_information is None else 0}
        observation = {**observation, "identity": digest(observation)}
        self._state = staged
        self._observations.append(observation)
        self._seen.add(evidence_id)
        self._pending = None

    def audit(self):
        return copy.deepcopy({"schema": SCHEMA, "feature_schema": FEATURE_SCHEMA,
            "feature_schema_hash": FEATURE_SCHEMA_HASH, "feature_names": FEATURE_NAMES, "utility": UTILITY,
            "dimension": DIMENSION, "task_identity": self.task_identity, "bank_identity": self.bank_identity,
            "scoring_identity": self.scoring_identity, "feature_context_identity": self.feature_context_identity,
            "prior_identity": self._prior_identity, "posterior": _posterior(self._state),
            "observation_count": len(self._observations),
            "update_count": sum(o["updated"] for o in self._observations),
            "inherited_observation_count": len(self._inherited),
            "pending_step": self._pending, "used_plan_ids": sorted(self._used),
            "decisions": list(self._decisions.values()), "observations": self._observations,
            "claim_boundary": "paired conditional working regression on search data; no calibrated, causal, efficacy or formula-truth claim"})

    def freeze_transfer_prior(self):
        if self._pending is not None:
            raise ValueError("cannot freeze a pending pair")
        evidence = copy.deepcopy(self._inherited + self._observations)
        if not evidence or not any(o["updated"] for o in evidence):
            raise ValueError("no informative training evidence to freeze")
        value = {"schema": SCHEMA, "feature_schema": FEATURE_SCHEMA,
                 "feature_schema_hash": FEATURE_SCHEMA_HASH, "utility": UTILITY,
                 "training_tasks": sorted({o["task_identity"] for o in evidence}),
                 "posterior": _posterior(self._state), "evidence": evidence}
        return {**value, "identity": digest(value)}
