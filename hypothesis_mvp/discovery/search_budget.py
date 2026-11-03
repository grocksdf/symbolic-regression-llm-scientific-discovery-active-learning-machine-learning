"""Explicit search stopping and cost accounting, not a statistical method.

Quality comparisons may use different declared resource ceilings. Mechanism
comparisons share candidate-opportunity ceilings and stopping rules; neither
contract claims equal realized CPU, token use, wall time or candidate quality.
Only search-side scores may enter this tracker. The owning harness must enforce
the data-role boundary: a number alone cannot prove where it was computed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from numbers import Real


COUNT_METRICS = frozenset({
    "planning_operations", "candidate_density_evaluations", "density_evaluations",
    "optimizer_fits", "provider_attempts", "prompt_tokens", "completion_tokens",
    "operator_preflights", "materializations",
})
WORK_METRICS = COUNT_METRICS | {"wall_time_seconds"}


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity")
    return value


@dataclass(frozen=True)
class SearchBudget:
    """A mandatory finite attempt cap and optional predeclared plateau rule.

    No small numerical cap is built in. ``patience=None`` spends the declared
    opportunity budget even if early rewards are all zero. When enabled, the
    plateau counter resets after the best observed search gain exceeds its
    last reset value by strictly more than ``minimum_improvement``. Small
    improvements can therefore accumulate. Invalid/failed attempts consume
    both the attempt budget and patience; they are never zero observations.
    """

    max_candidate_attempts: int
    patience: int | None = None
    minimum_improvement: float = 0.0

    def __post_init__(self):
        if type(self.max_candidate_attempts) is not int or self.max_candidate_attempts <= 0:
            raise ValueError("max_candidate_attempts must be a positive integer")
        if self.patience is not None and (type(self.patience) is not int or self.patience <= 0):
            raise ValueError("patience must be None or a positive integer")
        improvement = _finite_real(self.minimum_improvement, "minimum_improvement")
        if improvement < 0:
            raise ValueError("minimum_improvement must be nonnegative")
        object.__setattr__(self, "minimum_improvement", improvement)


def validate_comparison_budgets(*, purpose: str, arm_budgets: Mapping[str, SearchBudget]):
    """Freeze a quality or mechanism comparison without equating actual work.

    Mechanism equality covers nominal opportunities and stopping rules only.
    Data, initial Bank, grammar and search access need separate harness checks.
    This helper does not authorize an experiment or set a performance gate.
    """
    if purpose not in {"quality", "mechanism"}:
        raise ValueError("comparison purpose must be quality or mechanism")
    if not isinstance(arm_budgets, Mapping) or len(arm_budgets) < 2:
        raise ValueError("a comparison requires at least two named arm budgets")
    normalized = {}
    for name, budget in arm_budgets.items():
        _identity(name, "arm name")
        if not isinstance(budget, SearchBudget):
            raise ValueError("every arm needs a validated SearchBudget")
        normalized[name] = asdict(budget)
    policies = list(normalized.values())
    same_policy = all(value == policies[0] for value in policies[1:])
    if purpose == "mechanism" and not same_policy:
        raise ValueError("mechanism comparison requires common opportunity ceiling and stopping rule")
    payload = {
        "schema": "search-comparison-budget-v1", "purpose": purpose,
        "arm_budgets": dict(sorted(normalized.items())),
        "same_candidate_ceiling": len({v["max_candidate_attempts"] for v in policies}) == 1,
        "same_stopping_policy": same_policy,
        "equal_realized_compute_claimed": False,
        "actual_compute_matching_is_performance_gate": False,
        "measured_resource_use_required": True,
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**payload, "identity": sha256(material.encode()).hexdigest()}


class SearchBudgetTracker:
    """Record every attempted candidate once, with atomic input validation.

    Failure here is a candidate-level outcome. The caller must still terminate
    on protocol, data-integrity or provider infrastructure failures according
    to its registration. The tracker does not make such failures recoverable.
    """

    def __init__(self, budget: SearchBudget):
        if not isinstance(budget, SearchBudget):
            raise ValueError("tracker requires a validated SearchBudget")
        self._budget = budget
        self._records = []
        self._attempt_ids = set()
        self._counts = {"evaluated": 0, "invalid": 0, "failed": 0}
        self._work = {key: 0 for key in sorted(COUNT_METRICS)} | {"wall_time_seconds": 0.0}
        self._best = 0.0
        self._last_reset_gain = 0.0
        self._stagnant = 0
        self._stop_reason = None

    @property
    def can_attempt(self):
        return self._stop_reason is None

    @property
    def stop_reason(self):
        return self._stop_reason

    @property
    def attempt_count(self):
        return len(self._records)

    @property
    def best_search_gain(self):
        return self._best

    def record_attempt(self, *, attempt_id: str, outcome: str,
                       search_gain: float | None = None,
                       failure_reason: str | None = None, work: Mapping | None = None):
        if not self.can_attempt:
            raise ValueError("search budget is closed")
        attempt_id = _identity(attempt_id, "attempt_id")
        if attempt_id in self._attempt_ids:
            raise ValueError("duplicate attempt_id")
        if outcome not in self._counts:
            raise ValueError("outcome must be evaluated, invalid or failed")
        if outcome == "evaluated":
            score = _finite_real(search_gain, "search_gain")
            if failure_reason is not None:
                raise ValueError("an evaluated attempt cannot have a failure reason")
        else:
            if search_gain is not None:
                raise ValueError("invalid/failed attempts cannot carry a numerical reward")
            _identity(failure_reason, "failure_reason")
            score = None

        # Validate a complete staging state before publishing any mutation.
        supplied = {} if work is None else work
        if not isinstance(supplied, Mapping) or not set(supplied).issubset(WORK_METRICS):
            raise ValueError("unknown work accounting metric")
        increments = {}
        for name, value in supplied.items():
            if name in COUNT_METRICS:
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} must be a nonnegative integer count")
                increments[name] = value
            else:
                number = _finite_real(value, name)
                if number < 0:
                    raise ValueError(f"{name} must be nonnegative")
                increments[name] = number
        work_totals = dict(self._work)
        for name, value in increments.items():
            work_totals[name] += value
        if not math.isfinite(work_totals["wall_time_seconds"]):
            raise ValueError("cumulative wall time must remain finite")
        best = max(self._best, score) if score is not None else self._best
        significant = best - self._last_reset_gain > self._budget.minimum_improvement
        stagnant = 0 if significant else self._stagnant + 1
        next_count = self.attempt_count + 1
        reason = ("max_attempts" if next_count >= self._budget.max_candidate_attempts else
                  "plateau" if self._budget.patience is not None
                  and stagnant >= self._budget.patience else None)
        record = {"attempt_id": attempt_id, "outcome": outcome,
                  "search_gain": score, "failure_reason": failure_reason,
                  "work": increments, "best_search_gain_after": best,
                  "significant_search_improvement": significant}
        self._records.append(record)
        self._attempt_ids.add(attempt_id)
        self._counts[outcome] += 1
        self._best, self._stagnant, self._work = best, stagnant, work_totals
        if significant:
            self._last_reset_gain = best
        self._stop_reason = reason

    def mark_exhausted(self):
        """The fixed candidate/opportunity stream ended; no score is consulted."""
        if self._stop_reason not in {None, "exhausted"}:
            raise ValueError("cannot overwrite an existing stop reason")
        self._stop_reason = "exhausted"

    def audit(self):
        # Return an independent serializable snapshot, never mutable authority.
        return json.loads(json.dumps({
            "schema": "search-budget-accounting-v1", "budget": asdict(self._budget),
            "candidate_attempts": self.attempt_count, "outcomes": dict(self._counts),
            "best_search_gain": self._best, "attempts_since_significant_improvement": self._stagnant,
            "stop_reason": self._stop_reason, "can_attempt": self.can_attempt,
            "work": dict(self._work), "records": self._records,
            "reporting_score_used_for_stopping": False,
            "missing_scores_imputed": False,
            "equal_realized_compute_claimed": False,
        }, sort_keys=True, allow_nan=False))
