from __future__ import annotations

"""EvaluationRuntime: refit, pruning, semantic retention and Pareto policy.

It accepts and returns immutable EquationState objects. It never calls a
provider, retrieves knowledge, advances rounds or owns equation syntax.
"""

import dataclasses
from dataclasses import dataclass
import math
from collections import Counter
from typing import Any, Callable, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np
import sympy as sp

from .contracts import DiscoveryConfig, EquationState, LineageStep, json_safe
from .equation_runtime import EquationDAG, EquationRuntime, PrimitiveRegistry, RefitResult
if TYPE_CHECKING:
    from .proposal_runtime import ProposalCandidate

class EvaluationBudget:
    """Hard cap and audit ledger for unique candidate validation work."""

    def __init__(self, limit: Optional[int]) -> None:
        self.limit = max(1, int(limit)) if limit is not None else None
        self.used = 0
        self.denied = 0
        self.counts: Counter[str] = Counter()

    def reset(self) -> None:
        self.used = 0
        self.denied = 0
        self.counts.clear()

    def consume(self, category: str) -> bool:
        if self.limit is not None and self.used >= self.limit:
            self.denied += 1
            self.counts[f"denied:{category}"] += 1
            return False
        self.used += 1
        self.counts[str(category)] += 1
        return True

    def record(self, category: str, amount: int = 1) -> None:
        self.counts[str(category)] += max(0, int(amount))

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.used >= self.limit

    @property
    def enforced(self) -> bool:
        return self.limit is not None

    def snapshot(self) -> dict[str, Any]:
        return {
            "unit": "unique_candidate_validation",
            "limit": self.limit,
            "used": self.used,
            "remaining": None if self.limit is None else max(0, self.limit - self.used),
            "enforced": self.limit is not None,
            "exhausted": self.exhausted,
            "denied": self.denied,
            "counts": dict(sorted(self.counts.items())),
        }


@dataclass(frozen=True)
class MetricVector:
    train_nmse: float
    val_nmse: float
    val_strict: float
    val_p99: float
    val_p995: float
    val_fraction_01: float
    stress_nmse: float
    stress_strict: float
    stress_p99: float
    ood_proxy_nmse: float
    ood_proxy_strict: float
    ood_stability_penalty: float
    complexity: float
    actual_ood_nmse: Optional[float] = None
    actual_ood_strict: Optional[float] = None
    invalid_predictions: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def finite(self) -> bool:
        required = (
            self.train_nmse, self.val_nmse, self.val_strict, self.val_p99,
            self.val_p995, self.stress_nmse, self.stress_strict,
            self.stress_p99, self.ood_proxy_nmse, self.ood_proxy_strict,
            self.ood_stability_penalty, self.complexity,
        )
        return all(math.isfinite(float(v)) for v in required) and self.invalid_predictions == 0


@dataclass(frozen=True)
class EvaluatorConfig:
    stress_folds: int = 6
    random_seed: int = 0
    # Match LLM-SRBench strict-tail semantics by default:
    # abs(error) / max(abs(y), 1e-12).  A ratio floor is optional and must
    # never be silently enabled because it can hide failures around y=0.
    relative_error_floor_abs: float = 1.0e-12
    relative_error_floor_ratio: float = 0.0
    ood_probe_count: int = 128

class MultiDomainEvaluator:
    """Train/validation/strict-tail/stress/OOD-proxy evaluator.

    Actual benchmark OOD labels are audit-only when supplied. Search selection
    uses train, validation and train-derived proxies only.
    """

    def __init__(self, runtime: EquationRuntime, config: Optional[EvaluatorConfig] = None) -> None:
        self.runtime = runtime
        self.config = config or EvaluatorConfig()

    def _block(self, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        y, pred = np.asarray(y, dtype=float).reshape(-1), np.asarray(pred, dtype=float).reshape(-1)
        if len(y) == 0 or len(y) != len(pred) or not np.all(np.isfinite(pred)):
            return {"nmse": float("inf"), "strict": float("inf"), "p99": float("inf"), "p995": float("inf"), "fraction": 0.0}
        err = pred - y
        var = max(float(np.var(y)), 1.0e-15)
        scale = max(float(np.quantile(np.abs(y), 0.95)), math.sqrt(var), 1.0e-12)
        floor = max(
            float(self.config.relative_error_floor_abs),
            float(self.config.relative_error_floor_ratio) * scale,
        )
        rel = np.abs(err) / np.maximum(np.abs(y), floor)
        return {
            "nmse": float(np.mean(err * err) / var),
            "strict": float(np.max(rel)),
            "p99": float(np.quantile(rel, 0.99)),
            "p995": float(np.quantile(rel, 0.995)),
            "fraction": float(np.mean(rel <= 0.1)),
        }

    def _stress(self, expression: str, X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        pred, n = self.runtime.predict(expression, X), len(y)
        if n < 4:
            block = self._block(y, pred)
            return block["nmse"], block["strict"], block["p99"]
        rng = np.random.default_rng(int(self.config.random_seed) + 104729 * n)
        random_count = min(max(2, self.config.stress_folds // 2), n)
        ordered_count = min(max(2, self.config.stress_folds - random_count), n)
        folds = list(np.array_split(rng.permutation(n), random_count))
        folds += list(np.array_split(np.argsort(np.abs(y)), ordered_count))
        blocks = [self._block(y[idx], pred[idx]) for idx in folds if len(idx)]
        return (
            max((b["nmse"] for b in blocks), default=float("inf")),
            max((b["strict"] for b in blocks), default=float("inf")),
            max((b["p99"] for b in blocks), default=float("inf")),
        )

    def _ood_proxy(self, expression: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float, float]:
        low, high = np.quantile(X_train, 0.05, axis=0), np.quantile(X_train, 0.95, axis=0)
        extreme = np.any((X_val < low) | (X_val > high), axis=1)
        if int(np.sum(extreme)) < max(4, int(0.05 * len(X_val))):
            center, span = np.median(X_train, axis=0), np.maximum(high - low, 1.0e-12)
            distance = np.max(np.abs((X_val - center) / span), axis=1)
            count = min(len(X_val), max(4, int(math.ceil(0.2 * len(X_val)))))
            idx = np.argsort(distance)[-count:]
        else:
            idx = np.flatnonzero(extreme)
        proxy = self._block(y_val[idx], self.runtime.predict(expression, X_val[idx]))

        X_all, y_all = np.vstack([X_train, X_val]), np.concatenate([y_train, y_val])
        center = np.median(X_all, axis=0)
        span = np.maximum(np.quantile(X_all, 0.95, axis=0) - np.quantile(X_all, 0.05, axis=0), 1.0e-9)
        rng = np.random.default_rng(int(self.config.random_seed) + 65537 * X_all.shape[1] + len(X_all))
        n_probe = min(max(16, int(self.config.ood_probe_count)), max(16, len(X_all)))
        probe = center + rng.choice(np.array([-1.0, 1.0]), size=(n_probe, X_all.shape[1])) * rng.uniform(0.75, 1.50, size=(n_probe, X_all.shape[1])) * span
        y_scale = max(float(np.quantile(np.abs(y_all), 0.95)), math.sqrt(max(float(np.var(y_all)), 1.0e-15)), 1.0e-9)
        try:
            observed = self.runtime.predict(expression, X_all)
            far = self.runtime.predict(expression, probe)
            near = self.runtime.predict(expression, probe + 1.0e-4 * span)
        except Exception:
            return proxy["nmse"], proxy["strict"], float("inf")
        observed_amp = max(1.0, float(np.quantile(np.abs(observed), 0.99) / y_scale))
        far_amp = float(np.quantile(np.abs(far), 0.99) / y_scale)
        sensitivity = float(np.quantile(np.abs(near - far), 0.99) / y_scale / 1.0e-4)
        penalty = max(0.0, math.log1p(far_amp / observed_amp) - math.log(3.0))
        penalty += max(0.0, math.log1p(sensitivity) - math.log(1.0e5))
        return proxy["nmse"], proxy["strict"], penalty

    def evaluate(
        self,
        expression: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> MetricVector:
        try:
            dag = self.runtime.dag(expression)
            train = self._block(y_train, self.runtime.predict(dag.expression, X_train))
            val = self._block(y_val, self.runtime.predict(dag.expression, X_val))
            X_all, y_all = np.vstack([X_train, X_val]), np.concatenate([y_train, y_val])
            stress_nmse, stress_strict, stress_p99 = self._stress(dag.expression, X_all, y_all)
            ood_nmse, ood_strict, ood_stability = self._ood_proxy(dag.expression, X_train, y_train, X_val, y_val)
            return MetricVector(
                train["nmse"], val["nmse"], val["strict"], val["p99"], val["p995"], val["fraction"],
                stress_nmse, stress_strict, stress_p99, ood_nmse, ood_strict, ood_stability,
                dag.complexity, None, None, 0,
            )
        except Exception:
            return MetricVector(
                train_nmse=float("inf"), val_nmse=float("inf"), val_strict=float("inf"),
                val_p99=float("inf"), val_p995=float("inf"), val_fraction_01=0.0,
                stress_nmse=float("inf"), stress_strict=float("inf"), stress_p99=float("inf"),
                ood_proxy_nmse=float("inf"), ood_proxy_strict=float("inf"),
                ood_stability_penalty=float("inf"), complexity=float("inf"),
                actual_ood_nmse=None, actual_ood_strict=None, invalid_predictions=1,
            )


class ParetoPolicy:
    """Transparent multi-domain, complexity-aware acceptance and final dominance."""

    ERROR_FIELDS = (
        "train_nmse", "val_nmse", "val_strict", "val_p99", "val_p995",
        "stress_nmse", "stress_strict", "stress_p99",
        "ood_proxy_nmse", "ood_proxy_strict", "ood_stability_penalty",
    )
    FINAL_FIELDS = (
        "val_nmse", "val_strict", "val_p99", "val_p995",
        "stress_nmse", "stress_strict", "stress_p99",
        "ood_proxy_nmse", "ood_proxy_strict", "ood_stability_penalty",
    )

    def __init__(self, config: DiscoveryConfig) -> None:
        self.config = config

    @staticmethod
    def _safe_ratio(value: float, baseline: float) -> float:
        value, baseline = float(value), float(baseline)
        return value / max(abs(baseline), 1.0e-15)

    @staticmethod
    def _log_metric(value: float) -> float:
        value = float(value)
        return math.log1p(max(0.0, min(value, 1.0e18))) if math.isfinite(value) else 50.0

    def score(self, hypothesis: EquationState, island: str = "balanced", reference: Optional[EquationState] = None) -> float:
        m = hypothesis.metrics
        if not m.finite:
            return float("inf")
        common = (
            2.5 * self._log_metric(m.val_nmse)
            + 0.7 * self._log_metric(m.stress_nmse)
            + 0.5 * self._log_metric(m.ood_proxy_nmse)
            + 0.15 * self._log_metric(m.ood_stability_penalty)
        )
        if island == "low_complexity":
            return common + 0.050 * m.complexity + 0.25 * self._log_metric(m.val_p99)
        if island == "tail":
            return common + 1.50 * self._log_metric(m.val_strict) + 1.20 * self._log_metric(m.val_p99) + 0.70 * self._log_metric(m.stress_p99) + 0.012 * m.complexity
        if island == "novelty":
            novelty = 0.0
            if reference is not None and hypothesis.dag.structural_hash != reference.dag.structural_hash:
                novelty = 0.30 + 0.03 * len(hypothesis.edit.get("added_nodes", []))
            return common + 0.40 * self._log_metric(m.val_p99) + 0.015 * m.complexity - novelty
        if island == "nmse":
            return common + 0.25 * self._log_metric(m.val_p99) + 0.010 * m.complexity
        return common + 0.65 * self._log_metric(m.val_p99) + 0.35 * self._log_metric(m.val_strict) + 0.020 * m.complexity

    def _non_worse(self, candidate: float, reference: float, tolerance: float) -> bool:
        reference = float(reference)
        candidate = float(candidate)
        if not math.isfinite(candidate):
            return False
        return candidate <= reference * (1.0 + tolerance) + 1.0e-12

    def _normalized_complexity_cost(self, candidate: MetricVector, reference: MetricVector) -> tuple[float, float]:
        """Return a bounded cost for added structure.

        Relative-to-anchor complexity is unstable when f0 is a constant or a
        single variable: adding a compact, correct structure then appears to be
        an arbitrarily large percentage increase.  Use an absolute delta
        normalized by a fixed local structural budget instead.
        """
        delta = max(0.0, float(candidate.complexity) - float(reference.complexity))
        local_budget = max(8.0, min(float(self.config.max_complexity), float(reference.complexity) + 16.0))
        return delta, min(1.0, delta / max(local_budget, 1.0e-12))

    def _weighted_accuracy_gain(self, candidate: MetricVector, reference: MetricVector) -> tuple[float, dict[str, float]]:
        val_gain = 1.0 - self._safe_ratio(candidate.val_nmse, reference.val_nmse)
        tail_gain = 0.5 * (1.0 - self._safe_ratio(candidate.val_p99, reference.val_p99)) + 0.5 * (1.0 - self._safe_ratio(candidate.val_strict, reference.val_strict))
        stress_gain = 0.5 * (1.0 - self._safe_ratio(candidate.stress_nmse, reference.stress_nmse)) + 0.5 * (1.0 - self._safe_ratio(candidate.stress_p99, reference.stress_p99))
        ood_gain = 0.5 * (1.0 - self._safe_ratio(candidate.ood_proxy_nmse, reference.ood_proxy_nmse)) + 0.5 * (1.0 - self._safe_ratio(candidate.ood_proxy_strict, reference.ood_proxy_strict))
        weighted = 0.45 * val_gain + 0.25 * tail_gain + 0.20 * stress_gain + 0.10 * ood_gain
        return weighted, {
            "validation": val_gain,
            "tail": tail_gain,
            "stress": stress_gain,
            "ood_proxy": ood_gain,
        }

    def dominates(self, candidate: EquationState, reference: EquationState, final: bool = False) -> tuple[bool, dict[str, Any]]:
        cm, rm = candidate.metrics, reference.metrics
        tolerance = min(self.config.metric_worse_tolerance, 0.005) if final else self.config.metric_worse_tolerance
        fields = self.FINAL_FIELDS if final else self.ERROR_FIELDS
        comparisons: dict[str, dict[str, Any]] = {}
        for field in fields:
            candidate_value = float(getattr(cm, field))
            reference_value = float(getattr(rm, field))
            if field == "ood_stability_penalty":
                # The proxy is a bounded warning score, not a physical loss.
                # Smooth true equations (for example low-order polynomials) may
                # legitimately extrapolate farther than an underfit anchor.
                non_worse = candidate_value <= max(
                    self.config.ood_stability_warning_cap,
                    reference_value * (1.0 + tolerance) + 1.0e-12,
                )
                relative_gain = 1.0 - candidate_value / max(1.0, reference_value)
            else:
                non_worse = self._non_worse(candidate_value, reference_value, tolerance)
                relative_gain = 1.0 - self._safe_ratio(candidate_value, reference_value)
            comparisons[field] = {
                "candidate": candidate_value,
                "reference": reference_value,
                "non_worse": non_worse,
                "relative_gain": relative_gain,
            }
        # Near-zero relative-error metrics remain diagnostics but are not hard
        # vetoes: a physically useful candidate should not be rejected because
        # a target crosses zero. Safety is enforced on scale-normalized losses,
        # stress behavior, OOD proxy behavior, finiteness and complexity.
        safety_fields = ("val_nmse", "stress_nmse", "stress_p99", "ood_proxy_nmse", "ood_stability_penalty")
        safety_non_worse = all(comparisons[field]["non_worse"] for field in safety_fields)
        all_non_worse = all(row["non_worse"] for row in comparisons.values())
        meaningful_fields = ("val_nmse", "val_p99", "stress_nmse", "stress_p99", "ood_proxy_nmse")
        best_gain = max((float(comparisons[field]["relative_gain"]) for field in meaningful_fields if field in comparisons), default=-float("inf"))
        accuracy_gain, accuracy_components = self._weighted_accuracy_gain(cm, rm)
        added_complexity_delta, complexity_cost = self._normalized_complexity_cost(cm, rm)
        signed_complexity_delta = float(cm.complexity) - float(rm.complexity)
        complexity_tradeoff_pass = accuracy_gain >= self.config.complexity_penalty * complexity_cost - 1.0e-12
        complexity_cap_pass = cm.complexity <= float(self.config.max_complexity)
        val_strict_absolute_pass = bool(
            not final
            or not self.config.final_require_val_strict_pass
            or cm.val_strict <= self.config.final_val_strict_threshold + 1.0e-12
        )
        stress_strict_absolute_pass = bool(
            not final
            or not self.config.final_require_stress_strict_pass
            or cm.stress_strict <= self.config.final_stress_strict_threshold + 1.0e-12
        )
        changed = candidate.dag.canonical_hash != reference.dag.canonical_hash
        utility_gain = accuracy_gain - self.config.complexity_penalty * complexity_cost
        meaningful_gain = utility_gain >= self.config.min_objective_gain or (
            cm.complexity < rm.complexity and safety_non_worse
        )
        pass_gate = bool(
            cm.finite
            and safety_non_worse
            and meaningful_gain
            and complexity_tradeoff_pass
            and complexity_cap_pass
            and val_strict_absolute_pass
            and stress_strict_absolute_pass
            and changed
        )
        gate = {
            "pass": pass_gate,
            "final": bool(final),
            "metric_worse_tolerance": tolerance,
            "all_non_worse": all_non_worse,
            "safety_non_worse": safety_non_worse,
            "safety_fields": list(safety_fields),
            "diagnostic_only_fields": ["val_strict", "val_p99", "val_p995", "stress_strict", "ood_proxy_strict"],
            "meaningful_gain": meaningful_gain,
            "best_relative_gain": best_gain,
            "weighted_accuracy_gain": accuracy_gain,
            "utility_gain": utility_gain,
            "accuracy_gain_components": accuracy_components,
            "signed_complexity_delta": signed_complexity_delta,
            "added_complexity_delta": added_complexity_delta,
            "complexity_delta": added_complexity_delta,  # added cost only
            "normalized_added_complexity_cost": complexity_cost,
            "normalized_complexity_cost": complexity_cost,
            "complexity_tradeoff_pass": complexity_tradeoff_pass,
            "complexity_cap_pass": complexity_cap_pass,
            "val_strict_absolute_pass": val_strict_absolute_pass,
            "val_strict_absolute_threshold": self.config.final_val_strict_threshold,
            "stress_strict_absolute_pass": stress_strict_absolute_pass,
            "stress_strict_absolute_threshold": self.config.final_stress_strict_threshold,
            "changed": changed,
            "comparisons": comparisons,
        }
        return pass_gate, gate

    def accept_transition(self, candidate: EquationState, current: EquationState, island: str) -> tuple[bool, dict[str, Any]]:
        if not candidate.metrics.finite or candidate.metrics.complexity > self.config.max_complexity:
            return False, {"pass": False, "reason": "invalid_or_complexity_cap"}
        tolerance = self.config.metric_worse_tolerance
        safety_fields = (
            "val_nmse", "val_strict", "val_p99", "val_p995",
            "stress_nmse", "stress_strict", "stress_p99",
            "ood_proxy_nmse", "ood_proxy_strict",
        )
        safety = {field: self._non_worse(getattr(candidate.metrics, field), getattr(current.metrics, field), tolerance) for field in safety_fields}
        safety["ood_stability_penalty"] = bool(
            candidate.metrics.ood_stability_penalty
            <= max(
                self.config.ood_stability_warning_cap,
                current.metrics.ood_stability_penalty * (1.0 + tolerance) + 1.0e-12,
            )
        )
        before = self.score(current, island, current)
        after = self.score(candidate, island, current)
        relative_gain = (before - after) / max(abs(before), 1.0e-12)
        accuracy_gain, accuracy_components = self._weighted_accuracy_gain(candidate.metrics, current.metrics)
        added_complexity_delta, complexity_cost = self._normalized_complexity_cost(candidate.metrics, current.metrics)
        signed_complexity_delta = float(candidate.metrics.complexity) - float(current.metrics.complexity)
        pareto_rescue_gain = accuracy_gain - self.config.complexity_penalty * complexity_cost
        changed = candidate.dag.canonical_hash != current.dag.canonical_hash
        # The rescue is still guarded by every multi-domain safety check.  It
        # only prevents a scalarized island score from rejecting an otherwise
        # Pareto-safe, materially better complete equation because it adds a
        # few necessary operators.
        score_pass = relative_gain >= self.config.min_objective_gain
        pareto_rescue_pass = pareto_rescue_gain >= self.config.min_objective_gain
        passed = bool(all(safety.values()) and changed and (score_pass or pareto_rescue_pass))
        transition_utility = max(relative_gain, pareto_rescue_gain)
        return passed, {
            "pass": passed,
            "island": island,
            "objective_before": before,
            "objective_after": after,
            "relative_objective_gain": relative_gain,
            "weighted_accuracy_gain": accuracy_gain,
            "accuracy_gain_components": accuracy_components,
            "signed_complexity_delta": signed_complexity_delta,
            "added_complexity_delta": added_complexity_delta,
            "complexity_delta": added_complexity_delta,  # added cost only
            "normalized_added_complexity_cost": complexity_cost,
            "normalized_complexity_cost": complexity_cost,
            "pareto_rescue_gain": pareto_rescue_gain,
            "score_pass": score_pass,
            "pareto_rescue_pass": pareto_rescue_pass,
            "transition_utility": transition_utility,
            "changed": changed,
            "safety": safety,
        }

    def pareto_front(self, hypotheses: Sequence[EquationState]) -> list[EquationState]:
        unique: dict[str, EquationState] = {}
        for item in hypotheses:
            existing = unique.get(item.dag.canonical_hash)
            if existing is None or self.score(item) < self.score(existing):
                unique[item.dag.canonical_hash] = item
        items = list(unique.values())
        front: list[EquationState] = []
        for candidate in items:
            dominated = False
            for other in items:
                if other is candidate:
                    continue
                fields = ("val_nmse", "val_strict", "val_p99", "stress_nmse", "stress_p99", "ood_proxy_nmse", "complexity")
                non_worse = all(float(getattr(other.metrics, f)) <= float(getattr(candidate.metrics, f)) + 1.0e-12 for f in fields)
                strict_better = any(float(getattr(other.metrics, f)) < float(getattr(candidate.metrics, f)) - 1.0e-12 for f in fields)
                if non_worse and strict_better:
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        return sorted(front, key=lambda h: self.score(h))




class EvaluationRuntime:
    """Create and compare EquationState values under one evaluation contract."""

    def __init__(
        self,
        equation_runtime: EquationRuntime,
        config: DiscoveryConfig,
        on_rejection: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
        on_warning: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
    ) -> None:
        self.runtime = equation_runtime
        self.registry = equation_runtime.registry
        self.n_features = equation_runtime.n_features
        self.config = config
        self.evaluator = MultiDomainEvaluator(
            equation_runtime, EvaluatorConfig(stress_folds=config.stress_folds, random_seed=config.random_seed)
        )
        self.policy = ParetoPolicy(config)
        self.budget = EvaluationBudget(config.evaluation_budget)
        self._rejections: list[dict[str, Any]] = []
        self._warnings: list[dict[str, Any]] = []
        self._on_rejection = on_rejection
        self._on_warning = on_warning

    @property
    def rejections(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._rejections)

    @property
    def warnings(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._warnings)

    def reset(self) -> None:
        """Clear per-run audit state when a runtime instance is reused."""
        self._rejections.clear()
        self._warnings.clear()
        self.budget.reset()

    def _reject(self, reason: str, **payload: Any) -> None:
        row = {"reason": str(reason), **json_safe(payload)}
        self._rejections.append(row)
        if self._on_rejection is not None:
            self._on_rejection(reason, row)

    def _warn(self, reason: str, **payload: Any) -> None:
        row = {"reason": str(reason), **json_safe(payload)}
        self._warnings.append(row)
        if self._on_warning is not None:
            self._on_warning(reason, row)

    def _top_level_term_entries(self, expression: str) -> list[dict[str, Any]]:
        """Return executable top-level structures with numeric amplitudes separated.

        Entries retain the original additive-term index so semantic matching can
        later drive an actual ablation. Numeric literals are also masked for a
        cheap structural equality check, but numerical equivalence is decided by
        evaluated behavior rather than string identity.
        """
        expr = self.registry.parse(expression, self.n_features, evaluate=False)
        terms = list(expr.args) if getattr(expr, "is_Add", False) else [expr]
        out: list[dict[str, Any]] = []
        for term_index, term in enumerate(terms):
            _, structure = term.as_coeff_Mul(rational=False)
            if len(getattr(structure, "free_symbols", set())) == 0:
                continue
            structure_text = self.registry.normalize(sp.sstr(structure, order="lex"))
            try:
                numbers = sorted(structure.atoms(sp.Number), key=str)
                replacements = {number: sp.Symbol(f"C{i}", real=True) for i, number in enumerate(numbers)}
                masked = sp.sstr(structure.xreplace(replacements), order="lex")
            except Exception:
                masked = structure_text
            out.append({
                "term_index": term_index,
                "term_expression": self.registry.normalize(sp.sstr(term, order="lex")),
                "structure_expression": structure_text,
                "masked_structure": masked,
                "variables": sorted(str(symbol) for symbol in structure.free_symbols),
            })
        return out

    def _semantic_similarity(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        X: np.ndarray,
    ) -> dict[str, Any]:
        if str(left.get("masked_structure")) == str(right.get("masked_structure")):
            return {"match": True, "exact_masked": True, "abs_correlation": 1.0, "projection_nrmse": 0.0}
        try:
            a = self.runtime.predict(str(left["structure_expression"]), X)
            b = self.runtime.predict(str(right["structure_expression"]), X)
        except Exception as exc:
            return {"match": False, "exact_masked": False, "error": repr(exc)}
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        if len(a) != len(b) or len(a) < 3:
            return {"match": False, "exact_masked": False, "reason": "shape_or_sample_count"}
        ac = a - float(np.mean(a))
        bc = b - float(np.mean(b))
        an = max(float(np.linalg.norm(ac)), 1.0e-15)
        bn = max(float(np.linalg.norm(bc)), 1.0e-15)
        corr = abs(float(np.dot(ac, bc) / (an * bn)))
        scale_ab = float(np.dot(ac, bc) / max(float(np.dot(bc, bc)), 1.0e-15))
        scale_ba = float(np.dot(bc, ac) / max(float(np.dot(ac, ac)), 1.0e-15))
        err_ab = float(np.linalg.norm(ac - scale_ab * bc) / an)
        err_ba = float(np.linalg.norm(bc - scale_ba * ac) / bn)
        nrmse = min(err_ab, err_ba)
        matched = bool(
            corr >= self.config.structure_similarity_correlation
            or nrmse <= self.config.structure_similarity_nrmse
        )
        return {
            "match": matched,
            "exact_masked": False,
            "abs_correlation": corr,
            "projection_nrmse": nrmse,
        }

    def _semantic_matches(
        self,
        source: Sequence[Mapping[str, Any]],
        target: Sequence[Mapping[str, Any]],
        X: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[int], list[int]]:
        scored: list[tuple[float, float, int, int, dict[str, Any]]] = []
        for i, left in enumerate(source):
            for j, right in enumerate(target):
                evidence = self._semantic_similarity(left, right, X)
                if evidence.get("match"):
                    scored.append((
                        -float(evidence.get("abs_correlation", 0.0)),
                        float(evidence.get("projection_nrmse", float("inf"))),
                        i, j, evidence,
                    ))
        scored.sort(key=lambda row: (row[0], row[1]))
        used_source: set[int] = set()
        used_target: set[int] = set()
        matches: list[dict[str, Any]] = []
        for _, _, i, j, evidence in scored:
            if i in used_source or j in used_target:
                continue
            used_source.add(i)
            used_target.add(j)
            matches.append({
                "source_index": i,
                "target_index": j,
                "source": dict(source[i]),
                "target": dict(target[j]),
                **evidence,
            })
        return matches, [i for i in range(len(source)) if i not in used_source], [j for j in range(len(target)) if j not in used_target]

    def _prune_refitted_candidate(
        self,
        dag: EquationDAG,
        metrics: MetricVector,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        round_id: int,
    ) -> tuple[EquationDAG, MetricVector, dict[str, Any]]:
        """Remove numerically dispensable top-level terms before selection.

        This is canonical post-refit cleanup, not a proposal lane. A deletion is
        accepted only when every validation/stress/OOD-proxy safety metric stays
        within a small tolerance and complexity strictly decreases.
        """
        try:
            current_expr = self.registry.parse(dag.expression, self.n_features, evaluate=False)
        except Exception as exc:
            return dag, metrics, {"applied": False, "reason": "parse_failure", "error": repr(exc)}
        terms = list(current_expr.args) if getattr(current_expr, "is_Add", False) else [current_expr]
        if len(terms) <= 1 or len(terms) > self.config.candidate_prune_max_terms:
            return dag, metrics, {"applied": False, "reason": "term_count_out_of_range", "term_count": len(terms)}
        tolerance = self.config.candidate_prune_tolerance
        safety_fields = (
            "train_nmse", "val_nmse", "val_strict", "val_p99", "val_p995",
            "stress_nmse", "stress_strict", "stress_p99",
            "ood_proxy_nmse", "ood_proxy_strict",
        )
        current_dag, current_metrics = dag, metrics
        removed: list[str] = []
        trials = 0
        while True:
            current_expr = self.registry.parse(current_dag.expression, self.n_features, evaluate=False)
            current_terms = list(current_expr.args) if getattr(current_expr, "is_Add", False) else [current_expr]
            if len(current_terms) <= 1:
                break
            candidates: list[tuple[float, float, float, EquationDAG, MetricVector, str]] = []
            for index, removed_term in enumerate(current_terms):
                remaining = [term for j, term in enumerate(current_terms) if j != index]
                if not any(getattr(term, "free_symbols", set()) for term in remaining):
                    continue
                trial_expression = self.registry.normalize(sp.sstr(sp.Add(*remaining, evaluate=False), order="lex"))
                trials += 1
                if not self.budget.consume("post_refit_pruning_trial"):
                    return current_dag, current_metrics, {
                        "applied": bool(removed),
                        "reason": "evaluation_budget_exhausted",
                        "removed_terms": removed,
                        "trials_evaluated": trials - 1,
                    }
                try:
                    refit = self.runtime.refit_global_constants(
                        trial_expression, X_train, y_train, ridge=self.config.ridge_alpha,
                        maxiter=self.config.optimizer_maxiter,
                        random_seed=self.config.random_seed + 2003 * round_id + trials,
                    )
                    self.budget.record("parameter_refits")
                    trial_dag = self.runtime.dag(refit.expression)
                    trial_metrics = self.evaluator.evaluate(trial_dag.expression, X_train, y_train, X_val, y_val)
                    self.budget.record("multi_domain_metric_evaluations")
                except Exception:
                    continue
                if not trial_metrics.finite or trial_metrics.complexity >= current_metrics.complexity - 1.0e-12:
                    continue
                if any(
                    float(getattr(trial_metrics, field))
                    > float(getattr(current_metrics, field)) * (1.0 + tolerance) + 1.0e-12
                    for field in safety_fields
                ):
                    continue
                if trial_metrics.ood_stability_penalty > max(
                    self.config.ood_stability_warning_cap,
                    current_metrics.ood_stability_penalty * (1.0 + tolerance) + 1.0e-12,
                ):
                    continue
                candidates.append((
                    trial_metrics.complexity, trial_metrics.val_nmse, trial_metrics.val_p99,
                    trial_dag, trial_metrics, sp.sstr(removed_term, order="lex"),
                ))
            if not candidates:
                break
            _, _, _, current_dag, current_metrics, removed_text = min(candidates, key=lambda row: row[:3])
            removed.append(removed_text)
        return current_dag, current_metrics, {
            "applied": bool(removed),
            "reason": "terms_removed" if removed else "no_dispensable_terms",
            "removed_terms": removed,
            "trials_evaluated": trials,
            "before_expression": dag.expression,
            "after_expression": current_dag.expression,
            "before_complexity": metrics.complexity,
            "after_complexity": current_metrics.complexity,
            "tolerance": tolerance,
        }

    def _ablation_effect(
        self,
        final_dag: EquationDAG,
        final_metrics: MetricVector,
        term_index: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        seed_offset: int,
    ) -> dict[str, Any]:
        try:
            expr = self.registry.parse(final_dag.expression, self.n_features, evaluate=False)
            terms = list(expr.args) if getattr(expr, "is_Add", False) else [expr]
            if term_index < 0 or term_index >= len(terms) or len(terms) <= 1:
                return {"measurable": False, "reason": "invalid_ablation_index"}
            remaining = [term for index, term in enumerate(terms) if index != term_index]
            trial_expression = self.registry.normalize(sp.sstr(sp.Add(*remaining, evaluate=False), order="lex"))
            if not self.budget.consume("structure_ablation_trial"):
                return {"measurable": False, "reason": "evaluation_budget_exhausted"}
            refit = self.runtime.refit_global_constants(
                trial_expression, X_train, y_train, ridge=self.config.ridge_alpha,
                maxiter=self.config.optimizer_maxiter,
                random_seed=self.config.random_seed + 7919 + seed_offset,
            )
            self.budget.record("parameter_refits")
            trial_dag = self.runtime.dag(refit.expression)
            trial_metrics = self.evaluator.evaluate(trial_dag.expression, X_train, y_train, X_val, y_val)
            self.budget.record("multi_domain_metric_evaluations")
            final_pred = self.runtime.predict(final_dag.expression, X_val)
            trial_pred = self.runtime.predict(trial_dag.expression, X_val)
            y_scale = max(float(np.sqrt(np.mean(np.asarray(y_val, dtype=float) ** 2))), float(np.std(y_val)), 1.0e-12)
            prediction_effect = float(np.sqrt(np.mean((final_pred - trial_pred) ** 2)) / y_scale)
            val_nmse_effect = float((trial_metrics.val_nmse - final_metrics.val_nmse) / max(abs(final_metrics.val_nmse), 1.0e-15))
            tail_effect = float((trial_metrics.val_p99 - final_metrics.val_p99) / max(abs(final_metrics.val_p99), 1.0e-15))
            measurable = bool(
                prediction_effect >= self.config.structure_ablation_min_effect
                or val_nmse_effect >= self.config.structure_ablation_min_effect
                or tail_effect >= self.config.structure_ablation_min_effect
            )
            return {
                "measurable": measurable,
                "ablated_expression": trial_dag.expression,
                "prediction_effect": prediction_effect,
                "relative_val_nmse_degradation": val_nmse_effect,
                "relative_val_p99_degradation": tail_effect,
                "complexity_delta": float(trial_metrics.complexity - final_metrics.complexity),
            }
        except Exception as exc:
            return {"measurable": False, "reason": "ablation_failure", "error": repr(exc)}

    def _structure_retention_evidence(
        self,
        parent: EquationState,
        proposal_dag: EquationDAG,
        final_dag: EquationDAG,
        proposal_edit: Mapping[str, Any],
        final_metrics: MetricVector,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        parent_terms = self._top_level_term_entries(parent.dag.expression)
        proposal_terms = self._top_level_term_entries(proposal_dag.expression)
        final_terms = self._top_level_term_entries(final_dag.expression)
        proposal_parent_matches, added_proposal_indices, _ = self._semantic_matches(proposal_terms, parent_terms, X_train)
        parent_proposal_matches, deleted_parent_indices, _ = self._semantic_matches(parent_terms, proposal_terms, X_train)
        added_terms = [proposal_terms[index] for index in added_proposal_indices]
        deleted_terms = [parent_terms[index] for index in deleted_parent_indices]
        added_final_matches, unmatched_added, _ = self._semantic_matches(added_terms, final_terms, X_train)
        deleted_final_matches, _, _ = self._semantic_matches(deleted_terms, final_terms, X_train)
        matched_final_indices = sorted({int(row["target_index"]) for row in added_final_matches})
        ablations: list[dict[str, Any]] = []
        for offset, final_index in enumerate(matched_final_indices[:6]):
            entry = final_terms[final_index]
            evidence = self._ablation_effect(
                final_dag, final_metrics, int(entry["term_index"]),
                X_train, y_train, X_val, y_val, offset,
            )
            ablations.append({"final_term": entry, **evidence})
        retained_added = bool(added_terms and added_final_matches and any(row.get("measurable") for row in ablations))
        retained_deletion = bool(deleted_terms and not deleted_final_matches)
        try:
            parent_pred = self.runtime.predict(parent.dag.expression, X_val)
            final_pred = self.runtime.predict(final_dag.expression, X_val)
            scale = max(float(np.sqrt(np.mean(np.asarray(y_val, dtype=float) ** 2))), float(np.std(y_val)), 1.0e-12)
            parent_final_prediction_effect = float(np.sqrt(np.mean((parent_pred - final_pred) ** 2)) / scale)
        except Exception:
            parent_final_prediction_effect = 0.0
        actual_final_edit = parent.dag.diff(final_dag)
        reparameterized = bool(
            actual_final_edit.get("action") == "REPARAMETERIZE"
            and parent_final_prediction_effect >= self.config.structure_ablation_min_effect
        )
        deletion_effective = bool(
            retained_deletion
            and (
                final_metrics.complexity < parent.metrics.complexity - 1.0e-12
                or parent_final_prediction_effect >= self.config.structure_ablation_min_effect
            )
        )
        passed = bool(retained_added or deletion_effective or reparameterized)
        return {
            "pass": passed,
            "actual_final_action": actual_final_edit.get("action"),
            "proposal_action": proposal_edit.get("action"),
            "parent_proposal_semantic_matches": proposal_parent_matches,
            "parent_proposal_reverse_matches": parent_proposal_matches,
            "added_proposal_terms": added_terms,
            "deleted_parent_terms": deleted_terms,
            "added_final_matches": added_final_matches,
            "unmatched_added_term_indices": unmatched_added,
            "deleted_terms_still_present": deleted_final_matches,
            "ablation_evidence": ablations,
            "retained_nontrivial_added_structure": retained_added,
            "retained_deletion": retained_deletion,
            "deletion_effective": deletion_effective,
            "reparameterization_effective": reparameterized,
            "parent_final_prediction_effect": parent_final_prediction_effect,
            "semantic_match_policy": {
                "correlation_threshold": self.config.structure_similarity_correlation,
                "projection_nrmse_threshold": self.config.structure_similarity_nrmse,
                "ablation_min_effect": self.config.structure_ablation_min_effect,
            },
        }

    @staticmethod
    def _declared_action_matches(declared: str, actual: str) -> bool:
        action_aliases = {
            "ADD": {"ADD"},
            "DELETE": {"DELETE"},
            "REPLACE": {"REPLACE", "CHANGE_OPERATOR", "CHANGE_INTERACTION"},
            "REPARAMETERIZE": {"REPARAMETERIZE"},
            "CHANGE_OPERATOR": {"CHANGE_OPERATOR"},
            "CHANGE_INTERACTION": {"CHANGE_INTERACTION"},
        }
        return actual in action_aliases.get(str(declared).upper(), set())

    def _check_proposal(
        self, normalized: str, parent: Optional[EquationState], proposal: Optional[ProposalCandidate],
        origin: str, source: str, island: str, round_id: int,
    ) -> Optional[tuple[EquationDAG, dict[str, Any], bool]]:
        proposal_dag = self.runtime.dag(normalized)
        if parent is None:
            return proposal_dag, {}, True
        proposal_edit = parent.dag.diff(proposal_dag)
        candidate_id = proposal.candidate_id if proposal else ""
        if not proposal_edit.get("retained_llm_edit"):
            reason = "proposal_noop" if origin == "llm" else "deterministic_refinement_noop"
            self._reject(reason, source=source, island=island, round_id=round_id, candidate_id=candidate_id)
            return None
        if origin != "llm":
            return proposal_dag, proposal_edit, True
        if proposal is None:
            self._reject("missing_real_llm_proposal", source=source, island=island, round_id=round_id)
            return None
        if proposal.parent_hash != parent.dag.canonical_hash:
            self._reject(
                "proposal_parent_hash_mismatch", source=source, island=island, round_id=round_id,
                candidate_id=proposal.candidate_id, expected=parent.dag.canonical_hash,
                observed=proposal.parent_hash,
            )
            return None
        action_match = self._declared_action_matches(proposal.action, str(proposal_edit.get("action")))
        if not action_match:
            self._warn(
                "declared_action_mismatch", source=source, island=island, round_id=round_id,
                candidate_id=proposal.candidate_id, declared_action=proposal.action,
                actual_proposal_action=proposal_edit.get("action"), proposal_edit=proposal_edit,
                enforcement="warning_only_actual_dag_action_authoritative",
            )
        return proposal_dag, proposal_edit, action_match

    def _refit_and_validate(
        self, normalized: str, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, parent: Optional[EquationState],
        proposal: Optional[ProposalCandidate], proposal_dag: EquationDAG,
        proposal_edit: Mapping[str, Any], origin: str, source: str, island: str, round_id: int,
    ) -> Optional[tuple[RefitResult, EquationDAG, MetricVector, dict[str, Any], dict[str, Any]]]:
        refit = self.runtime.refit_global_constants(
            normalized, X_train, y_train, ridge=self.config.ridge_alpha,
            maxiter=self.config.optimizer_maxiter,
            random_seed=self.config.random_seed + 1009 * int(round_id),
        )
        self.budget.record("parameter_refits")
        dag = self.runtime.dag(refit.expression)
        metrics = self.evaluator.evaluate(dag.expression, X_train, y_train, X_val, y_val)
        self.budget.record("multi_domain_metric_evaluations")
        candidate_id = proposal.candidate_id if proposal else ""
        if not metrics.finite:
            self._reject(
                "nonfinite_multi_domain_metrics", source=source, island=island,
                round_id=round_id, candidate_id=candidate_id,
            )
            return None
        dag, metrics, pruning = self._prune_refitted_candidate(
            dag, metrics, X_train, y_train, X_val, y_val, round_id,
        )
        if dag.complexity > self.config.max_complexity:
            self._reject(
                "complexity_cap", source=source, island=island, round_id=round_id,
                candidate_id=candidate_id, complexity=dag.complexity,
                max_complexity=self.config.max_complexity, post_refit_pruning=pruning,
            )
            return None
        retention: dict[str, Any] = {}
        if parent is None:
            return refit, dag, metrics, pruning, retention
        if not parent.dag.diff(dag).get("retained_llm_edit"):
            reason = "refitted_candidate_noop" if origin == "llm" else "deterministic_refit_noop"
            self._reject(
                reason, source=source, island=island, round_id=round_id,
                candidate_id=candidate_id, post_refit_pruning=pruning,
            )
            return None
        if origin == "llm" and proposal is not None:
            retention = self._structure_retention_evidence(
                parent, proposal_dag, dag, proposal_edit, metrics, X_train, y_train, X_val, y_val,
            )
            if not retention.get("pass"):
                self._reject(
                    "llm_structure_not_retained_after_global_refit", source=source,
                    island=island, round_id=round_id, candidate_id=proposal.candidate_id,
                    proposal_expression=proposal_dag.expression, refitted_expression=dag.expression,
                    proposal_edit=proposal_edit, semantic_retention=retention,
                    post_refit_pruning=pruning,
                )
                return None
        return refit, dag, metrics, pruning, retention

    def _lineage_payload(
        self, parent: Optional[EquationState], proposal: Optional[ProposalCandidate],
        origin: str, source: str, island: str, round_id: int, dag: EquationDAG,
        metrics: MetricVector, proposal_edit: Mapping[str, Any], action_match: bool,
        retention: Mapping[str, Any], refit_payload: Mapping[str, Any],
    ) -> Optional[tuple[tuple[LineageStep, ...], str, str, dict[str, Any]]]:
        edit = parent.dag.diff(dag) if parent is not None else {}
        parent_hash = parent.dag.canonical_hash if parent is not None else ""
        if parent is None or proposal is None or origin != "llm":
            return tuple(), "", parent_hash, edit
        lineage_id = str(proposal.lineage_id or "")
        if not lineage_id:
            self._reject(
                "missing_lineage_id", source=source, island=island,
                round_id=round_id, candidate_id=proposal.candidate_id,
            )
            return None
        step = LineageStep(
            origin="llm", round_id=round_id, island=island, lineage_id=lineage_id,
            parent_lineage_id=parent.lineage_id, parent_hash=parent_hash,
            candidate_id=proposal.candidate_id, proposal_expression=proposal.equation,
            before_expression=parent.dag.expression, after_expression=dag.expression,
            declared_action=proposal.action,
            actual_action=str(edit.get("action") or proposal_edit.get("action") or ""),
            rationale=proposal.rationale, prompt_hash=proposal.prompt_hash,
            response_hash=proposal.response_hash, edit=dict(edit),
            before_metrics=parent.metrics.as_dict(), after_metrics=metrics.as_dict(),
            refit=dict(refit_payload),
            retention={**retention, "declared_action_matches_actual": action_match},
        )
        return tuple(parent.lineage) + (step,), lineage_id, parent_hash, edit

    def _make_hypothesis(
        self, expression: str, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, source: str, origin: str,
        island: str, round_id: int, parent: Optional[EquationState] = None,
        proposal: Optional[ProposalCandidate] = None,
    ) -> Optional[EquationState]:
        category = {
            "llm": "llm_candidate",
            "deterministic": "anchor_candidate" if parent is None else "deterministic_candidate",
        }.get(str(origin), "other_candidate")
        if not self.budget.consume(category):
            self._reject(
                "evaluation_budget_exhausted", source=source, island=island, round_id=round_id,
                candidate_id=(proposal.candidate_id if proposal else ""), budget=self.budget.snapshot(),
            )
            return None
        try:
            normalized = self.registry.normalize(expression)
            checked = self._check_proposal(normalized, parent, proposal, origin, source, island, round_id)
            if checked is None:
                return None
            proposal_dag, proposal_edit, action_match = checked
            fitted = self._refit_and_validate(
                normalized, X_train, y_train, X_val, y_val, parent, proposal,
                proposal_dag, proposal_edit, origin, source, island, round_id,
            )
            if fitted is None:
                return None
            refit, dag, metrics, pruning, retention = fitted
        except Exception as exc:
            self._reject(
                "candidate_build_exception", source=source, island=island, round_id=round_id,
                candidate_id=(proposal.candidate_id if proposal else ""), error=repr(exc),
            )
            return None
        refit_payload = {**dataclasses.asdict(refit), "post_refit_pruning": pruning}
        lineage_result = self._lineage_payload(
            parent, proposal, origin, source, island, round_id, dag, metrics,
            proposal_edit, action_match, retention, refit_payload,
        )
        if lineage_result is None:
            return None
        lineage, lineage_id, parent_hash, edit = lineage_result
        return EquationState(
            dag=dag, metrics=metrics, source=source, origin=origin,
            island=island, round_id=round_id, lineage_id=lineage_id,
            parent_hash=parent_hash, proposal_expression=(proposal.equation if proposal else expression),
            declared_action=(proposal.action if proposal else ""), rationale=(proposal.rationale if proposal else ""),
            candidate_id=(proposal.candidate_id if proposal else ""),
            prompt_hash=(proposal.prompt_hash if proposal else ""), response_hash=(proposal.response_hash if proposal else ""),
            edit=edit, refit=refit_payload, lineage=lineage,
        )


    def build_state(
        self, expression: str, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, *, source: str, origin: str,
        island: str, round_id: int, parent: Optional[EquationState] = None,
        proposal: Optional[ProposalCandidate] = None,
    ) -> Optional[EquationState]:
        return self._make_hypothesis(
            expression, X_train, y_train, X_val, y_val, source, origin, island, round_id,
            parent=parent, proposal=proposal,
        )

    def seed_survivor(
        self, base_candidates: Sequence[Any], X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> tuple[EquationState, list[EquationState]]:
        candidates: list[EquationState] = []
        for seed in base_candidates:
            if self.budget.exhausted:
                break
            if isinstance(seed, str):
                expression, source = seed, "deterministic_seed"
            elif isinstance(seed, Mapping):
                expression = str(seed.get("expression") or seed.get("program") or seed.get("equation") or "")
                source = str(seed.get("source") or "deterministic_seed")
            else:
                expression = str(getattr(seed, "expression", ""))
                source = str(getattr(seed, "source", "deterministic_seed"))
            if not expression.strip():
                continue
            state = self.build_state(expression, X_train, y_train, X_val, y_val, source=source, origin="deterministic", island="anchor", round_id=0)
            if state is not None:
                candidates.append(state)
        if not candidates:
            if self.budget.exhausted:
                raise RuntimeError("evaluation_budget_exhausted_before_valid_anchor")
            specs = self.registry.generic_anchor_basis(X_train)
            expression = self.registry.fit_sparse_linear_expression(
                specs, y_train, keep=min(12, max(3, X_train.shape[1] * 2)), ridge=self.config.ridge_alpha
            )
            state = self.build_state(expression, X_train, y_train, X_val, y_val, source="generic_sparse_anchor", origin="deterministic", island="anchor", round_id=0)
            if state is not None:
                candidates.append(state)
        if not candidates:
            raise RuntimeError("no_valid_deterministic_anchor")
        front = self.policy.pareto_front(candidates)
        survivor = min(front or candidates, key=lambda item: self.policy.score(item, "balanced"))
        return survivor, sorted(candidates, key=lambda item: self.policy.score(item, "balanced"))

    def failure_signature(self, state: EquationState, exploration: Optional[Any] = None) -> tuple[str, ...]:
        metrics = state.metrics
        tags: list[str] = []
        if metrics.val_nmse > 0.05: tags.append("nmse_underfit")
        if metrics.val_strict > 0.1: tags.append("strict_failure")
        if metrics.val_p99 > 0.1: tags.append("tail_failure")
        if metrics.stress_nmse > 1.25 * max(metrics.val_nmse, 1.0e-15) or metrics.stress_p99 > 1.25 * max(metrics.val_p99, 1.0e-15): tags.append("stress_instability")
        if metrics.ood_proxy_nmse > 1.25 * max(metrics.val_nmse, 1.0e-15) or metrics.ood_stability_penalty > 0.0: tags.append("ood_instability")
        if metrics.complexity > 80: tags.append("complexity_pressure")
        if exploration is not None:
            for tag in exploration.failure_summary.get("tags", []):
                if str(tag) not in tags: tags.append(str(tag))
        return tuple(tags or ["unexplained_signal"])
