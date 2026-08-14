from __future__ import annotations

"""Grammar-driven executable exploration-function solver.

The solver learns a complete executable program ``g(y_hat, x)`` by cross-fold
selection over a small compositional grammar.  The grammar is generic: it is
built only from the current prediction, raw inputs, current-DAG intermediates
and the primitive registry.  No benchmark-, task- or equation-specific motifs
are encoded here.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .equation_runtime import EquationDAG, EquationRuntime
if TYPE_CHECKING:
    from .evaluation_runtime import MetricVector


@dataclass(frozen=True)
class GrammarTerm:
    expression: str
    values: np.ndarray
    family: str
    depth: int
    parents: tuple[str, ...] = ()
    operator: str = "identity"
    cv_objective: float = float("inf")


@dataclass
class ExplorationProgram:
    expression: str
    correction_expression: str
    selected_terms: list[dict[str, Any]]
    train_nmse: float
    cv_nmse: float
    identity_cv_nmse: float
    cv_objective: float
    identity_cv_objective: float
    cv_p99: float
    identity_cv_p99: float
    relative_cv_nmse_gain: float
    relative_cv_tail_gain: float
    relative_cv_objective_gain: float
    failure_summary: dict[str, Any]
    grammar_summary: dict[str, Any] = field(default_factory=dict)
    search_trace: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "grammar-executable-g"

    def as_prompt_dict(self, include_search_trace: bool = False) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "executable_program": f"g(y_hat, x) = {self.expression}",
            "diagnostic_correction": f"delta(y_hat, x) = {self.correction_expression}",
            "selected_terms": self.selected_terms,
            "train_nmse": self.train_nmse,
            "cross_fold_nmse": self.cv_nmse,
            "identity_cross_fold_nmse": self.identity_cv_nmse,
            "cross_fold_objective": self.cv_objective,
            "identity_cross_fold_objective": self.identity_cv_objective,
            "cross_fold_p99": self.cv_p99,
            "identity_cross_fold_p99": self.identity_cv_p99,
            "relative_cross_fold_nmse_gain": self.relative_cv_nmse_gain,
            "relative_cross_fold_tail_gain": self.relative_cv_tail_gain,
            "relative_cross_fold_objective_gain": self.relative_cv_objective_gain,
            "failure_summary": self.failure_summary,
            "grammar_summary": self.grammar_summary,
        }
        if include_search_trace:
            payload["search_trace"] = self.search_trace
        return payload

    def as_audit_dict(self) -> dict[str, Any]:
        return self.as_prompt_dict(include_search_trace=True)


class ExplorationRuntime:
    """Solve ``g_t(f_t(x), x)`` with a generic compositional grammar.

    Search proceeds in two stages:
      1. build and beam-screen executable grammar terms by cross-fold loss;
      2. forward-select a sparse linear combination around the refitted
         identity program ``g(y_hat, x)=a*y_hat+b``.

    This is intentionally not residual feature screening: every term is ranked
    by the validation loss of an executable complete exploration program.
    """

    def __init__(self, runtime: EquationRuntime, config: Any, objective_profile: str = "balanced") -> None:
        self.runtime = runtime
        self.config = config
        self.objective_profile = str(objective_profile or "balanced")

    def for_island(self, island: str) -> "ExplorationRuntime":
        return ExplorationRuntime(self.runtime, self.config, objective_profile=str(island))

    @staticmethod
    def _nmse(y: np.ndarray, pred: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).reshape(-1)
        pred = np.asarray(pred, dtype=float).reshape(-1)
        if pred.size != y.size or not np.all(np.isfinite(pred)):
            return float("inf")
        return float(np.mean((pred - y) ** 2) / max(float(np.var(y)), 1.0e-15))

    @staticmethod
    def _relative_p99(y: np.ndarray, pred: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).reshape(-1)
        pred = np.asarray(pred, dtype=float).reshape(-1)
        if pred.size != y.size or not np.all(np.isfinite(pred)):
            return float("inf")
        rel = np.abs(pred - y) / np.maximum(np.abs(y), 1.0e-12)
        return float(np.quantile(rel, 0.99))

    def _objective(self, y: np.ndarray, pred: np.ndarray) -> float:
        nmse = self._nmse(y, pred)
        if not math.isfinite(nmse):
            return float("inf")
        base_tail_weight = max(0.0, float(getattr(self.config, "exploration_tail_weight", 0.08)))
        profile_multiplier = {
            "nmse": 0.25,
            "tail": 3.0,
            "low_complexity": 0.50,
            "novelty": 1.0,
            "balanced": 1.0,
        }.get(self.objective_profile, 1.0)
        tail_weight = base_tail_weight * profile_multiplier
        if tail_weight <= 0.0:
            return nmse
        scale = max(1.0, self._relative_p99(y, np.full_like(y, float(np.mean(y)))))
        return nmse + tail_weight * min(100.0, self._relative_p99(y, pred) / scale)

    @staticmethod
    def _ridge_fit(A: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        reg = max(0.0, float(alpha)) * np.eye(A.shape[1], dtype=float)
        if A.shape[1]:
            reg[0, 0] = 0.0
        try:
            return np.linalg.solve(A.T @ A + reg, A.T @ y)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, y, rcond=None)[0]

    @staticmethod
    def _safe_values(values: np.ndarray) -> Optional[np.ndarray]:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return None
        if float(np.std(values)) <= 1.0e-12:
            return None
        clip = max(1.0e5, float(np.quantile(np.abs(values), 0.999)) * 20.0)
        return np.clip(values, -clip, clip)

    @staticmethod
    def _fingerprint(values: np.ndarray) -> tuple[int, ...]:
        values = np.asarray(values, dtype=float).reshape(-1)
        sample = values if values.size <= 64 else values[np.linspace(0, values.size - 1, 64, dtype=int)]
        scale = max(float(np.std(sample)), 1.0e-12)
        normalized = np.clip((sample - float(np.mean(sample))) / scale, -8.0, 8.0)
        return tuple(np.rint(normalized * 1000.0).astype(np.int64).tolist())

    def _folds(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        folds = min(max(2, int(getattr(self.config, "exploration_cv_folds", 4))), max(2, n // 4))
        rng = np.random.default_rng(int(getattr(self.config, "random_seed", 0)) + 9176 * n)
        parts = [part for part in np.array_split(rng.permutation(n), folds) if len(part)]
        all_idx = np.arange(n, dtype=int)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for val_idx in parts:
            mask = np.ones(n, dtype=bool)
            mask[val_idx] = False
            train_idx = all_idx[mask]
            if len(train_idx) >= 2:
                out.append((train_idx, val_idx))
        return out

    def _fit_predict(
        self,
        terms: Sequence[GrammarTerm],
        selected: Sequence[int],
        y: np.ndarray,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
    ) -> np.ndarray:
        if not selected:
            return np.full(len(eval_idx), float(np.mean(y[train_idx])))
        B_train = np.column_stack([terms[j].values[train_idx] for j in selected])
        mean = np.mean(B_train, axis=0)
        scale = np.maximum(np.std(B_train, axis=0), 1.0e-12)
        A_train = np.column_stack([np.ones(len(train_idx)), (B_train - mean) / scale])
        coef = self._ridge_fit(A_train, y[train_idx], float(getattr(self.config, "ridge_alpha", 1.0e-8)))
        B_eval = np.column_stack([terms[j].values[eval_idx] for j in selected])
        return coef[0] + ((B_eval - mean) / scale) @ coef[1:]

    def _cv_score(self, terms: Sequence[GrammarTerm], selected: Sequence[int], y: np.ndarray) -> float:
        scores: list[float] = []
        for train_idx, val_idx in self._folds(len(y)):
            try:
                pred = self._fit_predict(terms, selected, y, train_idx, val_idx)
            except Exception:
                return float("inf")
            scores.append(self._objective(y[val_idx], pred))
        return float(np.mean(scores)) if scores else float("inf")

    def _cv_nmse(self, terms: Sequence[GrammarTerm], selected: Sequence[int], y: np.ndarray) -> float:
        scores: list[float] = []
        for train_idx, val_idx in self._folds(len(y)):
            try:
                pred = self._fit_predict(terms, selected, y, train_idx, val_idx)
            except Exception:
                return float("inf")
            scores.append(self._nmse(y[val_idx], pred))
        return float(np.mean(scores)) if scores else float("inf")

    def _cv_p99(self, terms: Sequence[GrammarTerm], selected: Sequence[int], y: np.ndarray) -> float:
        scores: list[float] = []
        for train_idx, val_idx in self._folds(len(y)):
            try:
                pred = self._fit_predict(terms, selected, y, train_idx, val_idx)
            except Exception:
                return float("inf")
            scores.append(self._relative_p99(y[val_idx], pred))
        return float(np.mean(scores)) if scores else float("inf")

    def _full_fit(
        self,
        terms: Sequence[GrammarTerm],
        selected: Sequence[int],
        y: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        idx = np.arange(len(y), dtype=int)
        if not selected:
            intercept = float(np.mean(y))
            return intercept, np.array([], dtype=float), np.full(len(y), intercept)
        B = np.column_stack([terms[j].values for j in selected])
        mean = np.mean(B, axis=0)
        scale = np.maximum(np.std(B, axis=0), 1.0e-12)
        A = np.column_stack([np.ones(len(y)), (B - mean) / scale])
        coef_z = self._ridge_fit(A, y, float(getattr(self.config, "ridge_alpha", 1.0e-8)))
        coef = coef_z[1:] / scale
        intercept = float(coef_z[0] - np.dot(coef, mean))
        return intercept, coef, intercept + B @ coef

    def _add_term(
        self,
        out: list[GrammarTerm],
        seen_expr: set[str],
        seen_values: set[tuple[int, ...]],
        expression: str,
        values: np.ndarray,
        family: str,
        depth: int,
        parents: tuple[str, ...] = (),
        operator: str = "identity",
    ) -> None:
        if expression in seen_expr:
            return
        safe = self._safe_values(values)
        if safe is None:
            return
        fingerprint = self._fingerprint(safe)
        if fingerprint in seen_values:
            return
        seen_expr.add(expression)
        seen_values.add(fingerprint)
        out.append(GrammarTerm(expression, safe, family, depth, parents, operator))

    def _base_terms(self, X: np.ndarray, y_hat: np.ndarray, dag: EquationDAG) -> list[GrammarTerm]:
        out: list[GrammarTerm] = []
        seen_expr: set[str] = set()
        seen_values: set[tuple[int, ...]] = set()
        self._add_term(out, seen_expr, seen_values, "y_hat", y_hat, "current_prediction", 0)
        for i in range(X.shape[1]):
            self._add_term(out, seen_expr, seen_values, f"x{i}", X[:, i], "raw_variable", 0)
        for text, values in self.runtime.intermediate_outputs(
            dag, X, int(getattr(self.config, "intermediate_node_limit", 20))
        ):
            self._add_term(
                out, seen_expr, seen_values, f"({text})", values,
                "current_dag_intermediate", 0,
            )
        return out

    def _expand_unary(self, seeds: Sequence[GrammarTerm]) -> list[tuple[str, np.ndarray, str, str]]:
        expanded: list[tuple[str, np.ndarray, str, str]] = []
        for term in seeds:
            v, e = term.values, term.expression
            expanded.extend([
                (f"({e})**2", v * v, "nonlinear_transform", "square"),
                (f"({e})**3", v * v * v, "nonlinear_transform", "cube"),
                (f"tanh({e})", np.tanh(v), "nonlinear_transform", "tanh"),
                (f"sin({e})", np.sin(np.clip(v, -1.0e4, 1.0e4)), "nonlinear_transform", "sin"),
                (f"cos({e})", np.cos(np.clip(v, -1.0e4, 1.0e4)), "nonlinear_transform", "cos"),
                (f"sign({e})*log(1+Abs({e}))", np.sign(v) * np.log1p(np.abs(v)), "nonlinear_transform", "signed_log"),
                (f"({e})/(1+Abs({e}))", v / (1.0 + np.abs(v)), "nonlinear_transform", "saturate"),
            ])
        return expanded

    def _expand_binary(self, seeds: Sequence[GrammarTerm]) -> list[tuple[str, np.ndarray, str, tuple[str, str], str]]:
        expanded: list[tuple[str, np.ndarray, str, tuple[str, str], str]] = []
        for i, left in enumerate(seeds):
            for j, right in enumerate(seeds):
                if i > j:
                    continue
                a, b = left.values, right.values
                ae, be = left.expression, right.expression
                expanded.append((f"({ae})*({be})", a * b, "interaction", (ae, be), "multiply"))
                if i != j:
                    expanded.append((f"({ae})/(1+Abs({be}))", a / (1.0 + np.abs(b)), "conditioned_transform", (ae, be), "safe_ratio"))
                    expanded.append((f"({be})/(1+Abs({ae}))", b / (1.0 + np.abs(a)), "conditioned_transform", (be, ae), "safe_ratio"))
        return expanded

    def _build_grammar(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, dag: EquationDAG) -> tuple[list[GrammarTerm], list[dict[str, Any]]]:
        max_terms = max(16, int(getattr(self.config, "exploration_max_primitives", 128)))
        beam_width = max(4, int(getattr(self.config, "exploration_beam_width", 16)))
        max_depth = max(1, int(getattr(self.config, "exploration_max_depth", 2)))
        terms = self._base_terms(X, y_hat, dag)
        seen_expr = {term.expression for term in terms}
        seen_values = {self._fingerprint(term.values) for term in terms}
        trace: list[dict[str, Any]] = []

        identity_idx = next(i for i, term in enumerate(terms) if term.expression == "y_hat")
        base_scores: list[tuple[float, int]] = []
        for idx, term in enumerate(terms):
            score = self._cv_score(terms, [identity_idx] if idx == identity_idx else [identity_idx, idx], y)
            base_scores.append((score, idx))
        base_scores.sort(key=lambda item: item[0])
        beam = [terms[idx] for _, idx in base_scores[:beam_width]]
        trace.append({"depth": 0, "generated": len(terms), "beam": [t.expression for t in beam]})

        for depth in range(1, max_depth + 1):
            additions: list[GrammarTerm] = []
            for expr, values, family, op in self._expand_unary(beam):
                self._add_term(additions, seen_expr, seen_values, expr, values, family, depth, operator=op)
                if len(terms) + len(additions) >= max_terms:
                    break
            if len(terms) + len(additions) < max_terms:
                binary_seeds = beam[:max(2, min(len(beam), beam_width // 2))]
                for expr, values, family, parents, op in self._expand_binary(binary_seeds):
                    self._add_term(additions, seen_expr, seen_values, expr, values, family, depth, parents, op)
                    if len(terms) + len(additions) >= max_terms:
                        break
            if not additions:
                break
            terms.extend(additions[: max(0, max_terms - len(terms))])
            scored: list[tuple[float, GrammarTerm]] = []
            identity_idx = next(i for i, term in enumerate(terms) if term.expression == "y_hat")
            for idx in range(max(0, len(terms) - len(additions)), len(terms)):
                score = self._cv_score(terms, [identity_idx, idx], y)
                term = terms[idx]
                scored.append((score, GrammarTerm(
                    term.expression, term.values, term.family, term.depth,
                    term.parents, term.operator, score,
                )))
            scored.sort(key=lambda item: item[0])
            beam = [term for _, term in scored[:beam_width]]
            trace.append({
                "depth": depth,
                "generated": len(additions),
                "total_terms": len(terms),
                "beam": [{"expression": term.expression, "cv_objective": score} for score, term in scored[:beam_width]],
            })
            if len(terms) >= max_terms:
                break
        return terms, trace

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        if len(a) < 3 or len(a) != len(b) or float(np.std(a)) <= 1.0e-14 or float(np.std(b)) <= 1.0e-14:
            return 0.0
        value = float(np.corrcoef(a, b)[0, 1])
        return value if math.isfinite(value) else 0.0

    def _select_terms(
        self, terms: Sequence[GrammarTerm], y: np.ndarray
    ) -> tuple[list[int], dict[str, float], list[dict[str, Any]]]:
        identity_index = next(i for i, term in enumerate(terms) if term.expression == "y_hat")
        selected = [identity_index]
        identity = {
            "objective": self._cv_score(terms, selected, y),
            "nmse": self._cv_nmse(terms, selected, y),
            "p99": self._cv_p99(terms, selected, y),
        }
        best_cv = identity["objective"]
        max_selected = max(1, int(getattr(self.config, "exploration_max_terms", 6)))
        min_gain = max(0.0, float(getattr(self.config, "exploration_min_cv_gain", 1.0e-4)))
        univariate = [
            (self._cv_score(terms, [identity_index, idx], y), idx)
            for idx in range(len(terms)) if idx != identity_index
        ]
        univariate.sort(key=lambda item: item[0])
        beam_width = int(getattr(self.config, "exploration_beam_width", 16))
        pool = [idx for _, idx in univariate[:max(8, min(len(univariate), beam_width * 2))]]
        trace: list[dict[str, Any]] = []
        while len(selected) < max_selected:
            scored = [
                (self._cv_score(terms, [*selected, idx], y), idx)
                for idx in pool if idx not in selected
            ]
            if not scored:
                break
            winner_score, winner = min(scored)
            gain = (best_cv - winner_score) / max(abs(best_cv), 1.0e-15)
            if gain < min_gain:
                break
            selected.append(winner)
            trace.append({
                "added_expression": terms[winner].expression,
                "before_cv_objective": best_cv,
                "after_cv_objective": winner_score,
                "relative_gain": gain,
            })
            best_cv = winner_score
        return selected, identity, trace

    def _render_fit(
        self, terms: Sequence[GrammarTerm], selected: Sequence[int],
        y: np.ndarray, y_hat: np.ndarray,
    ) -> tuple[str, list[dict[str, Any]], float, float, np.ndarray, np.ndarray]:
        intercept, coefficients, pred = self._full_fit(terms, selected, y)
        residual = y - y_hat
        pieces = [f"({intercept:.16g})"] if abs(intercept) > 1.0e-14 else []
        rows: list[dict[str, Any]] = []
        y_hat_coef = 0.0
        cap = max(1.0, float(getattr(self.config, "max_abs_coefficient", 1.0e4)))
        for coefficient, idx in zip(coefficients, selected):
            term = terms[idx]
            value = float(np.clip(coefficient, -cap, cap))
            if abs(value) <= 1.0e-14:
                continue
            pieces.append(f"({value:.16g})*({term.expression})")
            y_hat_coef = value if term.expression == "y_hat" else y_hat_coef
            rows.append({
                "expression": term.expression, "family": term.family,
                "operator": term.operator, "depth": term.depth,
                "parents": list(term.parents), "coefficient": value,
                "residual_correlation": self._corr(term.values, residual),
            })
        return " + ".join(pieces) if pieces else "0.0", rows, y_hat_coef, intercept, pred, residual

    @staticmethod
    def _suggest_actions(
        rows: Sequence[dict[str, Any]], y_hat_coef: float, intercept: float, n_features: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        remaining = [row for row in rows if row["expression"] != "y_hat"]
        actions: list[str] = []
        if abs(y_hat_coef) < 0.20:
            actions.append("REPLACE")
        elif abs(y_hat_coef - 1.0) > 0.10 or abs(intercept) > 1.0e-10:
            actions.append("REPARAMETERIZE")
        if any(row["family"] == "raw_variable" for row in remaining):
            actions.append("ADD")
        if any(row["family"] == "nonlinear_transform" and "y_hat" in row["expression"] for row in remaining):
            actions.extend(["CHANGE_OPERATOR", "REPLACE"])
        if any("y_hat" in row["expression"] and any(f"x{i}" in row["expression"] for i in range(n_features)) for row in remaining):
            actions.extend(["CHANGE_INTERACTION", "ADD"])
        if any(row["family"] == "current_dag_intermediate" for row in remaining):
            actions.extend(["DELETE", "REPLACE"])
        return list(dict.fromkeys(actions or (["ADD"] if remaining else ["REPARAMETERIZE"]))), remaining

    def _failure_summary(
        self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, residual: np.ndarray,
        remaining: Sequence[dict[str, Any]], actions: Sequence[str], y_hat_coef: float,
        current_metrics: MetricVector, identity: Mapping[str, float], observed: Mapping[str, float],
    ) -> dict[str, Any]:
        scale = max(float(np.quantile(np.abs(y), 0.95)), math.sqrt(max(float(np.var(y)), 1.0e-15)), 1.0e-12)
        abs_error = np.abs(residual)
        worst_count = min(len(y), max(4, int(math.ceil(0.05 * len(y)))))
        worst_idx = np.argsort(abs_error)[-worst_count:]
        correlations = [
            {"variable": f"x{i}", "residual_correlation": self._corr(X[:, i], residual)}
            for i in range(X.shape[1])
        ]
        correlations.sort(key=lambda row: abs(float(row["residual_correlation"])), reverse=True)
        interactions = any(
            "y_hat" in row["expression"] and any(f"x{i}" in row["expression"] for i in range(X.shape[1]))
            for row in remaining
        )
        prediction_nonlinearity = any(
            row["family"] == "nonlinear_transform" and "y_hat" in row["expression"] for row in remaining
        )
        tags = []
        if current_metrics.val_nmse > 0.05: tags.append("nmse_underfit")
        if current_metrics.val_p99 > max(0.1, 2.0 * math.sqrt(max(current_metrics.val_nmse, 0.0))): tags.append("tail_failure")
        if interactions: tags.append("missing_prediction_input_interaction")
        if prediction_nonlinearity: tags.append("missing_prediction_nonlinearity")
        if abs(float(np.mean(residual))) > 0.05 * scale: tags.append("systematic_bias")
        return {
            "tags": tags or ["unexplained_signal"], "suggested_structural_actions": list(actions),
            "identity_coefficient_after_refit": y_hat_coef,
            "identity_cross_fold_nmse": identity["nmse"], "g_cross_fold_nmse": observed["nmse"],
            "identity_cross_fold_objective": identity["objective"], "g_cross_fold_objective": observed["objective"],
            "identity_cross_fold_p99": identity["p99"], "g_cross_fold_p99": observed["p99"],
            "g_relative_cross_fold_nmse_gain": observed["nmse_gain"],
            "g_relative_cross_fold_tail_gain": observed["tail_gain"],
            "g_relative_cross_fold_objective_gain": observed["objective_gain"],
            "residual_mean_scaled": float(np.mean(residual) / scale), "residual_std_scaled": float(np.std(residual) / scale),
            "worst_slice_fraction": float(worst_count / max(1, len(y))),
            "worst_slice_y_range": [float(np.min(y[worst_idx])), float(np.max(y[worst_idx]))],
            "worst_slice_prediction_range": [float(np.min(y_hat[worst_idx])), float(np.max(y_hat[worst_idx]))],
            "worst_slice_error_p99_scaled": float(np.quantile(abs_error[worst_idx], 0.99) / scale),
            "top_variable_residual_correlations": correlations[:min(8, len(correlations))],
            "current_metrics": current_metrics.as_dict(),
            "g_improves_cross_fold_objective": bool(observed["objective"] + 1.0e-12 < identity["objective"]),
        }

    def solve(
        self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray,
        dag: EquationDAG, current_metrics: MetricVector,
    ) -> ExplorationProgram:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
        terms, trace = self._build_grammar(X, y, y_hat, dag)
        selected, identity, selection_trace = self._select_terms(terms, y)
        trace.append({"forward_selection": selection_trace})
        expression, rows, y_hat_coef, intercept, pred, residual = self._render_fit(terms, selected, y, y_hat)
        observed = {
            "objective": self._cv_score(terms, selected, y),
            "nmse": self._cv_nmse(terms, selected, y),
            "p99": self._cv_p99(terms, selected, y),
        }
        observed.update({
            "objective_gain": (identity["objective"] - observed["objective"]) / max(abs(identity["objective"]), 1.0e-15),
            "nmse_gain": (identity["nmse"] - observed["nmse"]) / max(abs(identity["nmse"]), 1.0e-15),
            "tail_gain": (identity["p99"] - observed["p99"]) / max(abs(identity["p99"]), 1.0e-15),
        })
        actions, remaining = self._suggest_actions(rows, y_hat_coef, intercept, X.shape[1])
        failure = self._failure_summary(
            X, y, y_hat, residual, remaining, actions, y_hat_coef, current_metrics, identity, observed,
        )
        return ExplorationProgram(
            expression=expression, correction_expression=f"({expression}) - y_hat",
            selected_terms=rows, train_nmse=self._nmse(y, pred),
            cv_nmse=observed["nmse"], identity_cv_nmse=identity["nmse"],
            cv_objective=observed["objective"], identity_cv_objective=identity["objective"],
            cv_p99=observed["p99"], identity_cv_p99=identity["p99"],
            relative_cv_nmse_gain=observed["nmse_gain"], relative_cv_tail_gain=observed["tail_gain"],
            relative_cv_objective_gain=observed["objective_gain"], failure_summary=failure,
            grammar_summary={
                "base_symbols": ["y_hat", *[f"x{i}" for i in range(X.shape[1])]],
                "uses_current_dag_intermediates": bool(int(getattr(self.config, "intermediate_node_limit", 20))),
                "unary_operators": ["square", "cube", "tanh", "sin", "cos", "signed_log", "saturate"],
                "binary_operators": ["multiply", "safe_ratio"],
                "max_depth": int(getattr(self.config, "exploration_max_depth", 2)),
                "beam_width": int(getattr(self.config, "exploration_beam_width", 16)),
                "candidate_count": len(terms), "selection_method": "cross_fold_complete-program-objective",
                "task_specific_templates": False,
            },
            search_trace=trace,
        )
