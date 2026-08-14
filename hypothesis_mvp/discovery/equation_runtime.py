from __future__ import annotations

"""EquationRuntime: the sole owner of equation syntax, DAGs and numerical refit.

No evaluation policy, provider behavior, controller state or learned experience
is allowed in this module.
"""

import dataclasses
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import sympy as sp

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

def sha256_text(value: Any, length: int = 24) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()[:length]


def finite_float(value: Any, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


class PrimitiveRegistry:
    """Single grammar registry. Learned experience is never added here."""

    ALLOWED_FUNCTIONS = {
        "Abs", "abs", "sign", "sin", "cos", "tan", "tanh", "exp", "log",
        "sqrt", "cbrt", "abspow", "signpow", "sigmoid",
    }

    def __init__(self, max_expression_chars: int = 24000) -> None:
        self.max_expression_chars = int(max_expression_chars)
        self.sympy_locals: dict[str, Any] = {
            "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tanh": sp.tanh,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "cbrt": sp.Function("cbrt"), "abspow": sp.Function("abspow"),
            "signpow": sp.Function("signpow"), "sigmoid": sp.Function("sigmoid"),
        }
        self.numpy_modules: dict[str, Callable[..., Any]] = {
            "Abs": np.abs, "abs": np.abs, "sign": np.sign,
            "cbrt": lambda x: np.cbrt(np.asarray(x, dtype=float)),
            "abspow": self._abspow, "signpow": self._signpow,
            "sigmoid": self._sigmoid,
        }

    @staticmethod
    def _abspow(x: Any, exponent: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        p = float(np.clip(float(np.asarray(exponent).reshape(-1)[0]), -8.0, 8.0))
        base = np.maximum(np.abs(arr), 1.0e-12) if p < 0 else np.abs(arr)
        return np.power(base, p)

    @classmethod
    def _signpow(cls, x: Any, exponent: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return np.sign(arr) * cls._abspow(arr, exponent)

    @staticmethod
    def _sigmoid(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))

    def normalize(self, expression: str) -> str:
        text = str(expression or "").strip().replace("^", "**")
        text = text.replace("np.", "").replace("math.", "")
        text = re.sub(r"\babs\s*\(", "Abs(", text)
        text = re.sub(r"\s+", " ", text)
        if not text:
            raise ValueError("empty_expression")
        if len(text) > self.max_expression_chars:
            raise ValueError("expression_too_long")
        lowered = text.lower()
        forbidden = ("__", "import", "lambda", "exec", "eval", "open(", "compile(", "globals(", "locals(", "subprocess", "os.", "sys.")
        if any(token in lowered for token in forbidden):
            raise ValueError("unsafe_expression")
        return text

    def parse(self, expression: str, n_features: int, evaluate: bool = False) -> sp.Expr:
        text = self.normalize(expression)
        symbols = {f"x{i}": sp.Symbol(f"x{i}", real=True) for i in range(int(n_features))}
        local = dict(self.sympy_locals)
        local.update(symbols)
        expr = sp.sympify(text, locals=local, evaluate=evaluate)
        unknown = sorted(str(s) for s in expr.free_symbols if str(s) not in symbols)
        if unknown:
            raise ValueError("unknown_symbols:" + ",".join(unknown))
        bad = set()
        for atom in expr.atoms(sp.Function):
            name = str(getattr(atom.func, "__name__", atom.func))
            if name not in self.ALLOWED_FUNCTIONS:
                bad.add(name)
        if bad:
            raise ValueError("unknown_functions:" + ",".join(sorted(bad)))
        return expr

    def canonical(self, expression: str, n_features: int) -> tuple[sp.Expr, str]:
        expr = self.parse(expression, n_features, evaluate=False)
        try:
            expr = sp.factor_terms(expr)
        except Exception:
            pass
        return expr, sp.sstr(expr, order="lex")

    def lambdify(self, expr: sp.Expr, symbols: Sequence[sp.Symbol]) -> Callable[..., Any]:
        return sp.lambdify(list(symbols), expr, modules=[self.numpy_modules, "numpy"], cse=False)

    def generic_anchor_basis(self, X: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Compact generic initializer grammar, not a learned candidate cache."""
        X = np.asarray(X, dtype=float)
        specs: list[tuple[str, np.ndarray]] = []
        for i in range(X.shape[1]):
            x = X[:, i]
            clipped = np.clip(x, -1.0e4, 1.0e4)
            specs.extend([
                (f"x{i}", x), (f"x{i}**2", x ** 2), (f"x{i}**3", x ** 3),
                (f"sin(x{i})", np.sin(clipped)), (f"cos(x{i})", np.cos(clipped)),
                (f"tanh(x{i})", np.tanh(x)),
                (f"log(1+Abs(x{i}))", np.log1p(np.abs(x))),
                (f"x{i}/(1+Abs(x{i}))", x / (1.0 + np.abs(x))),
            ])
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                specs.extend([
                    (f"x{i}*x{j}", X[:, i] * X[:, j]),
                    (f"x{i}**2*x{j}", X[:, i] ** 2 * X[:, j]),
                    (f"x{i}*x{j}**2", X[:, i] * X[:, j] ** 2),
                ])
        return [(name, np.asarray(v, dtype=float)) for name, v in specs if np.all(np.isfinite(v)) and float(np.std(v)) > 1.0e-12]

    @staticmethod
    def fit_sparse_linear_expression(specs: Sequence[tuple[str, np.ndarray]], y: np.ndarray, keep: int, ridge: float = 1.0e-8) -> str:
        """Fit a compact deterministic anchor using effect-aware screening.

        Ranking raw coefficients is scale-dependent and can discard a term with
        a small coefficient but large prediction effect.  Rank by
        ``abs(coef) * std(feature)`` and refit the selected design globally.
        """
        if not specs:
            return "0.0"
        y = np.asarray(y, dtype=float).reshape(-1)
        values = [np.asarray(v, dtype=float).reshape(-1) for _, v in specs]
        A = np.column_stack([np.ones(len(y))] + values)
        scales = np.maximum(np.linalg.norm(A, axis=0), 1.0e-12)
        As = A / scales
        try:
            coef_all = np.linalg.solve(As.T @ As + max(0.0, ridge) * np.eye(As.shape[1]), As.T @ y) / scales
        except np.linalg.LinAlgError:
            coef_all = np.linalg.lstsq(A, y, rcond=None)[0]
        effects = [abs(float(coef_all[i + 1])) * max(float(np.std(values[i])), 1.0e-12) for i in range(len(values))]
        selected = sorted(range(len(values)), key=lambda i: -effects[i])[:max(1, min(int(keep), len(values)))]
        A_selected = np.column_stack([np.ones(len(y))] + [values[i] for i in selected])
        selected_scales = np.maximum(np.linalg.norm(A_selected, axis=0), 1.0e-12)
        A_scaled = A_selected / selected_scales
        try:
            coef = np.linalg.solve(A_scaled.T @ A_scaled + max(0.0, ridge) * np.eye(A_scaled.shape[1]), A_scaled.T @ y) / selected_scales
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A_selected, y, rcond=None)[0]
        pieces = [f"({float(coef[0]):.16g})"] if abs(float(coef[0])) > 1.0e-12 else []
        for local_index, spec_index in enumerate(selected, 1):
            c = float(coef[local_index])
            if abs(c) > 1.0e-12:
                pieces.append(f"({c:.16g})*({specs[spec_index][0]})")
        return " + ".join(pieces) if pieces else "0.0"

@dataclass(frozen=True)
class EquationNode:
    path: str
    kind: str
    operator: str
    signature: str
    variables: tuple[str, ...]
    numeric_value: Optional[float] = None


@dataclass(frozen=True)
class EquationDAG:
    expression: str
    canonical: str
    canonical_hash: str
    structural_hash: str
    complexity: float
    nodes: tuple[EquationNode, ...]
    dimensional_metadata: tuple[tuple[str, str], ...]

    @classmethod
    def build(
        cls,
        expression: str,
        n_features: int,
        registry: PrimitiveRegistry,
        dimensional_metadata: Optional[Mapping[str, Any]] = None,
    ) -> "EquationDAG":
        expr, canonical = registry.canonical(expression, n_features)
        number_map = {num: sp.Symbol("C", real=True) for num in expr.atoms(sp.Number)}
        try:
            structural = sp.sstr(expr.xreplace(number_map), order="lex")
        except Exception:
            structural = canonical
        nodes: list[EquationNode] = []

        def visit(node: sp.Expr, path: str) -> None:
            if getattr(node, "is_Number", False):
                kind, operator, signature = "constant", "CONST", "CONST"
                numeric = finite_float(node, 0.0)
            elif getattr(node, "is_Symbol", False):
                kind, operator, signature = "variable", "VAR", str(node)
                numeric = None
            else:
                kind = "operator"
                operator = str(getattr(node.func, "__name__", node.func))
                try:
                    signature = sp.sstr(node.xreplace({n: sp.Symbol("C") for n in node.atoms(sp.Number)}), order="lex")
                except Exception:
                    signature = operator
                numeric = None
            variables = tuple(sorted(str(s) for s in node.free_symbols if str(s).startswith("x")))
            nodes.append(EquationNode(path, kind, operator, signature, variables, numeric))
            for index, child in enumerate(getattr(node, "args", ())):
                visit(child, f"{path}.{index}")

        visit(expr, "0")
        depth = max((node.path.count(".") for node in nodes), default=0)
        complexity = float(sp.count_ops(expr, visual=False)) + 0.20 * len(expr.atoms(sp.Number)) + 0.05 * depth
        metadata_rows = tuple(sorted(
            (str(key), str(value)) for key, value in dict(dimensional_metadata or {}).items()
        ))
        return cls(
            canonical, canonical, sha256_text(canonical), sha256_text(structural),
            complexity, tuple(nodes), metadata_rows,
        )

    def diff(self, other: "EquationDAG") -> dict[str, Any]:
        """Return a semantic structural edit instead of a path-only tree diff.

        Canonical reordering of commutative SymPy nodes must not turn a simple
        ADD into a false CHANGE_OPERATOR.  We therefore compare multisets of
        structural subtree signatures first, then use root operator/variables
        to distinguish operator and interaction edits.
        """
        before_counter = Counter(
            node.signature for node in self.nodes if node.kind != "constant"
        )
        after_counter = Counter(
            node.signature for node in other.nodes if node.kind != "constant"
        )
        added_counter = after_counter - before_counter
        deleted_counter = before_counter - after_counter
        added = list(added_counter.elements())[:32]
        deleted = list(deleted_counter.elements())[:32]
        changed = self.canonical_hash != other.canonical_hash

        before_root = self.nodes[0] if self.nodes else None
        after_root = other.nodes[0] if other.nodes else None
        operator_changes: list[dict[str, Any]] = []
        interaction_changes: list[dict[str, Any]] = []
        parameter_changes: list[dict[str, Any]] = []

        before_by_path = {n.path: n for n in self.nodes}
        after_by_path = {n.path: n for n in other.nodes}
        for path in sorted(set(before_by_path) & set(after_by_path)):
            left, right = before_by_path[path], after_by_path[path]
            if left.kind == right.kind == "constant" and left.numeric_value != right.numeric_value:
                parameter_changes.append({"path": path, "before": left.numeric_value, "after": right.numeric_value})

        if not changed:
            action = "NOOP"
        elif self.structural_hash == other.structural_hash:
            action = "REPARAMETERIZE"
        elif added and not deleted:
            action = "ADD"
        elif deleted and not added:
            action = "DELETE"
        elif (
            before_root is not None
            and after_root is not None
            and before_root.operator != after_root.operator
            and before_root.variables == after_root.variables
        ):
            action = "CHANGE_OPERATOR"
            operator_changes.append({
                "path": "0", "before": before_root.operator, "after": after_root.operator,
            })
        elif (
            before_root is not None
            and after_root is not None
            and before_root.operator == after_root.operator
            and before_root.variables != after_root.variables
        ):
            action = "CHANGE_INTERACTION"
            interaction_changes.append({
                "path": "0", "before": before_root.variables, "after": after_root.variables,
            })
        else:
            action = "REPLACE"

        return {
            "action": action,
            "changed": changed,
            "before_hash": self.canonical_hash,
            "after_hash": other.canonical_hash,
            "before_structural_hash": self.structural_hash,
            "after_structural_hash": other.structural_hash,
            "added_nodes": added,
            "deleted_nodes": deleted,
            "operator_changes": operator_changes[:16],
            "interaction_changes": interaction_changes[:16],
            "parameter_changes": parameter_changes[:16],
            "retained_llm_edit": bool(changed and action != "NOOP"),
        }



@dataclass(frozen=True)
class RefitResult:
    expression: str
    optimizer: str
    initial_loss: float
    final_loss: float
    optimized_parameter_count: int
    top_level_term_count: int
    success: bool
    message: str = ""


class EquationRuntime:
    def __init__(
        self,
        n_features: int,
        registry: Optional[PrimitiveRegistry] = None,
        external_build_lambda: Optional[Callable[..., Any]] = None,
        max_numeric_parameters: int = 18,
        max_abs_coefficient: float = 1.0e4,
        optimize_exponents: bool = False,
        variable_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.n_features = int(n_features)
        self.registry = registry or PrimitiveRegistry()
        self.external_build_lambda = external_build_lambda
        self.max_numeric_parameters = max(0, int(max_numeric_parameters))
        self.max_abs_coefficient = max(1.0, float(max_abs_coefficient))
        self.optimize_exponents = bool(optimize_exponents)
        self.variable_metadata = dict(variable_metadata or {})
        self.symbols = [sp.Symbol(f"x{i}", real=True) for i in range(self.n_features)]

    def normalize(self, expression: str) -> str:
        return self.registry.normalize(expression)

    def dag(self, expression: str) -> EquationDAG:
        return EquationDAG.build(
            expression, self.n_features, self.registry,
            dimensional_metadata=self.variable_metadata,
        )

    def predict(self, expression: str, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            if self.n_features == 1:
                X = X.reshape(-1, 1)
            elif X.size == self.n_features:
                X = X.reshape(1, -1)
            else:
                raise ValueError("ambiguous_1d_feature_shape")
        if X.ndim != 2 or X.shape[1] < self.n_features:
            raise ValueError("feature_count_mismatch")
        normalized = self.normalize(expression)
        if self.external_build_lambda is not None:
            built = self.external_build_lambda(normalized, self.n_features)
            fn = built[1] if isinstance(built, tuple) and len(built) >= 2 else built
            pred = np.asarray(fn(X), dtype=float)
        else:
            expr = self.registry.parse(normalized, self.n_features, evaluate=False)
            fn = self.registry.lambdify(expr, self.symbols)
            with np.errstate(all="ignore"):
                pred = np.asarray(fn(*[X[:, i] for i in range(self.n_features)]), dtype=float)
        if pred.ndim == 0:
            pred = np.full(len(X), float(pred), dtype=float)
        pred = pred.reshape(-1)
        if pred.size != len(X) or not np.all(np.isfinite(pred)):
            raise ValueError("nonfinite_or_shape_prediction")
        return pred

    @staticmethod
    def mse(y: np.ndarray, pred: np.ndarray) -> float:
        y, pred = np.asarray(y, dtype=float).reshape(-1), np.asarray(pred, dtype=float).reshape(-1)
        if len(y) != len(pred) or not np.all(np.isfinite(pred)):
            return float("inf")
        return float(np.mean((pred - y) ** 2))

    def _fit_top_level_amplitudes(self, expression: str, X: np.ndarray, y: np.ndarray, ridge: float) -> tuple[str, int]:
        expr = self.registry.parse(expression, self.n_features, evaluate=False)
        terms = list(expr.args) if isinstance(expr, sp.Add) else [expr]
        structures, columns = [], []
        for term in terms:
            _, structure = term.as_coeff_Mul(rational=False)
            text = sp.sstr(structure, order="lex")
            try:
                values = self.predict(text, X)
            except Exception:
                continue
            if float(np.std(values)) <= 1.0e-14:
                continue
            structures.append(text)
            columns.append(values)
        if not columns:
            return self.normalize(f"({float(np.mean(y)):.16g})"), 0
        A = np.column_stack([np.ones(len(X))] + columns)
        scales = np.maximum(np.linalg.norm(A, axis=0), 1.0e-12)
        As = A / scales
        gram = As.T @ As
        try:
            condition_number = float(np.linalg.cond(gram))
        except Exception:
            condition_number = float("inf")
        # A small adaptive ridge prevents strongly collinear basis terms from
        # exploding into large cancelling amplitudes after global refit.
        effective_ridge = max(0.0, float(ridge))
        if not math.isfinite(condition_number):
            effective_ridge = max(effective_ridge, 1.0e-4)
        elif condition_number > 1.0e8:
            effective_ridge = max(
                effective_ridge,
                min(1.0e-3, 1.0e-8 * (condition_number / 1.0e8) ** 0.5),
            )
        try:
            coef = np.linalg.solve(gram + effective_ridge * np.eye(As.shape[1]), As.T @ y) / scales
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
        max_abs = max(1.0, float(np.max(np.abs(coef))))
        y_scale = max(float(np.std(y)), float(np.sqrt(np.mean(np.asarray(y, dtype=float) ** 2))), 1.0e-12)
        contribution_floor = 1.0e-8
        pieces = []
        if abs(float(coef[0])) / y_scale > max(1.0e-12, contribution_floor):
            pieces.append(f"({float(np.clip(coef[0], -self.max_abs_coefficient, self.max_abs_coefficient)):.16g})")
        for c, structure, values in zip(coef[1:], structures, columns):
            c = float(np.clip(c, -self.max_abs_coefficient, self.max_abs_coefficient))
            relative_contribution = abs(c) * max(float(np.std(values)), 1.0e-12) / y_scale
            if abs(c) > max(1.0e-12, 1.0e-9 * max_abs) and relative_contribution > contribution_floor:
                pieces.append(f"({c:.16g})*({structure})")
        return self.normalize(" + ".join(pieces) if pieces else "0.0"), len(structures)

    def _parameterized_expression(self, expression: str) -> tuple[sp.Expr, list[sp.Symbol], np.ndarray, list[tuple[float, float]]]:
        """Replace numeric occurrences with independent parameters.

        ``xreplace`` by numeric value incorrectly ties equal literals appearing
        in unrelated locations. Rebuilding the tree gives each editable
        occurrence its own parameter. Exponents are structural operators and
        remain frozen by default; only an explicit runtime option may optimize
        them, so numerical refit cannot silently rewrite LLM structure.
        """
        expr = self.registry.parse(expression, self.n_features, evaluate=False)
        params: list[sp.Symbol] = []
        values: list[float] = []
        bounds: list[tuple[float, float]] = []

        def rebuild(node: sp.Expr, parent: Optional[sp.Expr] = None, arg_index: int = -1) -> sp.Expr:
            if getattr(node, "is_Number", False):
                is_exponent = isinstance(parent, sp.Pow) and arg_index == 1
                parent_name = str(getattr(getattr(parent, "func", None), "__name__", getattr(parent, "func", "")))
                is_power_argument = parent_name in {"abspow", "signpow"} and arg_index == 1
                structural_exponent = is_exponent or is_power_argument
                editable = (self.optimize_exponents or not structural_exponent) and not (
                    isinstance(node, sp.Integer) and int(node) in {-1, 0, 1}
                )
                if editable and len(params) < self.max_numeric_parameters:
                    value = finite_float(node, 0.0)
                    param = sp.Symbol(f"__p{len(params)}", real=True)
                    params.append(param)
                    values.append(value)
                    if is_exponent or is_power_argument:
                        bounds.append((-8.0, 8.0))
                    else:
                        width = min(self.max_abs_coefficient, max(10.0, abs(value) * 20.0 + 2.0))
                        bounds.append((-width, width))
                    return param
                return node
            if getattr(node, "is_Atom", False):
                return node
            args = [rebuild(child, node, index) for index, child in enumerate(node.args)]
            try:
                if isinstance(node, (sp.Add, sp.Mul, sp.Pow)):
                    return node.func(*args, evaluate=False)
                return node.func(*args)
            except Exception:
                return node.func(*args)

        parameterized = rebuild(expr)
        return parameterized, params, np.asarray(values, dtype=float), bounds


    def refit_global_constants(
        self,
        expression: str,
        X: np.ndarray,
        y: np.ndarray,
        ridge: float = 1.0e-8,
        maxiter: int = 220,
        random_seed: int = 0,
    ) -> RefitResult:
        """Refit all additive subtree amplitudes, then all editable literals.

        The old anchor coefficient is therefore never frozen at one. Terms may
        shrink to zero, change sign or be globally reweighted after structural edits.
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        raw = self.normalize(expression)
        try:
            initial_loss = self.mse(y, self.predict(raw, X))
        except Exception:
            initial_loss = float("inf")
        try:
            best_expr, term_count = self._fit_top_level_amplitudes(raw, X, y, ridge)
        except Exception as exc:
            best_expr, term_count, amplitude_error = raw, 0, repr(exc)
        else:
            amplitude_error = ""
        try:
            best_loss = self.mse(y, self.predict(best_expr, X))
        except Exception:
            best_expr, best_loss = raw, initial_loss
        if minimize is None or self.max_numeric_parameters <= 0:
            return RefitResult(best_expr, "ridge-amplitudes", initial_loss, best_loss, 0, term_count, math.isfinite(best_loss), amplitude_error)
        try:
            parameterized, params, x0, bounds = self._parameterized_expression(best_expr)
        except Exception as exc:
            return RefitResult(best_expr, "ridge-amplitudes", initial_loss, best_loss, 0, term_count, math.isfinite(best_loss), repr(exc))
        if not params:
            return RefitResult(best_expr, "ridge-amplitudes", initial_loss, best_loss, 0, term_count, math.isfinite(best_loss), amplitude_error)
        fn = self.registry.lambdify(parameterized, [*self.symbols, *params])
        y_scale = max(float(np.var(y)), 1.0e-15)

        def objective(values: np.ndarray) -> float:
            try:
                with np.errstate(all="ignore"):
                    pred = np.asarray(fn(*[X[:, i] for i in range(self.n_features)], *values), dtype=float)
            except Exception:
                return 1.0e100
            if pred.ndim == 0:
                pred = np.full(len(X), float(pred))
            pred = pred.reshape(-1)
            if pred.size != len(X) or not np.all(np.isfinite(pred)):
                return 1.0e100
            return float(np.mean((pred - y) ** 2) / y_scale + 1.0e-12 * np.sum(np.asarray(values) ** 2))

        rng = np.random.default_rng(int(random_seed))
        starts = [x0]
        if len(x0):
            lower, upper = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
            starts.append(np.clip(x0 + rng.normal(0.0, 0.05, len(x0)) * np.maximum(1.0, np.abs(x0)), lower, upper))
        best_result = None
        for start in starts:
            try:
                result = minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter), "ftol": 1.0e-12})
            except Exception:
                continue
            if best_result is None or finite_float(result.fun) < finite_float(best_result.fun):
                best_result = result
        if best_result is None:
            return RefitResult(best_expr, "ridge-amplitudes", initial_loss, best_loss, len(params), term_count, math.isfinite(best_loss), "numeric_optimizer_failed")
        substituted = parameterized.subs({p: float(v) for p, v in zip(params, best_result.x)})
        candidate = self.normalize(sp.sstr(substituted, order="lex"))
        try:
            candidate, _ = self._fit_top_level_amplitudes(candidate, X, y, ridge)
            candidate_loss = self.mse(y, self.predict(candidate, X))
        except Exception:
            candidate_loss = float("inf")
        if candidate_loss <= best_loss:
            best_expr, best_loss = candidate, candidate_loss
        return RefitResult(
            best_expr, "ridge-amplitudes+L-BFGS-B", initial_loss, best_loss,
            len(params), term_count, math.isfinite(best_loss), str(getattr(best_result, "message", "")),
        )

    def intermediate_outputs(self, dag: EquationDAG, X: np.ndarray, max_nodes: int = 20) -> list[tuple[str, np.ndarray]]:
        expr = self.registry.parse(dag.expression, self.n_features, evaluate=False)
        candidates: list[sp.Expr] = []
        for node in sp.preorder_traversal(expr):
            if getattr(node, "is_Atom", False) or node == expr or len(node.free_symbols) == 0:
                continue
            if node not in candidates:
                candidates.append(node)
        scored = []
        for node in candidates:
            text = sp.sstr(node, order="lex")
            try:
                values = self.predict(text, X)
            except Exception:
                continue
            if float(np.std(values)) <= 1.0e-12:
                continue
            score = math.log1p(float(np.std(values))) + 0.05 * len(node.free_symbols)
            scored.append((score, text, values))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(text, values) for _, text, values in scored[:max(0, int(max_nodes))]]




    def materialize_exploration(self, exploration_expression: str, current: EquationDAG) -> str:
        """Substitute the current complete equation into executable g(y_hat, x)."""
        materialized = re.sub(r"\by_hat\b", f"({current.expression})", str(exploration_expression))
        return self.normalize(materialized)
