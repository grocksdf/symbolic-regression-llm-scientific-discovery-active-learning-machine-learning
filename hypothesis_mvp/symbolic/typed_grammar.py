from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp


class ExprNode:
    def to_sympy(self) -> sp.Expr:  # pragma: no cover - interface
        raise NotImplementedError

    def size(self) -> int:  # number of nodes
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def to_string(self) -> str:
        try:
            return str(self.to_sympy())
        except Exception:
            # Fallback: structural string which must never raise (used for hashing/sorting).
            if isinstance(self, Var):
                return f"x{self.idx}"
            if isinstance(self, Const):
                return f"{float(self.value):.12g}"
            if isinstance(self, Unary):
                return f"{self.op}({self.child.to_string()})"
            if isinstance(self, Binary):
                return f"({self.left.to_string()}{self.op}{self.right.to_string()})"
            return repr(self)


@dataclass(frozen=True)
class Var(ExprNode):
    idx: int

    def to_sympy(self) -> sp.Expr:
        return sp.Symbol(f"x{self.idx}")

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1


@dataclass(frozen=True)
class Const(ExprNode):
    value: float

    def to_sympy(self) -> sp.Expr:
        return sp.Float(self.value)

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1


@dataclass(frozen=True)
class Unary(ExprNode):
    op: str
    child: ExprNode

    def to_sympy(self) -> sp.Expr:
        fn = getattr(sp, self.op, None)
        if fn is None:
            raise ValueError(f"Unknown unary op '{self.op}'")
        return fn(self.child.to_sympy())

    def size(self) -> int:
        return 1 + self.child.size()

    def depth(self) -> int:
        return 1 + self.child.depth()


@dataclass(frozen=True)
class Binary(ExprNode):
    op: str
    left: ExprNode
    right: ExprNode

    def to_sympy(self) -> sp.Expr:
        a = self.left.to_sympy()
        b = self.right.to_sympy()
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "/":
            return a / b
        if self.op in ("**", "pow"):
            return a ** b
        raise ValueError(f"Unknown binary op '{self.op}'")

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())


Path = Tuple[int, ...]  # child indices, e.g. () root, (0,) left child


def iter_paths(node: ExprNode, prefix: Path = ()) -> Iterable[Path]:
    yield prefix
    if isinstance(node, Unary):
        yield from iter_paths(node.child, prefix + (0,))
    elif isinstance(node, Binary):
        yield from iter_paths(node.left, prefix + (0,))
        yield from iter_paths(node.right, prefix + (1,))


def get_at(node: ExprNode, path: Path) -> ExprNode:
    cur = node
    for p in path:
        if isinstance(cur, Unary):
            if p != 0:
                raise ValueError("Invalid path for Unary")
            cur = cur.child
        elif isinstance(cur, Binary):
            cur = cur.left if p == 0 else cur.right
        else:
            raise ValueError("Path too deep")
    return cur


def replace_at(node: ExprNode, path: Path, new_sub: ExprNode) -> ExprNode:
    if not path:
        return new_sub
    head, *tail = path
    tail_t = tuple(tail)
    if isinstance(node, Unary):
        if head != 0:
            raise ValueError("Invalid path for Unary")
        return Unary(node.op, replace_at(node.child, tail_t, new_sub))
    if isinstance(node, Binary):
        if head == 0:
            return Binary(node.op, replace_at(node.left, tail_t, new_sub), node.right)
        return Binary(node.op, node.left, replace_at(node.right, tail_t, new_sub))
    raise ValueError("Invalid path")


def is_valid_ast(
    node: ExprNode,
    *,
    n_features: int,
    allowed_unary: Sequence[str],
    allowed_binary: Sequence[str],
    max_depth: int,
    max_nodes: int,
    X_sample: Optional[np.ndarray] = None,
    math_eps: float = 1e-6,
    math_max_exp: float = 20.0,
    math_max_pow_abs: int = 4,
) -> bool:
    if node.depth() > max_depth or node.size() > max_nodes:
        return False
    # Validate ops
    for p in iter_paths(node):
        sub = get_at(node, p)
        if isinstance(sub, Var):
            if sub.idx < 0 or sub.idx >= n_features:
                return False
        elif isinstance(sub, Unary):
            if allowed_unary and sub.op not in allowed_unary:
                return False
        elif isinstance(sub, Binary):
            op = sub.op
            if op == "pow":
                op = "**"
            if allowed_binary and op not in allowed_binary and sub.op not in allowed_binary:
                return False

    if X_sample is not None:
        y = eval_ast_numpy(
            node,
            X_sample,
            eps=math_eps,
            max_exp=math_max_exp,
            max_pow_abs=math_max_pow_abs,
        )
        if y is None:
            return False
    return True


def expand_ast(
    node: ExprNode,
    *,
    n_features: int,
    allowed_unary: Sequence[str],
    allowed_binary: Sequence[str],
    constants: Sequence[float],
    rng: np.random.Generator,
    max_depth: int,
    max_nodes: int,
    max_new: int,
    X_sample: Optional[np.ndarray] = None,
    math_eps: float = 1e-6,
    math_max_exp: float = 20.0,
    math_max_pow_abs: int = 4,
) -> List[ExprNode]:
    """Generate new ASTs by applying a small set of typed rewrite rules."""
    paths = list(iter_paths(node))
    rng.shuffle(paths)
    new_nodes: List[ExprNode] = []

    # Terminals we can attach
    terminals: List[ExprNode] = [Var(i) for i in range(n_features)]
    terminals += [Const(float(c)) for c in constants]

    for path in paths:
        focus = get_at(node, path)

        # Unary wrap
        for u in allowed_unary:
            cand = canonicalize_ast(replace_at(node, path, Unary(u, focus)))
            if is_valid_ast(
                cand,
                n_features=n_features,
                allowed_unary=allowed_unary,
                allowed_binary=allowed_binary,
                max_depth=max_depth,
                max_nodes=max_nodes,
                X_sample=X_sample,
                math_eps=math_eps,
                math_max_exp=math_max_exp,
                math_max_pow_abs=math_max_pow_abs,
            ):
                new_nodes.append(cand)
                if len(new_nodes) >= max_new:
                    return _unique(new_nodes)

        # Binary combine (focus with a random terminal)
        for b in allowed_binary:
            # normalize pow
            bop = "**" if b in ("pow", "^") else b
            t = terminals[int(rng.integers(0, len(terminals)))]
            cand1 = replace_at(node, path, Binary(bop, focus, t))
            cand2 = replace_at(node, path, Binary(bop, t, focus))
            for cand in (cand1, cand2):
                cand = canonicalize_ast(cand)
                if is_valid_ast(
                    cand,
                    n_features=n_features,
                    allowed_unary=allowed_unary,
                    allowed_binary=allowed_binary,
                    max_depth=max_depth,
                    max_nodes=max_nodes,
                    X_sample=X_sample,
                    math_eps=math_eps,
                    math_max_exp=math_max_exp,
                    math_max_pow_abs=math_max_pow_abs,
                ):
                    new_nodes.append(cand)
                    if len(new_nodes) >= max_new:
                        return _unique(new_nodes)

    return _unique(new_nodes)


def _unique(nodes: List[ExprNode]) -> List[ExprNode]:
    seen = set()
    out: List[ExprNode] = []
    for n in nodes:
        n = canonicalize_ast(n)
        s = n.to_string()
        if s in seen:
            continue
        seen.add(s)
        out.append(n)
    return out


def _constant_fold(op: str, left: ExprNode, right: ExprNode) -> Optional[Const]:
    if not isinstance(left, Const) or not isinstance(right, Const):
        return None
    try:
        value = float(Binary(op, left, right).to_sympy().evalf())
    except Exception:
        return None
    return Const(value) if np.isfinite(value) else None


def _canonical_add(left: ExprNode, right: ExprNode) -> ExprNode:
    terms = [t for t in _flatten_commutative("+", [left, right]) if not (isinstance(t, Const) and abs(t.value) < 1e-12)]
    constant = sum(t.value for t in terms if isinstance(t, Const))
    coefficients: dict[str, float] = {}
    bases: dict[str, ExprNode] = {}
    rest: List[ExprNode] = []
    for term in (t for t in terms if not isinstance(t, Const)):
        coefficient, base = _split_coeff(term)
        if base is None:
            rest.append(term)
            continue
        key = base.to_string()
        coefficients[key] = coefficients.get(key, 0.0) + float(coefficient)
        bases[key] = base
    combined = [
        bases[key] if abs(value - 1.0) < 1e-12 else Binary("*", Const(float(value)), bases[key])
        for key, value in coefficients.items() if abs(value) >= 1e-12
    ]
    output = rest + combined
    if abs(constant) >= 1e-12:
        output.append(Const(float(constant)))
    if not output:
        return Const(0.0)
    output.sort(key=lambda item: item.to_string())
    return _rebuild_binary("+", output)


def _canonical_multiply(left: ExprNode, right: ExprNode) -> ExprNode:
    terms = _flatten_commutative("*", [left, right])
    if any(isinstance(term, Const) and abs(term.value) < 1e-12 for term in terms):
        return Const(0.0)
    terms = [term for term in terms if not (isinstance(term, Const) and abs(term.value - 1.0) < 1e-12)]
    constant = 1.0
    remaining: List[ExprNode] = []
    for term in terms:
        if isinstance(term, Const):
            constant *= float(term.value)
        else:
            remaining.append(term)
    remaining = _combine_powers(remaining)
    if abs(constant - 1.0) >= 1e-12 or not remaining:
        remaining.append(Const(float(constant)))
    remaining = [
        term for term in remaining
        if not (isinstance(term, Const) and abs(term.value - 1.0) < 1e-12 and len(remaining) > 1)
    ]
    remaining.sort(key=lambda item: item.to_string())
    return remaining[0] if len(remaining) == 1 else _rebuild_binary("*", remaining)


def canonicalize_ast(node: ExprNode) -> ExprNode:
    """Canonicalize commutative order, associativity and safe constants."""
    if isinstance(node, (Var, Const)):
        return node
    if isinstance(node, Unary):
        child = canonicalize_ast(node.child)
        if isinstance(child, Const):
            try:
                value = float(Unary(node.op, child).to_sympy().evalf())
                if np.isfinite(value):
                    return Const(value)
            except Exception:
                pass
        return Unary(node.op, child)
    if not isinstance(node, Binary):
        return node
    op = "**" if node.op == "pow" else node.op
    left, right = canonicalize_ast(node.left), canonicalize_ast(node.right)
    folded = _constant_fold(op, left, right)
    if folded is not None:
        return folded
    if op == "+":
        return _canonical_add(left, right)
    if op == "*":
        return _canonical_multiply(left, right)
    if op == "-":
        return left if isinstance(right, Const) and abs(right.value) < 1e-12 else canonicalize_ast(Binary("+", left, Binary("*", Const(-1.0), right)))
    if op == "/":
        return left if isinstance(right, Const) and abs(right.value - 1.0) < 1e-12 else Binary("/", left, right)
    if op == "**" and isinstance(right, Const):
        if abs(right.value - 1.0) < 1e-12:
            return left
        if abs(right.value) < 1e-12:
            return Const(1.0)
    return Binary(op, left, right)


def _flatten_commutative(op: str, nodes: List[ExprNode]) -> List[ExprNode]:
    out: List[ExprNode] = []
    for n in nodes:
        if isinstance(n, Binary) and n.op == op:
            out.extend(_flatten_commutative(op, [n.left, n.right]))
        else:
            out.append(n)
    return out


def _rebuild_binary(op: str, terms: List[ExprNode]) -> ExprNode:
    """Rebuild a left-associated binary tree from a list (already canonical-sorted)."""
    cur = terms[0]
    for t in terms[1:]:
        cur = Binary(op, cur, t)
    return cur


def _combine_duplicate_powers(terms: List[ExprNode]) -> List[ExprNode]:
    """Combine duplicates: x*x -> x**2 (only for exact structural duplicates)."""
    counts: dict[str, int] = {}
    rep: dict[str, ExprNode] = {}
    for t in terms:
        key = t.to_string()
        counts[key] = counts.get(key, 0) + 1
        rep[key] = t
    out: List[ExprNode] = []
    for key, n in counts.items():
        base = rep[key]
        if n >= 2:
            out.append(Binary("**", base, Const(float(n))))
        else:
            out.append(base)
    return out


def _is_int(x: float) -> bool:
    return abs(x - int(x)) < 1e-9


def _split_coeff(term: ExprNode) -> tuple[float, Optional[ExprNode]]:
    """If term is (c*base) or base, return (c, base). Otherwise (1, None)."""
    if isinstance(term, Const):
        return float(term.value), None
    if isinstance(term, Binary) and term.op == "*":
        # c * base or base * c
        if isinstance(term.left, Const):
            return float(term.left.value), term.right
        if isinstance(term.right, Const):
            return float(term.right.value), term.left
    return 1.0, term


def _combine_powers(terms: List[ExprNode]) -> List[ExprNode]:
    """Combine repeated factors and integer powers with same base."""
    # base_str -> (base_node, exponent_sum)
    exps: dict[str, int] = {}
    base_rep: dict[str, ExprNode] = {}
    others: List[ExprNode] = []

    for t in terms:
        # t = base**k with integer k
        if isinstance(t, Binary) and t.op in ("**", "pow") and isinstance(t.right, Const) and _is_int(t.right.value):
            k = int(t.right.value)
            base = t.left
            key = base.to_string()
            exps[key] = exps.get(key, 0) + k
            base_rep[key] = base
            continue
        # plain base => exponent 1
        key = t.to_string()
        exps[key] = exps.get(key, 0) + 1
        base_rep[key] = t

    out: List[ExprNode] = []
    for key, k in exps.items():
        base = base_rep[key]
        if k == 0:
            continue
        if k == 1:
            out.append(base)
        else:
            out.append(Binary("**", base, Const(float(k))))
    out.extend(others)
    return out


def eval_ast_numpy(
    node: ExprNode,
    X: np.ndarray,
    *,
    eps: float = 1e-6,
    max_exp: float = 20.0,
    max_pow_abs: int = 4,
) -> Optional[np.ndarray]:
    """Evaluate AST on data with domain constraints; return None if invalid (NaN/Inf or domain violation)."""

    def _eval(n: ExprNode) -> Optional[np.ndarray]:
        if isinstance(n, Var):
            if n.idx < 0 or n.idx >= X.shape[1]:
                return None
            return X[:, n.idx].astype(float)
        if isinstance(n, Const):
            return np.full((X.shape[0],), float(n.value), dtype=float)
        if isinstance(n, Unary):
            a = _eval(n.child)
            if a is None:
                return None
            with np.errstate(all="ignore"):
                if n.op == "sin":
                    out = np.sin(a)
                elif n.op == "cos":
                    out = np.cos(a)
                elif n.op == "exp":
                    if np.any(np.abs(a) > max_exp):
                        return None
                    out = np.exp(a)
                elif n.op == "log":
                    if np.any(a <= eps):
                        return None
                    out = np.log(a)
                else:
                    return None
            if not np.all(np.isfinite(out)):
                return None
            return out
        if isinstance(n, Binary):
            a = _eval(n.left)
            b = _eval(n.right)
            if a is None or b is None:
                return None
            with np.errstate(all="ignore"):
                if n.op == "+":
                    out = a + b
                elif n.op == "-":
                    out = a - b
                elif n.op == "*":
                    out = a * b
                elif n.op == "/":
                    if np.any(np.abs(b) <= eps):
                        return None
                    out = a / b
                elif n.op in ("**", "pow"):
                    # Only allow small integer constant exponents for stability.
                    if not isinstance(n.right, Const):
                        return None
                    exp_val = float(n.right.value)
                    if abs(exp_val - int(exp_val)) > 1e-9:
                        return None
                    k = int(exp_val)
                    if abs(k) > int(max_pow_abs):
                        return None
                    if k < 0 and np.any(np.abs(a) <= eps):
                        return None
                    out = np.power(a, k)
                else:
                    return None
            if not np.all(np.isfinite(out)):
                return None
            return out
        return None

    return _eval(node)


def ast_from_sympy(expr: sp.Expr) -> Optional[ExprNode]:
    """Best-effort conversion for a restricted subset used by this project."""
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name.startswith("x") and name[1:].isdigit():
            return Var(int(name[1:]))
        return None
    if isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
        return Const(float(expr))
    if isinstance(expr, sp.Add):
        args = list(expr.args)
        node: Optional[ExprNode] = ast_from_sympy(args[0])
        if node is None:
            return None
        for a in args[1:]:
            rhs = ast_from_sympy(a)
            if rhs is None:
                return None
            node = Binary("+", node, rhs)
        return node
    if isinstance(expr, sp.Mul):
        args = list(expr.args)
        node = ast_from_sympy(args[0])
        if node is None:
            return None
        for a in args[1:]:
            rhs = ast_from_sympy(a)
            if rhs is None:
                return None
            node = Binary("*", node, rhs)
        return node
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        a = ast_from_sympy(base)
        b = ast_from_sympy(exp)
        if a is None or b is None:
            return None
        return Binary("**", a, b)
    if isinstance(expr, sp.Function):
        fname = expr.func.__name__
        if len(expr.args) != 1:
            return None
        child = ast_from_sympy(expr.args[0])
        if child is None:
            return None
        return Unary(fname, child)
    return None
