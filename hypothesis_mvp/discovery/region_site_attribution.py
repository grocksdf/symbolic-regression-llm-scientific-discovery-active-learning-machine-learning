"""Fitting-only region/subtree associations, not causal attribution.

The AST supplies legal edit addresses. Numeric behavior, a local Bank score,
and Bank state supply the context. Analytic derivatives concern the *raw
parent expression*, not a refitted posterior's derivative. Undefined
derivatives are explicit unavailable sites, never fabricated zero evidence.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import json

import numpy as np

from hypothesis_mvp.data.roles import DataRole
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import _check, _run, _constant, compile_fixed_formula_support
from .counterfactual_bank_symbolic_invention import _ast_paths, _node_at_path


SCHEMA = "bank-region-subtree-context-v1"
AST_FEATURES = ("root", "binary", "unary", "function", "leaf", "depth", "size", "dependencies")
LOCAL_FEATURES = ("coverage", "defect_direction", "defect_strength", "score_direction", "density_deficit", "uncertainty")
BEHAVIOR_FEATURES = ("output_mean", "output_spread", "region_axis_dependency")
BANK_FEATURES = ("parent_mass", "parent_redundancy")
ASSOCIATION_FEATURES = ("sensitivity", "contribution", "residual_association", "score_alignment", "association_defined")
FEATURE_NAMES = AST_FEATURES + LOCAL_FEATURES + BEHAVIOR_FEATURES + BANK_FEATURES + ASSOCIATION_FEATURES


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _diagnostic_root(expression, n_features):
    # A complete bank formula can be intercept-only. The basis-term parser
    # intentionally requires a covariate and is therefore the wrong entrypoint
    # here. Validate complete support first, retaining its grammar/node limits.
    compile_fixed_formula_support(expression, n_features)
    root = ast.parse(expression, mode="eval").body
    _check(root, n_features)
    return root


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def _ratio(a, b):
    # Defined degeneracy convention for features only; no change to scoring.
    return float(a / b) if b else 0.0


def _bounded(value):
    return float(value / (1.0 + abs(value)))


def _correlation(a, b):
    a, b = np.asarray(a) - np.mean(a), np.asarray(b) - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return _ratio(float(a @ b), denominator), bool(denominator)


def subtree_response(expression, site_path, inputs):
    """Return parent output, subtree output and d(parent)/d(subtree output).

    Supports the existing closed formula grammar only. Path indices are the
    original editor's ast.iter_child_nodes indices (including operator nodes).
    """
    x = np.asarray(inputs, dtype=float)
    if x.ndim != 2 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("finite fitting inputs required")
    root = _diagnostic_root(expression, int(x.shape[1]))
    path = tuple(site_path)
    if path not in _ast_paths(root):
        raise ValueError("site is not an editable expression")
    target = _node_at_path(root, path)

    def derivative(node):
        value = np.broadcast_to(_run(node, x), (len(x),))
        if node is target:
            return value, np.ones(len(x))
        if isinstance(node, (ast.Name, ast.Constant)):
            return value, np.zeros(len(x))
        if isinstance(node, ast.UnaryOp):
            _, d = derivative(node.operand)
            return value, d if isinstance(node.op, ast.UAdd) else -d
        if isinstance(node, ast.BinOp):
            a, da = derivative(node.left)
            if isinstance(node.op, ast.Pow):
                power = _constant(node.right)
                _, dpower = derivative(node.right)
                d = np.zeros(len(x))
                active = da != 0
                if power != 0 and np.any(active):
                    d[active] = power * np.power(a[active], power - 1) * da[active]
                active_exponent = dpower != 0
                if np.any(active_exponent):
                    if np.any(a[active_exponent] <= 0):
                        raise ValueError("exponent sensitivity undefined for nonpositive base")
                    d[active_exponent] += value[active_exponent] * np.log(a[active_exponent]) * dpower[active_exponent]
                return value, d
            b, db = derivative(node.right)
            if isinstance(node.op, ast.Add): d = da + db
            elif isinstance(node.op, ast.Sub): d = da - db
            elif isinstance(node.op, ast.Mult): d = da * b + a * db
            else: d = (da - (a / b) * db) / b
            return value, d
        a, da = derivative(node.args[0])
        active = da != 0
        d = np.zeros(len(x))
        if np.any(active):
            v = a[active]
            name = node.func.id
            if name == "sin": slope = np.cos(v)
            elif name == "cos": slope = -np.sin(v)
            elif name == "tanh": slope = 1 - np.tanh(v) ** 2
            elif name == "exp": slope = np.exp(v)
            elif name == "log": slope = 1 / v
            elif name == "sqrt": slope = .5 / np.sqrt(v)
            else:
                if np.any(v == 0):
                    raise ValueError("absolute-value derivative undefined at zero")
                slope = np.sign(v)
            d[active] = slope * da[active]
        return value, d

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            parent, jacobian = derivative(root)
            subtree = np.broadcast_to(_run(target, x), (len(x),))
        if not all(np.all(np.isfinite(v)) for v in (parent, subtree, jacobian)):
            raise ValueError("nonfinite subtree diagnostic")
    except (ArithmeticError, FloatingPointError) as error:
        raise ValueError("undefined subtree diagnostic") from error
    return parent.copy(), subtree.copy(), jacobian.copy()


def build_region_site_contexts(expression, fit, feedback, *, regions,
                               forbidden_row_ids, bank_location, bank_scale,
                               parent_mass, parent_redundancy):
    """All regions are fixed masks on fitting X; no selection/admission labels.

    ``bank_scale`` is a positive predictive scale surrogate, not necessarily
    a Student-t variance. Parent mass/redundancy must come from the same E0.
    The caller binds their provenance in the Bank identity.
    """
    x = np.asarray(fit.X)
    forbidden = tuple(forbidden_row_ids)
    if (fit.role is not DataRole.DEVELOPMENT or fit.fingerprint != feedback.search_data_identity
            or len(feedback.search_row_ids) != len(x) or not forbidden
            or set(feedback.search_row_ids).intersection(forbidden)):
        raise ValueError("region diagnostics require isolated fitting observations")
    location, scale, score, density = (np.asarray(v, dtype=float) for v in
        (bank_location, bank_scale, feedback.location_score, feedback.log_density))
    if (any(v.shape != (len(x),) or not np.all(np.isfinite(v)) for v in (location, scale, score, density))
            or np.any(scale <= 0) or not 0 <= parent_mass <= 1 or not 0 <= parent_redundancy <= 1):
        raise ValueError("invalid bound Bank behavior")
    root = _diagnostic_root(expression, int(x.shape[1]))
    paths, total = _ast_paths(root), len(list(ast.walk(root)))
    residual = fit.y - location
    regions = tuple(regions)
    if not regions or len({r["id"] for r in regions}) != len(regions):
        raise ValueError("unique nonempty regions required")
    result, unavailable = [], []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for region in regions:
            axis, side, cut = region["axis"], region["side"], region["cut"]
            if (type(axis) is not int or not 0 <= axis < x.shape[1]
                    or side not in {"le", "gt"} or not np.isfinite(cut)):
                raise ValueError("invalid frozen region")
            mask = x[:, axis] <= cut
            if side == "gt": mask = ~mask
            if not np.any(mask):
                raise ValueError("empty registered fitting region")
            for path in paths:
                try:
                    node = _node_at_path(root, path)
                    # Formula values must exist throughout fitting X. Local
                    # sensitivity need only exist inside this frozen region.
                    parent = np.broadcast_to(_run(root, x), (len(x),))
                    a = np.broadcast_to(_run(node, x), (len(x),))
                    if not np.all(np.isfinite([parent, a])):
                        raise ValueError("undefined parent/subtree on fitting domain")
                    _, _, j = subtree_response(expression, path, x[mask])
                except ValueError as error:
                    unavailable.append({"region_id": region["id"], "site_path": list(path), "reason": str(error)})
                    continue
                deps = sorted({int(n.id[1:]) for n in ast.walk(node)
                               if isinstance(n, ast.Name) and n.id.startswith("x") and n.id[1:].isdigit()})
                contribution = j * a[mask]
                corr, defined = _correlation(residual[mask], contribution)
                alignment = _ratio(float(np.mean(score[mask] * contribution)),
                                   _rms(score[mask]) * _rms(contribution))
                features = (
                    float(not path), float(isinstance(node, ast.BinOp)), float(isinstance(node, ast.UnaryOp)),
                    float(isinstance(node, ast.Call)), float(isinstance(node, ast.Name)),
                    len(path) / total, len(list(ast.walk(node))) / total, len(deps) / x.shape[1],
                    float(np.mean(mask)), _bounded(_ratio(np.mean(residual[mask]), _rms(residual))),
                    _bounded(_ratio(_rms(residual[mask]), _rms(residual))),
                    _bounded(_ratio(np.mean(score[mask]), _rms(score))),
                    _bounded(float(np.mean(density) - np.mean(density[mask]))),
                    _bounded(_ratio(_rms(scale[mask]), _rms(scale))),
                    _bounded(_ratio(np.mean(a[mask]), _rms(a))),
                    _bounded(_ratio(np.std(a[mask]), _rms(a))), float(axis in deps),
                    float(parent_mass), float(parent_redundancy),
                    _bounded(_ratio(_rms(j) * _rms(a), _rms(parent))),
                    _bounded(_ratio(_rms(contribution), _rms(parent))),
                    float(corr), float(alignment), float(defined))
                if not np.all(np.isfinite(features)):
                    raise ValueError("nonfinite region-site feature")
                row = {"region_id": region["id"], "site_path": list(path), "features": list(features),
                       "variables": deps, "subtree_expression": ast.unparse(node)}
                result.append(row)
    payload = {"schema": SCHEMA, "feature_names": list(FEATURE_NAMES), "parent": expression,
               "bank_identity": feedback.bank_identity, "feedback_identity": feedback.identity,
               "fit_fingerprint": fit.fingerprint, "regions": regions,
               "sites": result, "unavailable": unavailable,
               "causal_attribution_authorized": False}
    return {**payload, "identity": digest(payload)}
