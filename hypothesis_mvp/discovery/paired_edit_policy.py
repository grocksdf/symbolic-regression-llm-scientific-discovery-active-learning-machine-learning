"""Standard conjugate regression for matched structural-edit contrasts.

Working research model: y_b = (phi(a,c)-phi(control,c)) @ theta + epsilon_b,
epsilon_b | sigma2 ~ N(0,sigma2), theta | sigma2 ~ N(mu0,sigma2 Lambda0^-1),
sigma2 ~ InvGamma(alpha0,beta0). Each validated block contributes ONE averaged
fold contrast. Shared folds are not independent replicates. This conditional
working model does not establish calibration, efficacy, novelty, or formula
truth; its fixed noise prior is not fitted to evaluation outcomes.
"""
from __future__ import annotations

import ast
import copy
import json
from typing import Sequence

import numpy as np

from .counterfactual_bank_symbolic_invention import (
    INVENTION_OPERATORS, StructuralEditRequest, StructuralEditResult,
    _node_at_path, _parse_body, validate_matched_edit_pair,
)


CONTEXTS = tuple(f"{depth}:{kind}" for depth in ("root", "internal")
                 for kind in ("binary", "unary", "function", "leaf"))


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity")
    return value


def _validated_state(mean, precision, alpha, beta):
    """Validate a complete staged state; never repair it with jitter."""
    if (not np.all(np.isfinite(mean)) or not np.all(np.isfinite(precision))
            or not np.isfinite(alpha) or not np.isfinite(beta)
            or alpha <= 0 or beta <= 0
            or not np.array_equal(precision, precision.T)):
        raise ValueError("nonfinite or invalid paired posterior")
    try:
        cholesky = np.linalg.cholesky(precision)
        covariance = np.linalg.solve(precision, np.eye(len(mean)))
        covariance = (covariance + covariance.T) * 0.5
        if not np.all(np.isfinite(covariance)):
            raise ValueError("nonfinite paired posterior covariance")
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("paired posterior must be positive definite") from error
    return cholesky, covariance


class PairedEditPolicy:
    """Joint operator effects plus shrunk context-specific deviations.

    Feature coordinates are all global operators, followed by one operator
    block per fixed context. Operators are sorted, so caller ordering cannot
    change the prior, draws, or tie breaks. ``covariance`` is Lambda^-1, the
    conditional covariance FACTOR (conditional covariance is sigma2 times it).
    """

    def __init__(self, operators: Sequence[str] = INVENTION_OPERATORS,
                 seed: int = 0, *, scoring_identity: str, prior_mean=None,
                 prior_precision=None, prior_alpha: float = 1.0,
                 prior_beta: float = 1.0):
        supplied = tuple(operators)
        if (not supplied or len(set(supplied)) != len(supplied)
                or any(op not in INVENTION_OPERATORS for op in supplied)):
            raise ValueError("invalid paired operator set")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._operators, self._seed = tuple(sorted(supplied)), int(seed)
        self._scoring_identity = _identity(scoring_identity, "scoring_identity")
        dimension = len(self._operators) * (1 + len(CONTEXTS))
        self._mean = (np.zeros(dimension) if prior_mean is None
                      else np.array(prior_mean, dtype=float, copy=True))
        self._precision = (np.eye(dimension) if prior_precision is None
                           else np.array(prior_precision, dtype=float, copy=True))
        self._alpha, self._beta = float(prior_alpha), float(prior_beta)
        if (self._mean.shape != (dimension,)
                or self._precision.shape != (dimension, dimension)):
            raise ValueError("prior shape does not match canonical feature coordinates")
        self._cholesky, self._covariance = _validated_state(
            self._mean, self._precision, self._alpha, self._beta)
        self._observations, self._decisions, self._evidence_ids = [], [], set()
        self._prior = self.posterior_parameters()

    @property
    def operators(self):
        return self._operators

    @property
    def scoring_identity(self):
        return self._scoring_identity

    @property
    def observations(self):
        return copy.deepcopy(self._observations)

    @property
    def decisions(self):
        return copy.deepcopy(self._decisions)

    @property
    def observation_count(self):
        return len(self._observations)

    @staticmethod
    def site_context(parent_expression: str, site_path: Sequence[int]) -> str:
        node = _node_at_path(_parse_body(parent_expression), site_path)
        kind = ("binary" if isinstance(node, ast.BinOp) else
                "unary" if isinstance(node, ast.UnaryOp) else
                "function" if isinstance(node, ast.Call) else "leaf")
        return f"{'internal' if site_path else 'root'}:{kind}"

    def feature(self, operator: str, context: str) -> np.ndarray:
        if operator not in self._operators or context not in CONTEXTS:
            raise ValueError("unknown paired operator or context")
        index, count = self._operators.index(operator), len(self._operators)
        vector = np.zeros(len(self._mean))
        vector[index] = vector[count * (1 + CONTEXTS.index(context)) + index] = 1.0
        return vector

    def posterior_parameters(self) -> dict:
        return {"mean": self._mean.tolist(), "precision": self._precision.tolist(),
                "covariance": self._covariance.tolist(),
                "covariance_kind": "conditional coefficient covariance / sigma2",
                "alpha": self._alpha, "beta": self._beta,
                "count": self.observation_count}

    def observe_pair(self, *, primary_request: StructuralEditRequest,
                     control_request: StructuralEditRequest,
                     primary_result: StructuralEditResult,
                     control_result: StructuralEditResult,
                     primary_folds: Sequence[float], control_folds: Sequence[float],
                     context: str, evidence_id: str, scoring_identity: str) -> None:
        evidence_id = _identity(evidence_id, "evidence_id")
        if evidence_id in self._evidence_ids:
            raise ValueError("duplicate paired evidence_id")
        if scoring_identity != self._scoring_identity:
            raise ValueError("paired scoring identity mismatch")
        validate_matched_edit_pair(
            primary_request, control_request, primary_result, control_result)
        for request, result in ((primary_request, primary_result),
                                (control_request, control_result)):
            if any(getattr(request, key) != getattr(result, key)
                   for key in ("operator", "parent_expression", "donor_expression", "site_path")):
                raise ValueError("paired result does not match request")
        if context != self.site_context(primary_request.parent_expression,
                                        primary_request.site_path):
            raise ValueError("paired site context mismatch")
        x = (self.feature(primary_request.operator, context)
             - self.feature(control_request.operator, context))
        primary, control = (np.asarray(values, dtype=float)
                            for values in (primary_folds, control_folds))
        if (primary.ndim != 1 or control.shape != primary.shape or len(primary) < 2
                or not np.all(np.isfinite(primary)) or not np.all(np.isfinite(control))):
            raise ValueError("paired evidence needs at least two finite matched folds")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                contrasts = primary - control
                y = float(np.sum(contrasts / len(contrasts)))
                u = np.linalg.solve(self._precision, x)
                denominator = 1.0 + float(x @ u)
                residual = y - float(x @ self._mean)
                mean = self._mean + u * (residual / denominator)
                precision = self._precision + np.outer(x, x)
                alpha = self._alpha + 0.5
                scaled_residual = residual / np.sqrt(denominator)
                beta = float(self._beta + 0.5 * scaled_residual * scaled_residual)
                cholesky, covariance = _validated_state(mean, precision, alpha, beta)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise ValueError("paired posterior arithmetic failed") from error
        row = {"evidence_id": evidence_id, "scoring_identity": scoring_identity,
               "context": context, "primary_operator": primary_request.operator,
               "control_operator": control_request.operator,
               "primary_request_identity": primary_request.identity,
               "control_request_identity": control_request.identity,
               "primary_support_hash": primary_result.support_hash,
               "control_support_hash": control_result.support_hash,
               "fold_count": len(primary), "primary_folds": primary.tolist(),
               "control_folds": control.tolist(), "contrast": y,
               "feature_difference": x.tolist(), "observation_weight": 1}
        json.dumps(row, allow_nan=False)
        # Publish only after validation of the entire observation and new state.
        self._mean, self._precision = mean, precision
        self._alpha, self._beta = alpha, beta
        self._cholesky, self._covariance = cholesky, covariance
        self._observations.append(row)
        self._evidence_ids.add(evidence_id)

    def choose(self, step: int, *, context: str,
               eligible_operators: Sequence[str], mode: str = "thompson") -> str:
        eligible = tuple(eligible_operators)
        if (not eligible or len(set(eligible)) != len(eligible)
                or any(op not in self._operators for op in eligible)
                or context not in CONTEXTS or mode not in {"thompson", "greedy"}):
            raise ValueError("invalid paired policy choice")
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        allowed = tuple(op for op in self._operators if op in eligible)
        theta, sigma2 = self._mean.copy(), None
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if mode == "thompson":
                    rng = np.random.default_rng(np.random.SeedSequence([self._seed, int(step)]))
                    sigma2 = float(1.0 / rng.gamma(self._alpha, 1.0 / self._beta))
                    theta += np.sqrt(sigma2) * np.linalg.solve(
                        self._cholesky.T, rng.standard_normal(len(theta)))
                scores = {op: float(self.feature(op, context) @ theta) for op in allowed}
            if (not np.all(np.isfinite(theta)) or
                    (sigma2 is not None and (not np.isfinite(sigma2) or sigma2 <= 0))
                    or not all(np.isfinite(value) for value in scores.values())):
                raise ValueError("nonfinite paired posterior draw")
        except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
            raise ValueError("paired posterior draw failed") from error
        choice = min(allowed, key=lambda op: (-scores[op], op))
        self._decisions.append({"step": int(step), "mode": mode, "context": context,
                                "eligible": list(allowed), "choice": choice,
                                "scores": scores, "sampled_sigma2": sigma2,
                                "coefficients": theta.tolist(),
                                "observation_count": self.observation_count})
        return choice

    def audit(self) -> dict:
        return copy.deepcopy({
            "schema": "cbsi-paired-edit-policy-v1", "operator_set": list(self._operators),
            "contexts": list(CONTEXTS), "seed": self._seed,
            "scoring_identity": self._scoring_identity, "prior": self._prior,
            "posterior": self.posterior_parameters(),
            "observation_count": self.observation_count,
            "observation_unit": "one validated block's arithmetic mean fold contrast",
            "decisions": self._decisions, "observations": self._observations,
            "novelty_claim_authorized": False, "calibration_claim_authorized": False,
            "claim_boundary": "standard conditional working-model posterior; no formula-truth or efficacy claim",
        })
