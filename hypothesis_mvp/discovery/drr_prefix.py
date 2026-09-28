"""Fixed short-prefix continuous decision-risk diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .drr_readiness import evaluate_drr_readiness


PREFIXES = (8, 16, 32)
METHOD = "fixed-prefix-certified-normalized-decision-risk-aulc-v1"


@dataclass(frozen=True)
class DRRPrefixCurve:
    condition: str
    prefixes: tuple[int, ...]
    normalized_lower_bounds: tuple[float, ...]
    normalized_aulc: float
    positive_prefix_count: int
    certificates: tuple[Mapping[str, Any], ...]

    def to_dict(self):
        return {
            "schema": "scientific-drr-prefix-curve-v1",
            "method": METHOD,
            "condition": self.condition,
            "prefixes": list(self.prefixes),
            "normalized_lower_bounds": list(
                self.normalized_lower_bounds),
            "normalized_aulc": self.normalized_aulc,
            "positive_prefix_count": self.positive_prefix_count,
            "certificates": [dict(value) for value in self.certificates],
            "candidate_response_accessed": False,
            "action_response_accessed": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
        }


def evaluate_drr_prefix_curve(
    candidates: Sequence[Mapping[str, Any]],
    initial_X, initial_y, action_X, *, condition: str,
    exploration_identity: str,
    prefixes: tuple[int, ...] = PREFIXES,
    selection_method: str,
) -> DRRPrefixCurve:
    X = np.asarray(initial_X, dtype=float)
    y = np.asarray(initial_y, dtype=float).reshape(-1)
    chosen = tuple(int(value) for value in prefixes)
    if (chosen != PREFIXES or len(X) < chosen[-1] or len(X) != len(y)
            or any(left >= right for left, right in zip(chosen, chosen[1:]))):
        raise ValueError("DRR prefix curve requires fixed prefixes 8,16,32")
    scores, certificates = [], []
    for prefix in chosen:
        result = evaluate_drr_readiness(
            candidates, X[:prefix], y[:prefix], action_X,
            condition=condition,
            exploration_identity=exploration_identity,
            selection_method=selection_method)
        certificate = dict(result.certificate)
        viability = certificate.get("viability") or {}
        utility = certificate.get("decision_risk_utility") or {}
        prior_risk = float(viability.get("class_bayes_risk") or 0.0)
        lower = float(utility.get("selected_lower_bound") or 0.0)
        score = 0.0 if prior_risk <= 0.0 else lower / prior_risk
        scores.append(float(np.clip(score, 0.0, 1.0)))
        certificates.append(certificate)
    x = np.log2(np.asarray(chosen, dtype=float))
    aulc = float(np.trapezoid(scores, x) / (x[-1] - x[0]))
    return DRRPrefixCurve(
        condition, chosen, tuple(scores), aulc,
        sum(value > 0.0 for value in scores), tuple(certificates))


__all__ = [
    "DRRPrefixCurve", "METHOD", "PREFIXES",
    "evaluate_drr_prefix_curve",
]
