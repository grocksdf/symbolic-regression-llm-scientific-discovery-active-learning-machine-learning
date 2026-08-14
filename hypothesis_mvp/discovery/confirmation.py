"""Untouched confirmation stage, physically separate from structural selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.hypotheses import EvidenceEventType, EvidenceRegistry, HypothesisSpec

from .equation_runtime import EquationRuntime


@dataclass(frozen=True)
class ConfirmationResult:
    hypothesis_id: str
    passed: bool
    metrics: Mapping[str, float]
    heldout_fingerprint: str
    evidence_registry_path: Path


def confirm_frozen_hypothesis(
    *,
    hypothesis: HypothesisSpec,
    untouched_heldout: RoleDataset,
    evidence_registry_path: str | Path,
    maximum_nmse: float,
) -> ConfirmationResult:
    """Evaluate one already-frozen expression exactly once on held-out rows."""
    if untouched_heldout.role is not DataRole.UNTOUCHED_HELDOUT:
        raise ValueError("confirmation requires an untouched-heldout role")
    threshold = float(maximum_nmse)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("maximum_nmse must be a finite preregistered threshold")
    registry = EvidenceRegistry(evidence_registry_path)
    prior = registry.events(hypothesis_id=hypothesis.hypothesis_id)
    if not any(event.event_type is EvidenceEventType.PROPOSED for event in prior):
        raise RuntimeError(
            "confirmation registry does not contain the frozen hypothesis proposal"
        )
    if any(
        event.event_type is EvidenceEventType.TEST_OBSERVED
        and dict(event.payload).get("stage") == "untouched_heldout_confirmation"
        for event in prior
    ):
        raise RuntimeError("untouched confirmation already exists for this hypothesis")

    runtime = EquationRuntime(len(hypothesis.variables))
    prediction = runtime.predict(hypothesis.expression, untouched_heldout.X)
    truth = untouched_heldout.y
    residual = np.asarray(prediction, dtype=float).reshape(-1) - truth
    mse = float(np.mean(residual**2))
    variance = float(np.var(truth))
    nmse = float(mse / max(variance, 1.0e-15))
    mae = float(np.mean(np.abs(residual)))
    r2 = float(1.0 - np.sum(residual**2) / max(np.sum((truth - np.mean(truth)) ** 2), 1.0e-15))
    metrics = {"mse": mse, "nmse": nmse, "mae": mae, "r2": r2}
    passed = bool(nmse <= threshold)
    registry.append(
        hypothesis_id=hypothesis.hypothesis_id,
        event_type=EvidenceEventType.TEST_OBSERVED,
        payload={
            "stage": "untouched_heldout_confirmation",
            "independent_confirmation": True,
            "heldout_fingerprint": untouched_heldout.fingerprint,
            "preregistered_maximum_nmse": threshold,
            "metrics": metrics,
            "passed": passed,
        },
    )
    registry.append(
        hypothesis_id=hypothesis.hypothesis_id,
        event_type=EvidenceEventType.STATUS_CHANGED,
        payload={
            "from": hypothesis.status.value,
            "to": "supported" if passed else "refuted",
            "reason": "untouched_heldout_confirmation",
        },
    )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError("evidence registry verification failed after confirmation")
    return ConfirmationResult(
        hypothesis_id=hypothesis.hypothesis_id,
        passed=passed,
        metrics=metrics,
        heldout_fingerprint=untouched_heldout.fingerprint,
        evidence_registry_path=Path(evidence_registry_path),
    )


__all__ = ["ConfirmationResult", "confirm_frozen_hypothesis"]
