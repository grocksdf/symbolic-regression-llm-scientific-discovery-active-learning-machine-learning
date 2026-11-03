"""Freeze, test and deploy the SAME calibrated predictive mixture.

This is predictive stacking plus a fixed-law evidence screen, not a new law
posterior or a guarantee of improved future risk. Conditional on training and
calibration, the likelihood-ratio interpretation requires the registered E0
product predictive law as the null. It does not follow under arbitrary truth.
Neither stacking nor sample splitting is claimed as novel here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.special import logsumexp

from .regional_candidate_admission import global_protected_evidence_decision


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _ids(values, name):
    result = tuple(values)
    if (not result or any(not isinstance(v, str) or not v for v in result)
            or len(set(result)) != len(result)):
        raise ValueError(f"{name} needs unique nonempty identities")
    return result


@dataclass(frozen=True)
class ExpansionDesign:
    """Response-independent row identities and policy, fixed before calibration.

    training_ids includes ALL rows used by fitting or adaptive generation;
    calibration, admission and reporting must each be disjoint from that union.
    Region labels are frozen covariate-only assignments in admission row order.
    Even a rejected candidate/mixture remains in the registered attempt count.
    """

    training_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    admission_ids: tuple[str, ...]
    reporting_ids: tuple[str, ...]
    admission_regions: tuple[int, ...]
    eligible_regions: tuple[int, ...]
    alpha: float = .05
    tested_mixture_count: int = 1

    def __post_init__(self):
        seen = set()
        for name in ("training_ids", "calibration_ids", "admission_ids", "reporting_ids"):
            values = _ids(getattr(self, name), name)
            if seen.intersection(values):
                raise ValueError("generation/calibration/admission/reporting roles overlap")
            seen.update(values)
            object.__setattr__(self, name, values)
        labels, regions = tuple(self.admission_regions), tuple(self.eligible_regions)
        if (len(labels) != len(self.admission_ids) or not regions
                or any(type(v) is not int or v < 0 for v in (*labels, *regions))
                or len(set(regions)) != len(regions)
                or not set(labels).issubset(regions)):
            raise ValueError("invalid registered admission regions")
        if (not math.isfinite(self.alpha) or not 0 < self.alpha < 1
                or type(self.tested_mixture_count) is not int
                or self.tested_mixture_count < 1):
            raise ValueError("invalid fixed evidence policy")
        object.__setattr__(self, "admission_regions", labels)
        object.__setattr__(self, "eligible_regions", regions)

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class FrozenPredictiveMixture:
    """Component identities MUST bind fitted parameters/posteriors, not names."""

    component_ids: tuple[str, ...]
    weights: tuple[float, ...]
    calibration_digest: str
    design_identity: str

    def __post_init__(self):
        components = _ids(self.component_ids, "components")
        weights = tuple(float(v) for v in self.weights)
        if (len(weights) != len(components) or not all(math.isfinite(v) for v in weights)
                or any(v < 0 for v in weights)
                or abs(math.fsum(weights) - 1.) > 1e-12):
            raise ValueError("frozen mixture needs a finite simplex")
        if not self.calibration_digest or not self.design_identity:
            raise ValueError("frozen mixture needs calibration/design identities")
        object.__setattr__(self, "component_ids", components)
        object.__setattr__(self, "weights", weights)

    @property
    def identity(self):
        return digest(asdict(self))

    def log_density(self, values, *, component_ids):
        values = _matrix(values, component_ids, self.component_ids)
        weights = np.asarray(self.weights)
        active = weights > 0
        with np.errstate(over="raise", invalid="raise"):
            result = logsumexp(values[:, active] + np.log(weights[active]), axis=1)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite frozen mixture density")
        return result


def _matrix(values, component_ids, expected):
    if tuple(component_ids) != expected:
        raise ValueError("fitted component identity/order changed")
    values = np.array(values, dtype=float, copy=True)
    if (values.ndim != 2 or not len(values) or values.shape[1] != len(expected)
            or not np.all(np.isfinite(values))):
        raise ValueError("density matrix must be finite and component-aligned")
    return values


class PredictiveExpansionSession:
    """One calibration -> one admission -> one reporting transaction.

    Calibration, admission and reporting accept densities, never fit models.
    An evaluation exception terminally closes the session; no silent retry or
    replacement of negative rows. A new registered run needs a new session.
    Public audit objects are copies, so callers cannot alter the decision.
    """

    def __init__(self, design: ExpansionDesign, component_ids):
        self._design = design
        self._components = _ids(component_ids, "components")
        self._phase = "uncalibrated"
        self._mixture = None
        self._selected = None
        self._decision = None

    @property
    def phase(self):
        return self._phase

    @property
    def mixture(self):
        return self._mixture

    def _batch(self, values, ids, role, component_ids):
        if tuple(ids) != getattr(self._design, f"{role}_ids"):
            raise ValueError(f"{role} row identity/order mismatch")
        values = _matrix(values, component_ids, self._components)
        if len(values) != len(ids):
            raise ValueError(f"{role} row count mismatch")
        return values

    def freeze(self, calibration, weights, *, row_ids, component_ids):
        if self._phase != "uncalibrated":
            raise ValueError("mixture already frozen or session closed")
        self._phase = "failed"
        values = self._batch(calibration, row_ids, "calibration", component_ids)
        mixture = FrozenPredictiveMixture(
            self._components, tuple(weights), digest(values.tolist()), self._design.identity)
        means = np.sum(values / len(values), axis=0)
        if not np.all(np.isfinite(means)):
            raise FloatingPointError("nonfinite calibration component score")
        selected = min(range(len(means)), key=lambda i: (-means[i], self._components[i]))
        self._mixture, self._selected = mixture, selected
        self._phase = "calibration_frozen"
        return {"schema": "frozen-predictive-expansion-v1", "design": asdict(self._design),
                "mixture": asdict(mixture), "mixture_identity": mixture.identity,
                "independent_validation_selected_component": self._components[selected],
                "weights_are_posterior_probabilities": False}

    def admit(self, admission, *, row_ids, component_ids):
        if self._phase != "calibration_frozen":
            raise ValueError("admission requires a frozen, untested mixture")
        self._phase = "failed"
        values = self._batch(admission, row_ids, "admission", component_ids)
        with np.errstate(over="raise", invalid="raise"):
            ratios = self._mixture.log_density(values, component_ids=component_ids) - values[:, 0]
        threshold = math.log(self._design.tested_mixture_count) - math.log(self._design.alpha)
        decision = global_protected_evidence_decision(
            ratios, np.asarray(self._design.admission_regions),
            self._design.eligible_regions, threshold)
        if (not np.isfinite(decision["global_log_predictive_ratio"])
                or not np.all(np.isfinite(decision["region_log_predictive_ratios"]))):
            raise FloatingPointError("nonfinite aggregate admission evidence")
        self._decision = {"schema": "frozen-predictive-expansion-admission-v1",
                          **decision, "mixture_identity": self._mixture.identity,
                          "admission_density_digest": digest(values.tolist()),
                          "tested_mixture_count": self._design.tested_mixture_count,
                          "future_risk_improvement_guaranteed": False}
        self._phase = "admission_frozen"
        return json.loads(json.dumps(self._decision, allow_nan=False))

    def report(self, reporting, *, row_ids, component_ids):
        if self._phase != "admission_frozen":
            raise ValueError("reporting requires a frozen admission decision")
        self._phase = "failed"
        values = self._batch(reporting, row_ids, "reporting", component_ids)
        mixture = self._mixture.log_density(values, component_ids=component_ids)
        accepted = self._decision["passed"]
        deployed = mixture if accepted else values[:, 0]
        gains = {"engine": 0.,
                 "independent_validation": float(np.mean(values[:, self._selected] - values[:, 0])),
                 "predictive_stacking": float(np.mean(mixture - values[:, 0])),
                 "screened_predictive_stacking": float(np.mean(deployed - values[:, 0]))}
        if not all(np.isfinite(v) for v in gains.values()):
            raise FloatingPointError("nonfinite reporting gain")
        self._phase = "complete"
        return {"schema": "frozen-predictive-expansion-report-v1", "gains": gains,
                "mixture_accepted": accepted, "mixture_identity": self._mixture.identity,
                "deployed_identity": self._mixture.identity if accepted else self._components[0],
                "decision_identity": digest(self._decision),
                "reporting_density_digest": digest(values.tolist()),
                "reporting_used_for_selection": False,
                "claim_boundary": "Frozen bank-conditional predictive comparison only; "
                                  "not formula truth, posterior model probabilities or future-risk guarantee."}
