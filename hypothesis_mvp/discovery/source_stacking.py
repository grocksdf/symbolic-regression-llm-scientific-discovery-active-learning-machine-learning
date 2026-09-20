"""Conservative source-level stacking for frozen discovery banks.

The layer consumes only out-of-fold log predictive densities.  It never sees
candidate-pool, reporting-evaluation, or held-out responses.  A conventional
log-score stacking solution is shrunk toward a registered baseline along a
fixed dyadic path until every fold is no worse than that baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


SCHEMA = "scientific-safe-source-stacking-v1"
METHOD = "oof-log-score-stacking-half-core-reserve-dyadic-fold-safe-v2"


def source_family(candidate) -> str:
    """Collapse raw proposal labels into registered hierarchical sources."""
    if str(candidate.get("origin", "")) == "llm":
        return "llm"
    source = str(candidate.get("source", ""))
    return "engine:mcts" if source == "engine:mcts" else "core"


@dataclass(frozen=True)
class SafeSourceStackingCertificate:
    sources: tuple[str, ...]
    baseline_source: str
    unconstrained_weights: tuple[float, ...]
    weights: tuple[float, ...]
    dyadic_alpha: float
    fold_log_score_gains: tuple[float, ...]
    fold_numerical_tolerances: tuple[float, ...]
    passed: bool
    fallback_to_baseline: bool
    maximum_optional_mass: float = 0.5
    method: str = METHOD

    def __post_init__(self):
        arrays = tuple(np.asarray(value, dtype=float) for value in (
            self.unconstrained_weights, self.weights,
            self.fold_log_score_gains, self.fold_numerical_tolerances))
        if (not self.sources or len(set(self.sources)) != len(self.sources)
                or self.baseline_source not in self.sources
                or arrays[0].shape != (len(self.sources),)
                or arrays[1].shape != (len(self.sources),)
                or len(arrays[2]) < 2 or arrays[2].shape != arrays[3].shape
                or any(not np.all(np.isfinite(value)) for value in arrays)
                or np.any(arrays[0] < 0.0) or np.any(arrays[1] < 0.0)
                or not np.isclose(np.sum(arrays[0]), 1.0, atol=2e-12, rtol=0.0)
                or not np.isclose(np.sum(arrays[1]), 1.0, atol=2e-12, rtol=0.0)
                or np.any(arrays[3] < 0.0)
                or self.dyadic_alpha not in {0.0, *(2.0 ** -k for k in range(1, 9))}
                or self.maximum_optional_mass != 0.5
                or self.method != METHOD):
            raise ValueError("invalid safe source-stacking certificate")
        safe = bool(np.all(arrays[2] + arrays[3] >= 0.0))
        baseline = np.zeros(len(self.sources)); baseline[self.sources.index(self.baseline_source)] = 1.0
        fallback = bool(np.allclose(arrays[1], baseline, rtol=0.0, atol=2e-12))
        if self.passed is not safe or self.fallback_to_baseline is not fallback:
            raise ValueError("source-stacking safety decision is inconsistent")

    @property
    def source_weights(self) -> dict[str, float]:
        return dict(zip(self.sources, self.weights, strict=True))

    @property
    def stable_hash(self) -> str:
        payload = {"schema": SCHEMA, **self.to_dict()}
        return sha256(json.dumps(payload, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {"method": self.method, "sources": list(self.sources),
            "baseline_source": self.baseline_source,
            "unconstrained_weights": list(self.unconstrained_weights),
            "weights": list(self.weights), "dyadic_alpha": self.dyadic_alpha,
            "fold_log_score_gains": list(self.fold_log_score_gains),
            "fold_numerical_tolerances": list(self.fold_numerical_tolerances),
            "maximum_optional_mass": self.maximum_optional_mass,
            "passed": self.passed, "fallback_to_baseline": self.fallback_to_baseline}


def _stacking_weights(log_density: np.ndarray) -> np.ndarray:
    rows, columns = log_density.shape
    if columns == 1:
        return np.ones(1, dtype=float)
    start = np.full(columns, 1.0 / columns)

    def objective(weights):
        return -float(np.sum(logsumexp(log_density + np.log(
            np.maximum(weights, np.finfo(float).tiny))[None, :], axis=1)))

    result = minimize(objective, start, method="SLSQP",
        bounds=[(0.0, 1.0)] * columns,
        constraints={"type": "eq", "fun": lambda value: np.sum(value) - 1.0},
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("source stacking optimization did not converge")
    weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    return weights / np.sum(weights)


def safe_source_stacking(log_predictive_density, fold_ids, sources, *, baseline_source):
    """Stack predictive sources and certify non-inferiority on every fold."""
    values = np.asarray(log_predictive_density, dtype=float)
    folds = np.asarray(fold_ids)
    names = tuple(str(value) for value in sources)
    if (values.ndim != 2 or values.shape[1] != len(names) or values.shape[0] < 4
            or not np.all(np.isfinite(values)) or folds.shape != (len(values),)
            or len(np.unique(folds)) < 2 or any(np.sum(folds == fold) < 2 for fold in np.unique(folds))
            or len(set(names)) != len(names) or baseline_source not in names):
        raise ValueError("invalid out-of-fold source predictive profile")
    unconstrained = _stacking_weights(values)
    baseline = np.zeros(len(names)); baseline[names.index(baseline_source)] = 1.0
    chosen = baseline.copy(); chosen_alpha = 0.0; chosen_gains = None; chosen_tolerances = None
    # At least half of the production prior remains on the registered core.
    # This fixed safeguard prevents a finite calibration sample from replacing
    # the baseline family wholesale while retaining data-adaptive allocation
    # within the optional half.
    for alpha in (*(2.0 ** -k for k in range(1, 9)), 0.0):
        weights = alpha * unconstrained + (1.0 - alpha) * baseline
        mixture = logsumexp(values + np.log(np.maximum(
            weights, np.finfo(float).tiny))[None, :], axis=1)
        reference = values[:, names.index(baseline_source)]
        gains, tolerances = [], []
        for fold in np.unique(folds):
            active = folds == fold
            gain = float(np.sum(mixture[active] - reference[active]))
            scale = max(1.0, float(np.sum(np.abs(mixture[active]))),
                        float(np.sum(np.abs(reference[active]))))
            gains.append(gain)
            tolerances.append(float(1024.0 * np.finfo(float).eps * scale))
        if all(gain + tolerance >= 0.0 for gain, tolerance in zip(gains, tolerances)):
            chosen, chosen_alpha = weights, float(alpha)
            chosen_gains, chosen_tolerances = gains, tolerances
            break
    if chosen_gains is None:
        raise AssertionError("baseline must certify itself")
    return SafeSourceStackingCertificate(names, baseline_source,
        tuple(float(value) for value in unconstrained),
        tuple(float(value) for value in chosen), chosen_alpha,
        tuple(chosen_gains), tuple(chosen_tolerances), True,
        bool(np.allclose(chosen, baseline, rtol=0.0, atol=2e-12)), 0.5)


def arbitration_source_log_predictive(candidates, initial_data, arbitration, *,
                                      n_features, prior, exploration_identity,
                                      coefficient_policy, measurement_budget,
                                      action_domain):
    """Score frozen source families on an independent calibration split."""
    from dataclasses import replace
    from hypothesis_mvp.data.roles import DataRole
    from hypothesis_mvp.pcpi.reference import ExactPosterior
    from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target

    if (initial_data.role is not DataRole.DEVELOPMENT
            or arbitration.role is not DataRole.VALIDATION
            or len(arbitration.X) < 4 or len(arbitration.X) % 2):
        raise ValueError("source admission requires registered H0 and even arbitration data")
    families = tuple(sorted({source_family(row) for row in candidates}))
    if "core" not in families:
        raise ValueError("source admission requires the registered core baseline")
    expression_family = {(str(row["source"]), str(row["expression"])): source_family(row)
                         for row in candidates}
    model = freeze_discovery_model(candidates, n_features=n_features, prior=prior,
        exploration_identity=exploration_identity, coefficient_policy=coefficient_policy)
    target = freeze_discovery_target(model, initial_data, action_domain,
        measurement_budget=measurement_budget, expected_model_identity=model.stable_hash)
    identifier_family = {}
    for source, expression, identifier in model.candidate_bindings:
        family = expression_family[(source, expression)]
        previous = identifier_family.setdefault(identifier, family)
        if previous != family:
            raise ValueError("one structural support crossed source families")
    profile = np.empty((len(arbitration.X), len(families)), dtype=float)
    for column, family in enumerate(families):
        members = [member for member in target.initial_posterior.members
                   if identifier_family[member.structure.structure_id] == family]
        mass = float(sum(member.probability for member in members))
        if not members or mass <= 0.0:
            raise ValueError("source family has no calibration predictive mass")
        normalized = tuple(replace(member, probability=member.probability / mass)
                           for member in members)
        posterior = ExactPosterior(normalized, 0.0,
            target.initial_posterior.bank_hash,
            target.initial_posterior.likelihood_power)
        profile[:, column] = model.engine(model.stable_hash).predictive_logpdf(
            posterior, arbitration.X, arbitration.y)
    return profile, np.arange(len(arbitration.X)) % 2, families


def calibrate_source_admission(candidates, initial_data, arbitration, **kwargs):
    """Return a fold-safe source prior and per-source admission certificates."""
    values, folds, families = arbitration_source_log_predictive(
        candidates, initial_data, arbitration, **kwargs)
    baseline = values[:, families.index("core")]
    diagnostics = {}
    for column, family in enumerate(families):
        gains, tolerances = [], []
        for fold in np.unique(folds):
            active = folds == fold
            gain = float(np.sum(values[active, column] - baseline[active]))
            scale = max(1.0, float(np.sum(np.abs(values[active, column]))),
                        float(np.sum(np.abs(baseline[active]))))
            gains.append(gain)
            tolerances.append(float(1024.0 * np.finfo(float).eps * scale))
        diagnostics[family] = (gains, tolerances)
    eligible = tuple(family for family in families if family == "core" or all(
        gain + tolerance >= 0.0
        for gain, tolerance in zip(*diagnostics[family])))
    indices = [families.index(family) for family in eligible]
    admitted_certificate = safe_source_stacking(
        values[:, indices], folds, eligible, baseline_source="core")
    admitted_weights = admitted_certificate.source_weights
    weights = tuple(float(admitted_weights.get(family, 0.0)) for family in families)
    unconstrained_map = dict(zip(admitted_certificate.sources,
                                 admitted_certificate.unconstrained_weights, strict=True))
    unconstrained = tuple(float(unconstrained_map.get(family, 0.0)) for family in families)
    certificate = SafeSourceStackingCertificate(
        families, "core", unconstrained, weights,
        admitted_certificate.dyadic_alpha,
        admitted_certificate.fold_log_score_gains,
        admitted_certificate.fold_numerical_tolerances,
        admitted_certificate.passed,
        admitted_certificate.fallback_to_baseline,
        admitted_certificate.maximum_optional_mass)
    sources = {}
    for family in families:
        gains, tolerances = diagnostics[family]
        weight = certificate.source_weights[family]
        admitted = family == "core" or (family in eligible and weight > 2e-12)
        harmful = family != "core" and any(
            gain < -tolerance for gain, tolerance in zip(gains, tolerances))
        sources[family] = {"weight": weight, "admitted": admitted,
            "fold_log_score_gains_vs_core": gains,
            "fold_numerical_tolerances": tolerances,
            "negative_transfer_certified": bool(not admitted and harmful)}
    return certificate, sources


def filter_fold_safe_source_candidates(candidates, initial_data, arbitration, **kwargs):
    """Admit optional supports individually before family-level stacking.

    Each optional support is paired with the unchanged complete core bank and
    must beat the core predictive density on every arbitration fold.  The
    procedure never searches subsets jointly, so one candidate cannot mask
    another and the result is deterministic in candidate identity order.
    """
    rows = tuple(dict(row) for row in candidates)
    core = tuple(row for row in rows if source_family(row) == "core")
    optional = tuple(row for row in rows if source_family(row) != "core")
    if len(core) < 2:
        raise ValueError("candidate admission requires at least two core supports")
    retained, certificates = list(core), []
    ordered = sorted(optional, key=lambda row: sha256(json.dumps(
        row, sort_keys=True, default=str).encode()).hexdigest())
    for candidate in ordered:
        family = source_family(candidate)
        values, folds, families = arbitration_source_log_predictive(
            [*core, candidate], initial_data, arbitration, **kwargs)
        baseline = values[:, families.index("core")]
        proposed = values[:, families.index(family)]
        gains, tolerances = [], []
        for fold in np.unique(folds):
            active = folds == fold
            gain = float(np.sum(proposed[active] - baseline[active]))
            scale = max(1.0, float(np.sum(np.abs(proposed[active]))),
                        float(np.sum(np.abs(baseline[active]))))
            gains.append(gain)
            tolerances.append(float(1024.0 * np.finfo(float).eps * scale))
        admitted = all(gain > tolerance
                       for gain, tolerance in zip(gains, tolerances))
        if admitted:
            retained.append(candidate)
        certificates.append({
            "candidate_identity": sha256(json.dumps(candidate, sort_keys=True,
                default=str).encode()).hexdigest(),
            "source": str(candidate["source"]), "origin": str(candidate.get("origin", "")),
            "family": family, "admitted": admitted,
            "fold_log_score_gains_vs_core": gains,
            "fold_numerical_tolerances": tolerances,
            "negative_transfer_certified": bool(not admitted and any(
                gain < -tolerance for gain, tolerance in zip(gains, tolerances))),
            "candidate_response_accessed": False, "heldout_opened": False,
        })
    return tuple(retained), {
        "schema": "scientific-independent-candidatewise-admission-v1",
        "candidate_certificates": certificates,
        "input_candidate_count": len(rows), "retained_candidate_count": len(retained),
        "candidate_response_accessed": False, "heldout_opened": False,
    }


def filter_conditionally_complementary_engine_candidates(
        candidates, initial_data, arbitration, **kwargs):
    """Require optional-engine gain beyond the already safe core+LLM mixture."""
    rows = tuple(dict(row) for row in candidates)
    reference = tuple(row for row in rows
                      if source_family(row) in {"core", "llm"})
    engines = tuple(row for row in rows
                    if source_family(row) == "engine:mcts")
    if not engines:
        return rows, {
            "schema": "scientific-conditional-engine-complementarity-v1",
            "candidate_certificates": [], "retained_engine_count": 0,
            "candidate_response_accessed": False, "heldout_opened": False,
        }
    reference_values, folds, reference_families = arbitration_source_log_predictive(
        reference, initial_data, arbitration, **kwargs)
    if set(reference_families) != {"core", "llm"}:
        raise ValueError("conditional engine admission requires core and LLM reference")
    reference_certificate = safe_source_stacking(
        reference_values, folds, reference_families, baseline_source="core")
    reference_weights = np.asarray(reference_certificate.weights, dtype=float)
    reference_log_density = logsumexp(
        reference_values + np.log(np.maximum(
            reference_weights, np.finfo(float).tiny))[None, :], axis=1)
    retained, certificates = list(reference), []
    ordered = sorted(engines, key=lambda row: sha256(json.dumps(
        row, sort_keys=True, default=str).encode()).hexdigest())
    for candidate in ordered:
        values, candidate_folds, families = arbitration_source_log_predictive(
            [*reference, candidate], initial_data, arbitration, **kwargs)
        if not np.array_equal(candidate_folds, folds):
            raise ValueError("conditional engine admission fold identity changed")
        full_certificate = safe_source_stacking(
            values, folds, families, baseline_source="core")
        full_log_density = logsumexp(
            values + np.log(np.maximum(
                np.asarray(full_certificate.weights), np.finfo(float).tiny))[None, :],
            axis=1)
        gains, tolerances = [], []
        for fold in np.unique(folds):
            active = folds == fold
            gain = float(np.sum(full_log_density[active]
                                - reference_log_density[active]))
            scale = max(1.0, float(np.sum(np.abs(full_log_density[active]))),
                        float(np.sum(np.abs(reference_log_density[active]))))
            gains.append(gain)
            tolerances.append(float(1024.0 * np.finfo(float).eps * scale))
        engine_weight = full_certificate.source_weights.get("engine:mcts", 0.0)
        admitted = bool(engine_weight > 2e-12 and all(
            gain > tolerance for gain, tolerance in zip(gains, tolerances)))
        if admitted:
            retained.append(candidate)
        certificates.append({
            "candidate_identity": sha256(json.dumps(candidate, sort_keys=True,
                default=str).encode()).hexdigest(),
            "source": str(candidate["source"]), "admitted": admitted,
            "conditional_fold_log_score_gains": gains,
            "fold_numerical_tolerances": tolerances,
            "conditional_source_weight": float(engine_weight),
            "conditionally_redundant_certified": bool(
                not admitted and engine_weight <= 2e-12
                and all(gain + tolerance >= 0.0
                        for gain, tolerance in zip(gains, tolerances))),
            "negative_transfer_certified": bool(not admitted and any(
                gain < -tolerance for gain, tolerance in zip(gains, tolerances))),
            "candidate_response_accessed": False, "heldout_opened": False,
        })
    return tuple(retained), {
        "schema": "scientific-conditional-engine-complementarity-v1",
        "reference_stacking": reference_certificate.to_dict(),
        "candidate_certificates": certificates,
        "retained_engine_count": sum(row["admitted"] for row in certificates),
        "candidate_response_accessed": False, "heldout_opened": False,
    }


def crossfit_source_log_predictive(candidates, initial_data, action_domain, *,
                                   n_features, prior, exploration_identity,
                                   coefficient_policy, measurement_budget):
    """Build a two-fold source-predictive profile from H0 only."""
    from dataclasses import replace
    from hypothesis_mvp.data.roles import DataRole, RoleDataset
    from hypothesis_mvp.pcpi.reference import ExactPosterior
    from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target

    if initial_data.role is not DataRole.DEVELOPMENT or len(initial_data.X) < 4:
        raise ValueError("source stacking requires registered H0 development data")
    families = tuple(sorted({source_family(row) for row in candidates}))
    if "core" not in families or len(families) < 1:
        raise ValueError("source stacking requires the registered core baseline")
    expression_family = {(str(row["source"]), str(row["expression"])): source_family(row)
                         for row in candidates}
    profile = np.empty((len(initial_data.X), len(families)), dtype=float)
    fold_ids = np.arange(len(initial_data.X)) % 2
    for fold in range(2):
        held = fold_ids == fold
        training = RoleDataset(DataRole.DEVELOPMENT,
            initial_data.X[~held], initial_data.y[~held])
        model = freeze_discovery_model(candidates, n_features=n_features, prior=prior,
            exploration_identity=exploration_identity,
            coefficient_policy=coefficient_policy)
        target = freeze_discovery_target(model, training, action_domain,
            measurement_budget=measurement_budget,
            expected_model_identity=model.stable_hash)
        identifier_family = {}
        for source, expression, identifier in model.candidate_bindings:
            family = expression_family[(source, expression)]
            previous = identifier_family.setdefault(identifier, family)
            if previous != family:
                raise ValueError("one structural support crossed source families")
        for column, family in enumerate(families):
            members = [member for member in target.initial_posterior.members
                       if identifier_family[member.structure.structure_id] == family]
            mass = float(sum(member.probability for member in members))
            if not members or mass <= 0.0:
                raise ValueError("source family has no posterior predictive mass")
            normalized = tuple(replace(member, probability=member.probability / mass)
                               for member in members)
            posterior = ExactPosterior(normalized, 0.0,
                target.initial_posterior.bank_hash,
                target.initial_posterior.likelihood_power)
            profile[held, column] = model.engine(model.stable_hash).predictive_logpdf(
                posterior, initial_data.X[held], initial_data.y[held])
    return profile, fold_ids, families


__all__ = ["SafeSourceStackingCertificate", "arbitration_source_log_predictive",
           "calibrate_source_admission", "filter_fold_safe_source_candidates",
           "filter_conditionally_complementary_engine_candidates",
           "crossfit_source_log_predictive",
           "safe_source_stacking", "source_family"]
