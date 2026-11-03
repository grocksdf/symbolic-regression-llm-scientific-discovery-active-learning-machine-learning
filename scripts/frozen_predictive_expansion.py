"""Adapter for the calibrated Bank-mixture comparison (no execution on import).

One fitted component family serves all four arms. Reader callbacks enforce
calibration -> durable freeze -> admission -> durable decision -> reporting.
This adapter never calls a provider, refits after calibration or opens HDF5.
The owning registered runner must supply source row IDs and authorized readers.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from hypothesis_mvp.data.roles import DataRole
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model,
)
from hypothesis_mvp.discovery.predictive_bank_expansion import (
    ExpansionDesign, PredictiveExpansionSession, digest,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import compile_fixed_formula_support
from scripts.stacking_diagnostics import fit_stacking_weights


def _posterior_id(model, posterior, fit_identity):
    payload = json.dumps({"model": model.stable_hash, "posterior": asdict(posterior),
                          "fit_identity": fit_identity},
                         sort_keys=True, allow_nan=False, default=lambda v: v.tolist())
    return sha256(payload.encode()).hexdigest()


class FrozenBankComponents:
    """E0 and E0+c components fit ONCE, with identities binding actual posteriors."""

    def __init__(self, core, candidates, fit, *, fit_row_ids, design: ExpansionDesign,
                 prior=None):
        fit_ids = tuple(fit_row_ids)
        if (fit.role is not DataRole.DEVELOPMENT or len(fit_ids) != len(fit.X)
                or len(set(fit_ids)) != len(fit_ids)
                or not set(fit_ids).issubset(design.training_ids)):
            raise ValueError("fitting rows must belong to the registered training role")
        if not core:
            raise ValueError("protected core must be nonempty")
        core = json.loads(json.dumps(core, allow_nan=False))
        candidates = json.loads(json.dumps(candidates, allow_nan=False))
        self._design_identity = design.identity
        self._fit_identity = digest({"data": fit.fingerprint, "row_ids": fit_ids})
        seen_supports = {tuple(compile_fixed_formula_support(row["expression"], fit.X.shape[1]))
                         for row in core}
        for candidate in candidates:
            support = tuple(compile_fixed_formula_support(candidate["expression"], fit.X.shape[1]))
            if support in seen_supports:
                raise ValueError("optional candidates must have unique non-core supports")
            seen_supports.add(support)
        self._components = []
        banks = [core, *[[*core, candidate] for candidate in candidates]]
        for bank in banks:
            model = freeze_expanded_formula_model(
                bank, n_features=fit.X.shape[1], prior=prior or NormalInverseGammaPrior(),
                exploration_identity=digest(bank), coefficient_policy=EXPANDED_FORMULA_POLICY,
                minimum_supports=1)
            engine = model.engine(model.stable_hash)
            posterior = engine.fit_batch(fit.X, fit.y)
            self._components.append((model, engine, posterior,
                                     _posterior_id(model, posterior, self._fit_identity)))
        self._components = tuple(self._components)
        self._ids = tuple(component[3] for component in self._components)
        if len(set(self._ids)) != len(self._ids):
            raise ValueError("duplicate fitted component identities")
        self.evaluation_count = 0

    @property
    def component_ids(self):
        return self._ids

    @property
    def design_identity(self):
        return self._design_identity

    def densities(self, data):
        if data.role is not DataRole.VALIDATION:
            raise ValueError("only registered development validation roles may be read")
        columns = []
        for model, engine, posterior, expected in self._components:
            if (_posterior_id(model, posterior, self._fit_identity) != expected
                    or engine.bank.stable_hash != model.bank.stable_hash):
                raise ValueError("fitted component changed after freeze")
            values = np.asarray(engine.predictive_logpdf(posterior, data.X, data.y), dtype=float)
            self.evaluation_count += 1
            if values.shape != (len(data.X),) or not np.all(np.isfinite(values)):
                raise FloatingPointError("nonfinite component density")
            columns.append(values)
        return np.column_stack(columns)


def _publish(path, value):
    """Exclusive creation and durable flush. Never overwrite historical artifacts."""
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def run_comparison(*, design, components, read_calibration, read_admission,
                   read_reporting, output_root, seed):
    """Readers return (source row IDs, RoleDataset); no automatic retries.

    A fresh output directory is required, even if an old directory is empty.
    Raw responses are not serialized. Failure is terminal and preserves every
    earlier record. This is an adapter, not authorization for a real experiment.
    """
    root = Path(output_root)
    if components.design_identity != design.identity:
        raise ValueError("component design identity mismatch")
    session = PredictiveExpansionSession(design, components.component_ids)
    root.mkdir(parents=True, exist_ok=False)
    initial_evaluation_count = components.evaluation_count
    stage = "calibration"
    try:
        _publish(root / "PLAN.json", {"schema": "predictive-expansion-plan-v1",
                 "design": asdict(design), "component_ids": components.component_ids,
                 "optimizer_seed": seed, "optimizer_random_starts": 4,
                 "arms": ["engine", "independent_validation", "predictive_stacking",
                          "screened_predictive_stacking"]})

        def read_role(reader, role):
            ids, data = reader()
            if tuple(ids) != getattr(design, f"{role}_ids") or len(data.X) != len(ids):
                raise ValueError(f"{role} source row identity/count mismatch")
            return ids, components.densities(data)

        ids, calibration = read_role(read_calibration, "calibration")
        weights, optimizer = fit_stacking_weights(calibration, seed=seed, random_starts=4)
        frozen = session.freeze(calibration, weights, row_ids=ids,
                                component_ids=components.component_ids)
        _publish(root / "CALIBRATION_FROZEN.json", {**frozen, "optimizer": optimizer})
        stage = "admission"
        ids, admission = read_role(read_admission, "admission")
        decision = session.admit(admission, row_ids=ids, component_ids=components.component_ids)
        _publish(root / "ADMISSION_FROZEN.json", decision)
        stage = "reporting"
        ids, reporting = read_role(read_reporting, "reporting")
        result = session.report(reporting, row_ids=ids, component_ids=components.component_ids)
        result.update({"component_density_evaluations": components.evaluation_count - initial_evaluation_count,
                       "component_count": len(components.component_ids),
                       "candidate_count": len(components.component_ids) - 1,
                       "protocol_complete": True, "provider_calls": 0,
                       "post_calibration_refit": False})
        _publish(root / "RESULT.json", result)
        return result
    except Exception as error:
        _publish(root / "FAILURE.json", {"schema": "predictive-expansion-failure-v1",
                 "stage": stage, "error_type": type(error).__name__,
                 "message": str(error), "protocol_complete": False,
                 "retry_authorized": False})
        raise
