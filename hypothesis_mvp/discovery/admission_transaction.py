"""Admit-then-independent-update transaction for expanded global banks."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior

from .pcpi_adapter import (
    freeze_discovery_target, model_factory_for_policy,
)


@dataclass(frozen=True)
class AdmissionPosteriorTransaction:
    model_identity: str
    target_identity: str
    fit_identity: str
    admission_identity: str
    update_identity: str
    admitted_candidate_count: int
    update_row_count: int
    candidate_response_accessed: bool = False
    heldout_opened: bool = False


def freeze_admitted_posterior_transaction(
    core_rows: Sequence[Mapping], admitted_rows: Sequence[Mapping],
    fit: RoleDataset, admission: RoleDataset,
    posterior_update: RoleDataset, action_domain: np.ndarray,
    *, prior: NormalInverseGammaPrior, measurement_budget: int,
    coefficient_policy: str,
):
    """Freeze an expanded posterior only after independent admission.

    Admission responses authorize structure inclusion but are not reused for
    posterior fitting. A distinct update role is appended once to the original
    fit role. Reporting and held-out roles are never accepted here.
    """
    if (fit.role is not DataRole.DEVELOPMENT
            or admission.role is not DataRole.VALIDATION
            or posterior_update.role is not DataRole.VALIDATION
            or not admitted_rows
            or admission.row_fingerprints & fit.row_fingerprints
            or posterior_update.row_fingerprints & fit.row_fingerprints
            or posterior_update.row_fingerprints
               & admission.row_fingerprints):
        raise ValueError(
            "admitted posterior transaction requires three disjoint roles")
    rows = [*(dict(row) for row in core_rows),
            *(dict(row) for row in admitted_rows)]
    identity = sha256(json.dumps(
        rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
    model = model_factory_for_policy(coefficient_policy)(
        rows, n_features=fit.X.shape[1], prior=prior,
        exploration_identity=identity,
        coefficient_policy=coefficient_policy)
    combined = RoleDataset(
        DataRole.DEVELOPMENT,
        np.vstack((fit.X, posterior_update.X)),
        np.concatenate((fit.y, posterior_update.y)))
    target = freeze_discovery_target(
        model, combined, action_domain,
        measurement_budget=measurement_budget,
        expected_model_identity=model.stable_hash)
    audit = AdmissionPosteriorTransaction(
        model.stable_hash, target.stable_hash,
        fit.fingerprint, admission.fingerprint,
        posterior_update.fingerprint, len(admitted_rows),
        len(posterior_update.X))
    return model, target, audit


__all__ = [
    "AdmissionPosteriorTransaction",
    "freeze_admitted_posterior_transaction",
]
