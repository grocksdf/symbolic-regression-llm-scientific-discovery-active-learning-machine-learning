"""Frozen, target-blind role protocol for the registered real P2A datasets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np

from .real_registry import RealDatasetFrame
from .roles import (
    AcquisitionCovariates,
    covariate_fingerprint,
    DataRole,
    RoleDataset,
    SelectionData,
)
from .oracle import PoolOracle


P2A_REAL_DATASETS = (
    "uci_ccpp",
    "uci_gas_turbine_co",
    "uci_gas_turbine_nox",
)
SPLIT_SEED = 20260807


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _row_id_hash(row_ids: np.ndarray) -> str:
    return _canonical_hash([str(item) for item in row_ids])


def _source_hash(source_hashes: tuple[str, ...]) -> str:
    return _canonical_hash(list(source_hashes))


def _hash_partition(row_ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    order = np.argsort(
        np.asarray(
            [
                sha256(f"{seed}:{row_id}".encode("utf-8")).digest()
                for row_id in row_ids
            ],
            dtype="|S32",
        ),
        kind="stable",
    )
    count = len(order)
    boundaries = np.rint(np.asarray((0.60, 0.75, 0.90)) * count).astype(int)
    return {
        "development": order[: boundaries[0]],
        "validation": order[boundaries[0] : boundaries[1]],
        "acquisition-pool": order[boundaries[1] : boundaries[2]],
        "untouched-heldout": order[boundaries[2] :],
    }


def _group_partition(groups: np.ndarray) -> dict[str, np.ndarray]:
    years = np.asarray(groups, dtype=int)
    expected = {2011, 2012, 2013, 2014, 2015}
    if set(np.unique(years)) != expected:
        raise ValueError("Gas Turbine role protocol requires groups 2011 through 2015")
    return {
        "development": np.flatnonzero(np.isin(years, (2011, 2012))),
        "validation": np.flatnonzero(years == 2013),
        "acquisition-pool": np.flatnonzero(years == 2014),
        "untouched-heldout": np.flatnonzero(years == 2015),
    }


def _role_indices(frame: RealDatasetFrame, seed: int) -> tuple[dict[str, np.ndarray], str]:
    if frame.dataset_id == "uci_ccpp":
        return _hash_partition(frame.row_ids, seed), "target_blind_sha256_60_15_15_10"
    if frame.dataset_id in {"uci_gas_turbine_co", "uci_gas_turbine_nox"}:
        if frame.groups is None:
            raise ValueError("Gas Turbine data require year groups")
        return _group_partition(frame.groups), "year_group_2011_12_2013_2014_2015"
    raise ValueError(f"{frame.dataset_id!r} is not enabled for the P2A real protocol")


@dataclass(frozen=True)
class PreparedRealSelection:
    """Curator output; only ``selection`` crosses into the inference layer."""

    dataset_id: str
    feature_names: tuple[str, ...]
    target_name: str
    selection: SelectionData
    development_row_ids: np.ndarray
    validation_row_ids: np.ndarray
    acquisition_pool_row_ids: np.ndarray
    source_hashes: tuple[str, ...]
    split_manifest: dict[str, Any]

    @property
    def combined_source_hash(self) -> str:
        return _source_hash(self.source_hashes)


def prepare_real_selection(
    frame: RealDatasetFrame,
    *,
    split_seed: int = SPLIT_SEED,
) -> PreparedRealSelection:
    """Create visible roles and a commitment-only held-out record.

    Role assignment never reads target values.  No held-out array, path, shape,
    target range, summary statistic, or metadata is placed in ``SelectionData``.
    """

    if frame.dataset_id not in P2A_REAL_DATASETS:
        raise ValueError(f"dataset is outside the P2A real protocol: {frame.dataset_id}")
    indices, protocol = _role_indices(frame, split_seed)
    development_index = indices["development"]
    validation_index = indices["validation"]
    pool_index = indices["acquisition-pool"]
    development = RoleDataset(
        DataRole.DEVELOPMENT,
        frame.X[development_index],
        frame.y[development_index],
    )
    validation = RoleDataset(
        DataRole.VALIDATION,
        frame.X[validation_index],
        frame.y[validation_index],
    )
    pool_covariates = frame.X[pool_index]
    acquisition = AcquisitionCovariates(
        DataRole.ACQUISITION_POOL,
        pool_covariates,
        covariate_fingerprint(pool_covariates),
    )
    visible_manifest = (
        (DataRole.DEVELOPMENT.value, development.fingerprint),
        (DataRole.VALIDATION.value, validation.fingerprint),
        (DataRole.ACQUISITION_POOL.value, acquisition.source_fingerprint),
    )
    selection = SelectionData(development, validation, acquisition, visible_manifest)
    family = "uci_gas_turbine" if frame.dataset_id.startswith("uci_gas_turbine_") else frame.dataset_id
    role_hashes = {
        role: _row_id_hash(frame.row_ids[index])
        for role, index in indices.items()
    }
    assignment = {
        "schema": "pcpi-real-split-v1",
        "dataset_family": family,
        "source_hash": _source_hash(frame.source_hashes),
        "split_seed": split_seed,
        "protocol": protocol,
        "role_row_id_hashes": role_hashes,
    }
    split_manifest = {
        **assignment,
        "split_hash": _canonical_hash(assignment),
        "roles_exposed_to_inference": [
            "development",
            "validation",
            "acquisition-pool-covariates",
        ],
        "untouched_heldout": {
            "opened": False,
            "selection_used": False,
            "row_id_commitment": role_hashes["untouched-heldout"],
        },
    }
    return PreparedRealSelection(
        dataset_id=frame.dataset_id,
        feature_names=frame.feature_names,
        target_name=frame.target_name,
        selection=selection,
        development_row_ids=np.asarray(frame.row_ids[development_index]),
        validation_row_ids=np.asarray(frame.row_ids[validation_index]),
        acquisition_pool_row_ids=np.asarray(frame.row_ids[pool_index]),
        source_hashes=frame.source_hashes,
        split_manifest=split_manifest,
    )


def prepare_real_pool_oracle(
    frame: RealDatasetFrame,
    prepared: PreparedRealSelection,
    *,
    split_seed: int = SPLIT_SEED,
) -> PoolOracle:
    """Create the curator-only replay oracle for a visible measured pool."""

    if frame.dataset_id != prepared.dataset_id:
        raise ValueError("prepared selection and pool oracle dataset differ")
    indices, _ = _role_indices(frame, split_seed)
    pool_index = indices["acquisition-pool"]
    covariates = np.asarray(frame.X[pool_index], dtype=float)
    visible = prepared.selection.acquisition_pool
    if visible is None or visible.source_fingerprint != covariate_fingerprint(covariates):
        raise ValueError("pool oracle covariates do not match the visible pool commitment")
    if not np.array_equal(frame.row_ids[pool_index], prepared.acquisition_pool_row_ids):
        raise ValueError("pool oracle row identities do not match the split manifest")
    return PoolOracle(covariates, np.asarray(frame.y[pool_index], dtype=float))


__all__ = [
    "P2A_REAL_DATASETS",
    "PreparedRealSelection",
    "SPLIT_SEED",
    "prepare_real_selection",
    "prepare_real_pool_oracle",
]
