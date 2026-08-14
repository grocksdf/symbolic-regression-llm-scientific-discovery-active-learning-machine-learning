"""Strong data-role contracts for discovery and untouched confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Optional

import numpy as np


class DataRole(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    ACQUISITION_POOL = "acquisition-pool"
    UNTOUCHED_HELDOUT = "untouched-heldout"


def _matrix(values: np.ndarray, role: DataRole) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{role.value} inputs must be a non-empty finite matrix")
    output = np.ascontiguousarray(array)
    output.setflags(write=False)
    return output


def _vector(values: np.ndarray, role: DataRole) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{role.value} targets must be a non-empty finite vector")
    output = np.ascontiguousarray(array)
    output.setflags(write=False)
    return output


def _fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    digest = sha256()
    for array in (X, y):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def covariate_fingerprint(X: np.ndarray) -> str:
    """Hash visible covariates without incorporating any hidden target values."""
    contiguous = np.ascontiguousarray(np.asarray(X, dtype=float))
    digest = sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _row_fingerprints(X: np.ndarray, y: np.ndarray) -> frozenset[str]:
    return frozenset(
        sha256(
            np.ascontiguousarray(
                np.concatenate((np.asarray(row, dtype=float), [float(target)]))
            ).tobytes()
        ).hexdigest()
        for row, target in zip(X, y, strict=True)
    )


@dataclass(frozen=True)
class RoleDataset:
    role: DataRole
    X: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        X = _matrix(self.X, self.role)
        y = _vector(self.y, self.role)
        if len(X) != len(y):
            raise ValueError(f"{self.role.value} inputs and targets must be row-aligned")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.X, self.y)

    @property
    def row_fingerprints(self) -> frozenset[str]:
        return _row_fingerprints(self.X, self.y)


@dataclass(frozen=True)
class AcquisitionCovariates:
    """Pool covariates visible to planning; labels remain behind PoolOracle."""

    role: DataRole
    X: np.ndarray
    source_fingerprint: str

    def __post_init__(self) -> None:
        if self.role is not DataRole.ACQUISITION_POOL:
            raise ValueError("acquisition covariates require the acquisition-pool role")
        X = _matrix(self.X, self.role)
        if self.source_fingerprint != covariate_fingerprint(X):
            raise ValueError("acquisition source fingerprint must cover covariates only")
        object.__setattr__(self, "X", X)


@dataclass(frozen=True)
class SelectionData:
    """The complete data surface visible to structural selection."""

    development: RoleDataset
    validation: RoleDataset
    acquisition_pool: Optional[AcquisitionCovariates]
    role_manifest: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.development.role is not DataRole.DEVELOPMENT:
            raise ValueError("development split has the wrong role")
        if self.validation.role is not DataRole.VALIDATION:
            raise ValueError("validation split has the wrong role")
        if self.acquisition_pool is not None and self.acquisition_pool.role is not DataRole.ACQUISITION_POOL:
            raise ValueError("acquisition pool has the wrong role")


@dataclass(frozen=True)
class DiscoveryDataRoles:
    development: RoleDataset
    validation: RoleDataset
    acquisition_pool: Optional[RoleDataset] = None
    untouched_heldout: Optional[RoleDataset] = None

    def __post_init__(self) -> None:
        expected = (
            (self.development, DataRole.DEVELOPMENT),
            (self.validation, DataRole.VALIDATION),
            (self.acquisition_pool, DataRole.ACQUISITION_POOL),
            (self.untouched_heldout, DataRole.UNTOUCHED_HELDOUT),
        )
        for split, role in expected:
            if split is not None and split.role is not role:
                raise ValueError(f"expected {role.value}, received {split.role.value}")
        dimensions = {
            split.X.shape[1]
            for split, _ in expected
            if split is not None
        }
        if len(dimensions) != 1:
            raise ValueError("all data roles must have the same feature dimension")
        fingerprints = [
            split.fingerprint for split, _ in expected if split is not None
        ]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("data roles must not reuse an identical labelled split")
        present = [split for split, _ in expected if split is not None]
        for index, left in enumerate(present):
            for right in present[index + 1 :]:
                overlap = left.row_fingerprints & right.row_fingerprints
                if overlap:
                    raise ValueError(
                        f"data roles {left.role.value} and {right.role.value} "
                        f"share {len(overlap)} labelled rows"
                    )

    def selection_view(self) -> SelectionData:
        """Return a type that has no field through which held-out data can leak."""
        visible = [self.development, self.validation]
        if self.acquisition_pool is not None:
            visible.append(self.acquisition_pool)
        manifest = tuple(
            (
                split.role.value,
                covariate_fingerprint(split.X)
                if split.role is DataRole.ACQUISITION_POOL
                else split.fingerprint,
            )
            for split in visible
        )
        return SelectionData(
            development=self.development,
            validation=self.validation,
            acquisition_pool=(
                AcquisitionCovariates(
                    role=DataRole.ACQUISITION_POOL,
                    X=self.acquisition_pool.X,
                    source_fingerprint=covariate_fingerprint(self.acquisition_pool.X),
                )
                if self.acquisition_pool is not None
                else None
            ),
            role_manifest=manifest,
        )

    def heldout_for_confirmation(self) -> RoleDataset:
        """Expose held-out rows only to a separately invoked confirmation stage."""
        if self.untouched_heldout is None:
            raise RuntimeError("no untouched-heldout split was registered")
        return self.untouched_heldout


__all__ = [
    "AcquisitionCovariates",
    "covariate_fingerprint",
    "DataRole",
    "DiscoveryDataRoles",
    "RoleDataset",
    "SelectionData",
]
