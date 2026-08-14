"""Strict registry for paper-eligible real observational datasets.

Only explicitly registered, provenance-checked datasets can be loaded through
this module. Formula benchmarks and the unregistered mixed ``data/real_datasets``
directory are deliberately outside the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


UCI_AIRFOIL_URL = "https://archive.ics.uci.edu/dataset/291/airfoil+self+noise"
UCI_CCPP_URL = "https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant"
UCI_GAS_URL = "https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set"

AIRFOIL_HASH = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"
CCPP_HASH = "ccd490981db2a2f079963b3d9f0aea30d9d338900a0285428dfc6385396f4651"
GAS_HASHES = {
    "gt_2011.csv": "d87ceef9aa59533cc7d924d10de241b1b06ecd11f9b26bab59191ea0f8a76b9a",
    "gt_2012.csv": "be54b9d0e1a7de40c55d32fa489e75de892b000c066b5a09f09a19124ee29100",
    "gt_2013.csv": "13c437bb440ec2045bd12057e6654c41dd4107a661eac16ba2e878e897a08f9e",
    "gt_2014.csv": "c2a03c92c9c3207aad0c6be7de8d9b5b4bfa4720ad0efb2c1f21b6cec4d3f3fa",
    "gt_2015.csv": "9b08f35fde0d4b138232a605db4093c2b8bf9d6757e6f1fbd9534ad616c13591",
}

FORBIDDEN_SOURCE_COMPONENTS = {
    "feynman",
    "nguyen",
    "strict_ood_standard_sr",
    "real_datasets",
}


@dataclass(frozen=True)
class RealDatasetFrame:
    dataset_id: str
    X: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray
    groups: np.ndarray | None
    feature_names: tuple[str, ...]
    target_name: str
    source_paths: tuple[Path, ...]
    source_hashes: tuple[str, ...]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class RealDatasetSpec:
    dataset_id: str
    source_url: str
    target_name: str
    feature_names: tuple[str, ...]
    observation_type: str
    expected_rows: int


REAL_DATASET_SPECS: dict[str, RealDatasetSpec] = {
    "uci_airfoil": RealDatasetSpec(
        dataset_id="uci_airfoil",
        source_url=UCI_AIRFOIL_URL,
        target_name="scaled_sound_pressure_level_db",
        feature_names=(
            "frequency_hz",
            "angle_of_attack_deg",
            "chord_length_m",
            "free_stream_velocity_m_s",
            "displacement_thickness_m",
        ),
        observation_type="NASA anechoic wind-tunnel measurement",
        expected_rows=1503,
    ),
    "uci_ccpp": RealDatasetSpec(
        dataset_id="uci_ccpp",
        source_url=UCI_CCPP_URL,
        target_name="PE",
        feature_names=("AT", "V", "AP", "RH"),
        observation_type="combined-cycle power-plant operating observation",
        expected_rows=9568,
    ),
    "uci_gas_turbine_co": RealDatasetSpec(
        dataset_id="uci_gas_turbine_co",
        source_url=UCI_GAS_URL,
        target_name="CO",
        feature_names=("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"),
        observation_type="gas-turbine sensor and emission observation",
        expected_rows=36733,
    ),
    "uci_gas_turbine_nox": RealDatasetSpec(
        dataset_id="uci_gas_turbine_nox",
        source_url=UCI_GAS_URL,
        target_name="NOX",
        feature_names=("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"),
        observation_type="gas-turbine sensor and emission observation",
        expected_rows=36733,
    ),
}


def registered_real_dataset_ids() -> tuple[str, ...]:
    return tuple(REAL_DATASET_SPECS)


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_allowed_path(path: Path, data_root: Path) -> None:
    resolved_root = data_root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Real-data source escapes data root: {resolved_path}")
    lower_parts = {part.lower() for part in resolved_path.parts}
    blocked = sorted(lower_parts & FORBIDDEN_SOURCE_COMPONENTS)
    if blocked:
        raise ValueError(
            "Paper-ineligible source path contains blocked component(s): "
            + ", ".join(blocked)
        )


def _find_unique(data_root: Path, filename: str, parent_tokens: Sequence[str]) -> Path:
    candidates = []
    tokens = tuple(token.lower() for token in parent_tokens)
    for path in data_root.rglob(filename):
        parent_text = path.parent.as_posix().lower()
        if any(token in parent_text for token in tokens):
            candidates.append(path)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise FileNotFoundError(
            f"Expected exactly one official {filename} under {data_root}; found {len(candidates)}: {rendered}"
        )
    _assert_allowed_path(candidates[0], data_root)
    return candidates[0]


def _verify_hash(path: Path, expected: str, verify_hashes: bool) -> str:
    actual = _hash_file(path)
    if verify_hashes and actual.lower() != expected.lower():
        raise ValueError(
            f"Official-source SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def _finalize(
    spec: RealDatasetSpec,
    frame: pd.DataFrame,
    *,
    source_paths: Sequence[Path],
    source_hashes: Sequence[str],
    groups: np.ndarray | None,
    row_id_namespace: str | None = None,
) -> RealDatasetFrame:
    required = list(spec.feature_names) + [spec.target_name]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{spec.dataset_id} missing required columns: {missing}")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    numeric = numeric.loc[valid].reset_index(drop=True)
    if groups is not None:
        groups = np.asarray(groups)[valid]
    if len(numeric) != spec.expected_rows:
        raise ValueError(
            f"{spec.dataset_id} expected {spec.expected_rows} valid observations; got {len(numeric)}"
        )
    X = numeric[list(spec.feature_names)].to_numpy(dtype=float)
    y = numeric[spec.target_name].to_numpy(dtype=float)
    if X.shape[0] != y.size or X.shape[1] != len(spec.feature_names):
        raise AssertionError("Real-data matrix shape invariant failed")
    namespace = row_id_namespace or spec.dataset_id
    row_ids = np.asarray(
        [f"{namespace}:{index:06d}" for index in range(len(numeric))],
        dtype=object,
    )
    return RealDatasetFrame(
        dataset_id=spec.dataset_id,
        X=X,
        y=y,
        row_ids=row_ids,
        groups=groups,
        feature_names=spec.feature_names,
        target_name=spec.target_name,
        source_paths=tuple(source_paths),
        source_hashes=tuple(source_hashes),
        provenance={
            "source_url": spec.source_url,
            "observation_type": spec.observation_type,
            "expected_rows": spec.expected_rows,
            "synthetic": False,
            "formula_generated": False,
            "noise_added_by_loader": False,
        },
    )


def _load_airfoil(data_root: Path, verify_hashes: bool) -> RealDatasetFrame:
    spec = REAL_DATASET_SPECS["uci_airfoil"]
    path = _find_unique(data_root, "airfoil_self_noise.dat", ("airfoil",))
    actual_hash = _verify_hash(path, AIRFOIL_HASH, verify_hashes)
    columns = list(spec.feature_names) + [spec.target_name]
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    return _finalize(spec, frame, source_paths=(path,), source_hashes=(actual_hash,), groups=None)


def _load_ccpp(data_root: Path, verify_hashes: bool) -> RealDatasetFrame:
    spec = REAL_DATASET_SPECS["uci_ccpp"]
    path = _find_unique(data_root, "Folds5x2_pp.xlsx", ("combined", "ccpp"))
    actual_hash = _verify_hash(path, CCPP_HASH, verify_hashes)
    # The five sheets are permutations of the same 9,568 observations. Loading
    # Sheet1 once avoids five-fold duplication.
    frame = pd.read_excel(path, sheet_name="Sheet1")
    return _finalize(spec, frame, source_paths=(path,), source_hashes=(actual_hash,), groups=None)


def _load_gas(data_root: Path, dataset_id: str, verify_hashes: bool) -> RealDatasetFrame:
    spec = REAL_DATASET_SPECS[dataset_id]
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    hashes: list[str] = []
    groups: list[np.ndarray] = []
    for year in range(2011, 2016):
        filename = f"gt_{year}.csv"
        path = _find_unique(data_root, filename, ("gas", "turbine"))
        actual_hash = _verify_hash(path, GAS_HASHES[filename], verify_hashes)
        frame = pd.read_csv(path)
        frames.append(frame)
        paths.append(path)
        hashes.append(actual_hash)
        groups.append(np.full(len(frame), year, dtype=int))
    combined = pd.concat(frames, ignore_index=True)
    return _finalize(
        spec,
        combined,
        source_paths=paths,
        source_hashes=hashes,
        groups=np.concatenate(groups),
        row_id_namespace="uci_gas_turbine",
    )


def load_registered_real_dataset(
    dataset_id: str,
    data_root: str | Path,
    *,
    verify_hashes: bool = True,
) -> RealDatasetFrame:
    """Load one allowlisted real dataset and reject all other identifiers."""

    dataset_id = str(dataset_id).strip().lower()
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    if dataset_id not in REAL_DATASET_SPECS:
        raise ValueError(
            f"Dataset {dataset_id!r} is not real-data allowlisted. "
            f"Allowed: {', '.join(registered_real_dataset_ids())}"
        )
    if dataset_id == "uci_airfoil":
        return _load_airfoil(root, verify_hashes)
    if dataset_id == "uci_ccpp":
        return _load_ccpp(root, verify_hashes)
    if dataset_id in {"uci_gas_turbine_co", "uci_gas_turbine_nox"}:
        return _load_gas(root, dataset_id, verify_hashes)
    raise AssertionError(f"Missing loader for registered dataset {dataset_id}")
