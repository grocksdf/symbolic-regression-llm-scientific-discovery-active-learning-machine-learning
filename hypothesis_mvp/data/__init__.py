"""Measured-data contracts exposed to the production runtime."""

from .oracle import PoolOracle
from .real_data import DatasetBundle, load_real_data_from_file, split_real_arrays
from .real_registry import (
    RealDatasetFrame,
    load_registered_real_dataset,
    registered_real_dataset_ids,
)
from .real_protocol import (
    P2A_REAL_DATASETS,
    PreparedRealSelection,
    SPLIT_SEED,
    prepare_real_selection,
    prepare_real_pool_oracle,
)
from .roles import (
    AcquisitionCovariates,
    covariate_fingerprint,
    DataRole,
    DiscoveryDataRoles,
    RoleDataset,
    SelectionData,
)

__all__ = [
    "AcquisitionCovariates", "covariate_fingerprint", "DataRole", "DatasetBundle", "DiscoveryDataRoles",
    "PoolOracle", "RealDatasetFrame", "RoleDataset", "SelectionData",
    "load_real_data_from_file", "load_registered_real_dataset",
    "registered_real_dataset_ids", "split_real_arrays",
    "P2A_REAL_DATASETS", "PreparedRealSelection", "SPLIT_SEED",
    "prepare_real_selection",
    "prepare_real_pool_oracle",
]
