import numpy as np

from hypothesis_mvp.data import (
    DataRole, DiscoveryDataRoles, RoleDataset, split_real_arrays,
)


def test_heldout_is_absent_from_selection_surface() -> None:
    X = np.arange(240, dtype=float).reshape(80, 3)
    y = np.linspace(1.0, 4.0, 80)
    bundle = split_real_arrays(
        X, y, train_ratio=0.7, pool_ratio=0.15,
        heldout_ratio=0.15, strategy="pca1", seed=7,
    )
    roles = DiscoveryDataRoles(
        RoleDataset(DataRole.DEVELOPMENT, bundle.X_train, bundle.y_train),
        RoleDataset(DataRole.VALIDATION, bundle.X_val, bundle.y_val),
        RoleDataset(DataRole.ACQUISITION_POOL, bundle.X_pool, bundle.y_pool),
        RoleDataset(DataRole.UNTOUCHED_HELDOUT, bundle.X_heldout, bundle.y_heldout),
    )
    visible = roles.selection_view()
    assert not hasattr(visible, "untouched_heldout")
    assert not hasattr(visible.acquisition_pool, "y")
    assert roles.heldout_for_confirmation().role is DataRole.UNTOUCHED_HELDOUT
