"""Static contracts for the fresh short-prefix screening pilot."""

from bench.datamodules import TransformedFeynmanDataModule
from scripts.build_aistats_drr_prefix_pilot_freeze import _select


def test_task_selection_is_order_invariant_and_namespaced():
    names = ["a", "b", "c"]
    assert _select(names, "family") == _select(
        list(reversed(names)), "family")
    assert _select(names, "family") != _select(names, "other-family")


def test_lsr_transform_uses_loader_identifier_not_hdf5_group_spelling():
    module = TransformedFeynmanDataModule("fixture")
    assert module._dataset_identifier == "lsrtransform"
