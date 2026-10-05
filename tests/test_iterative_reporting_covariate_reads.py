"""Storage-access correctness fixture; no real benchmark data."""
import numpy as np
import pytest
from scripts.audit_iterative_bank_reporting import _read_role_rows


class GuardedDataset:
    def __init__(self, values):
        self.values = values
        self.reads = []

    def __getitem__(self, key):
        rows, columns = key
        assert columns.start == 1, 'response column was read'
        self.reads.append(key)
        return self.values[rows, columns]


@pytest.mark.parametrize('rows', [[3, 0, 2], [0], [2, 2, 0]])
def test_response_false_never_reads_labels_and_restores_original_order(rows):
    raw = np.arange(20, dtype=float).reshape(5, 4)
    guarded = GuardedDataset(raw)
    result = _read_role_rows(guarded, rows, response=False)
    assert np.array_equal(result, raw[rows, 1:])
    assert len(guarded.reads) == 1


def test_actual_hdf5_selection_excludes_response_column(tmp_path):
    import h5py
    path = tmp_path / 'fixture.h5'
    values = np.arange(12, dtype=float).reshape(4, 3)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('train', data=values)
    with h5py.File(path, 'r') as handle:
        points = _read_role_rows(handle['train'], [3, 0], response=False)
        features, labels = _read_role_rows(handle['train'], [2, 1], response=True)
    assert np.array_equal(points, values[[3, 0], 1:])
    assert np.array_equal(features, values[[2, 1], 1:])
    assert np.array_equal(labels, values[[2, 1], 0])
