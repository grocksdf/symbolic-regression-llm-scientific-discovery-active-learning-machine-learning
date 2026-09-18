"""Role/parse correctness fixtures only; no real observational source opened."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest

from hypothesis_mvp.data import system_protocol as protocol


def _registration(dataset="uci_gas_turbine_co"):
    return {"dataset": dataset, "source": str(Path(__file__).resolve().parent / "unopened-fixture"), "split_seed": 7,
            "counts": {role: 2 for role in protocol.ROLE_NAMES}}


def test_gas_only_opens_registered_years_and_disjoint_roles(monkeypatch):
    paths = []
    monkeypatch.setattr(protocol, "_verify_hash", lambda path, expected, verify: expected)
    def read(path, **kwargs):
        paths.append(Path(path).name)
        year = int(Path(path).stem.split("_")[1])
        spec = protocol.REAL_DATASET_SPECS["uci_gas_turbine_co"]
        return pd.DataFrame({column: np.arange(20.) + year * 100 + j
            for j, column in enumerate([*spec.feature_names, spec.target_name])})
    monkeypatch.setattr(protocol.pd, "read_csv", read)
    data = protocol.load_registered_system_data(_registration())
    assert paths == ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv", "gt_2014.csv"]
    rows = [set(split.X[:, 0]) for split in (data.selection.development,
        data.selection.validation, data.initial, data.evaluation)] + [set(data.pool.X_pool[:, 0])]
    assert all(not left & right for i, left in enumerate(rows) for right in rows[i + 1:])
    assert not data.manifest["heldout_opened"]
    assert data.manifest["scientific_context"]["task_name"]
    assert len(data.manifest["scientific_context"]["feature_names"]) == data.initial.X.shape[1]
    assert data.manifest["scientific_context_role"] == "public-source-metadata-no-observed-values"
    assert not hasattr(data.selection, "untouched_heldout")


def test_ccpp_excluded_cells_are_never_decoded(tmp_path):
    path = tmp_path / "source-fixture.xlsx"
    open_row = '<row r="2">' + ''.join(f'<c r="{c}2"><v>{i}</v></c>'
        for i, c in enumerate("ABCDE")) + '</row>'
    excluded = '<row r="3"><c r="E3"><v>DO_NOT_DECODE_EXCLUDED_RESPONSE</v></c></row>'
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", f'<worksheet><sheetData>{open_row}{excluded}</sheetData></worksheet>')
    assert protocol._ccpp_open_rows(path, {0}) == {0: [0., 1., 2., 3., 4.]}


def test_ccpp_outer_partition_preserves_historical_sealed_boundary(monkeypatch):
    from hypothesis_mvp.data.real_protocol import SPLIT_SEED
    monkeypatch.setattr(protocol, "_verify_hash", lambda path, expected, verify: expected)
    spec = replace(protocol.REAL_DATASET_SPECS["uci_ccpp"], expected_rows=100)
    monkeypatch.setitem(protocol.REAL_DATASET_SPECS, "uci_ccpp", spec)
    captured = []
    def read(source, selected):
        captured.extend(selected)
        return {i: [i, i + 1, i + 2, i + 3, i + 4] for i in selected}
    monkeypatch.setattr(protocol, "_ccpp_open_rows", read)
    protocol.load_registered_system_data(_registration("uci_ccpp"))
    order = sorted(range(100), key=lambda i: sha256(f"{SPLIT_SEED}:uci_ccpp:{i:06d}".encode()).digest())
    assert not set(captured) & set(order[90:])


def test_unknown_roles_reject_before_file_access(monkeypatch):
    def forbidden(*args): raise AssertionError("no file access allowed")
    monkeypatch.setattr(protocol, "_verify_hash", forbidden)
    registration = _registration(); registration["heldout"] = "forbidden"
    with pytest.raises(ValueError): protocol.load_registered_system_data(registration)


def test_official_hash_mismatch_blocks_before_data_parse(monkeypatch):
    def mismatch(*args): raise ValueError("official hash mismatch")
    def forbidden(*args, **kwargs): raise AssertionError("source must not be parsed")
    monkeypatch.setattr(protocol, "_verify_hash", mismatch)
    monkeypatch.setattr(protocol.pd, "read_csv", forbidden)
    with pytest.raises(ValueError, match="official hash"):
        protocol.load_registered_system_data(_registration())
