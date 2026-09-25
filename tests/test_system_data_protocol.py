"""Role/parse correctness fixtures only; no real observational source opened."""
from dataclasses import replace
from hashlib import sha256
import json
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


def test_airfoil_streams_only_registered_rows(tmp_path):
    path = tmp_path / "airfoil_self_noise.dat"
    path.write_text(
        "1 2 3 4 5 6\n"
        "DO NOT PARSE EXCLUDED RESPONSE\n"
        "7 8 9 10 11 12\n",
        encoding="utf-8")
    assert protocol._airfoil_open_rows(path, {0, 2}) == {
        0: [1., 2., 3., 4., 5., 6.],
        2: [7., 8., 9., 10., 11., 12.],
    }


def test_airfoil_outer_partition_preserves_sealed_boundary(monkeypatch):
    from hypothesis_mvp.data.real_protocol import SPLIT_SEED
    monkeypatch.setattr(
        protocol, "_verify_hash", lambda path, expected, verify: expected)
    spec = replace(protocol.REAL_DATASET_SPECS["uci_airfoil"],
                   expected_rows=100)
    monkeypatch.setitem(protocol.REAL_DATASET_SPECS, "uci_airfoil", spec)
    captured = []

    def read(source, selected):
        captured.extend(selected)
        return {i: [i, i + 1, i + 2, i + 3, i + 4, i + 5]
                for i in selected}

    monkeypatch.setattr(protocol, "_airfoil_open_rows", read)
    protocol.load_registered_system_data(_registration("uci_airfoil"))
    order = sorted(
        range(100), key=lambda i:
        sha256(f"{SPLIT_SEED}:uci_airfoil:{i:06d}".encode()).digest())
    assert not set(captured) & set(order[90:])
    assert (protocol.PUBLIC_SCIENTIFIC_CONTEXT["uci_airfoil"][
        "feature_names"])


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


def test_ved_system_entry_uses_only_registered_open_members(
        tmp_path, monkeypatch):
    source = {
        "schema": "scientific-ved-confirmation-source-registration-v1",
        "source_name": "fixture", "source_citation": "fixture",
        "archives": {
            "part1": {"path": str(tmp_path / "a.7z"), "sha256": "a" * 64},
            "part2": {"path": str(tmp_path / "b.7z"), "sha256": "b" * 64}},
        "extractor": {"path": str(tmp_path / "7z.exe"), "sha256": "c" * 64},
        "open_members": {
            "development": "development.csv",
            "validation": "validation.csv",
            "acquisition_pool": "pool.csv"},
        "reserved_confirmation_member_sha256": "d" * 64,
        "features": ["f0", "f1", "f2", "f3", "f4"],
        "target": "y", "grammar_contract": "pcpi-closed-basis-v1",
        "execution_authorized": False, "claim_boundary": "fixture"}
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(
        protocol, "run_ved_source_gate",
        lambda registration: {
            "passed": True, "archive_sha256": {"part1": "a", "part2": "b"},
            "extractor_sha256": "c"})
    monkeypatch.setattr(
        protocol, "_member_names",
        lambda extractor, archive: (
            ("development.csv", "validation.csv")
            if archive.name == "a.7z" else ("pool.csv",)))
    opened_members = []

    def stream(**kwargs):
        opened_members.append(kwargs["member"])
        header = "f0,f1,f2,f3,f4,y\n"
        rows = ["0,1,2,3,4,5\n" for _ in range(30)]
        return iter([header, *rows])

    monkeypatch.setattr(protocol, "stream_ved_archive_member", stream)
    registration = {
        "dataset": "ved_fuel_rate", "source": str(source_path),
        "split_seed": 7,
        "counts": {
            "exploration_development": 4, "exploration_validation": 3,
            "inference_initial": 2, "development_evaluation": 5,
            "acquisition_pool": 2}}
    data = protocol.load_registered_system_data(registration)
    assert opened_members == [
        "development.csv", "validation.csv", "pool.csv"]
    assert data.manifest["family"] == "ved_vehicle_energy"
    assert data.manifest["reserved_confirmation_member_opened"] is False
    assert data.initial.X.shape == (2, 5)
