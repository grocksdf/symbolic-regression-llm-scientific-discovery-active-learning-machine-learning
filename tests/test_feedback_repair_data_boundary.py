"""No-provider negative tests for the real data-boundary monitor.

Every denial here happens on a genuine h5py / pyarrow object, before any
prohibited value is returned, and is recorded in the monitor event log that the
child uses to derive the four generation-boundary flags.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pytest

from bench.data_boundary import (
    FLAG_NAMES, DataBoundaryMonitor, ProhibitedDataAccess,
    loader_train_only, registered_train_only,
)


SWITCHES = ("PCPI_TRAIN_ONLY", "LLMSRBENCH_TRAIN_ONLY",
            "PCPI_STRICT_TRAIN_VAL_ONLY", "PCPI_DISABLE_HELDOUT_PROMPT_DATA")


def _switches(monkeypatch, **values):
    for key in SWITCHES:
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return os.environ


@pytest.fixture
def train_only(monkeypatch):
    """All four runner switches, exactly as the registered runner injects them."""
    for key in SWITCHES:
        monkeypatch.setenv(key, "1")
    return True


def test_all_four_runner_switches_are_read_and_arm_the_boundary(monkeypatch):
    env = _switches(monkeypatch)
    assert registered_train_only(env) is False
    assert loader_train_only(env) is False
    # Any single switch arms the loader gate, but the registered child needs all
    # four and refuses a partial set instead of degrading quietly.
    monkeypatch.setenv("PCPI_TRAIN_ONLY", "1")
    assert loader_train_only(env) is True
    with pytest.raises(ValueError, match="partial train-only switch set"):
        registered_train_only(env)
    for key in SWITCHES:
        monkeypatch.setenv(key, "1")
    assert loader_train_only(env) is True
    assert registered_train_only(env) is True
    monkeypatch.delenv("LLMSRBENCH_TRAIN_ONLY")
    with pytest.raises(ValueError, match="partial train-only switch set"):
        registered_train_only(env)
    monkeypatch.setenv("PCPI_STRICT_TRAIN_VAL_ONLY", "0")
    with pytest.raises(ValueError, match="partial train-only switch set"):
        registered_train_only(env)


def test_every_injected_switch_is_consumed_by_the_loader_gate(monkeypatch):
    from bench import datamodules, pipelines
    for module in (datamodules, pipelines):
        for key in SWITCHES:
            _switches(monkeypatch)
            assert module._train_only() is False
            monkeypatch.setenv(key, "1")
            assert module._train_only() is True, key


@pytest.fixture
def hdf5(tmp_path):
    path = tmp_path / "benchmark.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("lsr_transform")
        task = group.create_group("I.10.7_1_0")
        train = np.asarray([[float(i), float(i + 1), float(i + 2)]
                            for i in range(20)])
        task.create_dataset("train", data=train)
        task.create_dataset("test", data=train)
        task.create_dataset("ood", data=train)
        task.create_dataset("holdout", data=train)
    return path


@pytest.fixture
def monitored(train_only):
    monitor = DataBoundaryMonitor(True)
    monitor.install()
    yield monitor
    monitor.uninstall()


def test_sealed_response_rows_outside_the_declared_horizon_are_denied(hdf5, monitored):
    with h5py.File(hdf5, "r") as handle:
        dataset = handle["lsr_transform/I.10.7_1_0/train"]
        assert dataset.shape == (20, 3)
        # Public covariates need no horizon declaration.
        covariates = np.asarray(dataset[:, 1:])
        assert covariates.shape == (20, 2)
        monitored.declare_open_rows(dataset.name, range(12))
        # Rows inside the horizon may be returned.
        assert np.asarray(dataset[sorted(range(12)), :]).shape == (12, 3)
        with pytest.raises(ProhibitedDataAccess, match="outside the declared response horizon"):
            np.asarray(dataset[:, :])
        with pytest.raises(ProhibitedDataAccess, match="outside the declared response horizon"):
            np.asarray(dataset[[19], :])
    assert monitored.flags() == dict.fromkeys(FLAG_NAMES, False) | {
        "sealed_response_values_decoded": True}
    assert [event["flag"] for event in monitored.events].count(
        "sealed_response_values_decoded") == 2


def test_sealed_column_without_a_declared_horizon_is_denied(hdf5, monitored):
    with h5py.File(hdf5, "r") as handle:
        with pytest.raises(ProhibitedDataAccess, match="no declared horizon"):
            np.asarray(handle["lsr_transform/I.10.7_1_0/train"][:, 0])


@pytest.mark.parametrize("split,flag", [
    ("test", "test_or_ood_accessed"),
    ("ood", "test_or_ood_accessed"),
    ("holdout", "heldout_opened")])
def test_prohibited_splits_are_denied_before_any_value_is_returned(hdf5, monitored, split, flag):
    with h5py.File(hdf5, "r") as handle:
        with pytest.raises(ProhibitedDataAccess, match=flag):
            handle["lsr_transform/I.10.7_1_0/" + split]
        with pytest.raises(ProhibitedDataAccess, match=flag):
            handle[f"lsr_transform/I.10.7_1_0/{split}"][...]
    assert monitored.flags()[flag] is True


@pytest.fixture
def metadata(tmp_path):
    path = tmp_path / "metadata.parquet"
    table = pa.table({
        "name": ["I.10.7_1_0"],
        "symbols": [["y", "x0", "x1"]],
        "symbol_descs": [["response", "a", "b"]],
        "symbol_properties": [["target", "variable", "variable"]],
        "expression": ["sin(x0) + x1"],
    })
    pq.write_table(table, path)
    return path


def test_projected_parquet_read_is_allowed(metadata, monitored):
    table = pq.read_table(metadata, columns=["name", "symbols"])
    assert table.column("name").to_pylist() == ["I.10.7_1_0"]
    assert monitored.flags() == dict.fromkeys(FLAG_NAMES, False)


def test_unprojected_parquet_read_is_denied(metadata, monitored):
    with pytest.raises(ProhibitedDataAccess, match="unprojected parquet read"):
        pq.read_table(metadata)


def test_ground_truth_expression_column_is_denied(metadata, monitored):
    with pytest.raises(ProhibitedDataAccess, match="forbidden metadata columns"):
        pq.read_table(metadata, columns=["name", "expression"])


def test_non_empty_ground_truth_expression_is_denied(monitored):
    with pytest.raises(ProhibitedDataAccess, match="ground-truth expression"):
        monitored.guard_expression("sin(x0) + x1", "fixture")
    assert monitored.flags()["ground_truth_expression_opened"] is True
    assert monitored.guard_expression("", "blanked") == ""


def test_flags_are_derived_from_the_event_log_only(monitored):
    assert monitored.flags() == dict.fromkeys(FLAG_NAMES, False)
    monitored.record("heldout_opened", "synthetic")
    assert monitored.flags()["heldout_opened"] is True
    assert monitored.flags()["test_or_ood_accessed"] is False


def test_arrays_handed_to_isolated_engine_workers_are_sealed(monitored):
    opened = {0, 1, 2, 3}
    sealed = np.zeros(10)
    sealed[list(opened)] = [1.0, 2.0, 3.0, 4.0]
    assert monitored.guard_arrays(sealed, 10, opened, "fixture") is sealed
    assert monitored.flags() == dict.fromkeys(FLAG_NAMES, False)
    leaked = sealed.copy()
    leaked[9] = 7.0
    with pytest.raises(ProhibitedDataAccess, match="sealed rows"):
        monitored.guard_arrays(leaked, 10, opened, "fixture")
    assert monitored.flags()["sealed_response_values_decoded"] is True
    with pytest.raises(ProhibitedDataAccess, match="response length"):
        monitored.guard_arrays(sealed[:4], 10, opened, "fixture")


def test_isolated_engine_worker_has_no_file_data_channel():
    """Static audit of the subprocess that actually receives the arrays."""
    mainline = Path(__file__).resolve().parents[2] / "hypothesis_mvp"
    scheduler = (mainline / "hypothesis_mvp" / "symbolic" / "scheduler.py")
    text = scheduler.read_text(encoding="utf-8")
    for forbidden in ("h5py", "pyarrow", "read_table", "np.load", "read_csv", "parquet"):
        assert forbidden not in text, forbidden
    assert "mp.get_context(\"spawn\")" in text
    assert "args=(sender, job, values, arrays)" in text


def test_no_unguarded_equivalent_read_path_in_the_execution_closure():
    """Static audit: every h5py / parquet read site must be a guarded site.

    The allowlist below is the complete set of sites the registered child and
    the benchmark loaders use. A new unguarded site fails this test.
    """
    root = Path(__file__).resolve().parents[1]
    allowed = {
        ("bench", "datamodules.py"): {"h5py.File", "parquet.read_table"},
        ("scripts", "run_aistats_three_arm_child.py"): {"pq.read_table", "h5py.File"},
        ("scripts", "run_aistats_three_arm_formula_child.py"): {"pq.read_table", "h5py.File"},
    }
    sites = {}
    for group, name in allowed:
        for line in (root / group / name).read_text(encoding="utf-8").splitlines():
            for call in ("h5py.File(", "parquet.read_table(", "pq.read_table("):
                if call in line:
                    sites.setdefault((group, name), set()).add(call[:-1])
    assert sites == allowed
    # The guarded sites must all be covered by an installed guard.
    monitor = DataBoundaryMonitor(True)
    assert set(monitor.install().installed) == {
        "h5py.Group.__getitem__", "pyarrow.parquet.read_table"}
    monitor.uninstall()


def test_no_child_script_hardcodes_an_isolation_flag():
    """Static audit: no child may write a boundary flag as a literal.

    The B2 blocker was exactly these four literals in the success payload.
    Both the C/D child and the R child must derive all four flags from one
    real DataBoundaryMonitor instead.
    """
    root = Path(__file__).resolve().parents[1]
    children = ("scripts/run_aistats_three_arm_child.py",
                "scripts/run_aistats_three_arm_formula_child.py")
    for relative in children:
        text = (root / relative).read_text(encoding="utf-8")
        for flag in FLAG_NAMES:
            assert f'"{flag}": False' not in text, (relative, flag)
            assert f"'{flag}': False" not in text, (relative, flag)
        assert "DataBoundaryMonitor(registered_train_only())" in text, relative
        assert ".install()" in text, relative
        assert "boundary.flags()" in text, relative
        assert "declare_open_rows(" in text, relative
        assert "guard_arrays(" in text, relative
