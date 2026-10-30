"""Real data-boundary monitoring for one train-only development child.

The four generation-boundary flags are derived from this monitor's event log.
They are never written as constants. Every guard denies *before* the prohibited
content is returned and records the attempt, so a child can only attest false
together with an empty event log; a denial aborts the child.

The four train-only environment switches injected by the registered runner are
read here and by ``bench.datamodules`` / ``bench.pipelines``. A partial switch
set is a hard error rather than a silently weaker guarantee.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

import numpy as np


TRAIN_ONLY_SWITCHES = (
    "PCPI_TRAIN_ONLY",
    "LLMSRBENCH_TRAIN_ONLY",
    "PCPI_STRICT_TRAIN_VAL_ONLY",
    "PCPI_DISABLE_HELDOUT_PROMPT_DATA",
)


def loader_train_only(env: dict[str, str] | None = None) -> bool:
    """Loader gate: any of the four injected switches arms train-only.

    All four names are read, so none of the switches the registered runner
    injects can stay inert. Arming on any one of them is the delivered
    ``bench.datamodules`` / ``bench.pipelines`` behaviour.
    """
    environment = os.environ if env is None else env
    return any(environment.get(key) == "1" for key in TRAIN_ONLY_SWITCHES)


def registered_train_only(env: dict[str, str] | None = None) -> bool:
    """Registered-child contract: all four switches must be armed together.

    The one registered runner injects all four. A partial set is a hard error
    rather than a silently weaker guarantee, and a missing set means the child
    is not running under the registered contract.
    """
    environment = os.environ if env is None else env
    armed = {key: environment.get(key) == "1" for key in TRAIN_ONLY_SWITCHES}
    if any(armed.values()) and not all(armed.values()):
        raise ValueError(
            "partial train-only switch set: "
            + ", ".join(key for key, on in armed.items() if not on))
    return all(armed.values())

FLAG_NAMES = (
    "test_or_ood_accessed",
    "heldout_opened",
    "ground_truth_expression_opened",
    "sealed_response_values_decoded",
)

#: Column zero of every benchmark HDF5 task group is the measured response.
SEALED_RESPONSE_COLUMN = 0

FORBIDDEN_METADATA_COLUMNS = frozenset({
    "expression", "gt_expression", "ground_truth", "ground_truth_expression",
    "sympy_expression", "target_values", "y_values", "response_values",
})

_TEST_TOKENS = ("test", "ood")
_HELD_TOKENS = ("holdout", "held_out", "held-out", "heldout_values", "hidden", "sealed")


class ProhibitedDataAccess(RuntimeError):
    """Raised before any prohibited scientific value is read or returned."""


def _classify_split(name: str) -> str | None:
    """Map an HDF5 split name onto the flag it would violate."""
    last = str(name).rstrip("/").split("/")[-1].lower()
    for token in _TEST_TOKENS:
        if token in last:
            return "test_or_ood_accessed"
    for token in _HELD_TOKENS:
        if token in last:
            return "heldout_opened"
    return None


def _column_touched(selection: Any, sealed: int, width: int) -> bool:
    if isinstance(selection, slice):
        start, stop = selection.start or 0, width if selection.stop is None else selection.stop
        return start <= sealed < stop
    if isinstance(selection, (int,)):
        return selection == sealed
    try:
        return any(_column_touched(item, sealed, width) for item in selection)
    except TypeError:
        return False


def _row_indices(selection: Any, count: int) -> Iterable[int]:
    if isinstance(selection, slice):
        return range(*selection.indices(count))
    if isinstance(selection, (int,)):
        return [selection]
    return [int(index) for index in selection]


class _GuardedDataset:
    """A train-split dataset that refuses to return sealed response values."""

    def __init__(self, dataset: Any, monitor: "DataBoundaryMonitor") -> None:
        self._dataset = dataset
        self._monitor = monitor

    @property
    def name(self) -> str:
        return self._dataset.name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._dataset.shape

    def __getattr__(self, item: str) -> Any:
        return getattr(self._dataset, item)

    def __getitem__(self, key: Any) -> Any:
        shape = self._dataset.shape
        width = shape[1] if len(shape) > 1 else 1
        if isinstance(key, tuple):
            row_key = key[0]
            column_key = key[1] if len(key) > 1 else slice(None)
        else:
            row_key, column_key = key, slice(None)
        if _column_touched(column_key, SEALED_RESPONSE_COLUMN, width):
            allowed = self._monitor.open_rows(self._dataset.name)
            if allowed is None:
                self._monitor.deny(
                    "sealed_response_values_decoded",
                    f"sealed response column read with no declared horizon: {self._dataset.name}")
            offenders = sorted(index for index in _row_indices(row_key, shape[0])
                               if index not in allowed)
            if offenders:
                self._monitor.deny(
                    "sealed_response_values_decoded",
                    f"rows {offenders[:8]} outside the declared response horizon "
                    f"of {self._dataset.name}")
        return self._dataset[key]


class DataBoundaryMonitor:
    """Record prohibited accesses and derive the four flags from that log."""

    def __init__(self, train_only: bool) -> None:
        self.train_only = bool(train_only)
        self.events: list[dict[str, Any]] = []
        self.installed: list[str] = []
        self._open_rows: dict[str, frozenset[int]] = {}
        self._patches: list[tuple[Any, str, Any]] = []

    # -- event log -----------------------------------------------------------
    def record(self, flag: str, detail: str) -> "DataBoundaryMonitor":
        if flag not in FLAG_NAMES:
            raise ValueError(f"unknown boundary flag {flag!r}")
        self.events.append({"flag": flag, "detail": str(detail)[:240]})
        return self

    def deny(self, flag: str, detail: str) -> None:
        self.record(flag, detail)
        raise ProhibitedDataAccess(f"{flag}: {detail}")

    def flags(self) -> dict[str, bool]:
        seen = {event["flag"] for event in self.events}
        return {name: name in seen for name in FLAG_NAMES}

    # -- declarations --------------------------------------------------------
    def declare_open_rows(self, dataset_name: str, rows: Iterable[int]) -> None:
        self._open_rows[str(dataset_name)] = frozenset(int(row) for row in rows)

    def open_rows(self, dataset_name: str) -> frozenset[int] | None:
        return self._open_rows.get(str(dataset_name))

    # -- guards --------------------------------------------------------------
    def guard_expression(self, value: Any, source: str) -> Any:
        """Refuse a ground-truth expression value, whatever hands it over.

        This is the value-level counterpart of the read-site guard in
        ``install()``: the only ground-truth expression in the benchmark is the
        ``expression`` column of the metadata parquet, and ``install()`` already
        refuses any projection that asks for it.  A caller that somehow obtains
        an expression string anyway must pass it through here so the flag is
        still derived from a real event.  Public task prose is not an
        expression and must not be routed through this guard.
        """
        if self.train_only and isinstance(value, str) and value.strip():
            self.deny("ground_truth_expression_opened",
                      f"non-empty ground-truth expression at {source}")
        return value

    def guard_arrays(self, response: Any, total: int, allowed: Iterable[int],
                     source: str) -> Any:
        """Check the exact arrays handed to the engines and their subprocesses.

        Isolated engine workers receive in-memory arrays only, so this is the
        boundary that actually covers them: every response outside the declared
        horizon must still be sealed.
        """
        if not self.train_only:
            return response
        allowed_rows = frozenset(int(row) for row in allowed)
        values = np.asarray(response, dtype=float).reshape(-1)
        if values.size != int(total):
            self.deny("sealed_response_values_decoded",
                      f"{source}: response length {values.size} != {total}")
        sealed = [index for index in range(int(total))
                  if index not in allowed_rows and float(values[index]) != 0.0]
        if sealed:
            self.deny("sealed_response_values_decoded",
                      f"{source}: sealed rows {sealed[:8]} carry response values")
        return response

    def install(self) -> "DataBoundaryMonitor":
        """Bind the real read boundaries in this process."""
        if self._patches:
            return self
        import h5py
        import pyarrow.parquet as parquet

        original_group_getitem = h5py.Group.__getitem__
        monitor = self

        def guarded_group_getitem(group: Any, key: Any) -> Any:
            if isinstance(key, (str, bytes)):
                text = key.decode() if isinstance(key, bytes) else key
                flag = _classify_split(text)
                if flag is not None:
                    monitor.deny(flag, f"hdf5 group read {group.name}/{text}")
            value = original_group_getitem(group, key)
            if monitor.train_only and isinstance(value, h5py.Dataset):
                return _GuardedDataset(value, monitor)
            return value

        original_read_table = parquet.read_table

        def guarded_read_table(source: Any, *args: Any, **kwargs: Any) -> Any:
            columns = kwargs.get("columns")
            if columns is None:
                monitor.deny("ground_truth_expression_opened",
                             f"unprojected parquet read of {source}")
            forbidden = sorted({str(name) for name in columns
                                if str(name).lower() in FORBIDDEN_METADATA_COLUMNS})
            if forbidden:
                monitor.deny("ground_truth_expression_opened",
                             f"forbidden metadata columns {forbidden} in {source}")
            return original_read_table(source, *args, **kwargs)

        h5py.Group.__getitem__ = guarded_group_getitem
        parquet.read_table = guarded_read_table
        self._patches.append((h5py.Group, "__getitem__", original_group_getitem))
        self._patches.append((parquet, "read_table", original_read_table))
        self.installed.extend(["h5py.Group.__getitem__", "pyarrow.parquet.read_table"])
        return self

    def uninstall(self) -> None:
        for owner, name, original in self._patches:
            setattr(owner, name, original)
        self._patches.clear()

    def coverage(self) -> dict[str, Any]:
        return {
            "train_only": self.train_only,
            "switches": dict.fromkeys(TRAIN_ONLY_SWITCHES),
            "guards_installed": list(self.installed),
            "subprocess_data_channels": (
                "engine workers receive in-memory arrays only; "
                "guard_arrays() re-checks every response outside the declared horizon"),
            "event_count": len(self.events),
        }
