"""Official-hash-verified open-development entry for the scientific system.

Gas reads exactly four explicit 2011--2014 CSVs, never the sealed source. CCPP
reads only target-blind registered open rows from Sheet1 via worksheet access.
No held-out object, role, record, shape or summary is returned. This is a new
development protocol; its seeds/counts must be frozen before user execution.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from zipfile import ZipFile
from xml.sax import make_parser, handler

from .real_registry import REAL_DATASET_SPECS, GAS_HASHES, CCPP_HASH, _verify_hash
from .roles import SelectionData, RoleDataset, DataRole
from .oracle import PoolOracle


ROLE_NAMES = ("exploration_development", "exploration_validation", "inference_initial",
              "development_evaluation", "acquisition_pool")


@dataclass(frozen=True)
class OpenSystemData:
    selection: SelectionData
    initial: RoleDataset
    evaluation: RoleDataset
    pool: PoolOracle
    manifest: dict


def _ordered(ids, seed):
    return sorted(ids, key=lambda row: sha256(f"{seed}:{row}".encode()).digest())


def validate_data_registration(registration):
    if (set(registration) != {"dataset", "source", "split_seed", "counts"}
            or registration["dataset"] not in {"uci_ccpp", "uci_gas_turbine_co", "uci_gas_turbine_nox"}
            or type(registration["split_seed"]) is not int or registration["split_seed"] < 0
            or not isinstance(registration["source"], str) or not Path(registration["source"]).is_absolute()
            or set(registration["counts"]) != set(ROLE_NAMES)
            or any(type(n) is not int or n < 2 for n in registration["counts"].values())):
        raise ValueError("invalid registered open-development roles")
    return registration


def _assign(available, counts):
    result = {}
    offsets = {group: 0 for group in available}
    for role in ROLE_NAMES:
        group = "pool" if role == "acquisition_pool" else "validation" if role == "exploration_validation" else "development"
        start, count = offsets[group], counts[role]
        chosen = available[group][start:start + count]
        if len(chosen) != count:
            raise ValueError("registered role counts exceed open-data capacity")
        result[role] = chosen
        offsets[group] += count
    return result


def _ccpp_open_rows(source, selected):
    """Stream the source container; decode/store only registered numeric cells.

    Unlike load_workbook/read_excel this never materializes excluded rows.
    Source-byte hashing/parsing is not a held-out evaluation. No workbook shape,
    sheet dimension, excluded target or excluded summary is consulted.
    """
    values = {}
    class OpenRows(handler.ContentHandler):
        def __init__(self):
            self.row = None; self.column = None; self.collect = False; self.text = []
        def startElement(self, name, attrs):
            if name == "row":
                index = int(attrs["r"]) - 2
                self.row = index if index in selected else None
                if self.row is not None: values[self.row] = [None] * 5
            elif name == "c" and self.row is not None:
                reference = attrs["r"]
                letter = reference.rstrip("0123456789")
                self.column = "ABCDE".find(letter) if len(letter) == 1 else -1
                if attrs.get("t", "n") != "n":
                    raise ValueError("registered CCPP response/input cell is not numeric")
            elif name == "v" and self.row is not None and self.column in range(5):
                self.collect = True; self.text = []
        def characters(self, content):
            if self.collect: self.text.append(content)
        def endElement(self, name):
            if name == "v" and self.collect:
                values[self.row][self.column] = float("".join(self.text))
                self.collect = False
            elif name == "c": self.column = None
            elif name == "row": self.row = None
    parser = make_parser()
    parser.setFeature(handler.feature_external_ges, False)
    parser.setContentHandler(OpenRows())
    with ZipFile(source) as archive, archive.open("xl/worksheets/sheet1.xml") as stream:
        parser.parse(stream)
    if set(values) != set(selected) or any(any(v is None for v in row) for row in values.values()):
        raise ValueError("registered CCPP open rows are incomplete")
    return values


def load_registered_system_data(registration):
    validate_data_registration(registration)
    dataset, seed = registration["dataset"], registration["split_seed"]
    source = Path(registration["source"])
    spec = REAL_DATASET_SPECS[dataset]
    hashes = {}
    values = {}
    if dataset.startswith("uci_gas_turbine_"):
        groups = {"development": [], "validation": [], "pool": []}
        for year, group in ((2011, "development"), (2012, "development"),
                            (2013, "validation"), (2014, "pool")):
            name = f"gt_{year}.csv"
            path = source / name
            hashes[name] = _verify_hash(path, GAS_HASHES[name], True)
            frame = pd.read_csv(path, usecols=[*spec.feature_names, spec.target_name])
            for index, row in enumerate(frame.loc[:, [*spec.feature_names, spec.target_name]].to_numpy(dtype=float)):
                identifier = f"{year}:{index}"
                groups[group].append(identifier)
                values[identifier] = row
        assigned = _assign({g: _ordered(rows, seed) for g, rows in groups.items()}, registration["counts"])
    else:
        hashes["Folds5x2_pp.xlsx"] = _verify_hash(source, CCPP_HASH, True)
        # Registered published row count, not a workbook/held-out shape query.
        # Match historical target-blind 60/15/15 partition; last decile stays
        # outside the access list and is never indexed/read by this layer.
        from .real_protocol import SPLIT_SEED
        ordered = sorted(range(spec.expected_rows), key=lambda i:
            sha256(f"{SPLIT_SEED}:uci_ccpp:{i:06d}".encode()).digest())
        b1, b2, b3 = np.rint(np.array([.60, .75, .90]) * spec.expected_rows).astype(int)
        assigned = _assign({"development": _ordered(ordered[:b1], seed),
                            "validation": _ordered(ordered[b1:b2], seed),
                            "pool": _ordered(ordered[b2:b3], seed)}, registration["counts"])
        values = _ccpp_open_rows(source, {i for rows in assigned.values() for i in rows})
    arrays = {role: np.array([values[i] for i in rows], dtype=float) for role, rows in assigned.items()}
    def opened(role, data_role):
        return RoleDataset(data_role, arrays[role][:, :-1], arrays[role][:, -1])
    development = opened("exploration_development", DataRole.DEVELOPMENT)
    validation = opened("exploration_validation", DataRole.VALIDATION)
    initial = opened("inference_initial", DataRole.DEVELOPMENT)
    evaluation = opened("development_evaluation", DataRole.VALIDATION)
    manifest = {"schema": "scientific-open-development-data-v1", "dataset": dataset,
        "family": "gas_turbine" if dataset.startswith("uci_gas") else "ccpp",
        "official_hashes": hashes, "split_seed": seed, "registered_counts": registration["counts"],
        "role_row_id_hashes": {role: sha256(json.dumps(rows).encode()).hexdigest() for role, rows in assigned.items()},
        "heldout_opened": False, "formula_generated": False, "noise_added": False}
    pool_values = arrays["acquisition_pool"]
    return OpenSystemData(SelectionData(development, validation, None, ()), initial,
        evaluation, PoolOracle(pool_values[:, :-1], pool_values[:, -1]), manifest)
