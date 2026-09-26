"""Response-isolated inspection of the official UCI Energy Efficiency archive."""

from __future__ import annotations

from hashlib import sha256
import io
from math import isfinite
from zipfile import ZipFile
from xml.sax import handler, make_parser


OPEN_ROLES = (
    "exploration_development", "exploration_validation",
    "inference_initial", "development_evaluation", "acquisition_pool")


def _ordered_rows(registration):
    count = int(registration["published_instances"])
    seed = int(registration["split_seed"])
    return sorted(range(count), key=lambda index:
        sha256(f"{seed}:energy_efficiency:{index}".encode()).digest())


def _assign_rows(registration):
    order = _ordered_rows(registration)
    counts = registration["role_counts"]
    roles, offset = {}, 0
    for role in (*OPEN_ROLES, "reserved_confirmation", "unused_open"):
        end = offset + int(counts[role])
        roles[role] = order[offset:end]
        offset = end
    if offset != len(order):
        raise ValueError("Energy Efficiency role counts do not cover the source")
    return roles


def _inspect_sheet(xlsx_payload, registration, roles):
    open_rows = {index for role in OPEN_ROLES for index in roles[role]}
    values = {}
    seen_rows = set()
    decoded_targets = 0

    class Sheet(handler.ContentHandler):
        def __init__(self):
            self.row = self.column = None
            self.collect = False
            self.text = []

        def startElement(self, name, attrs):
            if name == "row":
                index = int(attrs["r"]) - 2
                self.row = index if 0 <= index < registration[
                    "published_instances"] else None
                if self.row is not None:
                    seen_rows.add(self.row)
                    values.setdefault(self.row, [None] * 9)
            elif name == "c" and self.row is not None:
                letter = attrs["r"].rstrip("0123456789")
                self.column = "ABCDEFGHIJ".find(letter)
                opened = self.column in range(8) or (
                    self.column == 8 and self.row in open_rows)
                if attrs.get("t", "n") != "n" and opened:
                    raise ValueError("Energy Efficiency data cell is not numeric")
            elif name == "v" and self.row is not None:
                if self.column in range(8) or (
                        self.column == 8 and self.row in open_rows):
                    self.collect = True
                    self.text = []

        def characters(self, content):
            if self.collect:
                self.text.append(content)

        def endElement(self, name):
            nonlocal decoded_targets
            if name == "v" and self.collect:
                value = float("".join(self.text))
                if not isfinite(value):
                    raise ValueError("Energy Efficiency open value is nonfinite")
                values[self.row][self.column] = value
                if self.column == 8:
                    decoded_targets += 1
                self.collect = False
            elif name == "c":
                self.column = None
            elif name == "row":
                self.row = None

    parser = make_parser()
    parser.setFeature(handler.feature_external_ges, False)
    parser.setContentHandler(Sheet())
    with ZipFile(io.BytesIO(xlsx_payload)) as workbook:
        with workbook.open("xl/worksheets/sheet1.xml") as stream:
            parser.parse(stream)
    if seen_rows != set(range(registration["published_instances"])):
        raise ValueError("Energy Efficiency published row identity changed")
    if any(any(value is None for value in values[index][:8])
           for index in seen_rows):
        raise ValueError("Energy Efficiency input matrix is incomplete")
    if any(values[index][8] is None for index in open_rows):
        raise ValueError("Energy Efficiency open targets are incomplete")
    return decoded_targets


def inspect_energy_efficiency_zip(payload, registration):
    archive_hash = sha256(payload).hexdigest()
    if archive_hash != registration["download_sha256"]:
        raise ValueError("Energy Efficiency official archive hash changed")
    with ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if registration["data_member"] not in names:
            raise ValueError("Energy Efficiency workbook member is absent")
        workbook = archive.read(registration["data_member"])
    roles = _assign_rows(registration)
    decoded = _inspect_sheet(workbook, registration, roles)
    open_count = sum(len(roles[role]) for role in OPEN_ROLES)
    if decoded != open_count:
        raise ValueError("Energy Efficiency target isolation count changed")
    report = {
        "schema": "scientific-energy-efficiency-download-schema-gate-v1",
        "passed": True,
        "archive_sha256": archive_hash,
        "data_member": registration["data_member"],
        "data_member_sha256": sha256(workbook).hexdigest(),
        "row_count": registration["published_instances"],
        "feature_count": registration["published_features"],
        "role_row_indices": roles,
        "role_row_counts": {name: len(rows) for name, rows in roles.items()},
        "open_target_values_decoded": decoded,
        "reserved_confirmation_target_values_decoded": 0,
        "unused_open_target_values_decoded": 0,
        "sealed_secondary_target_values_decoded": 0,
        "observed_value_statistics_computed": False,
        "observed_values_published": False,
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "confirmation_responses_opened": False,
        "execution_authorized": False,
    }
    return report, workbook


__all__ = ["inspect_energy_efficiency_zip"]
