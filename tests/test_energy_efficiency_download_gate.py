"""Synthetic XLSX fixtures for response-isolated Energy Efficiency inspection."""

import io
from hashlib import sha256
from zipfile import ZipFile

from hypothesis_mvp.discovery.energy_efficiency_download_gate import (
    inspect_energy_efficiency_zip,
)


def _fixture(rows=20):
    sheet_rows = ['<row r="1"></row>']
    for index in range(rows):
        cells = []
        for column, letter in enumerate("ABCDEFGH"):
            cells.append(
                f'<c r="{letter}{index + 2}"><v>{index + column / 10}</v></c>')
        cells.append(f'<c r="I{index + 2}"><v>{index + .8}</v></c>')
        cells.append(
            f'<c r="J{index + 2}"><v>DO_NOT_DECODE_SECONDARY</v></c>')
        sheet_rows.append(
            f'<row r="{index + 2}">' + "".join(cells) + "</row>")
    workbook = io.BytesIO()
    with ZipFile(workbook, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            "<worksheet><sheetData>" + "".join(sheet_rows) +
            "</sheetData></worksheet>")
    outer = io.BytesIO()
    with ZipFile(outer, "w") as archive:
        archive.writestr("ENB2012_data.xlsx", workbook.getvalue())
    return outer.getvalue()


def test_energy_gate_never_decodes_reserved_or_secondary_targets():
    payload = _fixture()
    registration = {
        "download_sha256": sha256(payload).hexdigest(),
        "data_member": "ENB2012_data.xlsx",
        "published_instances": 20,
        "published_features": 8,
        "split_seed": 7,
        "role_counts": {
            "exploration_development": 2,
            "exploration_validation": 2,
            "inference_initial": 2,
            "development_evaluation": 2,
            "acquisition_pool": 2,
            "reserved_confirmation": 4,
            "unused_open": 6}}
    report, workbook = inspect_energy_efficiency_zip(payload, registration)
    assert workbook
    assert report["passed"] is True
    assert report["open_target_values_decoded"] == 10
    assert report["reserved_confirmation_target_values_decoded"] == 0
    assert report["unused_open_target_values_decoded"] == 0
    assert report["sealed_secondary_target_values_decoded"] == 0
