"""Synthetic ZIP fixtures for response-isolated Yacht download inspection."""

import io
from hashlib import md5
from zipfile import ZipFile

from hypothesis_mvp.discovery.yacht_download_gate import inspect_yacht_zip


def _fixture():
    lines = []
    for group in range(22):
        geometry = [group, group + .1, group + .2, group + .3, group + .4]
        for speed in range(14):
            row = [*geometry, speed / 10, group + speed / 100]
            lines.append(" ".join(str(value) for value in row))
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr(
            "yacht_hydrodynamics.data", "\n".join(lines) + "\n")
    return stream.getvalue()


def test_yacht_zip_inspection_never_decodes_confirmation_targets():
    payload = _fixture()
    registration = {
        "download_md5": md5(payload).hexdigest(),
        "published_instances": 308,
        "group_split": {
            "development": 8, "validation": 4, "acquisition_pool": 2,
            "unused_open": 3, "reserved_confirmation": 5}}
    report, data = inspect_yacht_zip(payload, registration)
    assert len(data) > 0
    assert report["passed"] is True
    assert report["hull_group_count"] == 22
    assert report["open_target_values_decoded"] == 196
    assert report["reserved_confirmation_target_values_decoded"] == 0
    assert report["unused_open_target_values_decoded"] == 0
