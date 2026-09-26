"""No-engine Yacht open loader smoke tests."""

import json
from pathlib import Path
import tempfile

from hypothesis_mvp.discovery.yacht_download_gate import _group_id
from hypothesis_mvp.discovery.yacht_open_loader_smoke import (
    smoke_yacht_open_loader,
)


def test_open_loader_smoke_binds_roles_and_closes_reserved_targets():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "yacht.data"
        rows = []
        groups = {}
        for group_index in range(2):
            inputs = (
                float(group_index), .5, 4., 4., 3., .1)
            group = _group_id(inputs[:5])
            groups.setdefault("development", []).append(group)
            for index in range(14):
                rows.append(" ".join(
                    [*(str(value) for value in inputs),
                     str(index + 1)]))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        schema = {
            "data_member_sha256": __import__("hashlib").sha256(
                path.read_bytes()).hexdigest(),
            "row_count": 28,
            "role_group_commitments": {
                "development": groups["development"],
                "validation": [], "acquisition_pool": [],
                "unused_open": [], "reserved_confirmation": []},
            "role_row_counts": {
                "development": 28, "validation": 0,
                "acquisition_pool": 0},
        }
        registration = {
            "group_split": {"development": 2, "validation": 0,
                "acquisition_pool": 0, "unused_open": 0,
                "reserved_confirmation": 0}}
        result = smoke_yacht_open_loader(path, schema, registration)
        assert result["passed"] is True
        assert result["confirmation_responses_opened"] is False
