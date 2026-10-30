"""Extract only registered symbol metadata for the frozen Gap task list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


COLUMNS = ("name", "symbols", "symbol_descs", "symbol_properties")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("symbols-only metadata output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    tasks = tuple(protocol["selection"]["tasks"])
    table = pq.read_table(args.source, columns=list(COLUMNS))
    names = [str(value) for value in table.column("name").to_pylist()]
    indices = []
    for task in tasks:
        if names.count(task) != 1:
            raise ValueError(f"registered task metadata missing or duplicated: {task}")
        indices.append(names.index(task))
    selected = table.take(pa.array(indices, type=pa.int64()))
    if selected.column_names != list(COLUMNS):
        raise ValueError("symbols-only projection opened an unregistered column")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(selected, args.output, compression="zstd")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "row_count": selected.num_rows,
        "columns": selected.column_names,
        "tasks": selected.column("name").to_pylist(),
        "expression_column_opened": False,
        "scientific_responses_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
