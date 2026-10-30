"""Combine symbols-only metadata for the frozen cross-family task set."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = ("name", "symbols", "symbol_descs", "symbol_properties")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("cross-family metadata already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    selected = protocol["selection"].get("selected_tasks_by_family")
    if selected is None:
        selected_rows = []
        seen = set()
        for row in protocol["selection"]["coordinates"]:
            key = (str(row["family"]), str(row["task"]))
            if key not in seen:
                seen.add(key); selected_rows.append(key)
    else:
        selected_rows = list(selected.items())
    files = {
        "bio_pop_growth": "lsr_synth_bio_pop_growth-00000-of-00001.parquet",
        "chem_react": "lsr_synth_chem_react-00000-of-00001.parquet",
        "matsci": "lsr_synth_matsci-00000-of-00001.parquet",
        "phys_osc": "lsr_synth_phys_osc-00000-of-00001.parquet"}
    rows = []
    for family, task in selected_rows:
        table = pq.read_table(
            args.data_dir / files[family], columns=list(COLUMNS))
        names = list(map(str, table.column("name").to_pylist()))
        if names.count(task) != 1:
            raise ValueError(f"metadata missing or duplicated: {family}/{task}")
        rows.append(table.take(pa.array(
            [names.index(task)], type=pa.int64())))
    combined = pa.concat_tables(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, args.output, compression="zstd")
    print(json.dumps({
        "rows": combined.num_rows, "columns": combined.column_names,
        "expression_column_opened": False,
        "scientific_responses_opened": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
