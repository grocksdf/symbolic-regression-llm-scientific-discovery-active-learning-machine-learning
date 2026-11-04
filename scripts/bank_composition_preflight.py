"""Response-free dependency and ancestry validation before generation/transport.

No task is replaced or excluded on failure. Row IDs must be source-bound by
the owning runner; a caller declaration is not itself proof of provenance.
"""
from pathlib import Path
import json
from hashlib import sha256


def validate_generation_ancestry(design, fit_row_ids, core_generation_row_ids, fit_count):
    def ids(value):
        if (not isinstance(value, tuple) or not value
                or any(not isinstance(v, str) or not v for v in value)
                or len(set(value)) != len(value)):
            raise ValueError("explicit unique source-bound generation row IDs required")
        return set(value)
    fit, core = ids(fit_row_ids), ids(core_generation_row_ids)
    if len(fit_row_ids) != fit_count or fit | core != set(design.training_ids):
        raise ValueError("training IDs must cover exactly fitting and core-generation ancestry")
    return {"fitting_row_ids": fit_row_ids, "core_generation_row_ids": core_generation_row_ids,
            "provenance_is_caller_declared": True}


def validate_historical_bank_manifest(coordinates):
    """Check every registered dependency without reading response arrays/results."""
    if not coordinates:
        raise ValueError("registered coordinate manifest is empty")
    seen, receipts, failures = set(), [], []
    for index, row in enumerate(coordinates):
        try:
            if not isinstance(row, dict):
                raise ValueError("invalid coordinate")
            identity = (row["family"], row["task"], row["seed"])
            if (any(not isinstance(v, str) or not v for v in identity[:2])
                    or type(identity[2]) is not int or identity[2] < 0 or identity in seen):
                raise ValueError("invalid or duplicate coordinate identity")
            seen.add(identity)
            if row.get("coordinate_failure") is not False:
                raise ValueError("historical failure cannot supply a frozen core")
            path = Path(row["coordinate_root"]) / "SHARED_BLOCK.json"
            content = path.read_bytes()
            value = json.loads(content.decode("utf-8"))
            shared = value["shared"]
            if shared["task"] != row["task"] or shared["seed"] != row["seed"]:
                raise ValueError("shared-bank coordinate mismatch")
            core = shared["E0_rows"]
            if not isinstance(core, list) or not core or any(
                    not isinstance(r, dict) or not isinstance(r.get("expression"), str)
                    or not r["expression"] or not isinstance(r.get("source"), str) for r in core):
                raise ValueError("invalid frozen core rows")
            receipts.append({"coordinate": identity, "core_path": str(path), "core_rows": core,
                             "core_file_sha256": sha256(content).hexdigest()})
        except (KeyError, TypeError, ValueError, OSError) as error:
            failures.append({"index": index, "error_type": type(error).__name__})
    if failures:
        raise ValueError("incomplete registered core manifest: " + json.dumps(failures))
    return receipts


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("preflight output exists")
    raw = args.manifest.read_bytes()
    receipts = validate_historical_bank_manifest(json.loads(raw.decode("utf-8"))["coordinates"])
    # A dependency receipt is not an efficacy registration or execution permit.
    result = {"schema":"bank-composition-dependency-preflight-v1", "passed":True,
              "source_manifest_sha256":sha256(raw).hexdigest(), "coordinates":receipts,
              "response_arrays_opened":False, "provider_calls":0, "execution_authorized":False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
