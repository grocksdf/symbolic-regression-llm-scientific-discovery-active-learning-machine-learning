"""Stage a CB-SI v2 zero-provider freeze; DOES NOT authorize execution.

Only updated source-code hash entries are changed. Data/artifact hashes are
never rehashed or modified. The caller must review and approve the new freeze.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

SOURCES = {
    "counterfactual_bank_symbolic_invention.py",
    "run_cbsi_zero_provider_experiment.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite existing v2 freeze")
    source = json.loads(args.v1_freeze.read_text(encoding="utf-8"))
    if source.get("schema") != "cbsi-zero-provider-development-freeze-v1":
        parser.error("Expected v1 CB-SI freeze")
    old_hashes = source.get("file_hashes")
    if not isinstance(old_hashes, dict):
        parser.error("Missing source hash map")
    seen = set()
    history = []
    for filename, expected in old_hashes.items():
        name = Path(filename).name
        if name not in SOURCES:
            continue
        if name in seen:
            parser.error(f"Duplicate source entry: {name}")
        seen.add(name)
        target = Path(filename)
        if not target.is_file():
            parser.error(f"Patched source file unavailable: {target}")
        new_hash = sha256(target.read_bytes()).hexdigest()
        old_hashes[filename] = new_hash
        history.append({
            "path": filename,
            "v1_sha256": expected,
            "v2_sha256": new_hash,
        })
    if seen != SOURCES:
        parser.error(
            "Freeze must identify both patched Python files; "
            f"found {sorted(seen)}")
    if not isinstance(source.get("budget"), dict):
        parser.error("Missing budget")
    source["schema"] = "cbsi-zero-provider-development-freeze-v2"
    source["protocol_id"] = "cbsi-correctness-hotfix-v2"
    # Preserve the prior zero complexity penalty for an isolated correctness
    # comparison. Any nonzero cost penalty requires a new pre-registration.
    source["budget"]["complexity_cost_weight"] = 0.0
    source["execution_authorized"] = False
    source["source_upgrade_audit"] = history
    source["v2_authorization_note"] = (
        "Fresh approval required after verifying new source, data roles "
        "and task identities; historical v1 authorization is not inherited.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(source, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"Wrote draft freeze (execution_authorized=false): {args.out}")
    for entry in history:
        print(
            f"  {entry['path']}: "
            f"{entry['v1_sha256']} -> {entry['v2_sha256']}")


if __name__ == "__main__":
    main()
