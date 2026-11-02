"""Freeze a provider-free staged-admission diagnostic over PMLB v2."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAINLINE = ROOT.parent / "hypothesis_mvp"


def digest(path):
    value = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("staged diagnostic freeze exists")
    run_freeze = json.loads(
        (args.source_output / "RUN_FREEZE.json").read_text())
    coordinates = list(run_freeze["coordinates"])
    artifacts = []
    for coordinate in coordinates:
        root = (args.source_output / "generation" /
                coordinate["family"] / coordinate["task"] /
                f"seed{int(coordinate['seed'])}")
        for path in (
                root / "SHARED_BLOCK.json",
                root / "B0" / "GENERATION_RESULT.json"):
            artifacts.append({
                "path": str(path.resolve()), "sha256": digest(path)})
    runner = ROOT / "scripts" / "run_staged_admission_diagnostic.py"
    module = (
        MAINLINE / "hypothesis_mvp" / "discovery" /
        "staged_candidate_admission.py")
    freeze = {
        "schema": "staged-admission-development-diagnostic-freeze-v1",
        "execution_authorized": True,
        "claim_boundary": (
            "Post-confirmation development diagnostic over already-opened "
            "PMLB v2 tasks. It evaluates a pre-specified quarantine appeal "
            "using the previously unused decision_calibration role. It is "
            "not independent confirmation and cannot alter historical results."),
        "source_output": str(args.source_output.resolve()),
        "source_run_freeze_sha256":
            digest(args.source_output / "RUN_FREEZE.json"),
        "generation_freeze_sha256":
            digest(args.source_output / "GENERATION_FROZEN.json"),
        "hdf5": str(args.hdf5.resolve()),
        "hdf5_sha256": digest(args.hdf5),
        "coordinates": coordinates,
        "coordinate_count": len(coordinates),
        "conditions": ["original_protected", "staged_protected"],
        "staged_rule": {
            "direct_threshold": "K/alpha unchanged",
            "quarantine_eligibility":
                "positive first-stage global log evidence",
            "appeal_role": "decision_calibration",
            "mixture": "0.5*E1 + 0.5*E1*E2",
            "regional_rule":
                "combined evidence nonnegative in every eligible region",
            "alpha": 0.025},
        "artifact_manifest": artifacts,
        "file_hashes": {
            str(runner.resolve()): digest(runner),
            str(module.resolve()): digest(module),
            str((ROOT / "scripts/run_gap_candidate_marginal_value.py"
                 ).resolve()): digest(
                     ROOT / "scripts/run_gap_candidate_marginal_value.py")},
        "provider_calls_authorized": False,
        "result_dependent_changes_authorized": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "coordinate_count": len(coordinates),
        "artifact_count": len(artifacts),
        "provider_calls_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
