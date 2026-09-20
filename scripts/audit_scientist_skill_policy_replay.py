"""Read-only cross-task replay of response-free scientist skill evidence."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.skill_policy import (
    SkillTaskEvidence, leave_one_task_out_skill_policy,
)
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _task_context(coordinate):
    manifest_path = coordinate / "DATA_MANIFEST.json"
    manifest = _read(manifest_path)
    identity = sha256(manifest_path.read_bytes()).hexdigest()
    return identity, str(manifest["family"])


def _candidate_evidence(coordinate):
    report = _read(coordinate / "CANDIDATE_ADMISSION.json")
    identity, family = _task_context(coordinate)
    grouped = {}
    for row in report["variants"]["full"]["candidate_certificates"]:
        skill = str(row["family"])
        if skill == "core":
            continue
        gains = tuple(float(value) for value in row["fold_log_score_gains_vs_core"])
        tolerances = tuple(float(value) for value in row["fold_numerical_tolerances"])
        score = min(gain - tolerance for gain, tolerance in zip(gains, tolerances))
        current = grouped.get(skill)
        if current is None or score > current[0]:
            grouped[skill] = (score, gains, tolerances)
    return [SkillTaskEvidence(identity, family, skill, values[1], values[2])
            for skill, values in grouped.items()]


def _source_evidence(coordinate):
    report = _read(coordinate / "SOURCE_ADMISSION.json")
    identity, family = _task_context(coordinate)
    output = []
    for skill, row in report["variants"]["full"]["sources"].items():
        if skill == "core":
            continue
        output.append(SkillTaskEvidence(
            identity, family, skill,
            tuple(float(value) for value in row["fold_log_score_gains_vs_core"]),
            tuple(float(value) for value in row["fold_numerical_tolerances"])))
    return output


def _coordinate(root):
    candidates = sorted(path.parent for path in Path(root).rglob("DATA_MANIFEST.json"))
    if len(candidates) != 1:
        raise ValueError("each replay root must contain exactly one coordinate")
    return candidates[0]


def collect(roots):
    by_key = {}
    for root in roots:
        coordinate = _coordinate(root)
        rows = (_candidate_evidence(coordinate)
                if (coordinate / "CANDIDATE_ADMISSION.json").is_file()
                else _source_evidence(coordinate))
        for row in rows:
            key = (row.task_identity, row.skill)
            if key in by_key and by_key[key] != row:
                raise ValueError("duplicate task/skill evidence is inconsistent")
            by_key[key] = row
    return tuple(by_key[key] for key in sorted(by_key))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = collect(args.roots)
    result = leave_one_task_out_skill_policy(evidence)
    result["source_roots"] = [str(path.resolve()) for path in args.roots]
    result["source_artifacts_immutable"] = True
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _publish(args.output.resolve(), result)
    print(text)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
