"""Build a read-only reporting correction for an external LLM-SR continuation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_aistats_external_llmsr_benchmark import _aggregate


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_means(rows: list[dict]) -> dict:
    output = {}
    tasks = sorted({(row["family"], row["task"]) for row in rows})
    conditions = sorted({row["condition"] for row in rows})
    for family, task in tasks:
        output[f"{family}/{task}"] = {}
        for condition in conditions:
            selected = [
                row for row in rows
                if row["family"] == family and row["task"] == task
                and row["condition"] == condition
            ]
            output[f"{family}/{task}"][condition] = {
                "id_nmse": float(np.mean([row["id_nmse"] for row in selected])),
                "ood_nmse": float(np.mean([row["ood_nmse"] for row in selected])),
                "id_acc_0_1": float(
                    np.mean([row["id_acc_0_1"] for row in selected])),
                "ood_acc_0_1": float(
                    np.mean([row["ood_acc_0_1"] for row in selected])),
                "failure_count": sum(
                    row["status"] != "success" for row in selected),
            }
    return output


def _wins(task_means: dict, metric: str) -> dict:
    full_wins = external_wins = ties = 0
    for values in task_means.values():
        full = values["full_scientist_v6"][metric]
        external = values["external_llmsr_glm53"][metric]
        tolerance = 1e-12 * max(1.0, abs(full), abs(external))
        if full < external - tolerance:
            full_wins += 1
        elif external < full - tolerance:
            external_wins += 1
        else:
            ties += 1
    return {
        "full_wins": full_wins,
        "external_llmsr_wins": external_wins,
        "ties": ties,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("external reporting correction output must be new")
    continuation = _read(args.continuation)
    result = _read(args.result)
    if (
        continuation.get("schema")
        != "scientific-aistats-external-llmsr-continuation-v1"
        or result.get("schema")
        != "scientific-aistats-external-llmsr-result-v1"
        or result.get("protocol_complete") is not True
        or result.get("row_count") != 24
    ):
        raise ValueError("invalid external reporting correction inputs")
    rows = result["rows"]
    counts = {
        condition: sum(row["condition"] == condition for row in rows)
        for condition in continuation["conditions"]
    }
    if counts != {
        "full_scientist_v6": 8,
        "no_llm_v6": 8,
        "external_llmsr_glm53": 8,
    }:
        raise ValueError("external result condition counts are invalid")
    task_means = _task_means(rows)
    full_no_llm_equal = {
        metric: int(sum(
            np.isclose(
                values["full_scientist_v6"][metric],
                values["no_llm_v6"][metric],
                rtol=0.0,
                atol=1e-12 * max(
                    1.0,
                    abs(values["full_scientist_v6"][metric]),
                    abs(values["no_llm_v6"][metric]),
                ),
            )
            for values in task_means.values()
        ))
        for metric in ("id_nmse", "ood_nmse", "id_acc_0_1", "ood_acc_0_1")
    }
    corrected = {
        "schema": "scientific-aistats-external-llmsr-reporting-correction-v1",
        "status": "eligible-for-corrected-reporting-only",
        "source_result": str(args.result.resolve()),
        "source_result_sha256": _sha(args.result),
        "source_continuation": str(args.continuation.resolve()),
        "source_continuation_sha256": _sha(args.continuation),
        "original_artifacts_immutable": True,
        "read_only": True,
        "correction_scope": (
            "reporting-count-alias-reconciliation-no-rerun-no-rescoring"),
        "original_reported_preserved_internal_row_count":
            result.get("preserved_internal_row_count"),
        "original_reported_rerun_external_row_count":
            result.get("rerun_external_row_count"),
        "corrected_preserved_internal_row_count": 16,
        "corrected_rerun_external_row_count": 8,
        "condition_row_counts": counts,
        "corrected_condition_summaries": _aggregate(rows),
        "task_means": task_means,
        "family_win_counts_lower_nmse_is_better": {
            "id_nmse": _wins(task_means, "id_nmse"),
            "ood_nmse": _wins(task_means, "ood_nmse"),
        },
        "full_no_llm_equal_task_count": full_no_llm_equal,
        "failure_count": sum(row["status"] != "success" for row in rows),
        "candidate_response_accessed": False,
        "test_or_ood_accessed": True,
        "heldout_opened": False,
        "rerun_authorized": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Read-only reporting correction and task-level external comparison; "
            "not a rerun, held-out confirmation, LLM synthesis effect, or "
            "universal superiority claim."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_EXTERNAL_LLMSR_REPORTING_CORRECTION.json"
    path.write_text(
        json.dumps(corrected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(corrected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
