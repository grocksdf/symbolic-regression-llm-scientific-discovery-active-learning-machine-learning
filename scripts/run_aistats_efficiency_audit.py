"""Audit matched-engine cost and resource use for passed synthesis screens."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


CONDITIONS = ("full_scientist_v6", "no_llm_v6")


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _engine_jobs(report: dict) -> int:
    evaluation = report.get("scientist_agent_system_evaluation") or {}
    cycles = evaluation.get("cycles") or ()
    values = []
    for cycle in cycles:
        state = cycle.get("scientist_state_after") or {}
        value = state.get("cumulative_engine_jobs")
        if value is not None:
            values.append(int(value))
    if values:
        return max(values)
    attempts = report.get("scientist_agent_engine_reports") or ()
    return int(sum(len(row.get("run_records") or ()) for row in attempts))


def _provider_calls(report: dict) -> int:
    evaluation = report.get("scientist_agent_system_evaluation") or {}
    if evaluation.get("provider_calls") is not None:
        return int(evaluation["provider_calls"])
    return int(report.get("scientist_agent_logical_llm_call_count") or 0)


def _collect(source: Path, screen_name: str) -> tuple[list[dict], dict]:
    summary_path = source / "AISTATS_SYNTHESIS_ONLY_PILOT_RESULT.json"
    summary = _read(summary_path)
    if (
        summary.get("schema")
        != "scientific-aistats-synthesis-only-pilot-result-v1"
        or summary.get("passed") is not True
        or summary.get("failure_count") != 0
        or summary.get("candidate_response_accessed") is not False
        or summary.get("action_response_accessed") is not False
        or summary.get("test_or_ood_accessed") is not False
        or summary.get("heldout_opened") is not False
        or not summary.get("decisions", {}).get(
            "identical_engine_execution_within_every_pair"
        )
    ):
        raise ValueError(f"source is not a passed response-free synthesis screen: {source}")
    output = []
    for row in summary["rows"]:
        condition = row["condition"]
        if condition not in CONDITIONS or row["status"] != "success":
            raise ValueError("efficiency audit requires complete paired conditions")
        run = (
            source
            / "runs"
            / row["family"]
            / row["task"]
            / f"seed{row['seed']}"
            / condition
        )
        artifact_path = run / "pcpi_artifacts" / f"{row['task']}.json"
        payload = _read(artifact_path)
        report = payload["scientific_discovery_runtime"]
        if (
            report.get("candidate_response_accessed") is not False
            or report.get("heldout_opened") is not False
        ):
            raise ValueError("protected response accessed in efficiency source")
        output.append(
            {
                "screen": screen_name,
                "family": row["family"],
                "task": row["task"],
                "seed": int(row["seed"]),
                "condition": condition,
                "normalized_aulc": float(row["normalized_aulc"]),
                "evaluation_budget_limit": int(report["evaluation_budget_limit"]),
                "evaluation_budget_used": int(report["evaluation_budget_used"]),
                "engine_jobs": _engine_jobs(report),
                "logical_llm_calls": int(
                    report.get("scientist_agent_logical_llm_call_count") or 0
                ),
                "provider_calls": _provider_calls(report),
                "provider_attempts": int(
                    report.get("scientist_agent_provider_attempt_count") or 0
                ),
                "artifact_sha256": _sha(artifact_path),
                "candidate_response_accessed": False,
                "heldout_opened": False,
            }
        )
    return output, {
        "path": str(summary_path.resolve()),
        "sha256": _sha(summary_path),
    }


def _condition_summary(rows: list[dict], condition: str) -> dict:
    selected = [row for row in rows if row["condition"] == condition]
    return {
        "run_count": len(selected),
        "mean_evaluation_budget_used": mean(
            row["evaluation_budget_used"] for row in selected
        ),
        "mean_engine_jobs": mean(row["engine_jobs"] for row in selected),
        "total_logical_llm_calls": sum(
            row["logical_llm_calls"] for row in selected
        ),
        "total_provider_calls": sum(row["provider_calls"] for row in selected),
        "total_provider_attempts": sum(
            row["provider_attempts"] for row in selected
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("efficiency audit output must be new")
    if len(args.source) < 2:
        raise ValueError("efficiency audit requires at least two passed screens")

    rows, sources = [], []
    for index, source in enumerate(args.source, start=1):
        collected, identity = _collect(source.resolve(), f"screen_{index}")
        rows.extend(collected)
        sources.append(identity)

    by = {
        (
            row["screen"],
            row["family"],
            row["task"],
            row["seed"],
            row["condition"],
        ): row
        for row in rows
    }
    pair_keys = sorted(
        {
            (row["screen"], row["family"], row["task"], row["seed"])
            for row in rows
        }
    )
    pair_checks = []
    for key in pair_keys:
        full = by[(*key, "full_scientist_v6")]
        no_llm = by[(*key, "no_llm_v6")]
        pair_checks.append(
            {
                "screen": key[0],
                "family": key[1],
                "task": key[2],
                "seed": key[3],
                "evaluation_budget_limit_equal": (
                    full["evaluation_budget_limit"]
                    == no_llm["evaluation_budget_limit"]
                ),
                "engine_jobs_equal": full["engine_jobs"] == no_llm["engine_jobs"],
                "no_llm_calls_are_zero": no_llm["logical_llm_calls"] == 0,
                "full_llm_calls_within_registered_limit": (
                    full["logical_llm_calls"] <= 4
                ),
                "full_minus_no_llm_aulc": (
                    full["normalized_aulc"] - no_llm["normalized_aulc"]
                ),
            }
        )

    full_summary = _condition_summary(rows, "full_scientist_v6")
    no_llm_summary = _condition_summary(rows, "no_llm_v6")
    decisions = {
        "all_registered_rows_accounted_for": (
            len(rows) == 2 * len(pair_keys) and len(pair_keys) == 16
        ),
        "all_pairs_share_evaluation_budget_limit": all(
            row["evaluation_budget_limit_equal"] for row in pair_checks
        ),
        "all_pairs_share_engine_job_count": all(
            row["engine_jobs_equal"] for row in pair_checks
        ),
        "no_llm_condition_has_zero_llm_calls": all(
            row["no_llm_calls_are_zero"] for row in pair_checks
        ),
        "full_condition_respects_llm_call_limit": all(
            row["full_llm_calls_within_registered_limit"] for row in pair_checks
        ),
        "candidate_responses_stayed_closed": all(
            row["candidate_response_accessed"] is False for row in rows
        ),
        "heldout_stayed_closed": all(
            row["heldout_opened"] is False for row in rows
        ),
    }
    result = {
        "schema": "scientific-aistats-synthesis-efficiency-audit-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "source_results": sources,
        "pair_count": len(pair_keys),
        "row_count": len(rows),
        "condition_summaries": {
            "full_scientist_v6": full_summary,
            "no_llm_v6": no_llm_summary,
        },
        "mean_full_minus_no_llm_aulc": mean(
            row["full_minus_no_llm_aulc"] for row in pair_checks
        ),
        "pair_checks": pair_checks,
        "rows": rows,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Matched-engine development resource accounting only; not "
            "external-baseline superiority, realized efficacy, or confirmation."
        ),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_SYNTHESIS_EFFICIENCY_AUDIT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"rows", "pair_checks"}},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
