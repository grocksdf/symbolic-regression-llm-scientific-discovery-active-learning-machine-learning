"""Audit a matched No-LLM engine bank plus additional typed LLM proposals.

Reads development candidate reports only; no acquisition, response reveal,
benchmark data loading, provider call, or formal efficacy evaluation.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from methods.hypothesis_mvp_pcpi.quality_first_augmentation import (
    build_quality_first_augmentation,
)


def audit_candidate_reports(root: Path, *, baseline_evaluations: int = 48,
                            llm_proposal_limit: int = 12) -> dict:
    root = Path(root)
    full_paths = sorted(root.glob("*/*/seed*/full_scientist_v6/pcpi_artifacts/*.json"))
    no_paths = sorted(root.glob("*/*/seed*/no_llm_v6/pcpi_artifacts/*.json"))
    if not full_paths or len(full_paths) != len(no_paths):
        raise ValueError("paired Full/No-LLM candidate reports are required")
    rows = []
    for path in full_paths:
        no_path = path.parent.parent.parent / "no_llm_v6" / "pcpi_artifacts" / path.name
        if not no_path.is_file():
            raise ValueError(f"missing No-LLM report for {path}")
        full = json.loads(path.read_text(encoding="utf-8"))["scientific_discovery_runtime"]
        no_llm = json.loads(no_path.read_text(encoding="utf-8"))["scientific_discovery_runtime"]
        n_features = full["task_context_audit"]["variable_description_count"]
        if no_llm["task_context_audit"]["variable_description_count"] != n_features:
            raise ValueError("paired reports have different feature counts")
        audit = build_quality_first_augmentation(
            full, no_llm, n_features=n_features,
            baseline_evaluations=baseline_evaluations,
            llm_proposal_limit=llm_proposal_limit)
        rows.append({
            "family": path.parts[-6], "task": path.parts[-5],
            "seed": path.parts[-4],
            "full_artifact_sha256": sha256(path.read_bytes()).hexdigest(),
            "no_llm_artifact_sha256": sha256(no_path.read_bytes()).hexdigest(),
            "baseline_evaluations_used": audit["baseline_evaluations_used"],
            "baseline_evaluated_candidate_count": audit["baseline_evaluated_candidate_count"],
            "llm_compiled_proposal_count": audit["llm_compiled_proposal_count"],
            "llm_novel_proposal_count": audit["llm_novel_proposal_count"],
            "duplicate_support_count": len(audit["excluded_llm_proposals"]),
            "additional_llm_supports": [row["support"] for row in
                                        audit["additional_llm_proposals"]],
        })
    if len(rows) != len(no_paths):
        raise ValueError("unmatched No-LLM candidate reports")
    return {
        "schema": "scientific-quality-first-response-free-archive-audit-v1",
        "pair_count": len(rows),
        "baseline_evaluations_per_pair": baseline_evaluations,
        "llm_proposal_limit_per_pair": llm_proposal_limit,
        "llm_compiled_proposal_total": sum(row["llm_compiled_proposal_count"] for row in rows),
        "llm_novel_proposal_total": sum(row["llm_novel_proposal_count"] for row in rows),
        "duplicate_support_total": sum(row["duplicate_support_count"] for row in rows),
        "rows": rows,
        "candidate_response_accessed": False, "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": "construction only; historic 48-evaluation No-LLM bank "
            "plus compiled LLM proposals, without independent rescoring",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--baseline-evaluations", type=int, default=48)
    parser.add_argument("--llm-proposal-limit", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_candidate_reports(
        args.candidates_root, baseline_evaluations=args.baseline_evaluations,
        llm_proposal_limit=args.llm_proposal_limit)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
