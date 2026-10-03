"""No-data correctness Gate for source-first Blind/Gap composition routing."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime
from hypothesis_mvp.discovery.scientist_policy import deterministic_plan
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


def _identity(value) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("round-two gap-routing Gate output must be new")

    runtime = ProposalRuntime(
        EquationRuntime(2, refit_policy="pcpi-expanded-fixed-inner-v1"),
        2, None, 1)
    raw = {
        "protocol_id": "scientific-engine-evidence-review-v1",
        "supported_mechanisms": ["fixture"],
        "contradicted_mechanisms": [],
        "cross_engine_conflicts": [],
        "synthesis_instructions": ["compose"],
        "stop": True,
        "stop_reason": "correctness fixture abstention",
        "synthesis_directives": [],
    }
    calls = []

    def complete_json(*, system_message, payload):
        calls.append({
            "system_message": system_message,
            "payload": payload,
        })
        return raw, {"fixture": "no-data-gap-routing"}

    runtime.complete_json = complete_json
    evidence = [
        {"engine": "polynomial_lasso", "expression": "x0",
         "lineage_id": "linear", "selection_score": 1.0},
        {"engine": "mcts", "expression": "sin(x1)",
         "lineage_id": "nonlinear", "selection_score": 1.1},
    ]
    kwargs = {
        "plan": deterministic_plan(("polynomial_lasso", "mcts"), 2),
        "engine_evidence": evidence,
        "require_typed_synthesis": True,
        "allow_interactions": True,
        "expanded_formula_synthesis": True,
    }
    runtime.review_engine_evidence(**kwargs)
    gap_context = {
        "posterior_gap_brief": {
            "propose_allowed": True,
            "eligible_regions": [{"region_id": "fixture-region"}],
        },
        "posterior_gap_existing_supports": [["x0"], ["sin_x1"]],
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }
    runtime.review_engine_evidence(
        **kwargs, scientific_context=gap_context)
    if len(calls) != 2:
        raise ValueError("Blind/Gap review call count changed")
    blind, gap = calls
    checks = {
        "same_engine_evidence": (
            _identity(blind["payload"]["engine_evidence"])
            == _identity(gap["payload"]["engine_evidence"])),
        "blind_has_no_gap_context": (
            blind["payload"]["scientific_context"] == {}),
        "gap_has_exact_registered_context": (
            gap["payload"]["scientific_context"] == gap_context),
        "gap_context_is_response_free": (
            gap_context["candidate_response_accessed"] is False
            and gap_context["heldout_opened"] is False),
        "gap_grants_no_epistemic_authority": (
            gap["payload"]["authority"][
                "may_access_pool_responses"] is False
            and gap["payload"]["authority"]["may_access_heldout"] is False),
    }
    result = {
        "schema": "formula-round2-gap-routing-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "engine_evidence_identity": _identity(evidence),
        "blind_context_identity": _identity({}),
        "gap_context_identity": _identity(gap_context),
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "provider_called": False,
        "real_data_accessed": False,
        "ground_truth_accessed": False,
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "No-data source-routing correctness only; not formula "
            "generation, admission, recovery, efficacy or confirmation."),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "GATE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
