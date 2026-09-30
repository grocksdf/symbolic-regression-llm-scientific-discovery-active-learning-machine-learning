"""Build familywise acquisition and synthesis deployment certificates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.discovery.realized_policy import (
    RealizedPolicyTaskEvidence, fit_familywise_realized_policy,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _symmetric(trajectory):
    if "symmetric_realized_risk_aulc" in trajectory:
        return float(trajectory["symmetric_realized_risk_aulc"])
    risks = np.asarray(trajectory["risk_curve"], dtype=float)
    initial = risks[0]
    absolute = initial - risks
    denominator = initial + risks
    curve = np.divide(
        absolute, denominator, out=np.zeros_like(absolute),
        where=denominator > 0.0)
    return float(np.trapezoid(
        np.clip(curve, -1.0, 1.0), np.arange(len(curve))) / 2.0)


def _evidence(paths, contrast):
    output = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        rows = result["rows"]
        by = {(row["family"], row["task"], row["seed"], row["label"]): row
              for row in rows}
        for family, task in sorted(set(
                (row["family"], row["task"]) for row in rows)):
            seeds = sorted(set(
                row["seed"] for row in rows
                if row["family"] == family and row["task"] == task))
            gains = []
            for seed in seeds:
                if contrast == "acquisition":
                    left, right = "full_targeted", "full_random"
                else:
                    left, right = "full_targeted", "no_llm_targeted"
                gains.append(
                    _symmetric(by[(family, task, seed, left)]["trajectory"])
                    - _symmetric(
                        by[(family, task, seed, right)]["trajectory"]))
            output.append(RealizedPolicyTaskEvidence(
                f"{family}/{task}", family, float(np.mean(gains)), 1e-12))
    return tuple(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realized-result", type=Path, action="append",
                        required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("realized policy output must be new")
    acquisition_evidence = _evidence(
        args.realized_result, "acquisition")
    synthesis_evidence = _evidence(
        args.realized_result, "synthesis")
    acquisition = fit_familywise_realized_policy(
        acquisition_evidence, policy_name="decision-risk-vs-random")
    synthesis = fit_familywise_realized_policy(
        synthesis_evidence, policy_name="synthesis-vs-no-llm")
    result = {
        "schema": "scientific-aistats-realized-policy-calibration-v1",
        "acquisition_policy": acquisition,
        "synthesis_policy": synthesis,
        "acquisition_evidence": [
            vars(row) | {"outcome": row.outcome}
            for row in acquisition_evidence],
        "synthesis_evidence": [
            vars(row) | {"outcome": row.outcome}
            for row in synthesis_evidence],
        "passed": bool(
            acquisition["authorized_families"]
            or synthesis["authorized_families"]),
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "candidate_response_accessed": True,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Development-only familywise deployment calibration; not "
            "confirmation or universal superiority."),
    }
    args.output_dir.mkdir(parents=True)
    _publish(
        args.output_dir / "AISTATS_REALIZED_POLICY_CALIBRATION.json",
        result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
