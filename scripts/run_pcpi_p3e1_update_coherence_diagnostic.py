"""Run the P3E.1 exact update-coherence correctness gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hypothesis_mvp.hypotheses import file_sha256, production_code_hash
from hypothesis_mvp.pcpi import REFERENCE_FALLBACK_MODE
from hypothesis_mvp.pcpi.reference.update_coherence import (
    UPDATE_COHERENT_REFERENCE_DOMINANCE_METHOD,
    UPDATE_COHERENT_TARGETED_HANDOVER_MODE,
    certified_update_coherent_reference_dominance,
    UPDATE_COHERENCE_FIXTURE_ROLE,
    exact_update_coherent_utility,
    generalized_update_ranking_reversal_fixture,
)


STAGE = "P3E.1"
EXPERIMENT = "generalized_posterior_update_coherence_correctness"
FIXTURE_ROLE = UPDATE_COHERENCE_FIXTURE_ROLE
CLAIM_BOUNDARY = (
    "This exact finite fixture checks loss/update alignment for generalized "
    "posterior decisions and a positive, registered-reference dominance rule. "
    "It is correctness evidence only: it does not validate posterior adequacy, "
    "real-data efficacy, held-out performance, or scientific discovery."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3E.1 config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "pcpi-p3e1-update-coherence-diagnostic-config-v1",
        "stage": STAGE,
        "fixture_role": FIXTURE_ROLE,
        "likelihood_power": 0.25,
        "reference_seed": 20260814,
        "interval_radius": 1e-10,
        "identity_tolerance": 1e-13,
        "heldout_state": "not-applicable",
    }
    if config != expected:
        raise ValueError("P3E.1 correctness contract was modified")
    return config


def _evaluate(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    structures, likelihoods, mapping, identifiers, reference = (
        generalized_update_ranking_reversal_fixture()
    )
    eta = float(config["likelihood_power"])
    result = exact_update_coherent_utility(
        structures, likelihoods, mapping, likelihood_power=eta
    )
    bayes = exact_update_coherent_utility(
        structures, likelihoods, mapping, likelihood_power=1.0
    )
    scores = result.update_coherent_entropy_reduction
    radius = float(config["interval_radius"])
    decision = certified_update_coherent_reference_dominance(
        scores,
        scores - radius,
        scores + radius,
        reference,
        identifiers,
        reference_seed=int(config["reference_seed"]),
    )
    negative = np.asarray([-0.10, -0.20])
    negative_decision = certified_update_coherent_reference_dominance(
        negative,
        negative - radius,
        negative + radius,
        reference,
        identifiers,
        reference_seed=int(config["reference_seed"]),
    )
    identity_tolerance = float(config["identity_tolerance"])
    direct = np.sum(
        result.outcome_probabilities * result.realized_entropy_reduction, axis=1
    )
    decisions = {
        "eta_one_recovers_ordinary_class_mi": bool(
            np.max(np.abs(
                bayes.ordinary_class_mi
                - bayes.update_coherent_entropy_reduction
            )) <= identity_tolerance
        ),
        "eta_quarter_differs_from_ordinary_class_mi": bool(
            np.max(np.abs(result.ordinary_class_mi - scores)) > 1e-4
        ),
        "ordinary_mi_selects_fixture_action_202": bool(
            identifiers[int(np.argmax(result.ordinary_class_mi))] == 202
        ),
        "aligned_utility_selects_fixture_action_101": bool(
            identifiers[int(np.argmax(scores))] == 101
        ),
        "expected_utility_matches_realized_direct_sum": bool(
            np.max(np.abs(scores - direct)) <= identity_tolerance
        ),
        "aligned_decision_targets_action_101": bool(
            decision.targeted_handover
            and decision.selected_candidate_id == 101
            and decision.utility_mode == UPDATE_COHERENT_TARGETED_HANDOVER_MODE
        ),
        "aligned_decision_uses_registered_method": bool(
            decision.method == UPDATE_COHERENT_REFERENCE_DOMINANCE_METHOD
        ),
        "targeted_lower_bound_exceeds_reference_and_zero": bool(
            decision.leader_lower_bound
            > max(0.0, decision.reference_upper_bound)
            + decision.numerical_tolerance
        ),
        "negative_utility_returns_to_reference": bool(
            not negative_decision.targeted_handover
            and negative_decision.utility_mode == REFERENCE_FALLBACK_MODE
            and negative_decision.selected_candidate_id
            == negative_decision.reference_sample_candidate_id
        ),
        "heldout_remains_closed": config["heldout_state"] == "not-applicable",
    }
    rows = [
        {
            "candidate_id": int(identifiers[position]),
            "ordinary_class_mi": float(result.ordinary_class_mi[position]),
            "update_coherent_entropy_reduction": float(scores[position]),
            "ordinary_mi_rank": int(
                np.where(np.argsort(-result.ordinary_class_mi) == position)[0][0] + 1
            ),
            "update_coherent_rank": int(
                np.where(np.argsort(-scores) == position)[0][0] + 1
            ),
        }
        for position in range(len(identifiers))
    ]
    summary = {
        "schema": "pcpi-p3e1-update-coherence-summary-v1",
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "claim_boundary": CLAIM_BOUNDARY,
        "likelihood_power": eta,
        "decision": asdict(decision),
        "negative_control_decision": asdict(negative_decision),
        "gate_decisions": decisions,
        "gate_decision_count": len(decisions),
        "gate_passed": all(decisions.values()),
        "formal_correctness_evidence": all(decisions.values()),
        "formal_efficacy_evidence": False,
        "heldout_opened": False,
    }
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3e_1_update_coherence_diagnostic.json"),
    )
    parser.add_argument("--phase", default=STAGE)
    parser.add_argument("--heldout-state", default="not-applicable")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("P3E.1 is correctness-only and has no held-out state")
    config = _load_config(args.config.resolve(), root)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = _evaluate(config)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_identity"] = {
        "production_code_hash": production_code_hash(root),
        "config_sha256": file_sha256(args.config.resolve()),
    }
    (output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    with (output / "action_utilities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(_canonical_json({
        "stage": STAGE,
        "gate_passed": summary["gate_passed"],
        "output_dir": str(output),
        "formal_efficacy_evidence": False,
    }), end="")
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
