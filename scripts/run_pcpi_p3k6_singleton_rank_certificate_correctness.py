"""Run the no-data P3K.6 singleton rank-certificate correctness Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3H_OPERATIONAL_POWERS,
    P3K_SINGLETON_RANK_CERTIFICATE,
    ClassPartition,
    build_p3j_formal_query_identity,
    initialize_operational_class_conditional_state,
    score_operational_class_conditional_candidates,
)
from hypothesis_mvp.pcpi.p3j_query_runner import _decision_payload
from hypothesis_mvp.pcpi.p3j_run_identity import P3JQueryWorkspace
from hypothesis_mvp.pcpi.reference import SequentialReferencePosterior
from hypothesis_mvp.pcpi.reference.models import (
    NormalInverseGammaPrior,
    ReferenceBank,
    ReferenceStructure,
)


CONFIG = (
    PROJECT_ROOT / "configs" / "p3k_6_singleton_rank_certificate_correctness.json"
)


def _algebraic_fixture():
    identifiers = (
        ("constant", "b0", ("1",)),
        ("linear", "b0 + b1*x", ("1", "x")),
        ("linear_alias", "b0 + b1*x_alias", ("1", "x_alias")),
        ("quadratic", "b0 + b1*x + b2*x^2", ("1", "x", "x2")),
        ("cubic", "b0 + b1*x + b2*x^2 + b3*x^3", ("1", "x", "x2", "x3")),
        ("sinusoid", "b0 + b1*sin(x)", ("1", "sin_x")),
        ("reciprocal", "b0 + b1/(1+x^2)", ("1", "reciprocal_1_plus_x2")),
    )
    bank = ReferenceBank(
        tuple(
            ReferenceStructure(name, expression, terms, 1.0 / len(identifiers))
            for name, expression, terms in identifiers
        ),
        NormalInverseGammaPrior(0.0, 0.5, 3.0, 0.05),
    )
    actions = np.linspace(-1.5, 1.5, 12)[:, None]
    targets = 0.4 - 0.8 * actions[:, 0] + 0.3 * np.square(actions[:, 0])
    engines = tuple(
        SequentialReferencePosterior(bank, power)
        for power in P3H_OPERATIONAL_POWERS
    )
    nominal = engines[-1].fit_batch(actions[:8], targets[:8])
    groups = ((0, 1, 2), tuple(range(3, len(nominal.members))))
    partition = ClassPartition(
        class_ids=("low-order", "higher-order"),
        member_indices=groups,
        class_probabilities=tuple(
            sum(nominal.members[index].probability for index in group)
            for group in groups
        ),
        structure_to_class=(0, 0, 0, 1, 1, 1, 1),
    )
    state = initialize_operational_class_conditional_state(
        engines,
        actions[:4],
        targets[:4],
        actions[4:8],
        targets[4:8],
        partition,
    )
    candidates = np.asarray([[2.0], [3.0], [4.0]])
    candidate_ids = np.asarray([8, 3, 5])
    return state, candidates, candidate_ids


def _evaluate(path: Path = CONFIG) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    state, candidates, candidate_ids = _algebraic_fixture()
    decision = score_operational_class_conditional_candidates(
        state,
        candidates,
        candidate_ids,
        candidates,
        candidates,
        eig_min_samples=8,
        eig_max_samples=8,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
    )
    scores = decision.scores
    identity = build_p3j_formal_query_identity(
        source_git_tree="0" * 40,
        config_sha256="1" * 64,
        dataset_id="algebraic_fixture",
        seed=0,
        query_index=1,
        candidate_ids=candidate_ids,
        candidate_actions=candidates,
        predictive_target_actions=candidates,
        representative_observed_actions=candidates,
        operational_state=state,
    )
    workspace = P3JQueryWorkspace(
        Path("."), Path("."), Path("."), Path("."), identity
    )
    payload = _decision_payload(workspace, decision)
    strict_json = json.dumps(payload, allow_nan=False, sort_keys=True)
    frozen = config["finite_representation"]
    decisions = {
        "projected_admissible_set_is_singleton": (
            scores.representative_safe_set_size == 1
            and np.count_nonzero(scores.representative_safe_mask) == 1
        ),
        "sole_admissible_candidate_is_selected": bool(
            scores.representative_safe_mask[decision.local_index]
            and decision.selected_candidate_id == candidate_ids[decision.local_index]
        ),
        "singleton_certificate_has_distinct_identity": (
            scores.ranking_certificate_method
            == config["method"]
            == P3K_SINGLETON_RANK_CERTIFICATE
        ),
        "vacuous_certificate_is_finite_and_neutral": (
            scores.ranking_certified is frozen["ranking_certified"]
            and scores.ranking_margin == frozen["ranking_margin"]
            and scores.ranking_error_bound == frozen["ranking_error_bound"]
            and scores.ranking_certificate_gap == frozen["ranking_certificate_gap"]
            and scores.possible_maximizer_count
            == frozen["possible_maximizer_count"]
            and np.all(
                np.isfinite(
                    [
                        scores.ranking_margin,
                        scores.ranking_error_bound,
                        scores.ranking_certificate_gap,
                    ]
                )
            )
        ),
        "strict_json_round_trip_succeeds": (
            json.loads(strict_json)["scores"]["ranking_certificate_method"]
            == P3K_SINGLETON_RANK_CERTIFICATE
        ),
        "selection_rule_is_unchanged": config["selection_rule_change"] is False,
        "no_experiment_or_response_surface_is_authorized": not any(
            config[key]
            for key in (
                "operational_execution_authorized",
                "real_data_access",
                "response_access",
                "validation_access",
                "heldout_access",
                "simulated_experiment",
            )
        ),
    }
    decisions = {key: bool(value) for key, value in decisions.items()}
    if not all(decisions.values()):
        raise AssertionError("P3K.6 singleton rank-certificate Gate failed")
    return {
        "schema": "pcpi-p3k6-singleton-rank-certificate-correctness-result-v1",
        "stage": "P3K.6",
        "status": "passed-correctness-no-real-experiment-authorized",
        "decisions": decisions,
        "selected_candidate_id": decision.selected_candidate_id,
        "ranking_certificate_method": scores.ranking_certificate_method,
        "ranking_margin": scores.ranking_margin,
        "ranking_error_bound": scores.ranking_error_bound,
        "ranking_certificate_gap": scores.ranking_certificate_gap,
        "real_data_access": False,
        "simulated_experiment": False,
        "heldout_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(_evaluate(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
