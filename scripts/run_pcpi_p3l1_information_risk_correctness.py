"""Run the no-data P3L.1 information-risk correctness gate."""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3L_INFORMATION_RISK_METHOD,
    P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    weighted_lower_tail_cvar,
)
from hypothesis_mvp.pcpi import (
    class_conditional_semiparametric as joint_source,
    operational_class_conditional as lifecycle_source,
    real_acquisition as acquisition_source,
)


CONFIG = PROJECT_ROOT / "configs" / "p3l_1_information_risk_correctness.json"
CONFIG_SHA256 = "c6add8a54a20855d2952b61f6495b84ff45d20858142596fd7e2491c77ee298a"


def _load_config(path: Path = CONFIG) -> dict[str, object]:
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3L.1 correctness config hash changed")
    config = json.loads(raw.decode("utf-8"))
    expected = {
        "schema": "pcpi-p3l1-information-risk-correctness-config-v1",
        "stage": "P3L.1",
        "tail_probability": P3L_INFORMATION_RISK_TAIL_PROBABILITY,
        "heldout_state": "closed",
        "real_data_access_authorized": False,
        "simulated_experiment_authorized": False,
        "operational_execution_authorized": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("P3L.1 correctness contract changed")
    return config


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    alpha = float(config["tail_probability"])
    rewards = np.asarray([-2.0, -1.0, 3.0])
    masses = np.asarray([0.1, 0.3, 0.6])
    cvar = weighted_lower_tail_cvar(rewards, masses, alpha)
    expected_cvar = (-2.0 * 0.1 - 1.0 * 0.15) / alpha

    certificate_rewards = np.asarray([-0.1, 0.5])
    certificate_masses = np.asarray([0.2, 0.8])
    certificate_cvar = weighted_lower_tail_cvar(
        certificate_rewards, certificate_masses, alpha
    )
    certificate_negative_mass = float(np.sum(
        certificate_masses[certificate_rewards < 0.0]
    ))

    chunk_source = inspect.getsource(
        joint_source._class_conditional_information_risk_chunk
    )
    scorer_source = inspect.getsource(
        acquisition_source._estimate_class_conditional_information_risk_until_ranked
    )
    lifecycle = inspect.getsource(
        lifecycle_source.score_operational_class_conditional_candidates
    )
    forbidden = (
        "candidate_targets", "validation_targets", "heldout_targets",
        "PoolOracle", "acquire_indices", "default_rng", "random.",
    )
    decisions = {
        "tail_probability_is_frozen_assessment_constant": alpha == 0.25,
        "weighted_boundary_atom_is_split_exactly": bool(np.isclose(
            cvar, expected_cvar, rtol=0.0, atol=1e-15
        )),
        "nonnegative_cvar_certificate_bounds_negative_mass": bool(
            certificate_cvar >= 0.0 and certificate_negative_mass <= alpha
        ),
        "entropy_gain_is_computed_before_any_response_reveal": (
            "prior_entropy - posterior_entropy" in chunk_source
            and not any(token in chunk_source for token in forbidden)
        ),
        "selection_is_maximin_over_complete_model_family": (
            "np.min(risk_by_model, axis=0)" in scorer_source
            and "len(states)" in scorer_source
        ),
        "mean_eig_is_not_the_selection_score": (
            "scores = np.min(risk_by_model, axis=0)" in scorer_source
            and "information_by_model" in scorer_source
        ),
        "operational_scorer_has_no_response_surface": not any(
            token in lifecycle for token in forbidden
        ),
        "risk_method_identity_is_explicit": (
            P3L_INFORMATION_RISK_METHOD
            == "lower-tail-cvar-of-frozen-class-entropy-reduction-v1"
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3L.1 information-risk correctness gate failed")
    return {
        "schema": "pcpi-p3l1-information-risk-correctness-gate-v1",
        "stage": "P3L.1",
        "status": "passed-correctness-no-real-experiment-authorized",
        "role": "information-risk-theory-leakage-and-source-composition-gate",
        "decisions": decisions,
        "tail_probability": alpha,
        "boundary_atom_cvar": cvar,
        "certificate_negative_mass": certificate_negative_mass,
        "formal_experiment": False,
        "real_data_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "operational_execution_authorized": False,
    }


def main() -> int:
    print(json.dumps(_evaluate(_load_config()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
