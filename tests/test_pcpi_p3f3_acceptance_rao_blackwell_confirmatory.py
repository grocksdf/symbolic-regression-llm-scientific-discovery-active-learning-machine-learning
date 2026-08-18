"""Static freeze and evidence-boundary tests for P3F.3-CF.RB.1."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/p3f_3_open_target_particle_acceptance_rao_blackwell_confirmatory.json"
)
RUNNER_PATH = (
    ROOT / "scripts/run_pcpi_p3f3_particle_acceptance_rao_blackwell_confirmatory.py"
)
FIRST_CONFIRMATORY_PATH = (
    ROOT / "configs/p3f_3_open_target_particle_confirmatory_fidelity_audit.json"
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_confirmatory_bank_and_seeds_are_frozen_and_unseen() -> None:
    config = _read(CONFIG_PATH)
    prior_paths = (
        ROOT / "configs/p3f_3_open_target_particle_confirmatory_fidelity_audit.json",
        ROOT / "configs/p3f_3_open_target_particle_variance_reduction_development.json",
        ROOT / "configs/p3f_3_open_target_particle_variance_reduction_strict_development.json",
        ROOT / "configs/p3f_3_open_target_particle_variance_reduction_full_population_development.json",
        ROOT / "configs/p3f_3_open_target_particle_variance_reduction_observation_terminal_development.json",
        ROOT / "configs/p3f_3_open_target_particle_acceptance_rao_blackwell_development.json",
    )
    assert config["stage"] == "P3F.3-CF.RB.1"
    assert len(config["fixtures"]) == 4
    assert len(config["seeds"]) == 5
    fixture_ids = {item["fixture_id"] for item in config["fixtures"]}
    seeds = set(config["seeds"])
    for path in prior_paths:
        prior = _read(path)
        assert seeds.isdisjoint(prior["seeds"])
        assert fixture_ids.isdisjoint(
            {item["fixture_id"] for item in prior["fixtures"]}
        )
    assert all(item["response_free_registration"] for item in config["fixtures"])
    freeze = config["freeze_state"]
    assert freeze["fixtures_frozen_before_first_confirmatory_response"] is True
    assert freeze["seeds_frozen_before_first_confirmatory_response"] is True
    assert freeze["envelopes_frozen_before_first_confirmatory_response"] is True
    assert freeze["result_adaptation"] == "forbidden"


def test_absolute_envelope_is_inherited_without_development_fitting() -> None:
    config = _read(CONFIG_PATH)
    first = _read(FIRST_CONFIRMATORY_PATH)
    inherited = config["absolute_fidelity_envelope"]
    assert inherited["candidate_worst_case_error_max"] == first[
        "fidelity_envelope"
    ]["worst_case_error_max"]
    assert inherited["candidate_cross_fixture_seed_median_span_max"] == first[
        "fidelity_envelope"
    ]["cross_fixture_seed_median_span_max"]
    assert inherited["candidate_worst_case_lower_bounds"] == first[
        "fidelity_envelope"
    ]["worst_case_lower_bounds"]
    assert inherited["candidate_worst_case_upper_bounds"] == first[
        "fidelity_envelope"
    ]["worst_case_upper_bounds"]
    assert config["development_authorization"][
        "mechanism_eligible_for_new_confirmatory_freeze"
    ] is True


def test_confirmatory_runner_gates_worst_case_and_fixture_stability() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    required = (
        "formal_fidelity_evidence",
        "confirmatory_fidelity_gate_passed",
        "candidate_cross_fixture_seed_median_spans",
        "paired_fixture_median_improvements",
        "log_evidence_exact_reference_signed_error",
        "density_signed_error",
        "cdf_signed_error",
        "maximum_ancestry_log_attrition_per_resampling_event",
        "maximum_root_entropy_loss_per_resampling_event",
        "terminal_distinct_root_ancestor_fraction",
        "terminal_normalized_root_entropy",
        "paired_resident_paths_identical",
        '"predictive_calibration": (',
        '"real_data": "blocked"',
        '"acquisition": "blocked"',
        '"heldout": "blocked"',
    )
    assert all(token in source for token in required)
    forbidden = (
        "run_pcpi_p3b_real",
        "run_pcpi_p3e3_real",
        "load_real_dataset",
        "acquisition_policy_loop",
    )
    assert all(token not in source for token in forbidden)
