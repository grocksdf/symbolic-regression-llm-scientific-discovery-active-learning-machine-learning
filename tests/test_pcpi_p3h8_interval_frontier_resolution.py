"""P3H.8 correctness and P3H.9 formal interval-resolution freeze."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3H_INTERVAL_FRONTIER_RESOLUTION,
    P3H_TERMINAL_ABSTENTION,
)
import hypothesis_mvp.pcpi.real_acquisition as acquisition
import hypothesis_mvp.pcpi.semiparametric_acquisition as semiparametric
from scripts import run_pcpi_p3b_real as shared_runner
from scripts.run_pcpi_p3h7_semiparametric_real_acquisition import P3H7_PROTOCOL
from scripts.run_pcpi_p3h9_interval_resolved_real_acquisition import (
    P3H9_PROTOCOL,
    RUNTIME_HASH,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3h_9_interval_resolved_real_acquisition.json"


def test_p3h7_history_is_not_retroactively_rewritten() -> None:
    assert P3H7_PROTOCOL.semiparametric_unresolved_action == P3H_TERMINAL_ABSTENTION
    old = shared_runner._load_config(
        ROOT / "configs" / "p3h_7_semiparametric_real_acquisition.json",
        ROOT,
        P3H7_PROTOCOL,
    )
    assert old["pcpi_uncertified_eig_action"] == P3H_TERMINAL_ABSTENTION


def test_p3h9_freezes_total_interval_frontier_rule_without_budget_changes() -> None:
    config = shared_runner._load_config(CONFIG, ROOT, P3H9_PROTOCOL)
    assert P3H9_PROTOCOL.semiparametric_unresolved_action == (
        P3H_INTERVAL_FRONTIER_RESOLUTION
    )
    assert config["pcpi_uncertified_eig_action"] == (
        P3H_INTERVAL_FRONTIER_RESOLUTION
    )
    assert config["eig_quadrature_min_evaluations"] == 32
    assert config["eig_quadrature_max_evaluations"] == 512
    assert config["likelihood_power_candidates"] == [0.125, 0.25, 0.5, 1.0]
    assert config["runtime_dependency_hash"] == RUNTIME_HASH


def test_possible_maximizer_rule_is_permutation_equivariant() -> None:
    lower = np.asarray([0.62, 0.55, 0.10, 0.60])
    upper = np.asarray([0.70, 0.68, 0.40, 0.71])
    eligible = np.asarray([True, True, True, True])
    expected = acquisition._lower_envelope_possible_maximizers(
        lower, upper, eligible
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted = acquisition._lower_envelope_possible_maximizers(
        lower[permutation], upper[permutation], eligible[permutation]
    )
    restored = np.empty_like(permuted)
    restored[permutation] = permuted
    np.testing.assert_array_equal(restored, expected)


def test_resolution_source_has_no_response_validation_rng_or_dataset_branch() -> None:
    source = "\n".join((
        inspect.getsource(acquisition._lower_envelope_possible_maximizers),
        inspect.getsource(acquisition.select_acquisition_candidate),
    )).lower()
    for forbidden in (
        "candidate_targets", "validation", "heldout", "dataset_id",
        "np.random", "default_rng", "posterior_epistemic_variance",
    ):
        assert forbidden not in source


def test_nested_refinement_source_reuses_preceding_scores_once() -> None:
    source = inspect.getsource(semiparametric.refine_semiparametric_class_eig)
    assert "coarse_estimate.scores" in source
    assert source.count("_semiparametric_scores(") == 1
    assert "estimate_class_eig" not in source
    assert "exact_class_eig" not in source


def test_shared_runner_dispatches_protocol_rule_and_explicit_selector() -> None:
    source = inspect.getsource(shared_runner._run_policy)
    assert 'score_kwargs["semiparametric_unresolved_action"]' in source
    assert "protocol.semiparametric_unresolved_action" in source
    assert "select_acquisition_candidate(scores, available)" in source
    assert "eig_possible_maximizer_candidate_ids" in source
    assert "eig_selection_admissible_candidate_ids" in source


def test_p3h9_cli_is_real_only_and_heldout_closed() -> None:
    parser = build_parser(P3H9_PROTOCOL)
    assert parser._option_string_actions["--phase"].choices == ("P3H.9",)
    assert parser._option_string_actions["--heldout-state"].choices == ("closed",)
    assert "failure-informed" in P3H9_PROTOCOL.claim_boundary
    assert "not independent confirmation" in P3H9_PROTOCOL.claim_boundary


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("pcpi_uncertified_eig_action", P3H_TERMINAL_ABSTENTION),
        ("eig_quadrature_max_evaluations", 1024),
        ("p3h_validation_state_policy", "reuse-validation"),
    ),
)
def test_p3h9_contract_tampering_fails_closed(
    tmp_path: Path, key: str, value: object
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[key] = value
    candidate = tmp_path / "tampered.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        shared_runner._load_config(candidate, tmp_path, P3H9_PROTOCOL)
