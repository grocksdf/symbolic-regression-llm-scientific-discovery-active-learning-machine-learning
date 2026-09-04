"""P3K.6 finite singleton-certificate correctness tests; no experiment data."""

from __future__ import annotations

import json

import numpy as np

from hypothesis_mvp.pcpi import (
    P3K_SINGLETON_RANK_CERTIFICATE,
    build_p3j_formal_query_identity,
    score_operational_class_conditional_candidates,
)
from hypothesis_mvp.pcpi import p3j_query_runner as query_runner
from hypothesis_mvp.pcpi.acquisition import EIGEstimate, _ranking_certificate
from hypothesis_mvp.pcpi.p3j_query_runner import run_p3j_formal_query
from hypothesis_mvp.pcpi.p3j_run_identity import open_p3j_query_workspace
from hypothesis_mvp.pcpi.real_acquisition import _lower_envelope_certificate
from scripts.run_pcpi_p3k6_singleton_rank_certificate_correctness import (
    _algebraic_fixture,
    _evaluate,
)


def test_singleton_lower_envelope_has_a_finite_vacuous_certificate() -> None:
    certified, margin, error, gap = _lower_envelope_certificate(
        np.asarray([0.4, 0.3, 0.2]),
        np.asarray([0.39, 0.29, 0.19]),
        np.asarray([0.41, 0.31, 0.21]),
        np.asarray([False, True, False]),
    )
    assert certified
    assert (margin, error, gap) == (0.0, 0.0, 0.0)
    assert np.all(np.isfinite([margin, error, gap]))


def test_singleton_base_ranking_has_the_same_finite_representation() -> None:
    estimate = EIGEstimate(
        scores=np.asarray([0.4, 0.3, 0.2]),
        error_bounds=np.asarray([0.01, 0.01, 0.01]),
        sample_count=8,
        structure_allocations=(8,),
        coarse_sample_count=4,
        integration_method="algebraic-fixture",
        error_safety_factor=4.0,
    )
    margin, error, gap, certified = _ranking_certificate(
        estimate,
        np.zeros(3),
        np.asarray([False, True, False]),
    )
    assert certified
    assert (margin, error, gap) == (0.0, 0.0, 0.0)
    assert np.all(np.isfinite([margin, error, gap]))


def test_singleton_decision_persists_and_loads_under_strict_json(
    tmp_path, monkeypatch
) -> None:
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
    tmp_path.mkdir(exist_ok=True)
    workspace = open_p3j_query_workspace(tmp_path, identity)
    monkeypatch.setattr(
        query_runner,
        "score_identity_bound_p3j_query",
        lambda *args, **kwargs: decision,
    )
    loaded = run_p3j_formal_query(
        workspace,
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
    payload = json.loads((workspace.query_root / "DECISION.json").read_text())
    assert loaded.scores.ranking_certificate_method == P3K_SINGLETON_RANK_CERTIFICATE
    assert loaded.scores.representative_safe_set_size == 1
    assert loaded.scores.possible_maximizer_count == 1
    assert loaded.local_index == decision.local_index
    assert payload["response_opened"] is False
    assert "Infinity" not in json.dumps(payload)
    assert "NaN" not in json.dumps(payload)
    assert not workspace.terminal_failure_path.exists()


def test_p3k6_no_data_gate_passes() -> None:
    result = _evaluate()
    assert result["status"] == "passed-correctness-no-real-experiment-authorized"
    assert all(result["decisions"].values())
    assert result["real_data_access"] is False
    assert result["simulated_experiment"] is False
    assert result["heldout_access"] is False
