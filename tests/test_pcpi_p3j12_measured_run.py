"""P3J.12 durable reveal recovery and measured-run tests; no data access."""

from __future__ import annotations

import inspect
import json
import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3JMeasuredPoolQueryResult,
    OperationalClassConditionalDecision,
    admit_p3j_formal_response,
    resume_or_run_p3j_measured_pool_query,
    run_p3j_measured_pool_acquisition,
)
from hypothesis_mvp.pcpi import p3j_measured_run
from hypothesis_mvp.pcpi.reference import DevelopmentStandardizer
from tests.test_pcpi_p3j2_operational_class_conditional import _scores
from tests.test_pcpi_p3j8_run_identity import CONFIG, TREE, _case
from tests.test_pcpi_p3j10_reveal_runner import _publish_decision


class _ForbiddenOracle:
    def acquire_indices(self, indices):
        raise AssertionError("durable reveal recovery reopened the oracle")


def _standardizer() -> DevelopmentStandardizer:
    return DevelopmentStandardizer(
        feature_mean=np.asarray([0.0]),
        feature_scale=np.asarray([1.0]),
        target_mean=0.0,
        target_scale=1.0,
    )


def test_durable_reveal_resumes_without_reopening_oracle(
    tmp_path, monkeypatch
) -> None:
    state, actions, ids, predictive, representative, _, workspace = (
        _publish_decision(tmp_path, monkeypatch)
    )
    expected = admit_p3j_formal_response(
        workspace, state, actions, ids, 3, actions[1], 0.31
    )
    resumed = resume_or_run_p3j_measured_pool_query(
        workspace,
        state,
        actions,
        ids,
        predictive,
        representative,
        _ForbiddenOracle(),
        _standardizer(),
        eig_min_samples=16,
        eig_max_samples=16,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
    )
    assert resumed.next_state.stable_hash == expected.stable_hash
    assert resumed.revealed_candidate_id == 3
    assert resumed.revealed_target == 0.31


def test_tampered_recovery_payload_fails_before_oracle(
    tmp_path, monkeypatch
) -> None:
    state, actions, ids, predictive, representative, _, workspace = (
        _publish_decision(tmp_path, monkeypatch)
    )
    admit_p3j_formal_response(
        workspace, state, actions, ids, 3, actions[1], 0.31
    )
    receipt_path = workspace.query_root / "REVEAL_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["target"] = "0.31"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="payload is invalid"):
        resume_or_run_p3j_measured_pool_query(
            workspace,
            state,
            actions,
            ids,
            predictive,
            representative,
            _ForbiddenOracle(),
            _standardizer(),
            eig_min_samples=16,
            eig_max_samples=16,
            eig_error_safety_factor=4.0,
            eig_growth_factor=2,
        )


def test_complete_run_shrinks_domain_and_extends_only_opened_history(
    tmp_path, monkeypatch
) -> None:
    state, actions, ids, predictive, representative, _, _ = _case(tmp_path)
    run_root = tmp_path / "coordinator"
    run_root.mkdir()
    seen: list[tuple[list[int], int]] = []

    def execute(workspace, current, visible, visible_ids, target, observed, *args, **kwargs):
        selected = int(visible_ids[0])
        seen.append((visible_ids.tolist(), len(observed)))
        decision = OperationalClassConditionalDecision(
            prior_state_hash=current.stable_hash,
            selected_candidate_id=selected,
            selected_action=visible[0],
            local_index=0,
            scores=_scores(current.target_partition.stable_hash),
        )
        return P3JMeasuredPoolQueryResult(
            decision=decision,
            next_state=current,
            revealed_candidate_id=selected,
            revealed_action=visible[0],
            revealed_target=0.2,
        )

    manifest = run_root / "RUN_MANIFEST.json"
    monkeypatch.setattr(
        p3j_measured_run, "resume_or_run_p3j_measured_pool_query", execute
    )
    monkeypatch.setattr(
        p3j_measured_run,
        "finalize_p3j_run_manifest",
        lambda root, identities: manifest,
    )
    result = run_p3j_measured_pool_acquisition(
        run_root,
        state,
        actions,
        ids,
        predictive,
        representative,
        _ForbiddenOracle(),
        _standardizer(),
        source_git_tree=TREE,
        config_sha256=CONFIG,
        dataset_id="uci_ccpp",
        seed=2026080701,
        acquisition_budget=2,
        eig_min_samples=16,
        eig_max_samples=16,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
    )
    assert seen == [([8, 3, 5], 2), ([3, 5], 3)]
    assert [item.query_index for item in result.identities] == [1, 2]
    assert result.identities[0].stable_hash != result.identities[1].stable_hash
    assert result.manifest_path == manifest


def test_measured_run_source_has_no_direct_response_or_evaluation_access() -> None:
    source = inspect.getsource(p3j_measured_run.run_p3j_measured_pool_acquisition)
    assert "resume_or_run_p3j_measured_pool_query" in source
    assert "finalize_p3j_run_manifest" in source
    assert "oracle.acquire_indices" not in source
    assert not any(token in source for token in (
        "validation", "heldout", "candidate_targets", "future_response", "rng"
    ))
