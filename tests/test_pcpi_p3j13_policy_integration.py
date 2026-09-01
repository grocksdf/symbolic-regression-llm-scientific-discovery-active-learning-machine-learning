"""P3J.13 post-transaction reporting and policy dispatch tests."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3JMeasuredPoolQueryResult,
    P3JMeasuredRunResult,
    OperationalClassConditionalDecision,
    admit_operational_class_conditional_response,
    build_p3j_policy_artifacts,
    dispatch_p3j_matched_policy,
    publish_p3j_policy_failure_snapshot,
    summarize_p3j_policy_artifacts,
)
from hypothesis_mvp.pcpi import p3j_policy_integration, p3j_reporting
from tests.test_pcpi_p3j2_operational_class_conditional import _scores
from tests.test_pcpi_p3j8_run_identity import _case


def _completed_run(tmp_path):
    state, actions, ids, predictive, representative, identity, _ = _case(tmp_path)
    decision = OperationalClassConditionalDecision(
        prior_state_hash=state.stable_hash,
        selected_candidate_id=3,
        selected_action=actions[1],
        local_index=1,
        scores=_scores(state.target_partition.stable_hash),
    )
    next_state = admit_operational_class_conditional_response(
        state, decision, 3, actions[1], 0.31
    )
    query = P3JMeasuredPoolQueryResult(
        decision=decision,
        next_state=next_state,
        revealed_candidate_id=3,
        revealed_action=actions[1],
        revealed_target=0.31,
    )
    run = P3JMeasuredRunResult(
        final_state=next_state,
        decisions=(decision,),
        query_results=(query,),
        identities=(identity,),
        manifest_path=tmp_path / "run" / "RUN_MANIFEST.json",
    )
    return state, actions, ids, predictive, representative, run


def test_reporting_occurs_only_after_admitted_query_state(tmp_path) -> None:
    state, actions, _, predictive, _, run = _completed_run(tmp_path)
    artifacts = build_p3j_policy_artifacts(
        run,
        state,
        predictive,
        actions,
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([f"row-{index}" for index in range(9)]),
        dataset_id="uci_ccpp",
        dataset_family="uci_ccpp",
        seed=2026080701,
        policy="pcpi_representative_safe_robust_class_eig",
        class_distance_threshold=0.2,
    )
    assert len(artifacts.curve_rows) == 2
    assert len(artifacts.query_rows) == 1
    row = artifacts.query_rows[0]
    assert row["p3j_prior_state_hash"] == state.stable_hash
    assert row["p3j_next_state_hash"] == run.final_state.stable_hash
    assert row["response_receipt_admitted_before_reporting"] is True
    assert row["selection_used_validation"] is False
    assert row["selected_row_id"] == "row-3"
    summary = summarize_p3j_policy_artifacts(artifacts, structure_count=7)
    assert summary["pcpi_decision_rule_valid_rate"] == 1.0
    assert summary["pcpi_target_only_class_eig_used_rate"] == 1.0
    assert np.isfinite(summary["normalized_aulc_validation_rmse"])


def test_dispatch_isolates_p3j_state_from_all_baselines(tmp_path) -> None:
    state, *_ = _completed_run(tmp_path)
    events: list[str] = []
    pcpi = dispatch_p3j_matched_policy(
        "pcpi", "pcpi", state,
        lambda: events.append("p3j") or "p3j-result",
        lambda: events.append("legacy") or "legacy-result",
    )
    baseline = dispatch_p3j_matched_policy(
        "random", "pcpi", None,
        lambda: events.append("p3j") or "p3j-result",
        lambda: events.append("legacy") or "legacy-result",
    )
    assert (pcpi, baseline, events) == (
        "p3j-result", "legacy-result", ["p3j", "legacy"]
    )
    with pytest.raises(ValueError, match="cannot enter a matched baseline"):
        dispatch_p3j_matched_policy(
            "qbc", "pcpi", state, lambda: None, lambda: None
        )


def test_failure_snapshot_records_progress_bits_but_no_response_values(tmp_path) -> None:
    root = tmp_path / "formal"
    query = root / "checkpoints" / "uci_ccpp" / "seed-2026080701" / "query-001"
    query.mkdir(parents=True)
    (query / "IDENTITY.json").write_text(
        json.dumps({"identity_hash": "a" * 64}), encoding="utf-8"
    )
    (query / "DECISION.json").write_text("{}", encoding="utf-8")
    path = publish_p3j_policy_failure_snapshot(
        root,
        dataset_id="uci_ccpp",
        seed=2026080701,
        failure_type="FloatingPointError",
        message="ranking unresolved",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["queries"][0]["decision_published"] is True
    assert payload["queries"][0]["reveal_receipt_published"] is False
    assert payload["response_values_recorded"] is False
    with pytest.raises(FileExistsError):
        publish_p3j_policy_failure_snapshot(
            root,
            dataset_id="uci_ccpp",
            seed=2026080701,
            failure_type="RuntimeError",
            message="replacement forbidden",
        )


def test_integration_source_has_no_direct_oracle_or_selection_from_validation() -> None:
    reporting = inspect.getsource(p3j_reporting)
    dispatch = inspect.getsource(p3j_policy_integration.dispatch_p3j_matched_policy)
    assert "oracle.acquire_indices" not in reporting + dispatch
    assert "validation_targets" not in dispatch
    assert dispatch.index("policy == pcpi_policy") < dispatch.index("p3j_runner()")
    assert "response_values_recorded\": False" in inspect.getsource(
        p3j_policy_integration.publish_p3j_policy_failure_snapshot
    )
