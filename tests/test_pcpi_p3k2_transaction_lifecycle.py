"""P3K.2 exact stage identity and transaction recovery tests; no data access."""

from __future__ import annotations

import json

from hypothesis_mvp.pcpi import (
    P3K_QUERY_DECISION_SCHEMA,
    P3K_QUERY_LEDGER_SCHEMA,
    P3K_REVEAL_RECEIPT_SCHEMA,
    P3K_RUN_IDENTITY_SCHEMA,
    P3K_RUN_MANIFEST_SCHEMA,
    admit_p3j_formal_response,
    finalize_p3j_run_manifest,
    resume_or_run_p3j_measured_pool_query,
)
from tests.test_pcpi_p3j12_measured_run import _ForbiddenOracle, _standardizer
from tests.test_pcpi_p3j10_reveal_runner import _publish_decision


def test_p3k_transaction_artifacts_never_claim_a_p3j_schema(tmp_path, monkeypatch) -> None:
    state, actions, ids, predictive, representative, identity, workspace = (
        _publish_decision(tmp_path, monkeypatch)
    )
    assert identity.schema == P3K_RUN_IDENTITY_SCHEMA
    identity_payload = json.loads((workspace.query_root / "IDENTITY.json").read_text())
    decision_payload = json.loads((workspace.query_root / "DECISION.json").read_text())
    assert identity_payload["schema"] == P3K_RUN_IDENTITY_SCHEMA
    assert decision_payload["schema"] == P3K_QUERY_DECISION_SCHEMA

    next_state = admit_p3j_formal_response(
        workspace, state, actions, ids, 3, actions[1], 0.31
    )
    receipt = json.loads((workspace.query_root / "REVEAL_RECEIPT.json").read_text())
    ledger = json.loads((workspace.query_root / "QUERY_LEDGER.json").read_text())
    assert receipt["schema"] == P3K_REVEAL_RECEIPT_SCHEMA
    assert ledger["schema"] == P3K_QUERY_LEDGER_SCHEMA
    assert ledger["prior_state_hash"] == state.stable_hash
    assert ledger["next_state_hash"] == next_state.stable_hash

    manifest = finalize_p3j_run_manifest(tmp_path / "run", (identity,))
    assert json.loads(manifest.read_text())["schema"] == P3K_RUN_MANIFEST_SCHEMA


def test_p3k_durable_recovery_does_not_reopen_the_oracle(tmp_path, monkeypatch) -> None:
    state, actions, ids, predictive, representative, _, workspace = (
        _publish_decision(tmp_path, monkeypatch)
    )
    expected = admit_p3j_formal_response(
        workspace, state, actions, ids, 3, actions[1], 0.31
    )
    recovered = resume_or_run_p3j_measured_pool_query(
        workspace, state, actions, ids, predictive, representative,
        _ForbiddenOracle(), _standardizer(), eig_min_samples=16,
        eig_max_samples=16, eig_error_safety_factor=4.0, eig_growth_factor=2,
    )
    assert recovered.next_state.stable_hash == expected.stable_hash
    assert recovered.revealed_candidate_id == 3
