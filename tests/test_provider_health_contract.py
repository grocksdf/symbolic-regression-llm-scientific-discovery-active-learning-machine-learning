"""Transport correctness fixtures; no benchmark or provider requests."""

import pytest

from scripts.provider_health_contract import (
    ProviderValidityError, bounded_messages, require_healthy_generation,
)
from scripts.provider_transport_preflight import classify_transport


def test_prompt_budget_is_a_ceiling_without_padding():
    messages = [{"role": "system", "content": "review"},
                {"role": "user", "content": "α"}]
    unchanged, length = bounded_messages(messages, 65536)
    assert unchanged is messages
    assert length == 2
    assert messages[-1]["content"] == "α"
    with pytest.raises(ProviderValidityError):
        bounded_messages(messages, 1)


def test_transport_failure_cannot_be_counted_as_negative_candidate_evidence():
    with pytest.raises(ProviderValidityError, match="transport"):
        require_healthy_generation([
            {"request_index": 1, "http_status": 429},
            {"request_index": 2, "http_status": 200},
        ])
    assert require_healthy_generation([])["provider_attempted"] is False
    assert require_healthy_generation([
        {"request_index": 1, "http_status": 200},
    ])["transport_valid"] is True
    with pytest.raises(ProviderValidityError, match="ledger"):
        require_healthy_generation([
            {"request_index": 2, "http_status": 200},
        ])


def test_429_does_not_imply_a_rate_limit_without_error_code():
    assert classify_transport(429, "") == "429_cause_unresolved"
    assert classify_transport(429, "insufficient_quota") == "account_quota_or_balance"
    assert classify_transport(429, "rate_limit_exceeded") == "rate_or_concurrency_limit"
