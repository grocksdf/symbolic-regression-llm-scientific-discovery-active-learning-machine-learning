"""Prospective provider transport and prompt-budget validity checks.

Neither a 429 nor a zero-response provider run is evidence about equation
generation. A new protocol must stop before candidate admission or truth.
"""
from __future__ import annotations

from collections import Counter


class ProviderValidityError(RuntimeError):
    pass


def bounded_messages(messages, maximum_user_prompt_bytes: int):
    """Check the ceiling; never pad a request to the ceiling."""
    if (type(maximum_user_prompt_bytes) is not int
            or maximum_user_prompt_bytes < 1 or not messages):
        raise ValueError("invalid prompt budget")
    actual = len(str(messages[-1].get("content") or "").encode("utf-8"))
    if actual > maximum_user_prompt_bytes:
        raise ProviderValidityError("user-prompt-byte-ceiling-exceeded")
    return messages, actual


def require_healthy_generation(provider_cost):
    """Fail on incomplete/failed transport before opening admission responses."""
    if not isinstance(provider_cost, list):
        raise ProviderValidityError("provider-cost-ledger-invalid")
    if not provider_cost:
        return {"request_count": 0, "status_counts": {},
                "provider_attempted": False, "transport_valid": True}
    counts = Counter()
    for index, row in enumerate(provider_cost, 1):
        if (not isinstance(row, dict) or row.get("request_index") != index
                or type(row.get("http_status")) is not int):
            raise ProviderValidityError("provider-cost-ledger-invalid")
        counts[row["http_status"]] += 1
    if any(not 200 <= status < 300 for status in counts):
        raise ProviderValidityError("provider-transport-failed")
    return {"request_count": len(provider_cost),
            "status_counts": {str(key): value for key, value in sorted(counts.items())},
            "provider_attempted": True,
            "transport_valid": True}
